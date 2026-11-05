#include "engine/backends/cuda/executor/execution.h"
#include "engine/models/session.h"
#include "engine/backends/cuda/native/scope.h"
#include "engine/weights/loader.h"
#include "engine/weights/routed_cache.h"
#include "engine/backends/cuda/ops/ffn/ffn.h"
#include "engine/backends/cuda/executor/tq_moe.h"
#include "dspark.h"
#include "engine/models/deepseek_v41/policy.h"

#include "engine/models/common/causal_model_ops.h"
#include "engine/models/common/mtp.h"
#include "ops.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace tyloi::cuda::deepseek_v41_runtime {

static std::pair<DenseTensor, DenseTensor>
dspark_full_attention_plan(std::int64_t batch, std::int64_t queries, std::int64_t keys,
                           const ::tyloi::Device &device) {
    TYLOINFER_RUNTIME_CHECK(batch > 0 && queries > 0 && keys > 0,
                      "invalid DeepSeek-V4.1 DSpark attention plan geometry");
    const auto selected = ((keys + 31) / 32) * 32;
    auto integer =
        ::tyloi::TensorOptions().device(device).dtype(::tyloi::kInt64);
    auto positions = ::tyloi::cuda::arange(selected, integer);
    auto valid = positions < static_cast<double>(keys);
    auto indices = ::tyloi::cuda::where(valid, positions, 0.0)
                       .to(::tyloi::kInt32)
                       .reshape({1, 1, selected})
                       .expand({batch, queries, selected})
                       .contiguous();
    auto mask =
        ::tyloi::cuda::where(
            valid,
            ::tyloi::cuda::zeros({selected}, integer.dtype(::tyloi::kFloat16)),
            ::tyloi::cuda::full({selected}, -std::numeric_limits<float>::infinity(),
                                     integer.dtype(::tyloi::kFloat16)))
            .reshape({1, 1, selected})
            .expand({batch, queries, selected})
            .contiguous();
    return {std::move(indices), std::move(mask)};
}

struct DeepseekV41Dspark final : tyloi::models::Mtp {
    CudaExecutionContext *execution = nullptr;
    std::shared_ptr<const ::tyloi::Tensor> embedding, output;
    double softcap = 0;
    int primary = 0;
    CommonConfig config;
    std::shared_ptr<const ::tyloi::Tensor> main_projection;
    DenseTensor main_norm;
    DenseTensor output_norm;
    std::shared_ptr<const ::tyloi::Tensor> markov_embedding;
    std::shared_ptr<const ::tyloi::Tensor> markov_output;
    std::shared_ptr<const ::tyloi::Tensor> confidence_projection;
    std::vector<std::unique_ptr<Block>> stages;
    std::vector<DenseTensor> rings;
    DenseTensor attention_meta;
    std::int64_t maximum_context = 0;
    std::int64_t batch = 0;
    std::int64_t position = 0;

    static std::unique_ptr<Block> load_stage(Pipelines& pipelines,
                                             const tyloi::weights::Loader& weights,
                                             const CommonConfig &config, std::int64_t stage) {
        const auto prefix = "predictor.stage." + std::to_string(stage) + ".";
        auto result = std::make_unique<Block>();
        result->config = config;
        result->layer = config.n_layers + stage;
        result->ratio = 0;
        result->max_context = config.max_position_embeddings;
        result->attention_norm = dynamic_cast<const ::tyloi::DenseTensor&>(*weights.tensor(prefix + "attention.norm.weight"))
                                     .to(::tyloi::kFloat32).contiguous();
        result->mlp_norm = dynamic_cast<const ::tyloi::DenseTensor&>(*weights.tensor(prefix + "mlp.norm.weight"))
                               .to(::tyloi::kFloat32).contiguous();
        result->query_a_norm =
            dynamic_cast<const ::tyloi::DenseTensor&>(*weights.tensor(prefix + "attention.query_a_norm.weight"));
        result->key_value_norm =
            dynamic_cast<const ::tyloi::DenseTensor&>(*weights.tensor(prefix + "attention.key_value_norm.weight"));
        result->sinks = dynamic_cast<const ::tyloi::DenseTensor&>(*weights.tensor(prefix + "attention.sink"))
                            .to(::tyloi::kFloat32)
                            .contiguous();
        result->attention_mhc_function =
            dynamic_cast<const ::tyloi::DenseTensor&>(*weights.tensor(prefix + "attention.mhc.pre.function"))
                .to(::tyloi::kFloat32)
                .contiguous();
        result->attention_mhc_scale =
            dynamic_cast<const ::tyloi::DenseTensor&>(*weights.tensor(prefix + "attention.mhc.pre.scale"))
                .to(::tyloi::kFloat32)
                .contiguous();
        result->attention_mhc_base =
            dynamic_cast<const ::tyloi::DenseTensor&>(*weights.tensor(prefix + "attention.mhc.pre.base"))
                .to(::tyloi::kFloat32)
                .contiguous();
        result->mlp_mhc_function = dynamic_cast<const ::tyloi::DenseTensor&>(*weights.tensor(prefix + "mlp.mhc.pre.function"))
                                       .to(::tyloi::kFloat32)
                                       .contiguous();
        result->mlp_mhc_scale = dynamic_cast<const ::tyloi::DenseTensor&>(*weights.tensor(prefix + "mlp.mhc.pre.scale"))
                                    .to(::tyloi::kFloat32)
                                    .contiguous();
        result->mlp_mhc_base = dynamic_cast<const ::tyloi::DenseTensor&>(*weights.tensor(prefix + "mlp.mhc.pre.base"))
                                   .to(::tyloi::kFloat32)
                                   .contiguous();
        result->query_a = weights.weight(prefix + "attention.query_a.weight");
        result->query_b = weights.weight(prefix + "attention.query_b.weight");
        result->key_value =
            weights.weight(prefix + "attention.key_value.weight");
        result->output_a =
            weights.weight(prefix + "attention.output_a.weight");
        result->output_b =
            weights.weight(prefix + "attention.output_b.weight");
        // Predictor bundles must not extend the backbone's final-layer marker:
        // residency updates also run when ordinary decoding skips DSpark.
        result->mlp = load_moe_at(pipelines, weights, config, prefix + "mlp.",
                                  -1 - stage, config.dspark_top_k);
        result->rope = TailRope(config.max_position_embeddings, config.rope_theta, 0);
        result->cuda_device = weights.placement.device;

        const auto function_width = config.hc_mult * config.hidden;
        const auto valid_mhc = [&](const DenseTensor &function, const DenseTensor &scale, const DenseTensor &base,
                                   const DenseTensor &norm) {
            return function.dim() == 2 && function.size(0) == 24 &&
                   function.size(1) == function_width && scale.numel() == 3 && base.numel() == 24 &&
                   norm.numel() == config.hidden;
        };
        const auto& routed = result->mlp.moe_split_gate_up ? result->mlp.moe_gate : result->mlp.moe_gate_up;
        const auto routed_experts = routed ? routed->size(0) : 0;
        TYLOINFER_RUNTIME_CHECK(valid_mhc(result->attention_mhc_function, result->attention_mhc_scale,
                                    result->attention_mhc_base, result->attention_norm) &&
                              valid_mhc(result->mlp_mhc_function, result->mlp_mhc_scale,
                                        result->mlp_mhc_base, result->mlp_norm) &&
                              result->query_a->size(1) == config.hidden &&
                              result->query_a->size(0) == config.q_lora_rank &&
                              result->query_b->size(1) == config.q_lora_rank &&
                              result->query_b->size(0) == config.n_heads * config.head_dim &&
                              result->key_value->size(1) == config.hidden &&
                              result->key_value->size(0) == config.head_dim &&
                              result->output_a->size(1) ==
                                  config.n_heads * config.head_dim / config.o_groups &&
                              result->output_a->size(0) == config.o_groups * config.o_lora_rank &&
                              result->output_b->size(1) ==
                                  config.o_groups * config.o_lora_rank &&
                              result->output_b->size(0) == config.hidden &&
                              result->query_a_norm.numel() == config.q_lora_rank &&
                              result->key_value_norm.numel() == config.head_dim &&
                              result->sinks.numel() == config.n_heads &&
                              routed_experts == config.dspark_n_experts &&
                              (result->mlp.moe_down ? result->mlp.moe_down->size(0) : 0) == config.dspark_n_experts,
                          "DeepSeek-V4.1 DSpark stage tensor geometry disagrees at stage ", stage);
        return result;
    }

    static std::optional<DeepseekV41Dspark> load_if_present(CudaExecutionContext &execution,
                                                            const tyloi::weights::Loader& weights,
                                                            const CommonConfig &config,
                                                            std::shared_ptr<const ::tyloi::Tensor> embedding, std::shared_ptr<const ::tyloi::Tensor> output,
                                                            double softcap) {
        const bool root = weights.has("predictor.stage.0.main_projection.weight");
        const bool any = root || weights.has("predictor.stage.0.attention.query_a.weight") ||
                         weights.has("predictor.stage.0.mlp.router.weight") ||
                         weights.has("predictor.stage.0.output_norm.weight");
        if (!root) {
            TYLOINFER_RUNTIME_CHECK(!any,
                              "DeepSeek-V4.1 model source contains an incomplete DSpark head");
            return std::nullopt;
        }
        TYLOINFER_RUNTIME_CHECK(config.has_dspark(),
                          "DeepSeek-V4.1 model source has DSpark tensors without configuration");
        DeepseekV41Dspark result;
        result.execution = &execution;
        result.embedding = std::move(embedding);
        result.output = std::move(output);
        result.softcap = softcap;
        result.primary = weights.placement.device;
        result.config = config;
        result.maximum_context = config.max_position_embeddings;
        const auto first = std::string("predictor.stage.0.");
        const auto last = "predictor.stage." + std::to_string(result.config.n_mtp_layers - 1) + ".";
        result.main_projection =
            weights.weight(first + "main_projection.weight");
        result.main_norm = dynamic_cast<const ::tyloi::DenseTensor&>(*weights.tensor(first + "main_norm.weight"));
        result.output_norm = dynamic_cast<const ::tyloi::DenseTensor&>(*weights.tensor(last + "output_norm.weight"));
        result.markov_embedding =
            weights.weight(last + "markov.embedding.weight");
        result.markov_output = weights.weight(last + "markov.output.weight");
        result.confidence_projection =
            weights.weight(last + "confidence.projection.weight");
        result.stages.reserve(static_cast<std::size_t>(result.config.n_mtp_layers));
        for (std::int64_t stage = 0; stage < result.config.n_mtp_layers; ++stage) {
            result.stages.push_back(load_stage(execution.executor().pipelines(), weights, result.config, stage));
        }
        const auto target_width =
            result.config.hidden *
            static_cast<std::int64_t>(result.config.dspark_target_layer_ids.size());
        TYLOINFER_RUNTIME_CHECK(
            result.config.hc_mult == 4 && result.config.n_heads == 64 &&
                result.config.head_dim == 512 && result.config.rope_head_dim == 64 &&
                result.main_projection->size(1) == target_width &&
                result.main_projection->size(0) == result.config.hidden &&
                result.main_norm.numel() == result.config.hidden &&
                result.output_norm.numel() == result.config.hidden &&
                result.markov_embedding->size(0) == result.config.vocab &&
                result.markov_embedding->size(1) == result.config.dspark_markov_rank &&
                result.markov_output->size(1) == result.config.dspark_markov_rank &&
                result.markov_output->size(0) == result.config.vocab &&
                result.confidence_projection->size(1) ==
                    result.config.hidden + result.config.dspark_markov_rank &&
                result.confidence_projection->size(0) == 1,
            "DeepSeek-V4.1 DSpark top-level tensor geometry disagrees");
        return result;
    }

    void reset(std::int64_t next_batch = 1) override {
        TYLOINFER_RUNTIME_CHECK(next_batch > 0, "DeepSeek-V4.1 DSpark batch must be positive");
        auto half = ::tyloi::TensorOptions()
                        .device(::tyloi::kCUDA)
                        .dtype(::tyloi::kFloat16);
        rings.clear();
        rings.reserve(stages.size());
        for (std::size_t stage = 0; stage < stages.size(); ++stage) {
            rings.push_back(::tyloi::cuda::zeros(
                {next_batch, config.sliding_window, config.head_dim}, half));
        }
        attention_meta =
            ::tyloi::cuda::empty({8 * 1024 * 1024}, half.dtype(::tyloi::kFloat32));
        batch = next_batch;
        position = 0;
    }

    tyloi::models::MtpStep step(const tyloi::Tensor&, const tyloi::Tensor&,
                                const tyloi::Tensor* = nullptr) override {
        throw std::runtime_error("DeepSeek-V4.1 DSpark uses blockwise drafting");
    }

    std::int64_t cache_position() const noexcept override { return position; }

    bool supports_session_state() const noexcept override {return true;}

    MtpSessionState capture_session_state(int64_t cache_pos,const tyloi::Tensor& input_hidden) const override {
        const auto& hidden=dynamic_cast<const DenseTensor&>(input_hidden);
        const auto width=config.hidden*int64_t(config.dspark_target_layer_ids.size());
        TYLOINFER_RUNTIME_CHECK(batch==1 && cache_pos==position && position>0 && rings.size()==stages.size() &&
            hidden.defined() && hidden.dim()==3 && hidden.size(0)==1 && hidden.size(1)>0 && hidden.size(2)==width,
            "DeepSeek-V4.1 DSpark session position or hidden state disagrees");
        MtpSessionState state;state.cache_pos=position;
        auto saved_hidden=hidden.clone();state.bytes=session_tensor_bytes(saved_hidden);
        state.last_target_hidden=std::make_shared<DenseTensor>(std::move(saved_hidden));
        auto& saved=state.payload.emplace<tyloi::models::deepseek_v41::MtpSnapshot>();
        for(const auto& ring:rings) {
            TYLOINFER_RUNTIME_CHECK(ring.dim()==3 && ring.size(0)==1 && ring.size(1)==config.sliding_window &&
                ring.size(2)==config.head_dim,"DeepSeek-V4.1 DSpark session ring disagrees");
            saved.rings.push_back(std::make_shared<DenseTensor>(ring.clone()));
            state.bytes+=session_tensor_bytes(*saved.rings.back());
        }
        return state;
    }

    void restore_session_state(const MtpSessionState& state) override {
        const auto width=config.hidden*int64_t(config.dspark_target_layer_ids.size());
        const auto* hidden=dynamic_cast<const DenseTensor*>(state.last_target_hidden.get());
        const auto* saved=std::get_if<tyloi::models::deepseek_v41::MtpSnapshot>(&state.payload);
        if(state.cache_pos<=0 || state.cache_pos>maximum_context ||
                !saved || saved->rings.size()!=stages.size() || !hidden || !hidden->defined() || !hidden->is_cuda() ||
                hidden->dim()!=3 || hidden->size(0)!=1 || hidden->size(1)<=0 || hidden->size(2)!=width)
            throw CudaSessionStateError("DeepSeek-V4.1 DSpark session state is incompatible");
        for(const auto& ring:saved->rings)
            if(!ring || !ring->defined() || !ring->is_cuda() || ring->scalar_type()!=::tyloi::kFloat16 ||
                    ring->dim()!=3 || ring->size(0)!=1 || ring->size(1)!=config.sliding_window ||
                    ring->size(2)!=config.head_dim || !dynamic_cast<const DenseTensor*>(ring.get()))
                throw CudaSessionStateError("DeepSeek-V4.1 DSpark saved ring geometry is incompatible");
        if(batch!=1 || rings.size()!=stages.size() || !attention_meta.defined())reset(1);
        for(std::size_t i=0;i<rings.size();++i)restore_session_tensor(rings[i],saved->rings[i].get());
        position=state.cache_pos;
    }

    void trim_cache_to(std::int64_t selected_position) override {
        TYLOINFER_RUNTIME_CHECK(selected_position == position,
                          "DeepSeek-V4.1 DSpark only stores committed context");
    }

    bool teacher_forced_prompt_prime() const noexcept override { return false; }

    bool target_bootstrap_decode() const noexcept override { return false; }
    bool preserve_output_dtype() const noexcept override { return true; }
    bool blockwise_drafting() const noexcept override { return true; }
    std::unique_ptr<tyloi::engine::mtp::DepthPolicy> depth_policy(
            int maximum_depth, tyloi::engine::mtp::PolicyState&) const override {
        return std::make_unique<tyloi::models::deepseek_v41::Policy>(maximum_depth);
    }
    MfqSamplingParams draft_sampling(MfqSamplingParams sampling, bool compact) const override {
        if (compact) {
            sampling.temperature = 0.6;
            sampling.top_p = 0.95;
        }
        return sampling;
    }
    bool retains_partial_target_prefix() const noexcept override { return true; }

    int maximum_draft_depth() const noexcept override {
        return static_cast<int>(config.dspark_block_size);
    }

    void seed_target_context(const tyloi::Tensor& input_hidden, std::int64_t tokens) override {
        const auto& hidden = dynamic_cast<const DenseTensor&>(input_hidden);
        TYLOINFER_RUNTIME_CHECK(position == 0 && hidden.dim() == 3 && hidden.size(1) > 0 &&
            tokens >= hidden.size(1) && tokens <= maximum_context &&
            hidden.size(1) >= std::min(tokens, config.sliding_window),
            "invalid DeepSeek-V4.1 DSpark prompt tail");
        position = tokens - hidden.size(1);
        append_target_context(hidden, position);
    }

    void append_target_context(const tyloi::Tensor& input_hidden, std::int64_t start_position) override {
        auto target_hidden = dynamic_cast<const DenseTensor&>(input_hidden);
        const auto target_width =
            config.hidden * static_cast<std::int64_t>(config.dspark_target_layer_ids.size());
        TYLOINFER_RUNTIME_CHECK(target_hidden.dim() == 3 && target_hidden.size(0) == batch &&
                              target_hidden.size(1) > 0 && target_hidden.size(2) == target_width &&
                              start_position >= position && position >= 0 &&
                              (start_position==position || target_hidden.size(1)>=config.sliding_window) &&
                              start_position + target_hidden.size(1) <= maximum_context &&
                              rings.size() == stages.size(),
                          "invalid DeepSeek-V4.1 DSpark context append");
        const auto tokens = target_hidden.size(1);
        const auto retained = std::min(tokens, config.sliding_window);
        target_hidden = target_hidden.narrow(1, tokens - retained, retained).contiguous();
        auto positions =
            ::tyloi::cuda::arange(start_position + tokens - retained, start_position + tokens,
                                       ::tyloi::TensorOptions()
                                           .device(::tyloi::kCUDA)
                                           .dtype(::tyloi::kInt64));
        auto slots = positions.remainder(static_cast<double>(config.sliding_window))
                         .to(::tyloi::kInt64)
                         .contiguous();
        auto next_position=start_position;
        tyloi::models::deepseek_v41::dspark_context(
            next_position, tokens, stages.size(),
            [&] {
                auto floating = target_hidden.to(::tyloi::kFloat32);
                auto scale = (floating.abs().amax(-1, true) / 64.0).clamp_min(1.0);
                return weighted_rms(
                    execution->executor().projector(execution->executor().mode().ffn, model_parallel_primary_device(execution->config)).run(*main_projection, (floating / scale).to(::tyloi::kFloat16).contiguous()),
                    main_norm, config.rms_eps);
            },
            [&](size_t index, const DenseTensor &main_x) {
                auto key_value = weighted_rms(execution->executor().projector(execution->executor().mode().ffn, model_parallel_primary_device(execution->config)).run(*stages[index]->key_value, main_x),
                                              stages[index]->key_value_norm, config.rms_eps);
                key_value = rotate_token_major_tail(
                    key_value,
                    ::tyloi::cuda::arange(start_position + tokens - retained, start_position + tokens,
                                               positions.options()),
                    stages[index]->rope);
                key_value = deepseek_v41_mxfp8_e4m3_sim_cuda(key_value.contiguous());
                return key_value;
            },
            [&](size_t index, DenseTensor key_value) { rings[index].index_copy_(1, slots, key_value); });
        position=next_position;
    }

    DenseTensor attention(const DenseTensor &input, Block &stage, const DenseTensor &ring) {
        const auto tokens = input.size(1);
        auto positions = ::tyloi::cuda::arange(position, position + tokens,
                                                    ::tyloi::TensorOptions()
                                                        .device(::tyloi::kCUDA)
                                                        .dtype(::tyloi::kInt64));
        const auto active = std::min(position, config.sliding_window);
        return tyloi::models::deepseek_v41::dspark_attention(
            active,
            [&] {
                auto q_rank = weighted_rms(execution->executor().projector(execution->executor().mode().ffn, model_parallel_primary_device(execution->config)).run(*stage.query_a, input),
                                           stage.query_a_norm, config.rms_eps);
                auto query = execution->executor().projector(execution->executor().mode().ffn, model_parallel_primary_device(execution->config)).run(*stage.query_b, q_rank)
                                 .reshape({batch, tokens, config.n_heads, config.head_dim})
                                 .transpose(1, 2)
                                 .contiguous()
                                 .to(::tyloi::kFloat32);
                query = stage.rope.forward(query, positions, false);
                return query;
            },
            [&] {
                auto current_keys = weighted_rms(execution->executor().projector(execution->executor().mode().ffn, model_parallel_primary_device(execution->config)).run(*stage.key_value, input),
                                                 stage.key_value_norm, config.rms_eps);
                current_keys = rotate_token_major_tail(current_keys, positions, stage.rope);
                current_keys = deepseek_v41_mxfp8_e4m3_sim_cuda(current_keys.contiguous());
                return current_keys;
            },
            [&](DenseTensor current_keys) {
                return ::tyloi::cuda::cat({ring.narrow(1, 0, active), current_keys}, 1)
                    .contiguous();
            },
            [&](const DenseTensor &keys) {
                return dspark_full_attention_plan(batch, tokens, keys.size(1), input.device());
            },
            [&](DenseTensor query, DenseTensor keys, const auto &plan) {
                return attention_dsv4_sparse_cuda(
                    query, keys, plan.first, plan.second, stage.sinks, attention_meta,
                    1.0 / std::sqrt(static_cast<double>(config.head_dim)));
            },
            [&](DenseTensor attended) {
                return stage.rope.forward(attended.transpose(1, 2).contiguous(), positions, true)
                    .transpose(1, 2)
                    .contiguous();
            },
            [&](DenseTensor attended) { return stage.output_projection(*execution, attended); });
    }

    std::shared_ptr<tyloi::Tensor> draft_block(const tyloi::Tensor& input_ids, int requested) override {
        auto anchor_ids = dynamic_cast<const DenseTensor&>(input_ids);
        const auto physical_width =
            std::min<std::int64_t>(config.dspark_block_size, maximum_context - position);
        TYLOINFER_RUNTIME_CHECK(
            batch == 1 && anchor_ids.dim() == 2 && anchor_ids.size(0) == batch &&
                anchor_ids.size(1) == 1 && requested > 0 && requested <= config.dspark_block_size &&
                requested <= physical_width && position > 0 && rings.size() == stages.size(),
            "invalid DeepSeek-V4.1 DSpark draft input");
        anchor_ids =
            anchor_ids.to(::tyloi::kCUDA, ::tyloi::kInt64).contiguous();
        auto draft_ids = anchor_ids;
        if (physical_width > 1) {
            draft_ids = ::tyloi::cuda::cat(
                            {anchor_ids, ::tyloi::cuda::full({batch, physical_width - 1},
                                                                  config.dspark_noise_token_id,
                                                                  anchor_ids.options())},
                            1)
                            .contiguous();
        }
        return std::make_shared<DenseTensor>(tyloi::models::deepseek_v41::dspark_draft(
            physical_width, requested, stages.size(),
            [&](int64_t) {
                return tyloi::cuda::embed(*embedding, draft_ids, execution->executor().collectives(),
                    execution->executor().profiler(), primary).to(::tyloi::kFloat16);
            },
            [&](DenseTensor embedded) {
                return embedded.unsqueeze(2)
                    .expand({batch, physical_width, config.hc_mult, config.hidden})
                    .contiguous();
            },
            [&](DenseTensor embedded) {
                auto previous_pre = ::tyloi::cuda::zeros(
                    {batch, physical_width, config.hc_mult},
                    embedded.options().dtype(::tyloi::kFloat32));
                previous_pre.narrow(2, 0, 1).fill_(1.0);
                return previous_pre;
            },
            [&](size_t index, DenseTensor hidden, DenseTensor &previous_pre, int64_t output_rows) {
                auto &stage = *stages[index];
                return tyloi::models::deepseek_v41::mega_layer(
                    std::move(hidden), previous_pre,
                    [&](DenseTensor value, const DenseTensor &pre, int branch) {
                        // The last attention still sees the complete block.
                        // Its following FFN is independent for each token.
                        const auto rows = branch == 0 ? physical_width : output_rows;
                        return stage.collapse(
                            execution->executor().profiler(), value.narrow(1, 0, rows), pre.narrow(1, 0, rows),
                            branch == 0 ? stage.attention_mhc_function : stage.mlp_mhc_function,
                            branch == 0 ? stage.attention_mhc_scale : stage.mlp_mhc_scale,
                            branch == 0 ? stage.attention_mhc_base : stage.mlp_mhc_base,
                            branch == 0 ? stage.attention_norm : stage.mlp_norm,
                            branch == 0 ? "deepseek_v41.dspark.mhc.attention.collapse"
                                        : "deepseek_v41.dspark.mhc.mlp.collapse");
                    },
                    [&](DenseTensor value) { return attention(value, stage, rings[index]); },
                    [&](DenseTensor value, DenseTensor residual, const auto &mix, int branch) {
                        return stage.expand(execution->executor().profiler(), value,
                                            residual.narrow(1, 0, value.size(1)), mix,
                                            branch == 0 ? "deepseek_v41.dspark.mhc.attention.expand"
                                                        : "deepseek_v41.dspark.mhc.mlp.expand");
                    },
                    [&](DenseTensor value) {
                        const auto rows = value.size(1);
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                        return stage.mlp.forward(value.reshape({-1, config.hidden}),
#else
                        auto executor = execution->executor().fork(model_parallel_primary_device(execution->config));
                        return stage.mlp.forward(value.reshape({-1, config.hidden}), *executor,
#endif
                                     execution->executor().forward_mode(execution->config.layer_placement.primary_device()).ffn,
                                     draft_ids.narrow(1, 0, rows))
                            .reshape({batch, rows, config.hidden});
                    });
            },
            [&](DenseTensor hidden, const DenseTensor &previous_pre) {
                return (previous_pre.to(::tyloi::kFloat32).unsqueeze(-1) *
                        hidden.to(::tyloi::kFloat32))
                    .sum(2)
                    .to(::tyloi::kFloat16)
                    .contiguous();
            },
            [&](DenseTensor head_hidden) {
                return tyloi::cuda::logits(*output, weighted_rms(head_hidden, output_norm, config.rms_eps),
                    execution->executor().projector(execution->executor().mode().ffn, model_parallel_primary_device(execution->config)), execution->executor().profiler(), softcap)
                    .to(::tyloi::kFloat32)
                    .contiguous();
            },
            [](DenseTensor logits, int64_t count) { return logits.narrow(1, 0, count); }));
    }

    std::shared_ptr<tyloi::Tensor> draft_next(const tyloi::Tensor& input_logits,
                                              const tyloi::Tensor& input_previous) override {
        const auto& logits = dynamic_cast<const DenseTensor&>(input_logits);
        const auto& previous = dynamic_cast<const DenseTensor&>(input_previous);
        auto markov = quant_embedding_lookup(*markov_embedding, previous)
                          .to(::tyloi::kFloat16)
                          .contiguous();
        return std::make_shared<DenseTensor>(logits + execution->executor().projector(execution->executor().mode().ffn, model_parallel_primary_device(execution->config)).run(*markov_output, markov).to(::tyloi::kFloat32));
    }
};

std::unique_ptr<tyloi::models::Mtp> load_dspark_if_present(CudaExecutionContext &execution,
                                                    const tyloi::weights::Loader& weights,
                                                    const CommonConfig &config,
                                                    std::shared_ptr<const ::tyloi::Tensor> embedding, std::shared_ptr<const ::tyloi::Tensor> output,
                                                    double softcap) {
    auto predictor = DeepseekV41Dspark::load_if_present(execution, weights, config,
        std::move(embedding), std::move(output), softcap);
    return predictor ? std::make_unique<DeepseekV41Dspark>(std::move(*predictor)) : nullptr;
}

void run_dspark_self_check() {
    // Seeding only the final window must address the same absolute RoPE and
    // ring slots as a complete target history.
    std::shared_ptr<tyloi::cuda::KvMemoryManager> kv;
    tyloi::cuda::RoutedCache weights;
    std::shared_ptr<tyloi::cuda::KvOffloadStore> offload;
    auto [execution_config, numeric_options] = tyloi::cuda::load_cuda_execution_config();
    auto executor = std::make_unique<tyloi::cuda::Executor>(std::move(numeric_options), 0);
    CudaExecutionContext execution(*executor, std::move(execution_config), weights, kv, offload);
    CudaScope scope(execution.config.layer_placement.primary_device(), execution.executor().contexts());
    DeepseekV41Dspark predictor;
    predictor.execution = &execution;
    predictor.config.hidden = 32;
    predictor.config.head_dim = 512;
    predictor.config.sliding_window = 128;
    predictor.config.dspark_target_layer_ids = {2, 3};
    predictor.maximum_context = 1024;
    const auto options = ::tyloi::TensorOptions().device(::tyloi::kCUDA)
                            .dtype(::tyloi::kFloat32);
    const auto projection = [&](int rows, int columns) {
        QuantLinear weight;
        weight.weight = std::make_shared<::tyloi::DenseTensor>(
            (::tyloi::cuda::randn({rows, columns}, options) / std::sqrt(double(columns)))
                .to(::tyloi::kFloat16).contiguous());
        return weight;
    };
    predictor.main_projection = projection(32, 64);
    predictor.main_norm = ::tyloi::cuda::ones({32}, options);
    auto stage = std::make_unique<Block>();
    stage->key_value = projection(512, 32);
    stage->key_value_norm = ::tyloi::cuda::ones({512}, options);
    stage->rope = TailRope(1024, 10000., 0);
    predictor.stages.push_back(std::move(stage));
    auto history = ::tyloi::cuda::randn({1, 257, 64}, options).to(::tyloi::kFloat16);
    predictor.reset();
    predictor.append_target_context(history, 0);
    auto expected_ring = predictor.rings.front().clone();
    predictor.reset();
    predictor.seed_target_context(history.narrow(1, 129, 128).contiguous(), 257);
    TYLOINFER_RUNTIME_CHECK(predictor.cache_position() == 257 && predictor.rings.front().equal(expected_ring),
                      "DeepSeek-V4.1 DSpark bounded prompt seed changed context");
    auto saved=predictor.capture_session_state(257,history.narrow(1,256,1).contiguous());
    predictor.append_target_context(history.narrow(1, 0, 1).contiguous(), 257);
    TYLOINFER_RUNTIME_CHECK(predictor.cache_position() == 258,
                      "DeepSeek-V4.1 DSpark append after bounded seed lost position");
    auto advanced=predictor.rings.front().clone();
    predictor.restore_session_state(saved);
    TYLOINFER_RUNTIME_CHECK(predictor.cache_position()==257 && predictor.rings.front().equal(expected_ring),
                      "DeepSeek-V4.1 DSpark session restore lost its committed ring");
    predictor.append_target_context(history.narrow(1,0,1).contiguous(),257);
    TYLOINFER_RUNTIME_CHECK(predictor.rings.front().equal(advanced) &&
                      dynamic_cast<const DenseTensor&>(*std::get<tyloi::models::deepseek_v41::MtpSnapshot>(saved.payload).rings.front()).equal(expected_ring),
                      "DeepSeek-V4.1 DSpark session changed continuation or mutated its snapshot");
    auto broken=saved;
    std::get<tyloi::models::deepseek_v41::MtpSnapshot>(broken.payload).rings.front()=
        std::make_shared<DenseTensor>(dynamic_cast<const DenseTensor&>(*std::get<tyloi::models::deepseek_v41::MtpSnapshot>(broken.payload).rings.front()).narrow(1,0,127));
    bool rejected=false;try{predictor.restore_session_state(broken);}catch(const CudaSessionStateError&){rejected=true;}
    TYLOINFER_RUNTIME_CHECK(rejected && predictor.cache_position()==258 && predictor.rings.front().equal(advanced),
                      "DeepSeek-V4.1 DSpark accepted an incompatible session ring");
    auto tail=history.narrow(1,129,128).contiguous();
    predictor.append_target_context(tail,400);
    auto skipped=predictor.rings.front().clone();
    predictor.reset();predictor.seed_target_context(tail,528);
    TYLOINFER_RUNTIME_CHECK(predictor.cache_position()==528 && predictor.rings.front().equal(skipped),
                      "DeepSeek-V4.1 DSpark restored suffix tail changed absolute ring positions");
    rejected=false;
    try{predictor.append_target_context(history.narrow(1,0,1),600);}
    catch(const std::runtime_error&){rejected=true;}
    TYLOINFER_RUNTIME_CHECK(rejected && predictor.cache_position()==528,
                      "DeepSeek-V4.1 DSpark skipped live history without a full replacement window");

    auto plan =
        dspark_full_attention_plan(1, 2, 33, ::tyloi::Device(::tyloi::kCUDA));
    auto indices = plan.first.to(::tyloi::kCPU, ::tyloi::kInt32).contiguous();
    auto mask = plan.second.to(::tyloi::kCPU, ::tyloi::kFloat32).contiguous();
    TYLOINFER_RUNTIME_CHECK(indices.sizes() == ::tyloi::IntArrayRef({1, 2, 64}) &&
                          indices.index({0, 0, 0}).item<std::int32_t>() == 0 &&
                          indices.index({0, 0, 32}).item<std::int32_t>() == 32 &&
                          indices.index({0, 0, 33}).item<std::int32_t>() == 0 &&
                          mask.index({0, 1, 32}).item<float>() == 0.0f &&
                          std::isinf(mask.index({0, 1, 33}).item<float>()) &&
                          mask.index({0, 1, 33}).item<float>() < 0.0f,
                      "DeepSeek-V4.1 DSpark attention plan contract failed");

    CommonConfig capture_config;
    capture_config.hidden = 2;
    capture_config.hc_mult = 4;
    capture_config.dspark_target_layer_ids = {3, 1};
    SharedState capture;
    capture.begin_forward(true, 2);
    auto first = ::tyloi::cuda::arange(8, ::tyloi::TensorOptions()
                                                   .device(::tyloi::kCUDA)
                                                   .dtype(::tyloi::kFloat32))
                     .reshape({1, 1, 4, 2});
    auto second = first + 8.0;
    capture.capture_dspark_target(1, capture_config, second);
    capture.capture_dspark_target(3, capture_config, first);
    auto captured = capture.dspark_target_hidden()
                        .to(::tyloi::kCPU, ::tyloi::kFloat32)
                        .contiguous();
    TYLOINFER_RUNTIME_CHECK(captured.sizes() == ::tyloi::IntArrayRef({1, 1, 4}) &&
                          captured.index({0, 0, 0}).item<float>() == 3.0f &&
                          captured.index({0, 0, 2}).item<float>() == 11.0f,
                      "DeepSeek-V4.1 DSpark target ordering contract failed");

    Block checkpoint;
    checkpoint.config.sliding_window = 4;
    checkpoint.layer = 0;
    checkpoint.shared = std::make_shared<SharedState>();
    checkpoint.shared->layers.resize(1);
    checkpoint.L().local_kv =
        ::tyloi::cuda::arange(8, ::tyloi::TensorOptions()
                                          .device(::tyloi::kCUDA)
                                          .dtype(::tyloi::kFloat16))
            .reshape({1, 4, 2});
    checkpoint.L().position = 4;
    checkpoint.begin_speculative(2);
    auto overwritten =
        ::tyloi::cuda::full({1, 2, 2}, 99.0, checkpoint.L().local_kv.options());
    checkpoint.L().local_kv.narrow(1, 0, 2).copy_(overwritten);
    checkpoint.L().position = 6;
    checkpoint.rollback_speculative(4);
    auto restored =
        checkpoint.L().local_kv.to(::tyloi::kCPU, ::tyloi::kFloat32)
            .contiguous();
    TYLOINFER_RUNTIME_CHECK(checkpoint.L().position == 4 &&
                          restored.index({0, 0, 0}).item<float>() == 0.0f &&
                          restored.index({0, 1, 1}).item<float>() == 3.0f,
                      "DeepSeek-V4.1 DSpark ring rollback contract failed");
}

} // namespace tyloi::cuda::deepseek_v41_runtime
