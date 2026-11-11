#ifndef TYLOINFER_NATIVE_CUDA
#include "engine/backends/cuda/executor/execution.h"
#endif
#include "engine/models/session/snapshot.h"
#include "engine/weights/loader.h"
#ifndef TYLOINFER_NATIVE_CUDA
#include "engine/backends/cuda/ops/ffn/ffn.h"
#endif
#include "engine/backends/cuda/executor/tq_moe.h"
#include "dspark.h"
#include "engine/models/deepseek_v41/policy.h"

#include "engine/models/common/mtp.h"
#include "engine/ops.h"
#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/ops/mhc.h"
#endif
#include "ops.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
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

// DSpark ring history belongs to the request; execution scratch belongs to the backend.
struct DsState final : tyloi::models::MtpState {
    explicit DsState(const tyloi::models::Mtp& owner)
        : tyloi::models::MtpState(owner) {}

    std::int64_t position() const noexcept override { return position_; }

#ifdef TYLOINFER_NATIVE_CUDA
    std::vector<::tyloi::Tensor> rings;
#else
    std::vector<DenseTensor> rings;
#endif
#ifndef TYLOINFER_NATIVE_CUDA
    DenseTensor attention_meta;
#endif
    std::int64_t batch = 0;
    std::int64_t position_ = 0;
};

#ifdef TYLOINFER_NATIVE_CUDA
struct Stage {
    tyloi::Tensor query_a, query_b, key_value, output_a, output_b;
    tyloi::Tensor query_a_norm, key_value_norm, sinks;
    tyloi::Tensor attention_norm, mlp_norm;
    tyloi::Tensor attention_mhc_function, attention_mhc_scale, attention_mhc_base;
    tyloi::Tensor mlp_mhc_function, mlp_mhc_scale, mlp_mhc_base;
    tyloi::ops::TailRopeTable rope;
    tyloi::ops::FfnWeights mlp;
};
#endif

struct DeepseekV41Dspark final : tyloi::models::Mtp {
#ifndef TYLOINFER_NATIVE_CUDA
    CudaExecutionContext *execution = nullptr;
#endif
    ::tyloi::Tensor embedding, output;
    double softcap = 0;
    int primary = 0;
    CommonConfig config;
    ::tyloi::Tensor main_projection;
    Parameter main_norm;
    Parameter output_norm;
    ::tyloi::Tensor markov_embedding;
    ::tyloi::Tensor markov_output;
    ::tyloi::Tensor confidence_projection;
#ifdef TYLOINFER_NATIVE_CUDA
    std::vector<std::unique_ptr<Stage>> stages;
#else
    std::vector<std::unique_ptr<Block>> stages;
#endif

    std::unique_ptr<tyloi::models::MtpState> make_state() const override {
        return std::make_unique<DsState>(*this);
    }

#ifdef TYLOINFER_NATIVE_CUDA
    static std::unique_ptr<Stage> load_stage(const tyloi::weights::Loader& weights,
#else
    static std::unique_ptr<Block> load_stage(const tyloi::weights::Loader& weights,
#endif
                                             const CommonConfig &config, std::int64_t stage) {
        const auto prefix = "predictor.stage." + std::to_string(stage) + ".";
#ifdef TYLOINFER_NATIVE_CUDA
        auto result = std::make_unique<Stage>();
#else
        auto result = std::make_unique<Block>();
        result->config = config;
        result->layer = config.n_layers + stage;
        result->ratio = 0;
        result->max_context = config.max_position_embeddings;
#endif
#ifdef TYLOINFER_NATIVE_CUDA
        result->attention_norm = std::make_shared<DenseTensor>(weights.tensor(prefix + "attention.norm.weight").dense()
                                     .to(::tyloi::kFloat32).contiguous());
        result->mlp_norm = std::make_shared<DenseTensor>(weights.tensor(prefix + "mlp.norm.weight").dense()
                               .to(::tyloi::kFloat32).contiguous());
        result->query_a_norm =
            weights.tensor(prefix + "attention.query_a_norm.weight");
        result->key_value_norm =
            weights.tensor(prefix + "attention.key_value_norm.weight");
        result->sinks = std::make_shared<DenseTensor>(weights.tensor(prefix + "attention.sink").dense()
                            .to(::tyloi::kFloat32)
                            .contiguous());
        result->attention_mhc_function =
            std::make_shared<DenseTensor>(weights.tensor(prefix + "attention.mhc.pre.function").dense()
                .to(::tyloi::kFloat32)
                .contiguous());
        result->attention_mhc_scale =
            std::make_shared<DenseTensor>(weights.tensor(prefix + "attention.mhc.pre.scale").dense()
                .to(::tyloi::kFloat32)
                .contiguous());
        result->attention_mhc_base =
            std::make_shared<DenseTensor>(weights.tensor(prefix + "attention.mhc.pre.base").dense()
                .to(::tyloi::kFloat32)
                .contiguous());
        result->mlp_mhc_function = std::make_shared<DenseTensor>(weights.tensor(prefix + "mlp.mhc.pre.function").dense()
                                       .to(::tyloi::kFloat32)
                                       .contiguous());
        result->mlp_mhc_scale = std::make_shared<DenseTensor>(weights.tensor(prefix + "mlp.mhc.pre.scale").dense()
                                    .to(::tyloi::kFloat32)
                                    .contiguous());
        result->mlp_mhc_base = std::make_shared<DenseTensor>(weights.tensor(prefix + "mlp.mhc.pre.base").dense()
                                   .to(::tyloi::kFloat32)
                                   .contiguous());
#else
        result->attention_norm = weights.tensor(prefix + "attention.norm.weight").dense()
                                     .to(::tyloi::kFloat32).contiguous();
        result->mlp_norm = weights.tensor(prefix + "mlp.norm.weight").dense()
                               .to(::tyloi::kFloat32).contiguous();
        result->query_a_norm =
            weights.tensor(prefix + "attention.query_a_norm.weight").dense();
        result->key_value_norm =
            weights.tensor(prefix + "attention.key_value_norm.weight").dense();
        result->sinks = weights.tensor(prefix + "attention.sink").dense()
                            .to(::tyloi::kFloat32)
                            .contiguous();
        result->attention_mhc_function =
            weights.tensor(prefix + "attention.mhc.pre.function").dense()
                .to(::tyloi::kFloat32)
                .contiguous();
        result->attention_mhc_scale =
            weights.tensor(prefix + "attention.mhc.pre.scale").dense()
                .to(::tyloi::kFloat32)
                .contiguous();
        result->attention_mhc_base =
            weights.tensor(prefix + "attention.mhc.pre.base").dense()
                .to(::tyloi::kFloat32)
                .contiguous();
        result->mlp_mhc_function = weights.tensor(prefix + "mlp.mhc.pre.function").dense()
                                       .to(::tyloi::kFloat32)
                                       .contiguous();
        result->mlp_mhc_scale = weights.tensor(prefix + "mlp.mhc.pre.scale").dense()
                                    .to(::tyloi::kFloat32)
                                    .contiguous();
        result->mlp_mhc_base = weights.tensor(prefix + "mlp.mhc.pre.base").dense()
                                   .to(::tyloi::kFloat32)
                                   .contiguous();
#endif
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
        result->mlp = load_moe_at(weights, config, prefix + "mlp.",
                                  -1 - stage, config.dspark_top_k);
#ifdef TYLOINFER_NATIVE_CUDA
        result->rope = ::tyloi::ops::tail_rope_table(config.max_position_embeddings,
            config.rope_theta, 0, ::tyloi::Device(::tyloi::kCUDA, weights.placement.device));
#else
        result->rope = TailRope(config.max_position_embeddings, config.rope_theta, 0);
#endif
#ifndef TYLOINFER_NATIVE_CUDA
        result->cuda_device = weights.placement.device;
#endif

        const auto function_width = config.hc_mult * config.hidden;
#ifdef TYLOINFER_NATIVE_CUDA
        const auto valid_mhc = [&](const Parameter &function, const Parameter &scale, const Parameter &base,
                                   const Parameter &norm) {
            return function.dim() == 2 && function.size(0) == 24 &&
                   function.size(1) == function_width && scale.numel() == 3 && base.numel() == 24 &&
                   norm.numel() == config.hidden;
        };
#else
        const auto valid_mhc = [&](const DenseTensor &function, const DenseTensor &scale, const DenseTensor &base,
                                   const DenseTensor &norm) {
            return function.dim() == 2 && function.size(0) == 24 &&
                   function.size(1) == function_width && scale.numel() == 3 && base.numel() == 24 &&
                   norm.numel() == config.hidden;
        };
#endif
        const auto& routed = result->mlp.moe_split_gate_up ? result->mlp.moe_gate : result->mlp.moe_gate_up;
        const auto routed_experts = routed ? routed.size(0) : 0;
#ifdef TYLOINFER_NATIVE_CUDA
        TYLOINFER_RUNTIME_CHECK(valid_mhc(result->attention_mhc_function, result->attention_mhc_scale,
                                    result->attention_mhc_base, result->attention_norm) &&
                              valid_mhc(result->mlp_mhc_function, result->mlp_mhc_scale,
                                        result->mlp_mhc_base, result->mlp_norm) &&
                              result->query_a.size(1) == config.hidden &&
                              result->query_a.size(0) == config.q_lora_rank &&
                              result->query_b.size(1) == config.q_lora_rank &&
                              result->query_b.size(0) == config.n_heads * config.head_dim &&
                              result->key_value.size(1) == config.hidden &&
                              result->key_value.size(0) == config.head_dim &&
                              result->output_a.size(1) ==
                                  config.n_heads * config.head_dim / config.o_groups &&
                              result->output_a.size(0) == config.o_groups * config.o_lora_rank &&
                              result->output_b.size(1) ==
                                  config.o_groups * config.o_lora_rank &&
                              result->output_b.size(0) == config.hidden &&
                              result->query_a_norm.numel() == config.q_lora_rank &&
                              result->key_value_norm.numel() == config.head_dim &&
                              result->sinks.numel() == config.n_heads &&
                              routed_experts == config.dspark_n_experts &&
                              (result->mlp.moe_down ? result->mlp.moe_down.size(0) : 0) == config.dspark_n_experts,
                          "DeepSeek-V4.1 DSpark stage tensor geometry disagrees at stage ", stage);
#else
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
#endif
        return result;
    }

    static std::optional<DeepseekV41Dspark> load_if_present(
#ifndef TYLOINFER_NATIVE_CUDA
        CudaExecutionContext &execution,
#endif
        const tyloi::weights::Loader& weights, const CommonConfig &config,
        ::tyloi::Tensor embedding, ::tyloi::Tensor output, double softcap) {
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
#ifndef TYLOINFER_NATIVE_CUDA
        result.execution = &execution;
#endif
        result.embedding = std::move(embedding);
        result.output = std::move(output);
        result.softcap = softcap;
        result.primary = weights.placement.device;
        result.config = config;
        const auto first = std::string("predictor.stage.0.");
        const auto last = "predictor.stage." + std::to_string(result.config.n_mtp_layers - 1) + ".";
        result.main_projection =
            weights.weight(first + "main_projection.weight");
#ifdef TYLOINFER_NATIVE_CUDA
        result.main_norm = weights.tensor(first + "main_norm.weight");
        result.output_norm = weights.tensor(last + "output_norm.weight");
#else
        result.main_norm = weights.tensor(first + "main_norm.weight").dense();
        result.output_norm = weights.tensor(last + "output_norm.weight").dense();
#endif
        result.markov_embedding =
            weights.weight(last + "markov.embedding.weight");
        result.markov_output = weights.weight(last + "markov.output.weight");
        result.confidence_projection =
            weights.weight(last + "confidence.projection.weight");
        result.stages.reserve(static_cast<std::size_t>(result.config.n_mtp_layers));
        for (std::int64_t stage = 0; stage < result.config.n_mtp_layers; ++stage) {
            result.stages.push_back(load_stage(weights, result.config, stage));
        }
        const auto target_width =
            result.config.hidden *
            static_cast<std::int64_t>(result.config.dspark_target_layer_ids.size());
#ifdef TYLOINFER_NATIVE_CUDA
        TYLOINFER_RUNTIME_CHECK(
            result.config.hc_mult == 4 && result.config.n_heads == 64 &&
                result.config.head_dim == 512 && result.config.rope_head_dim == 64 &&
                result.main_projection.size(1) == target_width &&
                result.main_projection.size(0) == result.config.hidden &&
                result.main_norm.numel() == result.config.hidden &&
                result.output_norm.numel() == result.config.hidden &&
                result.markov_embedding.size(0) == result.config.vocab &&
                result.markov_embedding.size(1) == result.config.dspark_markov_rank &&
                result.markov_output.size(1) == result.config.dspark_markov_rank &&
                result.markov_output.size(0) == result.config.vocab &&
                result.confidence_projection.size(1) ==
                    result.config.hidden + result.config.dspark_markov_rank &&
                result.confidence_projection.size(0) == 1,
            "DeepSeek-V4.1 DSpark top-level tensor geometry disagrees");
#else
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
#endif
        return result;
    }

    void reset(tyloi::models::MtpState& value, std::int64_t next_batch = 1) const override {
        auto& state = this->state<DsState>(value);
        TYLOINFER_RUNTIME_CHECK(next_batch > 0, "DeepSeek-V4.1 DSpark batch must be positive");
#ifdef TYLOINFER_NATIVE_CUDA
        state.rings.resize(stages.size());
        const auto options = ::tyloi::TensorOptions()
            .device(::tyloi::Device(::tyloi::kCUDA, primary)).dtype(::tyloi::kFloat16);
        for (auto& ring : state.rings) {
            if (!compatible_ring(ring, next_batch))
                ring = tyloi::ops::empty({next_batch, config.sliding_window, config.head_dim}, options);
            tyloi::ops::fill(ring, 0);
        }
#else
        auto half = ::tyloi::TensorOptions()
                        .device(::tyloi::kCUDA)
                        .dtype(::tyloi::kFloat16);
        state.rings.clear();
        state.rings.reserve(stages.size());
        for (std::size_t stage = 0; stage < stages.size(); ++stage) {
            state.rings.push_back(::tyloi::cuda::zeros(
                {next_batch, config.sliding_window, config.head_dim}, half));
        }
        state.attention_meta =
            ::tyloi::cuda::empty({8 * 1024 * 1024}, half.dtype(::tyloi::kFloat32));
#endif
        state.batch = next_batch;
        state.position_ = 0;
    }

#ifdef TYLOINFER_NATIVE_CUDA
    /** Estimate new ring backing required to reset the predictor for a batch. */
    std::vector<std::pair<::tyloi::Device, std::size_t>> reset_growth(
            const tyloi::models::MtpState& value, std::int64_t next_batch) const override {
        const auto& state = this->state<DsState>(value);
        TYLOINFER_RUNTIME_CHECK(next_batch > 0 && config.sliding_window > 0 && config.head_dim > 0,
            "DeepSeek-V4.1 DSpark batch or ring geometry is invalid");
        auto bytes = sizeof(std::uint16_t);
        for (const auto dimension : {next_batch, config.sliding_window, config.head_dim}) {
            if (bytes > std::numeric_limits<std::size_t>::max() / std::size_t(dimension))
                throw std::overflow_error("DeepSeek-V4.1 DSpark ring size overflow");
            bytes *= std::size_t(dimension);
        }
        std::size_t count = 0;
        for (std::size_t i = 0; i < stages.size(); ++i)
            if (i >= state.rings.size() || !compatible_ring(state.rings[i], next_batch)) ++count;
        if (count && bytes > std::numeric_limits<std::size_t>::max() / count)
            throw std::overflow_error("DeepSeek-V4.1 DSpark ring size overflow");
        return count ? std::vector<std::pair<::tyloi::Device, std::size_t>>{
            {::tyloi::Device(::tyloi::kCUDA, primary), bytes * count}} :
            std::vector<std::pair<::tyloi::Device, std::size_t>>{};
    }

    /** Report whether an existing ring can be reused for this batch. */
    bool compatible_ring(const ::tyloi::Tensor& ring, std::int64_t batch) const {
        return ring.defined() && ring.representation<DenseTensor>() && ring.dim() == 3 &&
            ring.size(0) == batch && ring.size(1) == config.sliding_window &&
            ring.size(2) == config.head_dim && ring.scalar_type() == ::tyloi::kFloat16 &&
            ring.device() == ::tyloi::Device(::tyloi::kCUDA, primary) && ring.is_contiguous();
    }

    /** Plan model-side temporary storage for a context update or physical draft block. */
    std::vector<std::pair<::tyloi::Device, std::size_t>> growth(
            const tyloi::models::MtpState& value, const tyloi::models::Mtp::Input& input) const override {
        const auto& state = this->state<DsState>(value);
        if (std::holds_alternative<Step>(input))
            throw std::runtime_error("DeepSeek-V4.1 DSpark uses blockwise drafting");
        const auto checked = [](std::size_t a, std::size_t b) {
            if (b && a > std::numeric_limits<std::size_t>::max() / b)
                throw std::overflow_error("DeepSeek-V4.1 DSpark workspace size overflow");
            return a * b;
        };
        const auto dimension = [](std::int64_t n) {
            if (n <= 0 || std::uint64_t(n) > std::numeric_limits<std::size_t>::max())
                throw std::invalid_argument("DeepSeek-V4.1 DSpark workspace geometry is invalid");
            return std::size_t(n);
        };
        const auto device = ::tyloi::Device(::tyloi::kCUDA, primary);
        if (state.batch != 1 || state.position_ < 0 || state.position_ > config.max_position_embeddings ||
            state.rings.size() != stages.size() || stages.empty())
            throw std::invalid_argument("DeepSeek-V4.1 DSpark state is unavailable");
        for (const auto& ring : state.rings)
            if (!compatible_ring(ring, state.batch))
                throw std::invalid_argument("DeepSeek-V4.1 DSpark ring geometry is incompatible");
        std::size_t bytes = 0;
        const auto add = [&](std::initializer_list<std::size_t> dimensions) {
            std::size_t term = 1;
            for (const auto n : dimensions) term = checked(term, n);
            if (bytes > std::numeric_limits<std::size_t>::max() - term)
                throw std::overflow_error("DeepSeek-V4.1 DSpark workspace size overflow");
            bytes += term;
        };
        const auto hidden_width = dimension(config.hidden);
        const auto head_width = dimension(config.head_dim);
        if (const auto* context = std::get_if<Context>(&input)) {
            const auto& hidden = context->hidden;
            const auto target_width = checked(hidden_width, config.dspark_target_layer_ids.size());
            if (!target_width || !hidden.defined() || !hidden.representation<DenseTensor>() ||
                hidden.dim() != 3 || hidden.size(0) != state.batch || hidden.size(1) <= 0 ||
                std::uint64_t(hidden.size(2)) != target_width || hidden.device() != device)
                throw std::invalid_argument("invalid DeepSeek-V4.1 DSpark context append");
            const auto tokens = hidden.size(1);
            std::int64_t start = context->position;
            if (context->seed) {
                if (state.position_ != 0 || context->position < tokens ||
                    context->position > config.max_position_embeddings ||
                    tokens < std::min(context->position, config.sliding_window))
                    throw std::invalid_argument("invalid DeepSeek-V4.1 DSpark prompt tail");
                start = context->position - tokens;
            }
            if (start < state.position_ || state.position_ < 0 ||
                (start != state.position_ && tokens < config.sliding_window) ||
                start > config.max_position_embeddings || tokens > config.max_position_embeddings - start)
                throw std::invalid_argument("invalid DeepSeek-V4.1 DSpark context append");
            const auto rows = dimension(std::min(tokens, config.sliding_window));
            // Account for conversions, normalization, projection and each stage's key temporaries.
            add({rows, target_width, 32});
            add({rows, hidden_width, 32});
            add({rows, head_width, stages.size(), 32});
            add({rows, 64});
        } else {
            const auto& draft = std::get<Draft>(input);
            const auto physical = std::min(config.dspark_block_size, config.max_position_embeddings - state.position_);
            if (!draft.ids.defined() || !draft.ids.representation<DenseTensor>() || draft.ids.dim() != 2 ||
                draft.ids.size(0) != state.batch || draft.ids.size(1) != 1 ||
                draft.depth <= 0 || draft.depth > config.dspark_block_size || draft.depth > physical ||
                state.position_ <= 0)
                throw std::invalid_argument("invalid DeepSeek-V4.1 DSpark draft input");
            const auto rows = dimension(physical);
            const auto active = std::min(state.position_, config.sliding_window);
            const auto keys = checked(dimension(active + physical), 1);
            if (keys > std::numeric_limits<std::size_t>::max() - 31)
                throw std::overflow_error("DeepSeek-V4.1 DSpark workspace size overflow");
            const auto aligned = (keys + 31) / 32 * 32;
            // The block retains MHC activations while attention and expert workspaces run.
            add({rows, dimension(config.hc_mult), hidden_width, 64});
            add({rows, dimension(config.n_heads), head_width, 64});
            add({keys, head_width, 16});
            add({rows, aligned, 16});
            add({rows, dimension(config.q_lora_rank), 32});
            add({rows, dimension(config.o_groups), dimension(config.o_lora_rank), 32});
            add({rows, dimension(config.dspark_n_experts), 16});
            add({rows, dimension(config.dspark_top_k), dimension(config.moe_inter), 32});
            add({rows, dimension(config.n_shared), dimension(config.moe_inter), 32});
            add({rows, dimension(config.vocab), 16});
            add({rows, dimension(config.dspark_markov_rank), 16});
            add({rows, 64});
        }
        return {{device, bytes}};
    }
#endif

    tyloi::models::MtpStep step(tyloi::models::MtpState&, const tyloi::Tensor&,
                                const tyloi::Tensor&, const tyloi::Tensor& = {},
                                const tyloi::models::ForwardOptions& = {}) const override {
        throw std::runtime_error("DeepSeek-V4.1 DSpark uses blockwise drafting");
    }

    bool supports_session_state() const noexcept override {return true;}

#ifdef TYLOINFER_NATIVE_CUDA
    /** DSpark keeps every committed target position in its context ring. */
    bool session_boundary(std::int64_t target, std::int64_t predictor) const noexcept override {
        return target > 0 && predictor == target;
    }
#endif

    MtpSessionState capture_session_state(const tyloi::models::MtpState& value,
                                          int64_t cache_pos,
                                          const tyloi::Tensor& input_hidden) const override {
        const auto& state = this->state<DsState>(value);
        const auto& hidden=input_hidden.dense();
        const auto width=config.hidden*int64_t(config.dspark_target_layer_ids.size());
#ifdef TYLOINFER_NATIVE_CUDA
        const bool boundary = session_boundary(cache_pos, state.position_);
#else
        const bool boundary = cache_pos == state.position_ && state.position_ > 0;
#endif
        TYLOINFER_RUNTIME_CHECK(state.batch==1 && boundary &&
            state.rings.size()==stages.size() &&
            hidden.defined() && hidden.dim()==3 && hidden.size(0)==1 && hidden.size(1)>0 && hidden.size(2)==width,
            "DeepSeek-V4.1 DSpark session position or hidden state disagrees");
        MtpSessionState snapshot;snapshot.cache_pos=state.position_;
#ifdef TYLOINFER_NATIVE_CUDA
        snapshot.last_target_hidden=tyloi::ops::clone(input_hidden);
        snapshot.bytes=session_tensor_bytes(snapshot.last_target_hidden);
#else
        auto saved_hidden=hidden.clone();snapshot.bytes=session_tensor_bytes(saved_hidden);
        snapshot.last_target_hidden=std::make_shared<DenseTensor>(std::move(saved_hidden));
#endif
        auto& saved=snapshot.payload.emplace<tyloi::models::deepseek_v41::MtpSnapshot>();
        for(const auto& ring:state.rings) {
            TYLOINFER_RUNTIME_CHECK(ring.dim()==3 && ring.size(0)==1 && ring.size(1)==config.sliding_window &&
                ring.size(2)==config.head_dim,"DeepSeek-V4.1 DSpark session ring disagrees");
#ifdef TYLOINFER_NATIVE_CUDA
            saved.rings.push_back(tyloi::ops::clone(ring));
#else
            saved.rings.push_back(std::make_shared<DenseTensor>(ring.clone()));
#endif
            snapshot.bytes+=session_tensor_bytes(saved.rings.back());
        }
        return snapshot;
    }

#ifdef TYLOINFER_NATIVE_CUDA
    /** Validate and retain the DSpark ring snapshot for joint restoration. */
    std::unique_ptr<tyloi::models::Restore> prepare_restore(
            tyloi::models::MtpState& value, const MtpSessionState& snapshot) const override {
        auto& state = this->state<DsState>(value);
        struct Candidate final : tyloi::models::Restore {
            const DeepseekV41Dspark& model;
            DsState& state;
            MtpSessionState saved;
            std::vector<::tyloi::Tensor> rings;

            Candidate(const DeepseekV41Dspark& model, DsState& state, const MtpSessionState& snapshot)
                : model(model), state(state), saved(snapshot), rings(state.rings) {
                const auto* payload = std::get_if<tyloi::models::deepseek_v41::MtpSnapshot>(&saved.payload);
                const auto* hidden = saved.last_target_hidden.representation<DenseTensor>();
                const auto width = model.config.hidden * std::int64_t(model.config.dspark_target_layer_ids.size());
                const auto device = ::tyloi::Device(::tyloi::kCUDA, model.primary);
                if (saved.cache_pos <= 0 || saved.cache_pos > model.config.max_position_embeddings ||
                    !payload || payload->rings.size() != model.stages.size() || !hidden || !hidden->defined() ||
                    hidden->dim() != 3 || hidden->size(0) != 1 || hidden->size(1) <= 0 ||
                    hidden->size(2) != width || hidden->device() != device ||
                    (hidden->scalar_type() != ::tyloi::kFloat16 && hidden->scalar_type() != ::tyloi::kBFloat16 &&
                     hidden->scalar_type() != ::tyloi::kFloat32))
                    throw CudaSessionStateError("DeepSeek-V4.1 DSpark session state is incompatible");
                for (const auto& ring : payload->rings) {
                    if (!model.compatible_ring(ring, 1))
                        throw CudaSessionStateError("DeepSeek-V4.1 DSpark saved ring geometry is incompatible");
                }
                rings.resize(model.stages.size());
            }
            std::vector<std::pair<::tyloi::Device, std::size_t>> growth() const override {
                const auto& source = std::get<tyloi::models::deepseek_v41::MtpSnapshot>(saved.payload).rings;
                std::size_t bytes = 0;
                for (std::size_t i = 0; i < rings.size(); ++i) {
                    const auto added = restore_tensor_bytes(rings[i], source[i]);
                    if (bytes > std::numeric_limits<std::size_t>::max() - added)
                        throw std::overflow_error("DeepSeek-V4.1 DSpark session allocation overflow");
                    bytes += added;
                }
                return bytes ? std::vector<std::pair<::tyloi::Device, std::size_t>>{
                    {::tyloi::Device(::tyloi::kCUDA, model.primary), bytes}} :
                    std::vector<std::pair<::tyloi::Device, std::size_t>>{};
            }
            void stage() override {
                const auto& source = std::get<tyloi::models::deepseek_v41::MtpSnapshot>(saved.payload).rings;
                for (std::size_t i = 0; i < rings.size(); ++i) {
                    if (restore_tensor_bytes(rings[i], source[i]))
                        rings[i] = tyloi::ops::empty(source[i].dense().sizes(), source[i].options());
                    tyloi::ops::copy_into(rings[i], source[i]);
                }
            }
            void publish() noexcept override {
                state.rings.swap(rings);
                state.position_ = saved.cache_pos;
                state.batch = 1;
            }
        };
        return std::make_unique<Candidate>(*this, state, snapshot);
    }
#else
    void restore_session_state(tyloi::models::MtpState& value,
                               const MtpSessionState& snapshot) const override {
        auto& state = this->state<DsState>(value);
        const auto width=config.hidden*int64_t(config.dspark_target_layer_ids.size());
        const auto* hidden=snapshot.last_target_hidden.representation<DenseTensor>();
        const auto* saved=std::get_if<tyloi::models::deepseek_v41::MtpSnapshot>(&snapshot.payload);
        if(snapshot.cache_pos<=0 || snapshot.cache_pos>config.max_position_embeddings ||
                !saved || saved->rings.size()!=stages.size() || !hidden || !hidden->defined() || !hidden->is_cuda() ||
                hidden->dim()!=3 || hidden->size(0)!=1 || hidden->size(1)<=0 || hidden->size(2)!=width)
            throw CudaSessionStateError("DeepSeek-V4.1 DSpark session state is incompatible");
        for(const auto& ring:saved->rings)
            if(!ring.defined() || !ring.is_cuda() || ring.scalar_type()!=::tyloi::kFloat16 ||
                    ring.dim()!=3 || ring.size(0)!=1 || ring.size(1)!=config.sliding_window ||
                    ring.size(2)!=config.head_dim || !ring.representation<DenseTensor>())
                throw CudaSessionStateError("DeepSeek-V4.1 DSpark saved ring geometry is incompatible");
        if(state.batch!=1 || state.rings.size()!=stages.size() || !state.attention_meta.defined())reset(value, 1);
        for(std::size_t i=0;i<state.rings.size();++i) {
            auto ring=restore_session_tensor(state.rings[i],saved->rings[i]);
            state.rings[i]=ring.defined()?ring.dense():DenseTensor{};
        }
        state.position_=snapshot.cache_pos;
    }
#endif

    void trim_cache_to(tyloi::models::MtpState& value,
                       std::int64_t selected_position) const override {
        const auto& state = this->state<DsState>(value);
        TYLOINFER_RUNTIME_CHECK(selected_position == state.position_,
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

    void seed_target_context(tyloi::models::MtpState& value,
                             const tyloi::Tensor& input_hidden,
                             std::int64_t tokens) const override {
        auto& state = this->state<DsState>(value);
        const auto& hidden = input_hidden.dense();
        TYLOINFER_RUNTIME_CHECK(state.position_ == 0 && hidden.dim() == 3 && hidden.size(1) > 0 &&
            tokens >= hidden.size(1) && tokens <= config.max_position_embeddings &&
            hidden.size(1) >= std::min(tokens, config.sliding_window),
            "invalid DeepSeek-V4.1 DSpark prompt tail");
        state.position_ = tokens - hidden.size(1);
        append_target_context(value, hidden, state.position_);
    }

    void append_target_context(tyloi::models::MtpState& value,
                               const tyloi::Tensor& input_hidden,
                               std::int64_t start_position) const override {
        auto& state = this->state<DsState>(value);
        auto target_hidden = input_hidden.dense();
        const auto target_width =
            config.hidden * static_cast<std::int64_t>(config.dspark_target_layer_ids.size());
        TYLOINFER_RUNTIME_CHECK(target_hidden.dim() == 3 && target_hidden.size(0) == state.batch &&
                              target_hidden.size(1) > 0 && target_hidden.size(2) == target_width &&
                              start_position >= state.position_ && state.position_ >= 0 &&
                              (start_position==state.position_ || target_hidden.size(1)>=config.sliding_window) &&
                              start_position + target_hidden.size(1) <= config.max_position_embeddings &&
                              state.rings.size() == stages.size(),
                          "invalid DeepSeek-V4.1 DSpark context append");
        const auto tokens = target_hidden.size(1);
        const auto retained = std::min(tokens, config.sliding_window);
        target_hidden = target_hidden.narrow(1, tokens - retained, retained).contiguous();
        auto positions =
            ::tyloi::cuda::arange(start_position + tokens - retained, start_position + tokens,
                                       ::tyloi::TensorOptions()
#ifdef TYLOINFER_NATIVE_CUDA
                                           .device(::tyloi::Device(::tyloi::kCUDA, primary))
#else
                                           .device(::tyloi::kCUDA)
#endif
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
#ifdef TYLOINFER_NATIVE_CUDA
                    tyloi::ops::project((floating / scale).to(::tyloi::kFloat16).contiguous(), main_projection).dense(),
#else
                    execution->executor().projector(execution->executor().mode().ffn, model_parallel_primary_device(execution->config)).run(main_projection, (floating / scale).to(::tyloi::kFloat16).contiguous()),
#endif
#ifdef TYLOINFER_NATIVE_CUDA
                    main_norm, config.rms_eps);
#else
                    main_norm, config.rms_eps);
#endif
            },
            [&](size_t index, const DenseTensor &main_x) {
                auto key_value = weighted_rms(
#ifdef TYLOINFER_NATIVE_CUDA
                    tyloi::ops::project(main_x, stages[index]->key_value).dense(),
#else
                    execution->executor().projector(execution->executor().mode().ffn, model_parallel_primary_device(execution->config)).run(stages[index]->key_value, main_x),
#endif
#ifdef TYLOINFER_NATIVE_CUDA
                                              stages[index]->key_value_norm, config.rms_eps);
#else
                                              stages[index]->key_value_norm, config.rms_eps);
#endif
                key_value = rotate_token_major_tail(
                    key_value,
                    ::tyloi::cuda::arange(start_position + tokens - retained, start_position + tokens,
                                               positions.options()),
                    stages[index]->rope);
                key_value = deepseek_v41_mxfp8_e4m3_sim_cuda(key_value.contiguous());
                return key_value;
            },
            [&](size_t index, DenseTensor key_value) {
#ifdef TYLOINFER_NATIVE_CUDA
                state.rings[index].dense().index_copy_(1, slots, key_value);
#else
                state.rings[index].index_copy_(1, slots, key_value);
#endif
            });
        state.position_=next_position;
    }

    DenseTensor attention(const DsState& state, const DenseTensor &input,
#ifdef TYLOINFER_NATIVE_CUDA
                          const Stage &stage, const DenseTensor &ring) const {
#else
                          const Block &stage, const DenseTensor &ring) const {
#endif
        const auto tokens = input.size(1);
        auto positions = ::tyloi::cuda::arange(state.position_, state.position_ + tokens,
                                                    ::tyloi::TensorOptions()
#ifdef TYLOINFER_NATIVE_CUDA
                                                        .device(::tyloi::Device(::tyloi::kCUDA, primary))
#else
                                                        .device(::tyloi::kCUDA)
#endif
                                                        .dtype(::tyloi::kInt64));
        const auto active = std::min(state.position_, config.sliding_window);
        return tyloi::models::deepseek_v41::dspark_attention(
            active,
            [&] {
                auto q_rank = weighted_rms(
#ifdef TYLOINFER_NATIVE_CUDA
                                           tyloi::ops::project(input, stage.query_a).dense(),
#else
                                           execution->executor().projector(execution->executor().mode().ffn, model_parallel_primary_device(execution->config)).run(stage.query_a, input),
#endif
#ifdef TYLOINFER_NATIVE_CUDA
                                           stage.query_a_norm, config.rms_eps);
#else
                                           stage.query_a_norm, config.rms_eps);
#endif
                auto query =
#ifdef TYLOINFER_NATIVE_CUDA
                    tyloi::ops::project(q_rank, stage.query_b).dense()
#else
                    execution->executor().projector(execution->executor().mode().ffn, model_parallel_primary_device(execution->config)).run(stage.query_b, q_rank)
#endif
                                 .reshape({state.batch, tokens, config.n_heads, config.head_dim})
                                 .transpose(1, 2)
                                 .contiguous()
                                 .to(::tyloi::kFloat32);
#ifdef TYLOINFER_NATIVE_CUDA
                auto rotated = ::tyloi::ops::tail_rope(query, positions, stage.rope);
                query = rotated.dense();
#else
                query = stage.rope.forward(query, positions, false);
#endif
                return query;
            },
            [&] {
                auto current_keys = weighted_rms(
#ifdef TYLOINFER_NATIVE_CUDA
                                                 tyloi::ops::project(input, stage.key_value).dense(),
#else
                                                 execution->executor().projector(execution->executor().mode().ffn, model_parallel_primary_device(execution->config)).run(stage.key_value, input),
#endif
#ifdef TYLOINFER_NATIVE_CUDA
                                                 stage.key_value_norm, config.rms_eps);
#else
                                                 stage.key_value_norm, config.rms_eps);
#endif
                current_keys = rotate_token_major_tail(current_keys, positions, stage.rope);
                current_keys = deepseek_v41_mxfp8_e4m3_sim_cuda(current_keys.contiguous());
                return current_keys;
            },
            [&](DenseTensor current_keys) {
                return ::tyloi::cuda::cat({ring.narrow(1, 0, active), current_keys}, 1)
                    .contiguous();
            },
            [&](const DenseTensor &keys) {
                return dspark_full_attention_plan(state.batch, tokens, keys.size(1), input.device());
            },
            [&](DenseTensor query, DenseTensor keys, const auto &plan) {
#ifdef TYLOINFER_NATIVE_CUDA
                return tyloi::ops::sparse_attention(
                           query, keys, plan.first, plan.second, stage.sinks,
                           1.0 / std::sqrt(static_cast<double>(config.head_dim)))
                    .dense();
#else
                return attention_dsv4_sparse_cuda(
                    query, keys, plan.first, plan.second,
                    stage.sinks, state.attention_meta,
                    1.0 / std::sqrt(static_cast<double>(config.head_dim)));
#endif
            },
            [&](DenseTensor attended) {
#ifdef TYLOINFER_NATIVE_CUDA
                auto value = attended.transpose(1, 2).contiguous();
                auto rotated = ::tyloi::ops::tail_rope(value, positions, stage.rope, true);
                return rotated.dense()
#else
                return stage.rope.forward(attended.transpose(1, 2).contiguous(), positions, true)
#endif
                    .transpose(1, 2)
                    .contiguous();
            },
            [&](DenseTensor attended) {
#ifdef TYLOINFER_NATIVE_CUDA
                return tyloi::ops::low_rank(attended, stage.output_a, stage.output_b, config.o_groups, true, false).dense();
#else
                return stage.output_projection(attended);
#endif
            });
    }

    ::tyloi::Tensor draft_block(tyloi::models::MtpState& value,
                                const tyloi::Tensor& input_ids,
                                int requested) const override {
        const auto& state = this->state<DsState>(value);
        auto anchor_ids = input_ids.dense();
        const auto physical_width =
            std::min<std::int64_t>(config.dspark_block_size,
                                   config.max_position_embeddings - state.position_);
        TYLOINFER_RUNTIME_CHECK(
            state.batch == 1 && anchor_ids.dim() == 2 && anchor_ids.size(0) == state.batch &&
                anchor_ids.size(1) == 1 && requested > 0 && requested <= config.dspark_block_size &&
                requested <= physical_width && state.position_ > 0 && state.rings.size() == stages.size(),
            "invalid DeepSeek-V4.1 DSpark draft input");
        anchor_ids =
#ifdef TYLOINFER_NATIVE_CUDA
            anchor_ids.to(::tyloi::Device(::tyloi::kCUDA, primary), ::tyloi::kInt64).contiguous();
#else
            anchor_ids.to(::tyloi::kCUDA, ::tyloi::kInt64).contiguous();
#endif
        auto draft_ids = anchor_ids;
        if (physical_width > 1) {
            draft_ids = ::tyloi::cuda::cat(
                            {anchor_ids, ::tyloi::cuda::full({state.batch, physical_width - 1},
                                                                  config.dspark_noise_token_id,
                                                                  anchor_ids.options())},
                            1)
                            .contiguous();
        }
        return ::tyloi::Tensor{tyloi::models::deepseek_v41::dspark_draft(
            physical_width, requested, stages.size(),
            [&](int64_t) {
#ifdef TYLOINFER_NATIVE_CUDA
                auto local_ids = tyloi::ops::copy(draft_ids, ::tyloi::Device(::tyloi::kCUDA, primary));
                return tyloi::ops::measure("model.embed", [&] {
                    return tyloi::ops::embedding(local_ids, embedding).dense()
                        .to(::tyloi::kFloat16);
                });
#else
                return tyloi::cuda::embed(embedding, draft_ids, execution->executor().collectives(),
                    execution->executor().profiler(), primary).to(::tyloi::kFloat16);
#endif
            },
            [&](DenseTensor embedded) {
                return embedded.unsqueeze(2)
                    .expand({state.batch, physical_width, config.hc_mult, config.hidden})
                    .contiguous();
            },
            [&](DenseTensor embedded) {
                auto previous_pre = ::tyloi::cuda::zeros(
                    {state.batch, physical_width, config.hc_mult},
                    embedded.options().dtype(::tyloi::kFloat32));
                previous_pre.narrow(2, 0, 1).fill_(1.0);
                return previous_pre;
            },
            [&](size_t index, DenseTensor hidden, DenseTensor &previous_pre, int64_t output_rows) {
                const auto &stage = *stages[index];
                // Keep the two MHC branches explicit here. Attention consumes
                // the complete physical block, while the following FFN only
                // produces the requested rows on the final predictor stage.
#ifdef TYLOINFER_NATIVE_CUDA
                const tyloi::ops::MhcConfig mhc{config.hc_sinkhorn_iters, config.rms_eps, config.hc_eps};
                auto attention_mix = tyloi::ops::measure("deepseek_v41.dspark.mhc.attention.collapse", [&] {
                    return tyloi::ops::mhc_collapse(hidden.narrow(1, 0, physical_width),
                        previous_pre.narrow(1, 0, physical_width), stage.attention_mhc_function,
                        stage.attention_mhc_scale, stage.attention_mhc_base, stage.attention_norm, mhc);
                });
                const auto attended = attention(state, attention_mix.branch.dense(), stage, state.rings[index].dense());
                auto after_attention = tyloi::ops::measure("deepseek_v41.dspark.mhc.attention.expand", [&] {
                    return tyloi::ops::mhc_expand(attended, hidden.narrow(1, 0, attended.size(1)), attention_mix).dense();
                });
                const auto mlp_rows = output_rows;
                auto mlp_mix = tyloi::ops::measure("deepseek_v41.dspark.mhc.mlp.collapse", [&] {
                    return tyloi::ops::mhc_collapse(after_attention.narrow(1, 0, mlp_rows),
                        attention_mix.next_pre.dense().narrow(1, 0, mlp_rows), stage.mlp_mhc_function,
                        stage.mlp_mhc_scale, stage.mlp_mhc_base, stage.mlp_norm, mhc);
                });
                const auto rows = mlp_mix.branch.size(1);
                auto mlp = tyloi::ops::ffn(stage.mlp, mlp_mix.branch.dense().reshape({-1, config.hidden}),
                    draft_ids.narrow(1, 0, rows)).dense().reshape({state.batch, rows, config.hidden});
                auto result = tyloi::ops::measure("deepseek_v41.dspark.mhc.mlp.expand", [&] {
                    return tyloi::ops::mhc_expand(mlp, after_attention.narrow(1, 0, rows), mlp_mix).dense();
                });
                previous_pre = std::move(mlp_mix.next_pre.dense());
                return result;
#else
                auto attention_mix = stage.collapse(
                            hidden.narrow(1, 0, physical_width),
                            previous_pre.narrow(1, 0, physical_width),
                            stage.attention_mhc_function, stage.attention_mhc_scale,
                            stage.attention_mhc_base, stage.attention_norm,
                            "deepseek_v41.dspark.mhc.attention.collapse");
                const auto attended = attention(state, attention_mix.branch, stage,
                    state.rings[index]);
                auto after_attention = stage.expand(
                    attended, hidden.narrow(1, 0, attended.size(1)), attention_mix,
                    "deepseek_v41.dspark.mhc.attention.expand");
                const auto mlp_rows = output_rows;
                auto mlp_mix = stage.collapse(
                    after_attention.narrow(1, 0, mlp_rows),
                    attention_mix.next_pre.narrow(1, 0, mlp_rows),
                    stage.mlp_mhc_function, stage.mlp_mhc_scale, stage.mlp_mhc_base,
                    stage.mlp_norm,
                    "deepseek_v41.dspark.mhc.mlp.collapse");
                const auto rows = mlp_mix.branch.size(1);
                auto mlp = [&] {
                        auto executor = execution->executor().fork(model_parallel_primary_device(execution->config));
                        return stage.mlp.forward(mlp_mix.branch.reshape({-1, config.hidden}), *executor,
                                     execution->executor().forward_mode(execution->config.layer_placement.primary_device()).ffn,
                                     draft_ids.narrow(1, 0, rows))
                            .reshape({state.batch, rows, config.hidden});
                }();
                auto result = stage.expand(mlp, after_attention.narrow(1, 0, rows), mlp_mix,
                                           "deepseek_v41.dspark.mhc.mlp.expand");
                previous_pre = std::move(mlp_mix.next_pre);
                return result;
#endif
            },
            [&](DenseTensor hidden, const DenseTensor &previous_pre) {
                return (previous_pre.to(::tyloi::kFloat32).unsqueeze(-1) *
                        hidden.to(::tyloi::kFloat32))
                    .sum(2)
                    .to(::tyloi::kFloat16)
                    .contiguous();
            },
            [&](DenseTensor head_hidden) {
                auto normalized = weighted_rms(head_hidden,
#ifdef TYLOINFER_NATIVE_CUDA
                    output_norm, config.rms_eps);
#else
                    output_norm, config.rms_eps);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
                auto logits = tyloi::ops::measure("model.lm_head", [&] {
                    return tyloi::ops::project(normalized, output).dense();
                });
                if (softcap > 0.0)
                    logits = ::tyloi::cuda::tanh(logits / softcap) * softcap;
                return logits
                    .to(::tyloi::kFloat32)
                    .contiguous();
#else
                return tyloi::cuda::logits(output, std::move(normalized),
                    execution->executor().projector(execution->executor().mode().ffn, model_parallel_primary_device(execution->config)), execution->executor().profiler(), softcap)
                    .to(::tyloi::kFloat32)
                    .contiguous();
#endif
            },
            [](DenseTensor logits, int64_t count) { return logits.narrow(1, 0, count); })};
    }

    ::tyloi::Tensor draft_next(const tyloi::Tensor& input_logits,
                                              const tyloi::Tensor& input_previous) const override {
        const auto& logits = input_logits.dense();
        const auto& previous = input_previous.dense();
#ifdef TYLOINFER_NATIVE_CUDA
        auto markov = tyloi::ops::embedding(previous, markov_embedding);
        auto projected = tyloi::ops::project(markov.dense().to(::tyloi::kFloat16).contiguous(), markov_output).dense()
                             .to(::tyloi::kFloat32);
#else
        auto markov = quant_embedding_lookup(markov_embedding, previous)
                          .to(::tyloi::kFloat16)
                          .contiguous();
        auto projected = execution->executor().projector(execution->executor().mode().ffn, model_parallel_primary_device(execution->config)).run(markov_output, markov).to(::tyloi::kFloat32);
#endif
        return ::tyloi::Tensor{logits + projected};
    }
};

std::unique_ptr<tyloi::models::Mtp> load_dspark_if_present(
#ifndef TYLOINFER_NATIVE_CUDA
    CudaExecutionContext &execution,
#endif
    const tyloi::weights::Loader& weights, const CommonConfig &config,
    ::tyloi::Tensor embedding, ::tyloi::Tensor output, double softcap) {
    auto predictor = DeepseekV41Dspark::load_if_present(
#ifndef TYLOINFER_NATIVE_CUDA
        execution,
#endif
        weights, config,
        std::move(embedding), std::move(output), softcap);
    return predictor ? std::make_unique<DeepseekV41Dspark>(std::move(*predictor)) : nullptr;
}

} // namespace tyloi::cuda::deepseek_v41_runtime
