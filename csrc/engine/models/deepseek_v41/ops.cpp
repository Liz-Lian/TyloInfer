#include "engine/backends/cuda/executor/execution.h"
#include "engine/models/session.h"
#include "engine/models/common/load.h"
#include "engine/weights/loader.h"
#include "ops.h"
#include "vision.h"
#include "engine/models/common/check.h"
#include "engine/models/session_codec.h"

namespace tyloi::cuda::deepseek_v41_runtime {

tyloi::cuda::Ffn load_moe_at(Pipelines& pipelines, const tyloi::weights::Loader& weights,
                const CommonConfig &config, const std::string &prefix, std::int64_t layer,
                std::int64_t top_k) {
    auto result = load_moe_weights(weights, prefix,
                                   {.layer = static_cast<int>(layer),
                                    .router_bias_required = true,
                                    .shared_gate_up_compatible_prefix = 0});
    result.moe_top_k = static_cast<int>(top_k);
    result.moe_use_sqrt_softplus = true;
    result.moe_normalize = config.norm_topk_prob;
    result.moe_delayed_softmax = false;
    result.moe_shared_ungated = true;
    result.moe_router_scale = config.routed_scaling;
    result.swiglu_limit = config.swiglu_limit;
    result.shared->swiglu_limit = config.swiglu_limit;
    pipelines.prepare(result);
    return result;
}

std::unique_ptr<Block> load_block(Pipelines& pipelines, const tyloi::weights::Loader& weights,
                                    std::int64_t layer,
                                    const std::shared_ptr<SharedState> &shared, std::size_t engram_cache_rows) {
    TYLOINFER_RUNTIME_CHECK(shared, "missing DeepSeek-V4.1 CUDA state");
    const auto &config = shared->config;
    const auto prefix = "model.block." + std::to_string(layer) + ".";
    auto result = std::make_unique<Block>();
    result->config = config;
    result->layer = layer;
    result->ratio = config.compress_ratios.at(static_cast<std::size_t>(layer));
    result->max_context = config.max_position_embeddings;
    result->shared = shared;
    if (config.has_engram(layer)) {
        result->engram = Engram::load(weights, config, static_cast<int>(layer),
                                      engram_cache_rows);
        shared->engram_prefetch.push_back(result->engram.get());
    }
#ifdef TYLOINFER_NATIVE_CUDA
    result->attention_norm = std::make_shared<DenseTensor>(dynamic_cast<const DenseTensor&>(*weights.tensor(prefix + "attention.norm.weight"))
                                 .to(::tyloi::kFloat32).contiguous());
    result->mlp_norm = std::make_shared<DenseTensor>(dynamic_cast<const DenseTensor&>(*weights.tensor(prefix + "mlp.norm.weight"))
                           .to(::tyloi::kFloat32).contiguous());
    result->query_a_norm =
        weights.tensor(prefix + "attention.query_a_norm.weight");
    result->key_value_norm =
        weights.tensor(prefix + "attention.key_value_norm.weight");
    result->sinks = std::make_shared<DenseTensor>(dynamic_cast<const DenseTensor&>(*weights.tensor(prefix + "attention.sink"))
                        .to(::tyloi::kFloat32)
                        .contiguous());
    result->attention_mhc_function =
        std::make_shared<DenseTensor>(dynamic_cast<const DenseTensor&>(*weights.tensor(prefix + "attention.mhc.pre.function"))
            .to(::tyloi::kFloat32)
            .contiguous());
    result->attention_mhc_scale =
        std::make_shared<DenseTensor>(dynamic_cast<const DenseTensor&>(*weights.tensor(prefix + "attention.mhc.pre.scale"))
            .to(::tyloi::kFloat32)
            .contiguous());
    result->attention_mhc_base = std::make_shared<DenseTensor>(dynamic_cast<const DenseTensor&>(*weights.tensor(prefix + "attention.mhc.pre.base"))
                                     .to(::tyloi::kFloat32)
                                     .contiguous());
    result->mlp_mhc_function = std::make_shared<DenseTensor>(dynamic_cast<const DenseTensor&>(*weights.tensor(prefix + "mlp.mhc.pre.function"))
                                   .to(::tyloi::kFloat32)
                                   .contiguous());
    result->mlp_mhc_scale = std::make_shared<DenseTensor>(dynamic_cast<const DenseTensor&>(*weights.tensor(prefix + "mlp.mhc.pre.scale"))
                                .to(::tyloi::kFloat32)
                                .contiguous());
    result->mlp_mhc_base = std::make_shared<DenseTensor>(dynamic_cast<const DenseTensor&>(*weights.tensor(prefix + "mlp.mhc.pre.base"))
                               .to(::tyloi::kFloat32)
                               .contiguous());
#else
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
    result->attention_mhc_base = dynamic_cast<const ::tyloi::DenseTensor&>(*weights.tensor(prefix + "attention.mhc.pre.base"))
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
#endif
    result->query_a = weights.weight(prefix + "attention.query_a.weight");
    result->query_b = weights.weight(prefix + "attention.query_b.weight");
    result->key_value = weights.weight(prefix + "attention.key_value.weight");
    result->output_a = weights.weight(prefix + "attention.output_a.weight");
    result->output_b = weights.weight(prefix + "attention.output_b.weight");
    if (result->kv_source()) {
        result->compressor_key_value =
            weights.weight(prefix + "attention.compressor.key_value.weight");
#ifdef TYLOINFER_NATIVE_CUDA
        result->compressor_norm = weights.tensor(prefix + "attention.compressor.norm.weight");
#else
        result->compressor_norm =
            dynamic_cast<const ::tyloi::DenseTensor&>(*weights.tensor(prefix + "attention.compressor.norm.weight"));
#endif
        result->index_key =
            weights.weight(prefix + "attention.indexer.key.weight");
#ifdef TYLOINFER_NATIVE_CUDA
        result->index_key_norm = weights.tensor(prefix + "attention.indexer.key_norm.weight");
#else
        result->index_key_norm =
            dynamic_cast<const ::tyloi::DenseTensor&>(*weights.tensor(prefix + "attention.indexer.key_norm.weight"));
#endif
        if (result->ratio > 1) {
            result->compressor_gate =
                weights.weight(prefix + "attention.compressor.gate.weight");
        }
    }
    if (result->index_source()) {
        result->index_query =
            weights.weight(prefix + "attention.indexer.query.weight");
        result->index_score =
            weights.weight(prefix + "attention.indexer.score.weight");
    }
    result->mlp = load_moe_at(pipelines, weights, config, prefix + "mlp.", layer, config.top_k);
    if(weights.has(prefix+"mlp.router.vision_bias")) {
#ifdef TYLOINFER_NATIVE_CUDA
        result->vision_router_bias=std::make_shared<DenseTensor>(
            dynamic_cast<const DenseTensor&>(*weights.tensor(prefix+"mlp.router.vision_bias"))
                .to(::tyloi::kFloat32).contiguous());
        TYLOINFER_RUNTIME_CHECK(result->vision_router_bias->dim()==1 && result->vision_router_bias->size(0)==config.n_experts,
                          "DeepSeek-V4.1 vision router bias shape mismatch");
#else
        result->vision_router_bias=dynamic_cast<const ::tyloi::DenseTensor&>(*weights.tensor(prefix+"mlp.router.vision_bias"))
            .to(::tyloi::kFloat32).contiguous();
        TYLOINFER_RUNTIME_CHECK(result->vision_router_bias.dim()==1 && result->vision_router_bias.size(0)==config.n_experts,
                          "DeepSeek-V4.1 vision router bias shape mismatch");
#endif
    }
    if(shared->router_prefetch.empty())shared->router_prefetch.resize(config.n_layers,nullptr);
    shared->router_prefetch.at(static_cast<std::size_t>(layer))=&result->mlp;
#ifdef TYLOINFER_NATIVE_CUDA
    result->rope = ::tyloi::ops::tail_rope_table(
        config.max_position_embeddings, config.rope_theta, result->ratio,
        ::tyloi::Device(::tyloi::kCUDA, weights.placement.device),
        config.compress_rope_theta, config.rope_scaling.original_max_position_embeddings,
        config.rope_scaling.factor, config.rope_scaling.beta_fast, config.rope_scaling.beta_slow);
#else
    result->rope = TailRope(
        config.max_position_embeddings, config.rope_theta, result->ratio,
        config.compress_rope_theta, config.rope_scaling.original_max_position_embeddings,
        config.rope_scaling.factor, config.rope_scaling.beta_fast, config.rope_scaling.beta_slow);
#endif

    const auto function_width = config.hc_mult * config.hidden;
#ifdef TYLOINFER_NATIVE_CUDA
    const auto valid_mhc = [&](const Parameter &function, const Parameter &scale, const Parameter &base,
                               const Parameter &norm) {
        return function->dim() == 2 && function->size(0) == 24 &&
               function->size(1) == function_width && scale->numel() == 3 && base->numel() == 24 &&
               norm->numel() == config.hidden;
    };
#else
    const auto valid_mhc = [&](const DenseTensor &function, const DenseTensor &scale, const DenseTensor &base,
                               const DenseTensor &norm) {
        return function.dim() == 2 && function.size(0) == 24 &&
               function.size(1) == function_width && scale.numel() == 3 && base.numel() == 24 &&
               norm.numel() == config.hidden;
    };
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    const bool base_shapes =
        valid_mhc(result->attention_mhc_function, result->attention_mhc_scale,
                  result->attention_mhc_base, result->attention_norm) &&
        valid_mhc(result->mlp_mhc_function, result->mlp_mhc_scale, result->mlp_mhc_base,
                  result->mlp_norm) &&
        result->query_a_norm->numel() == config.q_lora_rank &&
        result->key_value_norm->numel() == config.head_dim &&
        result->sinks->numel() == config.n_heads && result->query_a->size(1) == config.hidden &&
        result->query_a->size(0) == config.q_lora_rank &&
        result->query_b->size(1) == config.q_lora_rank &&
        result->query_b->size(0) == config.n_heads * config.head_dim &&
        result->key_value->size(1) == config.hidden &&
        result->key_value->size(0) == config.head_dim &&
        result->output_a->size(1) == config.n_heads * config.head_dim / config.o_groups &&
        result->output_a->size(0) == config.o_groups * config.o_lora_rank &&
        result->output_b->size(1) == config.o_groups * config.o_lora_rank &&
        result->output_b->size(0) == config.hidden;
    const bool source_shapes =
        !result->kv_source() ||
        (result->compressor_key_value->size(1) == config.hidden &&
         result->compressor_key_value->size(0) == config.head_dim &&
         result->compressor_norm->numel() == config.head_dim &&
         result->index_key->size(1) == config.head_dim &&
         result->index_key->size(0) == config.index_head_dim &&
         result->index_key_norm->numel() == config.index_head_dim &&
         (result->ratio == 1 || (result->compressor_gate->size(1) == config.hidden &&
                                 result->compressor_gate->size(0) == config.head_dim)));
#else
    const bool base_shapes =
        valid_mhc(result->attention_mhc_function, result->attention_mhc_scale,
                  result->attention_mhc_base, result->attention_norm) &&
        valid_mhc(result->mlp_mhc_function, result->mlp_mhc_scale, result->mlp_mhc_base,
                  result->mlp_norm) &&
        result->query_a_norm.numel() == config.q_lora_rank &&
        result->key_value_norm.numel() == config.head_dim &&
        result->sinks.numel() == config.n_heads && result->query_a->size(1) == config.hidden &&
        result->query_a->size(0) == config.q_lora_rank &&
        result->query_b->size(1) == config.q_lora_rank &&
        result->query_b->size(0) == config.n_heads * config.head_dim &&
        result->key_value->size(1) == config.hidden &&
        result->key_value->size(0) == config.head_dim &&
        result->output_a->size(1) == config.n_heads * config.head_dim / config.o_groups &&
        result->output_a->size(0) == config.o_groups * config.o_lora_rank &&
        result->output_b->size(1) == config.o_groups * config.o_lora_rank &&
        result->output_b->size(0) == config.hidden;
    const bool source_shapes =
        !result->kv_source() ||
        (result->compressor_key_value->size(1) == config.hidden &&
         result->compressor_key_value->size(0) == config.head_dim &&
         result->compressor_norm.numel() == config.head_dim &&
         result->index_key->size(1) == config.head_dim &&
         result->index_key->size(0) == config.index_head_dim &&
         result->index_key_norm.numel() == config.index_head_dim &&
         (result->ratio == 1 || (result->compressor_gate->size(1) == config.hidden &&
                                 result->compressor_gate->size(0) == config.head_dim)));
#endif
    const bool index_shapes =
        !result->index_source() ||
        (result->index_query->size(1) == config.q_lora_rank &&
         result->index_query->size(0) == config.index_n_heads * config.index_head_dim &&
         result->index_score->size(1) == config.hidden &&
         result->index_score->size(0) == config.index_n_heads);
    TYLOINFER_RUNTIME_CHECK(config.hc_mult == 4 && config.n_heads == 64 && config.head_dim == 512 &&
                          config.rope_head_dim == 64 && config.index_n_heads == 32 &&
                          config.index_head_dim == 128 && base_shapes && source_shapes &&
                          index_shapes,
                      "DeepSeek-V4.1 CUDA tensor geometry disagrees at layer ", layer);
    return result;
}

void validate_load_options(const CudaExecutionContext &execution) {
    if (execution.config.tensor_parallel.enabled() || execution.config.layer_placement.enabled() ||
        execution.config.n_gpu_layers >= 0) {
        throw std::runtime_error("DeepSeek-V4.1 native CUDA currently supports single-device dense "
                                 "placement or expert parallelism");
    }
}

std::shared_ptr<SharedState> load_shared_state(const tyloi::source::Source &source,
                                               const CommonConfig &config) {
    auto state = std::make_shared<SharedState>();
    state->config = config;
    state->layers.resize(static_cast<std::size_t>(config.n_layers));
    state->engram_hash = EngramHashState::load(source, config);
    return state;
}

DenseTensor finalize_hidden(const std::shared_ptr<SharedState> &state, const DenseTensor &hidden,
                       CudaProfiler &profiler) {
    TYLOINFER_RUNTIME_CHECK(state, "DeepSeek-V4.1 final state is unavailable");
    return profiler.measure("model.deepseek_v41.final_collapse", [&]() {
        return state->final_collapse(hidden, state->config.n_layers);
    });
}

} // namespace tyloi::cuda::deepseek_v41_runtime

namespace tyloi::cuda::deepseek_v41 {

using deepseek_v41_runtime::Block;
using tyloi::models::require_model;

void Model::load_config(std::string_view payload) {
    auto& config = this->config;
    auto& metadata = this->metadata;
    config = CausalConfig::from_json(payload);
    metadata.vocab_size = config.vocab;
    metadata.hidden_size = config.hidden;
    metadata.num_hidden_layers = config.n_layers;
    metadata.num_attention_heads = config.n_heads;
    metadata.num_key_value_heads = config.n_kv_heads;
    metadata.head_dim = config.head_dim;
    metadata.max_position_embeddings = config.max_position_embeddings;
    metadata.rotary_dim = config.rope_head_dim;
    metadata.num_experts = config.n_experts;
    metadata.hc_mult = config.hc_mult;
    metadata.rope_base = config.rope_theta;
    metadata.rms_norm_eps = config.rms_eps;
    metadata.hc_eps = config.hc_eps;
    metadata.model_type = config.text_model_type;
    metadata.layer_types.assign(static_cast<std::size_t>(config.n_layers), "deepseek_v41");
}

tyloi::models::Info Model::info() const {
    return {metadata.vocab_size, config.max_position_embeddings, config.text_model_type,
            std::size_t(metadata.hidden_size) * sizeof(float) * 8, true, true};
}

MfqModelCapabilities Model::capabilities() const {
    const auto state = cuda_component_state(graph, plan, bool(vision), false);
    const auto caps = cuda_runtime_capabilities(graph, plan, state, bool(vision), false);
    return {graph.architecture, caps.text, caps.image_input, caps.video_input,
        caps.audio_input, caps.audio_output, caps.full_duplex, caps.mtp,
        "model-graph+cuda-adapters"};
}

tyloi::engine::Metrics Model::metrics() const {
    const auto state = cuda_component_state(graph, plan, bool(vision), false);
    return {{"kv_memory_supported", 0.0},
        {"kv_quantization_supported", 0.0},
        {"kv_quantization_bits", 0.0},
        {"vision_declared", double(state.vision_declared)},
        {"vision_supported", double(state.vision_supported)},
        {"vision_available", double(state.vision_available)},
        {"mtp_declared", double(state.mtp_declared)},
        {"mtp_supported", double(state.mtp_supported)},
        {"mtp_available", double(state.mtp_available)}};
}

std::unique_ptr<tyloi::models::State> Model::make_state() const {
    // shortcut: states are single-flight per engine; the bridge is released in ~State
    return std::make_unique<State>(const_cast<Model&>(*this));
}

tyloi::StepSequence<tyloi::models::Prepared> Model::prepare(
    const std::vector<std::int64_t>& tokens, const MfqMultimodalInput& media,
    tyloi::models::State&) const {
    if (!vision)
        throw std::invalid_argument("model has no multimodal component");
    return vision->prepare(const_cast<Model&>(*this), const_cast<Model&>(*this).media, tokens, media);
}

std::shared_ptr<tyloi::Tensor> Model::embedding(const tyloi::Tensor& ids, int) const {
    auto scope = execution_scope();
    return std::make_shared<DenseTensor>(embed_tokens(device_ids(dynamic_cast<const DenseTensor&>(ids))));
}

std::shared_ptr<tyloi::Tensor> Model::normalize(const tyloi::Tensor& hidden, std::int64_t, std::int64_t) const {
    return std::make_shared<DenseTensor>(dynamic_cast<const DenseTensor&>(hidden));
}

std::shared_ptr<tyloi::Tensor> Model::softcap(const tyloi::Tensor& logits) const {
    return std::make_shared<DenseTensor>(dynamic_cast<const DenseTensor&>(logits));
}

std::shared_ptr<tyloi::Tensor> Model::logits(const tyloi::Tensor& hidden, bool) const {
    return std::make_shared<DenseTensor>(adapter_logits(*lm_head, dynamic_cast<const DenseTensor&>(hidden)));
}

bool Model::supports_speculation() const {
    return !blocks.empty();
}

void Model::adapter_validate_load_options() const {
    deepseek_v41_runtime::validate_load_options(*execution);
}

void Model::adapter_load_final_state(const tyloi::weights::Loader& weights) {
#ifdef TYLOINFER_NATIVE_CUDA
    output_norm = weights.tensor("model.output_norm.weight");
#else
    output_norm = dynamic_cast<const ::tyloi::DenseTensor&>(*weights.tensor("model.output_norm.weight"));
#endif
}

void Model::adapter_prepare_blocks(const tyloi::weights::Loader& weights) {
    shared = deepseek_v41_runtime::load_shared_state(weights.source, config);
}

bool Model::adapter_prefetch_token_rows(Model&, const ::tyloi::DenseTensor& ids) {
    const auto cache_pos = active ? active->cache_pos : shared->cache_position;
    if(!shared || shared->batch!=1 || !shared->engram_hash || shared->engram_prefetch.empty())return false;
    if(!ids.is_cpu() || ids.dim()!=2 || ids.size(0)!=1 || ids.size(1)<=0 ||
            ids.size(1)>std::numeric_limits<int>::max() || ids.scalar_type()!=::tyloi::kInt64 ||
            !ids.is_contiguous())throw std::invalid_argument("Engram prefetch needs contiguous host token IDs");
    const auto hashes=shared->engram_hash->preview(ids.data_ptr<int64_t>(),1,int(ids.size(1)),cache_pos);
    for(const auto* memory:shared->engram_prefetch)memory->prefetch(hashes);
    return true;
}

std::unique_ptr<deepseek_v41_runtime::Block> Model::adapter_load_block(
        const tyloi::weights::Loader& weights, int layer, int,
        const std::string& type) {
    TYLOINFER_RUNTIME_CHECK(type == "deepseek_v41" && shared, "invalid DeepSeek-V4.1 block loader state");
    return deepseek_v41_runtime::load_block(execution->executor().pipelines(), weights, layer, shared,
                                            execution->config.deepseek_v41_engram_cache_rows);
}

void Model::adapter_begin_forward(bool capture_raw_hidden) const {
    TYLOINFER_RUNTIME_CHECK(shared, "DeepSeek-V4.1 target capture state is unavailable");
    shared->begin_forward(capture_raw_hidden, shared->config.dspark_target_layer_ids.size());
}

::tyloi::DenseTensor Model::collapse_hidden(::tyloi::DenseTensor hidden, std::int64_t,
                                            std::int64_t) const {
    return deepseek_v41_runtime::finalize_hidden(shared, hidden, execution->executor().profiler());
}

::tyloi::DenseTensor Model::normalize_hidden(::tyloi::DenseTensor hidden,
#ifdef TYLOINFER_NATIVE_CUDA
                                             const ::tyloi::Tensor& output_norm,
#else
                                             const ::tyloi::DenseTensor& output_norm,
#endif
                                             std::int64_t batch, std::int64_t tokens) const {
    return execution->executor().profiler().measure("model.output_norm", [&]() {
        return qwen_rms_norm(hidden.reshape({batch * tokens, metadata.hidden_size})
                                 .to(::tyloi::kFloat32),
                             output_norm, metadata.rms_norm_eps, metadata.norm_weight_offset)
            .reshape({batch, tokens, metadata.hidden_size});
    });
}

::tyloi::DenseTensor Model::adapter_prepare_hidden(::tyloi::DenseTensor hidden, std::int64_t batch,
                                                   std::int64_t tokens) const {
    const auto width = hidden.size(-1);
    return hidden.to(::tyloi::kFloat16)
        .unsqueeze(2)
        .expand({batch, tokens, metadata.hc_mult, width})
        .contiguous();
}

::tyloi::DenseTensor Model::adapter_finalize_hidden(::tyloi::DenseTensor hidden,
                                                    std::int64_t batch, std::int64_t tokens) const {
    auto collapsed = collapse_hidden(std::move(hidden), batch, tokens);
#ifdef TYLOINFER_NATIVE_CUDA
    return normalize_hidden(std::move(collapsed), *output_norm, batch, tokens);
#else
    return normalize_hidden(std::move(collapsed), output_norm, batch, tokens);
#endif
}

::tyloi::DenseTensor Model::adapter_raw_hidden(const ::tyloi::DenseTensor&,
                                               const ::tyloi::DenseTensor&) const {
    return shared->dspark_target_hidden();
}

void Model::adapter_validate_forward(std::int64_t, std::int64_t tokens, std::int64_t cache_position,
                                     bool has_position_override, bool has_cache_position_override) const {
    require_model(
        tokens > 0 && cache_position + tokens <= metadata.max_position_embeddings &&
            !has_position_override && !has_cache_position_override,
        "DeepSeek-V4.1 currently requires contiguous causal cache positions");
}

void Model::adapter_begin_speculative() {
    TYLOINFER_RUNTIME_CHECK(shared, "DeepSeek-V4.1 speculative state is unavailable");
    shared->begin_speculative();
}

void Model::adapter_commit_speculative() {
    TYLOINFER_RUNTIME_CHECK(shared, "DeepSeek-V4.1 speculative state is unavailable");
    shared->commit_speculative();
}

void Model::adapter_rollback_speculative(std::int64_t position) {
    TYLOINFER_RUNTIME_CHECK(shared, "DeepSeek-V4.1 speculative state is unavailable");
    shared->rollback_speculative(position);
}

::tyloi::DenseTensor Model::hidden_forward_speculative_suffix(::tyloi::DenseTensor ids,
                                                              ::tyloi::DenseTensor* raw_hidden) {
    auto& state = *active;
    require_model(state.speculative_start >= 0 && state.speculative_confirmed == 0 &&
                      !state.speculative_suffix_forward,
                  "speculative suffix is not active");
    state.speculative_suffix_forward = true;
    try {
        std::shared_ptr<tyloi::Tensor> raw;
        auto result = forward(tyloi::models::Input{std::make_shared<DenseTensor>(std::move(ids)), {}, {}, {}, {}, nullptr,
                                                   raw_hidden ? &raw : nullptr, false, 0, 0, 0},
                              state);
        if (raw_hidden) *raw_hidden = dynamic_cast<const DenseTensor&>(*raw);
        state.speculative_suffix_forward = false;
        return dynamic_cast<const DenseTensor&>(*result);
    } catch (...) {
        state.speculative_suffix_forward = false;
        throw;
    }
}

void Model::rollback_speculative(std::int64_t accepted) {
    auto& state = *active;
    require_model(state.speculative_start >= 0 && accepted >= 0, "no speculative transaction");
    const auto keep = state.speculative_start + state.speculative_confirmed + accepted;
    for (auto& block : blocks)
        block->rollback_speculative(keep);
    adapter_rollback_speculative(keep);
    state.cache_pos = keep;
    state.speculative_start = -1;
    state.speculative_confirmed = 0;
}

// The decode forward, ported from the retired generic causal loop. The block
// records own the layer math; this owns geometry, speculation bookkeeping and
// device placement of the walked tensors.
std::shared_ptr<tyloi::Tensor> Model::forward(tyloi::models::Input input, tyloi::models::State& state_value,
                                    const tyloi::models::ForwardOptions&) const {
    auto& state = static_cast<State&>(state_value);
    auto scope = execution_scope();
    const auto primary = execution->config.layer_placement.primary_device();
    require_model(input.ids && input.ids->defined(), "token ids are required");
    auto ids = device_ids(dynamic_cast<const DenseTensor&>(*input.ids));
    auto embeddings = input.embeddings
        ? execution->executor().collectives().copy(dynamic_cast<const DenseTensor&>(*input.embeddings), primary).contiguous()
        : embed_tokens(ids);
    const auto B = ids.size(0), T = ids.size(1);
    require_model(embeddings.dim() == 3 && embeddings.size(0) == B && embeddings.size(1) == T &&
                      embeddings.size(2) == hidden_size(),
                  "inputs_embeds shape must match [batch,tokens,hidden_size]");
    adapter_validate_forward(B, T, state.cache_pos, bool(input.positions),
                             bool(input.cache_positions));
    require_model(state.speculative_start < 0 || state.speculative_suffix_forward,
                  "commit or roll back the pending speculative pass before forwarding");
    const auto confirmed = input.confirmed;
    require_model(confirmed >= 0 && (confirmed == 0 ||
                      (confirmed < T && B == 1 && state.cache_pos > 0 && supports_speculation())),
                  "unsupported speculative backbone geometry");
    if (confirmed > 0) {
        require_model(state.cache_pos + T <= max_position_embeddings(),
                      "speculative pass exceeds context capacity");
        state.speculative_start = state.cache_pos;
        state.speculative_confirmed = confirmed;
    }
    if (state.cache_pos == 0)
        state.reset(B);
    auto positions = ::tyloi::cuda::arange(state.cache_pos, state.cache_pos + T,
                                           ::tyloi::TensorOptions()
                                               .device(::tyloi::kCUDA)
                                               .dtype(::tyloi::kInt64));
    if (state.decode_position_delta != 0)
        positions = positions + state.decode_position_delta;
    DenseTensor host_ids, cpu_positions;
    if (execution->config.cache.pipeline || execution->config.dense_cpu_layer_count > 0) {
        host_ids = ids.to(::tyloi::kCPU, ::tyloi::kInt64).contiguous();
        cpu_positions = positions.to(::tyloi::kCPU, ::tyloi::kInt64).contiguous();
    }
    auto hidden = adapter_prepare_hidden(std::move(embeddings), B, T);
    adapter_begin_forward(input.raw_hidden != nullptr);
    if (input.trace)
        input.trace->push_back(std::make_shared<DenseTensor>(hidden.to(::tyloi::kFloat32).clone()));
    const auto trace_layer = [&] {
        if (!input.trace) return;
        input.trace->push_back(std::make_shared<DenseTensor>(
            execution->executor().collectives().copy(hidden, primary).to(::tyloi::kFloat32).clone()));
    };
    for (auto& block : blocks) {
        MfqCudaGuard guard(block->cuda_device);
        auto local_ids = block->cpu_offloaded
            ? host_ids
            : execution->executor().collectives().copy(ids, block->cuda_device);
        auto local_pos = block->cpu_offloaded
            ? cpu_positions
            : execution->executor().collectives().copy(positions, block->cuda_device);
        block->set_token_ids(local_ids);
        BlockContext context;
        context.token_ids = local_ids;
        context.host_token_ids = host_ids;
        context.positions = local_pos;
        context.full_positions = local_pos;
        context.cache_position = state.cache_pos;
        context.confirmed_prefix = confirmed;
        context.planned_kv_length = input.kv_length;
        context.decode_attention_parts = input.attention_parts;
        if (block->cpu_offloaded)
            hidden = hidden.to(::tyloi::kCPU).contiguous();
        else
            hidden = execution->executor().collectives().copy(hidden, block->cuda_device);
        hidden = block->forward_context(*execution, std::move(hidden), context);
        trace_layer();
    }
    // finish hook is the CausalResources no-op for this family
    state.cache_pos += T;
    hidden = execution->executor().collectives().copy(hidden, primary);
    auto finalized = adapter_finalize_hidden(std::move(hidden), B, T);
    if (input.raw_hidden)
        *input.raw_hidden = std::make_shared<DenseTensor>(adapter_raw_hidden(DenseTensor(), finalized));
    return std::make_shared<DenseTensor>(std::move(finalized));
}

} // namespace tyloi::cuda::deepseek_v41
