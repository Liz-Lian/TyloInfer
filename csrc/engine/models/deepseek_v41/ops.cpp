#include "engine/backends/cuda/executor/moe_ffn_pipeline.h"
#include "engine/models/session/snapshot.h"
#include "engine/models/common/load.h"
#include "engine/weights/loader.h"
#include "ops.h"
#include "vision.h"
#include "engine/models/common/check.h"
#include "engine/models/session/codec.h"

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
                                    const CommonConfig& config, ModelTables& tables,
                                    std::size_t engram_cache_rows) {
    const auto prefix = "model.block." + std::to_string(layer) + ".";
    auto result = std::make_unique<Block>();
    result->config = config;
    result->layer = layer;
    result->ratio = config.compress_ratios.at(static_cast<std::size_t>(layer));
    result->max_context = config.max_position_embeddings;
    if (config.has_engram(layer)) {
        result->engram = Engram::load(weights, config, static_cast<int>(layer),
                                      engram_cache_rows);
        tables.engram_prefetch.push_back(result->engram.get());
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
    if(tables.router_prefetch.empty())tables.router_prefetch.resize(config.n_layers,nullptr);
    tables.router_prefetch.at(static_cast<std::size_t>(layer))=&result->mlp;
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

DenseTensor finalize_hidden(const SharedState &state, const DenseTensor &hidden) {
    return tyloi::ops::measure("model.deepseek_v41.final_collapse", [&]() {
        return state.final_collapse(hidden, state.config.n_layers);
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
    return std::make_unique<State>(const_cast<Model&>(*this));
}

tyloi::StepSequence<tyloi::models::Prepared> Model::prepare(
    const std::vector<std::int64_t>& tokens, const MfqMultimodalInput& media,
    tyloi::models::State&) const {
    if (!vision)
        throw std::invalid_argument("model has no multimodal component");
    return vision->prepare(const_cast<Model&>(*this), const_cast<Model&>(*this).media, tokens, media);
}

::tyloi::Tensor Model::embedding(const tyloi::Tensor& ids, int) const {
#ifndef TYLOINFER_NATIVE_CUDA
    auto scope = execution_scope();
#endif
    return std::make_shared<DenseTensor>(embed_tokens(device_ids(ids.dense())));
}

std::shared_ptr<tyloi::Tensor> Model::normalize(const tyloi::Tensor& hidden, std::int64_t, std::int64_t) const {
    return std::make_shared<DenseTensor>(dynamic_cast<const DenseTensor&>(hidden));
}

std::shared_ptr<tyloi::Tensor> Model::softcap(const tyloi::Tensor& logits) const {
    return std::make_shared<DenseTensor>(dynamic_cast<const DenseTensor&>(logits));
}

::tyloi::Tensor Model::logits(const tyloi::Tensor& hidden, bool) const {
    return tyloi::ops::measure("model.lm_head", [&] {
        return tyloi::ops::project(hidden, lm_head);
    });
}

bool Model::supports_speculation() const {
    return !blocks.empty();
}

bool Model::prefetch_token_rows(State& state, const ::tyloi::Tensor& input_ids) const {
    const auto& ids = input_ids.dense();
    auto& shared = *state.shared;
    const auto cache_pos = state.cache_pos;
    if(shared.batch!=1 || !shared.engram_hash || tables.engram_prefetch.empty())return false;
    if(!ids.is_cpu() || ids.dim()!=2 || ids.size(0)!=1 || ids.size(1)<=0 ||
            ids.size(1)>std::numeric_limits<int>::max() || ids.scalar_type()!=::tyloi::kInt64 ||
            !ids.is_contiguous())throw std::invalid_argument("Engram prefetch needs contiguous host token IDs");
    const auto hashes=shared.engram_hash->preview(ids.data_ptr<int64_t>(),1,int(ids.size(1)),cache_pos);
    for(const auto* memory:tables.engram_prefetch)
        memory->prefetch(*shared.engram_states.at(static_cast<std::size_t>(memory->layer_id())), hashes);
    return true;
}

::tyloi::DenseTensor Model::normalize_hidden(::tyloi::DenseTensor hidden,
#ifdef TYLOINFER_NATIVE_CUDA
                                             const ::tyloi::Tensor& output_norm,
#else
                                             const ::tyloi::DenseTensor& output_norm,
#endif
                                             std::int64_t batch, std::int64_t tokens) const {
    return tyloi::ops::measure("model.output_norm", [&]() {
        return qwen_rms_norm(hidden.reshape({batch * tokens, metadata.hidden_size})
                                 .to(::tyloi::kFloat32),
                             output_norm, metadata.rms_norm_eps, metadata.norm_weight_offset)
            .reshape({batch, tokens, metadata.hidden_size});
    });
}

::tyloi::DenseTensor Model::prepare_hidden(::tyloi::DenseTensor hidden, std::int64_t batch,
                                                   std::int64_t tokens) const {
    const auto width = hidden.size(-1);
    return hidden.to(::tyloi::kFloat16)
        .unsqueeze(2)
        .expand({batch, tokens, metadata.hc_mult, width})
        .contiguous();
}

::tyloi::DenseTensor Model::finalize_hidden(State& state, ::tyloi::DenseTensor hidden,
                                                    std::int64_t batch, std::int64_t tokens) const {
    auto collapsed = deepseek_v41_runtime::finalize_hidden(*state.shared, hidden);
#ifdef TYLOINFER_NATIVE_CUDA
    return normalize_hidden(std::move(collapsed), *output_norm, batch, tokens);
#else
    return normalize_hidden(std::move(collapsed), output_norm, batch, tokens);
#endif
}

void Model::validate_forward(std::int64_t, std::int64_t tokens, std::int64_t cache_position,
                                     bool has_position_override, bool has_cache_position_override) const {
    require_model(
        tokens > 0 && cache_position + tokens <= metadata.max_position_embeddings &&
            !has_position_override && !has_cache_position_override,
        "DeepSeek-V4.1 currently requires contiguous causal cache positions");
}

::tyloi::Tensor Model::hidden_forward_speculative_suffix(State& state, ::tyloi::Tensor ids,
                                                       ::tyloi::Tensor* raw_hidden) {
    require_model(state.speculative_start >= 0 && state.speculative_confirmed == 0 &&
                      !state.speculative_suffix_forward,
                  "speculative suffix is not active");
    state.speculative_suffix_forward = true;
    try {
        auto result = forward(tyloi::models::Input{std::move(ids), {}, {}, {}, {}, nullptr,
                                                   raw_hidden, false, 0, 0, 0},
                              state);
        state.speculative_suffix_forward = false;
        return result;
    } catch (...) {
        state.speculative_suffix_forward = false;
        throw;
    }
}

// The decode forward, ported from the retired generic causal loop. The block
// records own the layer math; this owns geometry, speculation bookkeeping and
// device placement of the walked tensors.
std::shared_ptr<tyloi::Tensor> Model::forward(tyloi::models::Input input, tyloi::models::State& state_value,
                                    const tyloi::models::ForwardOptions&) const {
    auto& state = static_cast<State&>(state_value);
#ifndef TYLOINFER_NATIVE_CUDA
    auto scope = execution_scope();
#endif
    const auto primary = ::tyloi::Device(::tyloi::kCUDA, primary_device);
    require_model(input.ids && input.ids.defined(), "token ids are required");
    auto ids = device_ids(input.ids.dense());
    auto embeddings = input.embeddings
        ? tyloi::ops::contiguous(tyloi::ops::copy(input.embeddings, primary)).dense()
        : embed_tokens(ids);
    const auto B = ids.size(0), T = ids.size(1);
    require_model(embeddings.dim() == 3 && embeddings.size(0) == B && embeddings.size(1) == T &&
                      embeddings.size(2) == hidden_size(),
                  "inputs_embeds shape must match [batch,tokens,hidden_size]");
    validate_forward(B, T, state.cache_pos, bool(input.positions),
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
    if (needs_host_tokens) {
        host_ids = ids.to(::tyloi::kCPU, ::tyloi::kInt64).contiguous();
        cpu_positions = positions.to(::tyloi::kCPU, ::tyloi::kInt64).contiguous();
    }
    auto hidden = prepare_hidden(std::move(embeddings), B, T);
    state.begin_forward(input.raw_hidden != nullptr);
    if (input.trace)
        input.trace->push_back(std::make_shared<DenseTensor>(hidden.to(::tyloi::kFloat32).clone()));
    const auto trace_layer = [&] {
        if (!input.trace) return;
        input.trace->push_back(std::make_shared<DenseTensor>(
            tyloi::ops::copy(hidden, primary).dense().to(::tyloi::kFloat32).clone()));
    };
    for (auto& block : blocks) {
        MfqCudaGuard guard(block->cuda_device);
        auto local_ids = block->cpu_offloaded
            ? host_ids
            : tyloi::ops::copy(ids, ::tyloi::Device(::tyloi::kCUDA, block->cuda_device)).dense();
        auto local_pos = block->cpu_offloaded
            ? cpu_positions
            : tyloi::ops::copy(positions, ::tyloi::Device(::tyloi::kCUDA, block->cuda_device)).dense();
        block->set_token_ids(*state.shared, local_ids);
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
            hidden = tyloi::ops::copy(hidden, ::tyloi::Device(::tyloi::kCUDA, block->cuda_device)).dense();
        hidden = block->forward_context(*state.shared,
#ifndef TYLOINFER_NATIVE_CUDA
                                        *execution,
#endif
                                        std::move(hidden), context);
        trace_layer();
    }
    state.cache_pos += T;
    hidden = tyloi::ops::copy(hidden, primary).dense();
    auto finalized = finalize_hidden(state, std::move(hidden), B, T);
    if (input.raw_hidden)
        *input.raw_hidden = state.shared->dspark_target_hidden();
    return std::make_shared<DenseTensor>(std::move(finalized));
}

} // namespace tyloi::cuda::deepseek_v41
