#include "engine/models/session/snapshot.h"
#include "engine/models/common/load.h"
#include "engine/weights/loader.h"
#ifndef TYLOINFER_NATIVE_CUDA
#include "engine/backends/cuda/ops/projection/projection.h"
#endif
#include "engine/models/glm_dsa/snapshot.h"
#include "engine/models/glm_dsa/model.h"
#include "engine/models/glm_dsa/state.h"
#include "engine/models/glm_dsa/forward.h"
#include "engine/models/glm_dsa/load.h"
#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/models/load.h"
#endif
#include "engine/models/glm_dsa/weights.h"
#include "engine/backends/cuda/native/collectives.h"
#include "engine/backends/cuda/ops/norm/rms.h"
#include "engine/backends/cuda/kernels/mfq_cuda_attention_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_cache_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_norm_ops.h"
#include "engine/ops.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using ::tyloi::indexing::Slice;

namespace tyloi::models::glm_dsa {
template <class Input, class QueryNorm, class Query, class Prepare, class Cache, class Index,
          class Absorb, class Attend, class Unembed, class Output>
auto attention(bool full_indexer, Input input, QueryNorm query_norm, Query query, Prepare prepare,
               Cache cache, Index index, Absorb absorb, Attend attend, Unembed unembed,
               Output output) {
    auto first = input();
    require_model(first.size() == (full_indexer ? 4u : 2u),
                  "GLM DSA input projection count mismatch");
    auto second = query(query_norm(first[0]));
    require_model(second.size() == (full_indexer ? 2u : 1u), "GLM DSA q projection count mismatch");
    auto qkv = prepare(first[1], second[0]);
    cache(qkv);
    if (full_indexer)
        index(first[2], first[3], second[1]);
    return output(unembed(attend(absorb(qkv))));
}

template <class Dense, class DensePrefix, class Sparse, class Concat>
auto indexed_attention(bool indexed, int64_t tokens, int64_t prefix, int64_t sparse_rows,
                       Dense dense, DensePrefix dense_prefix, Sparse sparse, Concat concat) {
    if (!indexed)
        return dense();
    require_model(prefix >= 0 && sparse_rows > 0 && prefix + sparse_rows == tokens,
                  "GLM DSA shared index state has the wrong row count");
    if (prefix == 0)
        return sparse();
    auto first = dense_prefix();
    auto rest = sparse();
    return concat(std::move(first), std::move(rest));
}


} // namespace tyloi::models::glm_dsa


namespace tyloi::cuda::glm_dsa {

State::State(const Model& model) : layers(model.layers.size()), model_(&model) {
    layout_.reserve(model.layers.size());
    for (const auto& layer : model.layers) {
        layout_.push_back({layer.device, layer.full_indexer,
                           layer.config.max_position_embeddings, layer.config.index_head_dim});
        selections.try_emplace(layer.device);
    }
}

void State::reset(int64_t next_batch) {
    cache_pos = decode_position_delta = 0;
    batch = next_batch;
    for (auto& [device, selection] : selections) {
        MfqCudaGuard guard(device);
        selection = {};
    }
    for (size_t i = 0; i < layers.size(); ++i) {
        auto& layer = layers[i];
        if (layer.kv_cache.defined() && layer.kv_cache.size(0) == next_batch) continue;
        MfqCudaGuard guard(layout_[i].device);
        layer = {};
    }
}

std::vector<std::pair<tyloi::Device, std::size_t>> State::growth(
    std::int64_t tokens, const tyloi::models::ForwardOptions&, std::int64_t confirmed) const {
    if (confirmed) throw std::invalid_argument("GLM DSA does not support speculative verification");
    if (batch <= 0 || tokens <= 0 || cache_pos < 0)
        throw std::invalid_argument("GLM DSA cache planning requires positive batch and tokens and a nonnegative position");
    if (cache_pos > std::numeric_limits<std::int64_t>::max() - tokens)
        throw std::overflow_error("GLM DSA cache planning position overflow");
    const auto multiply = [](std::size_t a, std::size_t b) {
        if (b && a > std::numeric_limits<std::size_t>::max() / b)
            throw std::overflow_error("GLM DSA cache planning byte count overflow");
        return a * b;
    };
    std::vector<std::pair<tyloi::Device, std::size_t>> result;
    for (std::size_t i = 0; i < layers.size(); ++i) {
        if (layers[i].kv_cache.defined()) continue;
        const auto& layout = layout_[i];
        if (layout.capacity <= 0 || (layout.full_indexer && layout.index_width <= 0))
            throw std::invalid_argument("GLM DSA cache planning requires a positive layout");
        auto rows = multiply(static_cast<std::size_t>(batch), static_cast<std::size_t>(layout.capacity));
        auto width = std::size_t(576);
        if (layout.full_indexer) {
            const auto index_width = static_cast<std::size_t>(layout.index_width);
            if (index_width > std::numeric_limits<std::size_t>::max() - width)
                throw std::overflow_error("GLM DSA cache planning byte count overflow");
            width += index_width;
        }
        const auto bytes = multiply(multiply(rows, width), sizeof(std::uint16_t));
        const tyloi::Device device(tyloi::DeviceType::cuda, layout.device);
        auto entry = std::find_if(result.begin(), result.end(),
            [device](const auto& item) { return item.first == device; });
        if (entry == result.end()) result.emplace_back(device, bytes);
        else {
            if (entry->second > std::numeric_limits<std::size_t>::max() - bytes)
                throw std::overflow_error("GLM DSA cache planning byte count overflow");
            entry->second += bytes;
        }
    }
    return result;
}

std::vector<::tyloi::Tensor> State::storage() const {
    std::vector<::tyloi::Tensor> result;
    for (const auto& layer : layers) {
        if (layer.kv_cache.defined()) result.push_back(layer.kv_cache);
        if (layer.index_cache.defined()) result.push_back(layer.index_cache);
    }
    for (const auto& [device, selection] : selections) {
        (void)device;
        if (selection.topk_indices.defined()) result.push_back(selection.topk_indices);
    }
    return result;
}

TextSessionStateKind State::text_session_state_kind() const {
    return layers.empty() ? TextSessionStateKind::Unsupported : TextSessionStateKind::GlmDsa;
}
bool State::supports_text_session_state() const {
    return text_session_state_kind() != TextSessionStateKind::Unsupported;
}

TextSessionState State::capture_text_session_state(const std::vector<int64_t>& tokens) const {
    if (!supports_text_session_state() || cache_pos <= 0 ||
        static_cast<size_t>(cache_pos) != tokens.size())
        throw std::runtime_error("GLM DSA text session state is unavailable");
    TextSessionState state;
    state.tokens = tokens;
    state.cache_pos = cache_pos;
    state.decode_position_delta = decode_position_delta;
    state.payload = std::vector<tyloi::models::glm_dsa::BlockSnapshot>{};
    auto& saved_layers = std::get<std::vector<tyloi::models::glm_dsa::BlockSnapshot>>(state.payload);
    saved_layers.reserve(layers.size());
    for (size_t i = 0; i < layers.size(); ++i) {
        MfqCudaGuard guard(layout_[i].device);
        const auto& layer = layers[i];
        if (!layer.kv_cache.defined() || layer.kv_cache.dim() != 4 ||
            layer.kv_cache.size(0) != 1 || cache_pos > layer.kv_cache.size(2))
            throw std::runtime_error("GLM DSA session MLA cache is unavailable");
        tyloi::models::glm_dsa::BlockSnapshot saved;
        saved.full_indexer = layout_[i].full_indexer;
        saved.kv_capacity = layer.kv_cache.size(2);
        saved.kv_cache = std::make_shared<DenseTensor>(layer.kv_cache.narrow(2, 0, cache_pos).clone());
        state.bytes += session_tensor_bytes(saved.kv_cache);
        if (saved.full_indexer) {
            if (!layer.index_cache.defined() || layer.index_cache.dim() != 3 ||
                layer.index_cache.size(0) != 1 || cache_pos > layer.index_cache.size(1))
                throw std::runtime_error("GLM DSA session index cache is unavailable");
            saved.index_capacity = layer.index_cache.size(1);
            saved.index_cache = std::make_shared<DenseTensor>(layer.index_cache.narrow(1, 0, cache_pos).clone());
            state.bytes += session_tensor_bytes(saved.index_cache);
        }
        saved_layers.push_back(std::move(saved));
    }
    return state;
}

std::vector<std::pair<tyloi::Device, std::size_t>> State::restore_growth(
        const TextSessionState& state) const {
    const auto* saved_layers = std::get_if<std::vector<tyloi::models::glm_dsa::BlockSnapshot>>(&state.payload);
    if (!supports_text_session_state() || state.kind() != TextSessionStateKind::GlmDsa ||
        state.cache_pos <= 0 || static_cast<size_t>(state.cache_pos) != state.tokens.size() ||
        saved_layers == nullptr || saved_layers->size() != layers.size())
        throw CudaSessionStateError("GLM DSA text session state is incompatible");
    std::vector<std::pair<tyloi::Device, std::size_t>> result;
    const auto add = [&](const tyloi::Tensor& source, std::size_t bytes) {
        if (!bytes) return;
        const auto device = source.device();
        auto entry = std::find_if(result.begin(), result.end(),
            [device](const auto& item) { return item.first == device; });
        if (entry == result.end()) result.emplace_back(device, bytes);
        else {
            if (entry->second > std::numeric_limits<std::size_t>::max() - bytes)
                throw std::overflow_error("GLM DSA session planning byte count overflow");
            entry->second += bytes;
        }
    };
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const auto& layer = layers[i];
        const auto& saved = (*saved_layers)[i];
        if (layout_[i].full_indexer != saved.full_indexer || !saved.kv_cache || !saved.kv_cache.defined() ||
            saved.kv_cache.dim() != 4 || saved.kv_cache.size(0) != 1 ||
            saved.kv_cache.size(2) != state.cache_pos)
            throw CudaSessionStateError("GLM DSA saved MLA cache is invalid");
        if (saved.full_indexer && (!saved.index_cache || !saved.index_cache.defined() ||
            saved.index_cache.dim() != 3 || saved.index_cache.size(0) != 1 ||
            saved.index_cache.size(1) != state.cache_pos))
            throw CudaSessionStateError("GLM DSA saved index cache is invalid");
        add(saved.kv_cache, restore_prefix_bytes(layer.kv_cache, saved.kv_cache, 2, saved.kv_capacity));
        if (saved.full_indexer)
            add(saved.index_cache, restore_prefix_bytes(layer.index_cache, saved.index_cache, 1, saved.index_capacity));
    }
    return result;
}

void State::restore_text_session_state(const TextSessionState& state) {
    const auto* saved_layers = std::get_if<std::vector<tyloi::models::glm_dsa::BlockSnapshot>>(&state.payload);
    if (!supports_text_session_state() || state.kind() != TextSessionStateKind::GlmDsa ||
        state.cache_pos <= 0 || static_cast<size_t>(state.cache_pos) != state.tokens.size() ||
        saved_layers == nullptr || saved_layers->size() != layers.size())
        throw CudaSessionStateError("GLM DSA text session state is incompatible");
    for (size_t i = 0; i < layers.size(); ++i) {
        MfqCudaGuard guard(layout_[i].device);
        auto& layer = layers[i];
        const auto& saved = (*saved_layers)[i];
        if (layout_[i].full_indexer != saved.full_indexer || !saved.kv_cache || !saved.kv_cache.defined() ||
            saved.kv_cache.dim() != 4 || saved.kv_cache.size(0) != 1 ||
            saved.kv_cache.size(2) != state.cache_pos)
            throw CudaSessionStateError("GLM DSA saved MLA cache is invalid");
#ifdef TYLOINFER_NATIVE_CUDA
        layer.kv_cache = restore_session_prefix_tensor(layer.kv_cache, saved.kv_cache, 2, saved.kv_capacity).dense();
#else
        restore_session_prefix_tensor(layer.kv_cache, saved.kv_cache, 2, saved.kv_capacity);
#endif
        if (saved.full_indexer) {
            if (!saved.index_cache || !saved.index_cache.defined() || saved.index_cache.dim() != 3 ||
                saved.index_cache.size(0) != 1 || saved.index_cache.size(1) != state.cache_pos)
                throw CudaSessionStateError("GLM DSA saved index cache is invalid");
#ifdef TYLOINFER_NATIVE_CUDA
            layer.index_cache = restore_session_prefix_tensor(layer.index_cache, saved.index_cache, 1, saved.index_capacity).dense();
#else
            restore_session_prefix_tensor(layer.index_cache, saved.index_cache, 1, saved.index_capacity);
#endif
        } else layer.index_cache = {};
    }
    for (auto& [device, selection] : selections) {
        MfqCudaGuard guard(device);
        selection = {};
    }
    cache_pos = state.cache_pos;
    decode_position_delta = state.decode_position_delta;
    batch = 1;
}

static void validate_geometry(const Config& config) {
    if (config.q_lora_rank <= 0 || config.kv_lora_rank <= 0 || config.qk_nope_head_dim <= 0 ||
        config.qk_rope_head_dim <= 0 || config.v_head_dim <= 0 || config.index_head_dim <= 0 ||
        config.index_n_heads <= 0 || config.index_topk <= 0 || config.num_experts <= 0 ||
        config.num_experts_per_tok <= 0 || config.num_attention_heads != 64 ||
        config.num_key_value_heads != 64 || config.kv_lora_rank != 512 ||
        config.qk_nope_head_dim != 192 || config.qk_rope_head_dim != 64 ||
        config.v_head_dim != 256 || config.index_head_dim != 128 || config.index_n_heads != 32 ||
        config.index_topk != 2048 ||
        config.qk_head_dim != config.qk_nope_head_dim + config.qk_rope_head_dim ||
        config.attention_bias || !config.rope_interleave || !config.indexer_rope_interleave ||
        config.hidden_act != "silu" || config.expert_group_count != 1 ||
        config.selected_group_count != 1 || config.shared_expert_count != 1 ||
        config.scoring_func != "sigmoid" || config.topk_method != "noaux_tc") {
        throw std::runtime_error("unsupported GLM DSA CUDA configuration");
    }
}

namespace {
DenseTensor normalize_ids(Collectives& collectives, int primary, DenseTensor ids) {
    tyloi::models::require_model(ids.dim() == 1 || ids.dim() == 2,
                               "token IDs must have shape [tokens] or [batch,tokens]");
    if (ids.dim() == 1) ids = ids.unsqueeze(0);
    tyloi::models::require_model(ids.size(0) > 0 && ids.size(1) > 0,
                               "token IDs must not be empty");
    return collectives.copy(ids.to(::tyloi::kInt64), primary).contiguous();
}

} // namespace

Layer load_layer(const ::tyloi::weights::Loader& weights, const Config& config, int index,
                 const std::string& type) {
    MfqCudaGuard guard(weights.placement.device);
    Layer layer;
    layer.device = weights.placement.device;
    TransformerWeightLoader loader{weights};
    tyloi::models::glm_dsa::load_layer_weights(layer, loader, config, index, type);
    return layer;
}

Model read(std::shared_ptr<const tyloi::ModelSource> source, std::string_view definition,
           int64_t context) {
    validate_model_source(*source);
    Model model;
    model.source = std::move(source);
    model.graph = tyloi::models::graph(*model.source);
    model.plan = cuda_model_plan(model.graph);
    tyloi::models::require_model(model.graph.backbone == "glm_dsa",
                               "model backbone does not match the requested causal model");
    auto& c = model.config;
    auto& m = model.metadata;
    tyloi::models::glm_dsa::load_config(definition, c, m);
    validate_geometry(c);
    tyloi::models::require_model(c.num_hidden_layers == model.graph.topology.text_layers,
                               "model graph/config text-layer topology mismatch");
    if (context > 0) {
        tyloi::models::require_model(context <= m.max_position_embeddings,
                                   "--ctx-size exceeds max_position_embeddings");
        c.max_position_embeddings = m.max_position_embeddings = context;
    }
    return model;
}

void load(Model& model, const ::tyloi::weights::Loader& weights, LayerPlacementConfig& placement,
          int gpu_layers, bool load_layers, bool load_input_output, const tyloi::ops::YarnScaling& yarn) {
    const auto& c = model.config;
    placement.prepare(c.num_hidden_layers);
    const int cpu_layers = gpu_layers < 0 ? 0 :
        static_cast<int>(std::max<int64_t>(c.num_hidden_layers - gpu_layers, 0));
    if (cpu_layers > 0) {
        if (weights.placement.tensor.enabled() || weights.placement.experts.enabled() || placement.enabled())
            throw std::runtime_error("--n-gpu-layers cannot be combined with tensor/expert/layer parallelism");
        throw std::runtime_error("--n-gpu-layers currently supports dense Qwen-style blocks only");
    }
    MfqCudaGuard guard(placement.primary_device());
    const auto make_rope = [&](int device) {
        RopeCache rope(c.max_position_embeddings, c.rotary_dim, c.rope_base, 0, -1,
                         ::tyloi::Device(::tyloi::kCUDA, device),
                         model.metadata.rope_interleaved, yarn);

#ifdef TYLOINFER_NATIVE_CUDA
        return rope.view();
#else
        return rope;
#endif
    };
    model.rope = make_rope(placement.primary_device());
    if (placement.enabled()) {
        for (int device : placement.devices) {
            MfqCudaGuard layer_guard(device);
            model.device_ropes.emplace(device, make_rope(device));
        }
    }
    if (load_input_output) {
        const bool separate = !c.tie_word_embeddings && weights.has("model.output.weight");
        model.embed = weights.embedding("model.token_embedding.weight", separate);
#ifdef TYLOINFER_NATIVE_CUDA
        model.output_norm = weights.tensor("model.output_norm.weight");
#else
        model.output_norm = weights.tensor("model.output_norm.weight").dense();
#endif
        model.lm_head = separate ? weights.weight("model.output.weight", TensorParallelAxis::Output)
            : weights.placement.tensor.enabled()
                ? weights.weight("model.token_embedding.weight", TensorParallelAxis::Output)
                : model.embed;
        if (load_layers) {
            model.layers.reserve(static_cast<size_t>(c.num_hidden_layers));
            for (int index = 0; index < c.num_hidden_layers; ++index) {
                const int device = placement.device_for_layer(index);
                MfqCudaGuard layer_guard(device);
                const auto& type = model.metadata.layer_types.at(static_cast<size_t>(index));
                std::cerr << "loading layer " << index << ' ' << type << " CUDA" << std::endl;
                model.layers.push_back(load_layer(weights.for_layer(device, false), c, index, type));
            }
        }
    } else {
        if (load_layers) throw std::invalid_argument("streamed model setup requires deferred blocks");
#ifdef TYLOINFER_NATIVE_CUDA
        model.output_norm = weights.tensor("model.output_norm.weight");
#else
        model.output_norm = weights.tensor("model.output_norm.weight").dense();
#endif
    }
}

#ifdef TYLOINFER_NATIVE_CUDA
std::unique_ptr<tyloi::models::CausalModel> load(
        std::shared_ptr<const tyloi::ModelSource> source, std::string_view definition,
        const tyloi::weights::Loader& weights, LayerPlacementConfig& placement,
        const tyloi::models::LoadOptions& options) {
    if (!options.cpu_offload_layers.empty())
        throw std::invalid_argument("GLM DSA does not support --cpu-offload-layers");
    auto model = std::make_unique<Model>(read(std::move(source), definition, options.context));
    load(*model, weights, placement, options.gpu_layers, true, true, options.yarn);
    if (model->plan.vision != CudaVisionAdapter::none || model->plan.predictor != CudaPredictorAdapter::none)
        throw std::runtime_error("CUDA component adapter is unsupported for this model");
    return model;
}

tyloi::models::Info Model::info() const {
    auto result = tyloi::models::model_info(metadata);
    result.graph_weights = tyloi::models::GraphWeights::resident;
    return result;
}

MfqModelCapabilities Model::capabilities() const {
    return {graph.architecture, graph.has_component("text"), false, false,
        false, false, false, false, "model-graph+cuda-adapters"};
}

tyloi::engine::Metrics Model::metrics() const {
    const auto state = cuda_component_state(graph, plan, false, false);
    return {{"kv_memory_supported", 0.0}, {"kv_quantization_supported", 0.0},
        {"kv_quantization_bits", 0.0}, {"vision_declared", double(state.vision_declared)},
        {"vision_supported", double(state.vision_supported)}, {"vision_available", 0.0},
        {"mtp_declared", double(state.mtp_declared)}, {"mtp_supported", double(state.mtp_supported)},
        {"mtp_available", 0.0}};
}

std::unique_ptr<tyloi::models::State> Model::make_state() const {
    return std::make_unique<State>(*this);
}

tyloi::StepSequence<tyloi::models::Prepared> Model::prepare(
        const std::vector<std::int64_t>&, const MfqMultimodalInput&, tyloi::models::State&) const {
    throw std::invalid_argument("model has no multimodal component");
}
#endif

#ifdef TYLOINFER_NATIVE_CUDA
::tyloi::Tensor Model::embedding(const ::tyloi::Tensor& input_ids, int primary) const {
    const auto& model = *this;
    auto& executor = Executor::current();
    auto& collectives = executor.collectives();
    auto ids = normalize_ids(collectives, primary, input_ids.dense());
    auto scope = executor.enter(mfq_current_cuda_device(), Executor::current_ffn());
    return ::tyloi::ops::measure("model.embed", ids.device(), [&] {
        return ::tyloi::ops::embedding(ids, model.embed);
    });
#else
DenseTensor embed(const Model& model, Executor& executor, Collectives& collectives,
             int primary, DenseTensor ids, const Executor::Mode& mode) {
    auto normalized = normalize_ids(collectives, primary, std::move(ids));
    return executor.profiler().measure("model.embed", [&] {
        return quant_embedding_lookup(model.embed, std::move(normalized));
    });
#endif
}

#ifdef TYLOINFER_NATIVE_CUDA
::tyloi::Tensor Model::normalize(const ::tyloi::Tensor& input_hidden,
                 std::int64_t batch, std::int64_t tokens) const {
    const auto& model = *this;
    auto& executor = Executor::current();
    auto hidden = input_hidden.dense();
    hidden = hidden.reshape({batch, tokens, model.metadata.hidden_size});
    auto scope = executor.enter(mfq_current_cuda_device(), FfnMode{});
    return ::tyloi::ops::measure("model.output_norm", hidden.device(), [&] {
        return ::tyloi::ops::rms_norm(hidden, model.output_norm,
                                      model.metadata.rms_norm_eps,
                                      model.metadata.norm_weight_offset);
    });
#else
DenseTensor normalize(const Model& model, Executor& executor, DenseTensor hidden,
                 std::int64_t batch, std::int64_t tokens) {
    hidden = hidden.reshape({batch, tokens, model.metadata.hidden_size});
    return executor.profiler().measure("model.output_norm", [&] {
        return qwen_rms_norm(hidden.reshape({-1, hidden.size(-1)}).to(::tyloi::kFloat32),
                             model.output_norm, model.metadata.rms_norm_eps,
                             model.metadata.norm_weight_offset)
            .reshape(hidden.sizes());
    });
#endif
}

#ifdef TYLOINFER_NATIVE_CUDA
::tyloi::Tensor Model::softcap(const ::tyloi::Tensor& input_logits) const {
    const auto& model = *this;
    auto& executor = Executor::current();
    auto scope = executor.enter(mfq_current_cuda_device(), FfnMode{});
    return ::tyloi::ops::softcap(input_logits, model.metadata.final_logit_softcapping);
#else
DenseTensor softcap(const Model& model, Executor&, DenseTensor logits) {
    const double cap = model.metadata.final_logit_softcapping;
    return cap > 0.0 ? ::tyloi::cuda::tanh(logits / cap) * cap : std::move(logits);
#endif
}

#ifdef TYLOINFER_NATIVE_CUDA
::tyloi::Tensor Model::logits(const ::tyloi::Tensor& input_hidden, bool capped) const {
    const auto& model = *this;
    auto& executor = Executor::current();
    auto scope = executor.enter(mfq_current_cuda_device(), Executor::current_ffn());
    auto value = ::tyloi::ops::measure("model.lm_head", input_hidden.device(), [&] {
        return ::tyloi::ops::project(input_hidden, model.lm_head);
    });
    return capped ? ::tyloi::ops::softcap(value, model.metadata.final_logit_softcapping) : value;
#else
DenseTensor logits(const Model& model, Executor& executor, DenseTensor hidden, bool capped,
              const Executor::Mode& mode) {
    auto value = executor.profiler().measure("model.lm_head", [&] {
        return executor.projector(mode.ffn).run(model.lm_head, std::move(hidden));
    });
    if (capped) {
        const double cap = model.metadata.final_logit_softcapping;
        if (cap > 0.0) return ::tyloi::cuda::tanh(value / cap) * cap;
    }
    return value;
#endif
}
#ifdef TYLOINFER_NATIVE_CUDA
::tyloi::Tensor Model::forward(tyloi::models::Input input, tyloi::models::State& state_value,
               const tyloi::models::ForwardOptions& forward_options) const {
    auto* state_ptr = dynamic_cast<State*>(&state_value);
    tyloi::models::require_model(state_ptr && state_ptr->belongs_to(*this),
                               "GLM DSA state belongs to a different model");
    auto& state = *state_ptr;
    const auto& model = *this;
    auto& executor = Executor::current();
    auto& collectives = executor.collectives();
    auto mode = executor.forward_mode(forward_options.primary);
    mode.ffn = Executor::current_ffn();
    mode.kv_capacity = forward_options.kv_capacity;
    mode.trace = forward_options.trace;
    mode.trace_layer = forward_options.trace_layer;
#else
DenseTensor forward(const Model& model, State& state, Executor& executor, Collectives& collectives,
               const Executor::Mode& mode, tyloi::models::Input input) {
#endif
    MfqCudaGuard guard(mode.primary);
    auto ids = normalize_ids(collectives, mode.primary, std::move(input.ids).dense());
#ifdef TYLOINFER_NATIVE_CUDA
    auto embeddings = input.embeddings.defined() ? input.embeddings.dense()
        : embedding(ids, mode.primary).dense();
#else
    auto embeddings = input.embeddings.has_value() ? DenseTensor(std::move(*input.embeddings))
        : embed(model, executor, collectives, mode.primary, ids, mode);
#endif
    const auto B = ids.size(0), T = ids.size(1);
    tyloi::models::require_model(embeddings.dim() == 3 && embeddings.size(0) == B &&
                                   embeddings.size(1) == T &&
                                   embeddings.size(2) == model.metadata.hidden_size,
                               "inputs_embeds shape must match [batch,tokens,hidden_size]");
    tyloi::models::require_model(input.confirmed >= 0 && input.confirmed == 0,
                               "unsupported speculative backbone geometry");
    if (state.cache_pos == 0) state.reset(B);

    const auto options = ::tyloi::TensorOptions().device(::tyloi::Device(::tyloi::kCUDA, mode.primary))
                             .dtype(::tyloi::kInt64);
#ifdef TYLOINFER_NATIVE_CUDA
    auto physical = input.cache_positions.defined()
        ? collectives.copy(input.cache_positions.dense().to(::tyloi::kInt64), mode.primary).contiguous()
        : ::tyloi::cuda::arange(state.cache_pos, state.cache_pos + T, options);
    auto semantic = input.positions.defined()
        ? collectives.copy(input.positions.dense().to(::tyloi::kInt64), mode.primary).contiguous()
        : (state.decode_position_delta == 0 ? physical : physical + state.decode_position_delta);
#else
    auto physical = input.cache_positions.has_value()
        ? collectives.copy(DenseTensor(*input.cache_positions).to(::tyloi::kInt64), mode.primary).contiguous()
        : ::tyloi::cuda::arange(state.cache_pos, state.cache_pos + T, options);
    auto semantic = input.positions.has_value()
        ? collectives.copy(DenseTensor(*input.positions).to(::tyloi::kInt64), mode.primary).contiguous()
        : (state.decode_position_delta == 0 ? physical : physical + state.decode_position_delta);
#endif
    tyloi::models::require_model((physical.dim() == 1 && physical.numel() == T) ||
                                   (physical.dim() == 2 && physical.size(0) == B && physical.size(1) == T),
                               "cache_positions must have shape [tokens] or [batch,tokens]");
    tyloi::models::require_model((semantic.dim() == 1 && semantic.size(0) == T) ||
                                   (semantic.dim() == 2 && semantic.size(0) == B && semantic.size(1) == T),
                               "position_ids must have shape [tokens] or [batch,tokens]");
#ifdef TYLOINFER_NATIVE_CUDA
    if (!input.lengths.defined() && state.cache_pos > 0 && T == 1)
        input.lengths = ::tyloi::cuda::full({B}, state.cache_pos + 1, options);
    AttentionStep step{semantic, input.lengths,
                       input.cache_positions.defined() ? physical : ::tyloi::DenseTensor{},
                       state.cache_pos, input.kv_length, input.attention_parts};
#else
    if (!input.lengths.has_value() && state.cache_pos > 0 && T == 1)
        input.lengths = ::tyloi::cuda::full({B}, state.cache_pos + 1, options);
    AttentionStep step{semantic, input.lengths ? MfqOptional<DenseTensor>(DenseTensor(*input.lengths)) : mfq_nullopt,
                       input.cache_positions.has_value() ? MfqOptional<DenseTensor>(physical) : mfq_nullopt,
                       state.cache_pos, input.kv_length, input.attention_parts};
#endif
    auto hidden = collectives.copy(std::move(embeddings), mode.primary).contiguous();
    const auto trace = [&](const DenseTensor& value) {
        if (input.trace)
            input.trace->push_back(collectives.copy(value, mode.primary).to(::tyloi::kFloat32).clone());
    };
    trace(hidden);
    for (std::size_t index = 0; index < model.layers.size(); ++index) {
        const auto& layer = model.layers[index];
        {
            MfqCudaGuard layer_guard(layer.device);
            auto local = step;
#ifdef TYLOINFER_NATIVE_CUDA
            local.positions = collectives.copy(step.positions.dense(), layer.device);
            if (step.sequence_lengths.defined())
                local.sequence_lengths = collectives.copy(step.sequence_lengths.dense(), layer.device);
            if (step.cache_positions.defined())
                local.cache_positions = collectives.copy(step.cache_positions.dense(), layer.device);
#else
            local.positions = collectives.copy(step.positions, layer.device);
            if (step.sequence_lengths.has_value())
                local.sequence_lengths = collectives.copy(DenseTensor(*step.sequence_lengths), layer.device);
            if (step.cache_positions.has_value())
                local.cache_positions = collectives.copy(DenseTensor(*step.cache_positions), layer.device);
#endif
            const auto& rope = model.device_ropes.empty() ? model.rope : model.device_ropes.at(layer.device);
            hidden = forward_layer(layer, state.layers[index], state.selections.at(layer.device), executor,
                                   collectives.copy(std::move(hidden), layer.device), local, rope, mode);
        }
        trace(hidden);
    }
#ifdef TYLOINFER_NATIVE_CUDA
    if (!input.positions.defined() || input.advance) state.cache_pos += T;
#else
    if (!input.positions.has_value() || input.advance) state.cache_pos += T;
#endif
    hidden = collectives.copy(std::move(hidden), mode.primary);
#ifdef TYLOINFER_NATIVE_CUDA
    auto finalized = normalize(hidden, B, T);
#else
    auto finalized = normalize(model, executor, hidden, B, T);
#endif
    if (input.raw_hidden) *input.raw_hidden = std::move(hidden);
    return finalized;
}

namespace {
DenseTensor headwise_project(const Projector& projector, MlaScratch& scratch, const tyloi::Tensor& weight,
                        DenseTensor x, int64_t B, int64_t T, int64_t heads) {
#if TYLOINFER_ENABLE_MFQ_COMPAT

    if (x.dim() != 4 || x.size(0) != B || x.size(1) != T || x.size(2) != heads ||
        x.size(3) != weight.size(2) || weight.size(0) != heads) {
        throw std::runtime_error("GLM head-wise MFE projection shape mismatch");
    }
    const int64_t rows = B * T * heads;
    const auto &route = scratch.head_route(rows, static_cast<int>(heads));
    auto y = projector.run(weight, x.contiguous().reshape({rows, weight.size(2)}), route);
    return y.reshape({B, T, heads, weight.size(1)});

#else
    throw std::runtime_error("Legacy head-wise projections require a compatibility build");
#endif
}

DenseTensor dense_attention(MlaScratch& scratch, const DenseTensor& kv_cache, DenseTensor q,
                       int64_t logical_len, int64_t B, const MfqOptional<DenseTensor>& seq_len,
                       double scale, int64_t planned_kv_length) {
    if (seq_len.has_value() && q.size(2) == 1 && B == 1) {
        const int64_t planned_len = planned_kv_length > 0 ? planned_kv_length : logical_len;
        scratch.ensure_decode_workspace(B, planned_len);
        return attention_glm_mla576_decode_cuda(
            q, kv_cache, seq_len.value(), scale, planned_len, scratch.decode_mask,
            scratch.kv_max, scratch.attention_meta);
    }
    scratch.ensure_dense_workspace(B, q.size(2), logical_len);
    return attention_glm_mla576_cached_cuda(q, kv_cache, logical_len, scratch.dense_mask,
                                            scratch.kv_max, scratch.attention_meta,
                                            scale);
}

void update_indexer(const Config& c, Selection& selection, const DenseTensor& index_cache,
                    CudaProfiler& profiler, DenseTensor index_q, DenseTensor index_weights, int64_t B,
                    int64_t T, int64_t cache_pos, const MfqOptional<DenseTensor>& seq_len,
                    int64_t planned_kv_length) {
    const int64_t logical_len = cache_pos + T;
    if (logical_len <= c.index_topk) {
        selection.topk_indices = ::tyloi::DenseTensor();
        selection.dense_prefix_rows = T;
        return;
    }

    if (seq_len.has_value() && T == 1 && B == 1) {
        const int64_t planned_len = planned_kv_length > 0 ? planned_kv_length : logical_len;
        auto scores = profiler.measure("glm.indexer_scores", [&]() {
            return glm_dsa_indexer_scores_decode_cuda(index_q, index_cache, index_weights,
                                                      seq_len.value(), planned_len);
        });
        auto selected = profiler.measure("glm.indexer_topk", [&]() {
            return ::tyloi::cuda::topk(scores, c.index_topk, -1, true, false);
        });
        selection.topk_indices =
            std::get<1>(selected).to(::tyloi::kInt32).contiguous();
        selection.dense_prefix_rows = 0;
        return;
    }

    const int64_t prefix_rows =
        std::max<int64_t>(0, std::min<int64_t>(T, c.index_topk - cache_pos));
    const int64_t sparse_rows = T - prefix_rows;
    if (sparse_rows <= 0) {
        selection.topk_indices = ::tyloi::DenseTensor();
        selection.dense_prefix_rows = T;
        return;
    }
    auto indices = ::tyloi::cuda::empty({B, sparse_rows, c.index_topk},
                                             ::tyloi::TensorOptions()
                                                 .device(::tyloi::kCUDA)
                                                 .dtype(::tyloi::kInt32));
    constexpr int64_t kMaxScoreElements = 32 * 1024 * 1024;
    int64_t rows_per_chunk =
        std::max<int64_t>(1, kMaxScoreElements / std::max<int64_t>(1, B * logical_len));
    rows_per_chunk = std::min<int64_t>(rows_per_chunk, 256);
    if (rows_per_chunk >= 64)
        rows_per_chunk = rows_per_chunk / 64 * 64;
    for (int64_t start = 0; start < sparse_rows; start += rows_per_chunk) {
        const int64_t count = std::min<int64_t>(rows_per_chunk, sparse_rows - start);
        auto q_chunk = index_q.narrow(1, prefix_rows + start, count).contiguous();
        auto w_chunk = index_weights.narrow(1, prefix_rows + start, count).contiguous();
        auto scores = profiler.measure("glm.indexer_scores", [&]() {
            return glm_dsa_indexer_scores_cuda(q_chunk, index_cache, w_chunk,
                                               cache_pos + prefix_rows + start, logical_len);
        });
        auto selected = profiler.measure("glm.indexer_topk", [&]() {
            return ::tyloi::cuda::topk(scores, c.index_topk, -1, true, false);
        });
        indices.narrow(1, start, count)
            .copy_(std::get<1>(selected).to(::tyloi::kInt32));
    }
    selection.topk_indices = indices;
    selection.dense_prefix_rows = prefix_rows;
}

} // namespace

DenseTensor forward_layer(const Layer& layer, LayerState& state, Selection& selection,
                     Executor& executor, DenseTensor x, const AttentionStep& step,
#ifdef TYLOINFER_NATIVE_CUDA
                     const tyloi::ops::RopeTable& rope,
#else
                     const RopeCache& rope,
#endif
                     const Executor::Mode& mode) {
    auto& profiler = executor.profiler();
    auto& scratch = executor.mla_scratch(layer.device);
    const auto project = executor.projector(mode.ffn);
    const auto& c = layer.config;
#ifndef TYLOINFER_NATIVE_CUDA
    const auto& attn_norm = layer.attn_norm;
    const auto& ffn_norm = layer.ffn_norm;
    const auto& q_a_norm = layer.q_a_norm;
    const auto& kv_a_norm = layer.kv_a_norm;
    const auto& index_k_norm = layer.index_k_norm;
    const auto& index_k_bias = layer.index_k_bias;
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    const auto input_proj = project.group(layer.input_proj);
    const auto q_proj = project.group(layer.q_proj);
#else
    const auto input_proj = make_quant_group(layer.input_proj, true, false);
    const auto q_proj = make_quant_group(layer.q_proj, true, false);
#endif
    const auto& embed_q = *layer.embed_q;
    const auto& unembed_out = *layer.unembed_out;
    const auto& o_proj = *layer.o_proj;
    const auto& ffn = layer.ffn;
    const bool full_indexer = layer.full_indexer;
    auto& kv_cache = state.kv_cache;
    auto& index_cache = state.index_cache;
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& pos = step.positions.dense();
    const MfqOptional<DenseTensor> seq_len = step.sequence_lengths
        ? MfqOptional<DenseTensor>(step.sequence_lengths.dense()) : mfq_nullopt;
#else
    const auto& pos = step.positions;
    const auto& seq_len = step.sequence_lengths;
#endif
    const auto cache_pos = step.cache_position;
    const auto planned_kv_length = step.planned_kv_length;
    const int64_t B = x.size(0);
    const int64_t T = x.size(1);
    const int64_t H = x.size(2);
    constexpr int64_t kHeads = 64;
    constexpr int64_t kNope = 192;
    constexpr int64_t kRope = 64;
    constexpr int64_t kLatent = 512;
    constexpr int64_t kMlaWidth = kLatent + kRope;
    constexpr int64_t kValue = 256;
    const int64_t logical_len = cache_pos + T;
    if (!kv_cache.defined()) {
        auto options = ::tyloi::TensorOptions()
                           .device(::tyloi::kCUDA)
                           .dtype(::tyloi::kFloat16);
        kv_cache =
            ::tyloi::cuda::empty({B, 1, c.max_position_embeddings, kMlaWidth}, options);
        if (full_indexer) {
            index_cache = ::tyloi::cuda::empty(
                {B, c.max_position_embeddings, c.index_head_dim}, options);
        }
    }

    x = x.to(::tyloi::kFloat16).contiguous();
    auto xn = profiler.measure("glm.attn_norm", [&]() {
#ifdef TYLOINFER_NATIVE_CUDA
        const auto& attn_norm = layer.attn_norm.dense();
#endif
        return rms_norm_f16_cuda(x.reshape({B * T, H}), attn_norm, c.rms_norm_eps, 0.0)
            .reshape({B, T, H});
    });
    using DenseTensor = ::tyloi::DenseTensor;
    struct Projection {
        DenseTensor query, rotary, rows;
    };
    auto attn_out = tyloi::models::glm_dsa::attention(
                full_indexer,
                [&] {
                    return profiler.measure(
                        "glm.input_proj", [&]() { return project.run(input_proj, xn); });
                },
                [&](DenseTensor projected) {
                    return profiler.measure("glm.q_a_norm", [&]() {
#ifdef TYLOINFER_NATIVE_CUDA
                        const auto& q_a_norm = layer.q_a_norm.dense();
#endif
                        return rms_norm_f16_cuda(projected.reshape({B * T, c.q_lora_rank})
                                                     .to(::tyloi::kFloat16)
                                                     .contiguous(),
                                                 q_a_norm, 1e-6, 0.0)
                            .reshape({B, T, c.q_lora_rank});
                    });
                },
                [&](DenseTensor qr) {
                    return profiler.measure("glm.q_proj",
                                            [&]() { return project.run(q_proj, qr); });
                },
                [&](DenseTensor key_value, DenseTensor query) {
                    auto q_main = query.to(::tyloi::kFloat16)
                                      .reshape({B, T, kHeads, kNope + kRope});
                    auto q_nope =
                        q_main.index({Slice(), Slice(), Slice(), Slice(0, kNope)}).contiguous();
                    auto q_pe =
                        q_main.index({Slice(), Slice(), Slice(), Slice(kNope, kNope + kRope)})
                            .permute({0, 2, 1, 3})
                            .contiguous();
                    q_pe = profiler.measure("glm.q_rope", [&]() {
#ifdef TYLOINFER_NATIVE_CUDA
                        const auto& rope_cos = rope.cos.dense();
                        const auto& rope_sin = rope.sin.dense();
#else
                        const auto& rope_cos = rope.cos;
                        const auto& rope_sin = rope.sin;
#endif
                        return glm_interleaved_rope_cuda(q_pe, pos.contiguous(), rope_cos,
                                                         rope_sin, kRope);
                    });

                    auto compressed = key_value.to(::tyloi::kFloat16)
                                          .reshape({B, T, kLatent + kRope});
                    auto kv_latent = profiler.measure("glm.kv_a_norm", [&]() {
#ifdef TYLOINFER_NATIVE_CUDA
                        const auto& kv_a_norm = layer.kv_a_norm.dense();
#endif
                        auto raw = compressed.index({Slice(), Slice(), Slice(0, kLatent)})
                                       .contiguous();
                        return rms_norm_f16_cuda(raw.reshape({B * T, kLatent}), kv_a_norm, 1e-6,
                                                 0.0)
                            .reshape({B, T, kLatent});
                    });
                    auto k_pe =
                        compressed.index({Slice(), Slice(), Slice(kLatent, kLatent + kRope)})
                            .reshape({B, T, 1, kRope})
                            .permute({0, 2, 1, 3})
                            .contiguous();
                    k_pe = profiler.measure("glm.k_rope", [&]() {
#ifdef TYLOINFER_NATIVE_CUDA
                        const auto& rope_cos = rope.cos.dense();
                        const auto& rope_sin = rope.sin.dense();
#else
                        const auto& rope_cos = rope.cos;
                        const auto& rope_sin = rope.sin;
#endif
                        return glm_interleaved_rope_cuda(k_pe, pos.contiguous(), rope_cos,
                                                         rope_sin, kRope);
                    });
                    auto kv_rows =
                        ::tyloi::cuda::cat(
                            {kv_latent, k_pe.permute({0, 2, 1, 3}).reshape({B, T, kRope})}, -1)
                            .contiguous();
                    return Projection{q_nope, q_pe, kv_rows};
                },
                [&](const Projection &p) {
                    profiler.measure("glm.kv_write", [&]() {
                        return glm_dsa_cache_write_cuda(
                            kv_cache.view({B, c.max_position_embeddings, kMlaWidth}), p.rows,
                            pos.contiguous());
                    });
                },
                [&](DenseTensor index_key, DenseTensor index_weight, DenseTensor index_query) {
                    auto index_k = profiler.measure("glm.indexer_k_norm", [&]() {
#ifdef TYLOINFER_NATIVE_CUDA
                        const auto& index_k_norm = layer.index_k_norm.dense();
                        const auto& index_k_bias = layer.index_k_bias.dense();
#endif
                        return glm_dsa_indexer_layer_norm_cuda(
                            index_key.to(::tyloi::kFloat16)
                                .reshape({B, T, c.index_head_dim})
                                .contiguous(),
                            index_k_norm, index_k_bias, 1e-5);
                    });
#ifdef TYLOINFER_NATIVE_CUDA
                    const auto& rope_cos = rope.cos.dense();
                    const auto& rope_sin = rope.sin.dense();
#else
                    const auto& rope_cos = rope.cos;
                    const auto& rope_sin = rope.sin;
#endif
                    index_k =
                        glm_interleaved_rope_cuda(index_k.reshape({B, T, 1, c.index_head_dim})
                                                      .permute({0, 2, 1, 3})
                                                      .contiguous(),
                                                  pos.contiguous(), rope_cos, rope_sin, kRope)
                            .permute({0, 2, 1, 3})
                            .reshape({B, T, c.index_head_dim})
                            .contiguous();
                    profiler.measure("glm.indexer_k_write", [&]() {
                        return glm_dsa_cache_write_cuda(index_cache, index_k, pos.contiguous());
                    });
                    auto index_q = index_query.to(::tyloi::kFloat16)
                                       .reshape({B, T, c.index_n_heads, c.index_head_dim})
                                       .permute({0, 2, 1, 3})
                                       .contiguous();
                    index_q = glm_interleaved_rope_cuda(index_q, pos.contiguous(), rope_cos,
                                                        rope_sin, kRope)
                                  .permute({0, 2, 1, 3})
                                  .contiguous();
                    auto index_weights = index_weight.reshape({B, T, c.index_n_heads})
                                             .to(::tyloi::kFloat32)
                                             .contiguous();
                    update_indexer(c, selection, index_cache, profiler, index_q, index_weights,
                                   B, T, cache_pos, seq_len, planned_kv_length);
                },
                [&](const Projection &p) {
                    auto q_absorbed =
                        profiler
                            .measure("glm.embed_q",
                                     [&]() {
                                         return headwise_project(project, scratch, embed_q,
                                                                 p.query, B, T, kHeads);
                                     })
                            .permute({0, 2, 1, 3})
                            .contiguous();
                    auto q_mla = ::tyloi::cuda::cat({q_absorbed, p.rotary}, -1)
                                     .to(::tyloi::kFloat32)
                                     .contiguous();
                    return q_mla;
                },
                [&](DenseTensor q_mla) {
                    const double scale = 1.0 / std::sqrt(static_cast<double>(kNope + kRope));
                    const bool indexed = selection.topk_indices.defined();
                    const int64_t prefix = selection.dense_prefix_rows;
                    const int64_t sparse_rows =
                        indexed ? selection.topk_indices.size(1) : 0;
                    return tyloi::models::glm_dsa::indexed_attention(
                        indexed, T, prefix, sparse_rows,
                        [&] {
                            return profiler.measure("glm.attention_dense", [&]() {
                                return dense_attention(scratch, kv_cache, q_mla, logical_len, B,
                                                       seq_len, scale, planned_kv_length);
                            });
                        },
                        [&] {
                            return profiler.measure("glm.attention_dense_prefix", [&]() {
                                return dense_attention(
                                    scratch, kv_cache, q_mla.narrow(2, 0, prefix).contiguous(),
                                    cache_pos + prefix, B, mfq_nullopt, scale, planned_kv_length);
                            });
                        },
                        [&] {
                            scratch.ensure_meta();
                            return profiler.measure("glm.attention_sparse", [&]() {
                                return attention_glm_mla_sparse_cuda(
                                    q_mla.narrow(2, prefix, sparse_rows).contiguous(),
                                    kv_cache.view({B, c.max_position_embeddings, kMlaWidth}),
                                    selection.topk_indices, scratch.attention_meta,
                                    scale);
                            });
                        },
                        [](DenseTensor first, DenseTensor rest) {
                            return ::tyloi::cuda::cat({first, rest}, 1);
                        });
                },
                [&](DenseTensor attended) {
                    return profiler.measure("glm.unembed_out", [&]() {
                        return headwise_project(
                            project, scratch, unembed_out,
                            attended.to(::tyloi::kFloat16).contiguous(), B, T,
                            kHeads);
                    });
                },
                [&](DenseTensor value_heads) {
                    return profiler.measure("glm.o_proj", [&]() {
                        return project.run(o_proj, value_heads.reshape({B, T, kHeads * kValue}).contiguous());
                    });
                });
    xn = {};
    std::array<DenseTensor, 2> prepared;
    {
        auto attn_pair = profiler.measure("glm.attn_residual_ffn_norm", [&]() {
#ifdef TYLOINFER_NATIVE_CUDA
            const auto& ffn_norm = layer.ffn_norm.dense();
#endif
            return acc_rms_norm_f16_cuda(x.reshape({B * T, H}),
                                         attn_out.reshape({B * T, H})
                                             .to(::tyloi::kFloat16)
                                             .contiguous(),
                                         ffn_norm, c.rms_norm_eps, 0.0);
        });
        prepared = std::array<::tyloi::DenseTensor, 2>{
            attn_pair[0].reshape({B, T, H}), attn_pair[1].reshape({B * T, H})};
    }
    x = {};
    attn_out = {};
    auto hidden = std::move(prepared[0]);
    auto ffn_input = std::move(prepared[1]);
#ifdef TYLOINFER_NATIVE_CUDA
    auto ffn_out = ffn.forward(ffn_input).reshape({B * T, H});
#else
    auto ffn_out = ffn.forward(ffn_input, executor, mode.ffn).reshape({B * T, H});
#endif
    ffn_input = {};
    return profiler.measure("glm.ffn_residual", [&]() {
        return acc_cuda(hidden.reshape({B * T, H}), ffn_out).reshape({B, T, H});
    });
}
} // namespace tyloi::cuda::glm_dsa
