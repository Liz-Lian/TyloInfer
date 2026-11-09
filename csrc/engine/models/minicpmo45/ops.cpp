#include "engine/models/minicpmo45/text/load.h"
#include "engine/models/minicpmo45/text/forward.h"
#include "engine/models/common/ffn.h"
#include "engine/models/common/mlp.h"
#include "engine/models/common/load.h"
#include "engine/weights/loader.h"
#include "engine/models/common/causal_forward.h"
#include "engine/models/session/state.h"
#include "engine/models/source/model_graph.h"
#include "engine/backends/cuda/native/collectives.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/backends/cuda/ops/norm/rms.h"
#include "engine/backends/cuda/ops/projection/projection.h"

#include <algorithm>
#include <iostream>

namespace tyloi::cuda::minicpmo45::text {

Model read(std::shared_ptr<const tyloi::ModelSource> source, std::string_view definition, int64_t context) {
    validate_model_source(*source);
    Model model;
    model.source = std::move(source);
    model.graph = tyloi::models::graph(*model.source);
    model.plan = cuda_model_plan(model.graph);
    tyloi::models::require_model(model.graph.backbone == "minicpmo45",
        "model backbone does not match the requested causal model");
    auto& config = model.config;
    config = Config::from_json(definition);
    if (config.model_type.empty()) config.model_type = model.graph.architecture;
    config.rotary_dim = config.head_dim;
    config.layer_types.assign(static_cast<size_t>(config.num_hidden_layers), "full_attention");
    if (config.hidden_size != 4096 || config.intermediate_size != 12288 ||
        config.num_hidden_layers != 36 || config.num_attention_heads != 32 ||
        config.num_key_value_heads != 8 || config.head_dim != 128 ||
        config.hidden_act != "silu" || config.attention_bias || config.use_sliding_window)
        throw std::runtime_error("unsupported MiniCPM-o 4.5 Qwen3 configuration");
    auto& metadata = model.metadata;
    metadata.vocab_size = config.vocab_size;
    metadata.hidden_size = config.hidden_size;
    metadata.num_hidden_layers = config.num_hidden_layers;
    metadata.num_attention_heads = config.num_attention_heads;
    metadata.num_key_value_heads = config.num_key_value_heads;
    metadata.head_dim = config.head_dim;
    metadata.max_position_embeddings = config.max_position_embeddings;
    metadata.rotary_dim = config.rotary_dim;
    metadata.rope_base = config.rope_base;
    metadata.rms_norm_eps = config.rms_norm_eps;
    metadata.tie_word_embeddings = config.tie_word_embeddings;
    metadata.model_type = config.model_type;
    metadata.layer_types = config.layer_types;
    metadata.rope_interleaved = true;
    metadata.decode_graph_double_warmup = true;
    tyloi::models::require_model(config.num_hidden_layers == model.graph.topology.text_layers,
        "model graph/config text-layer topology mismatch");
    if (context > 0) {
        tyloi::models::require_model(context <= model.metadata.max_position_embeddings,
            "--ctx-size exceeds max_position_embeddings");
        config.max_position_embeddings = model.metadata.max_position_embeddings = context;
    }
    return model;
}

Layer load_layer(const tyloi::weights::Loader& weights, const Config& config, int index,
                 const std::string& type) {
    if (type != "full_attention")
        throw std::runtime_error("unsupported MiniCPM layer type: " + type);
    Layer layer;
    layer.device = weights.placement.device;
    layer.cpu = weights.placement.cpu;
    layer.attention.official_bf16 = true;
    TransformerWeightLoader loader{weights};
    loader.preserve_projection_boundaries = true;
    const auto prefix = "model.block." + std::to_string(index) + ".";
    tyloi::models::load_attention_mixer(layer.attention, loader, config, index, prefix, 0.0);
#ifdef TYLOINFER_NATIVE_CUDA
    layer.ffn_norm = weights.tensor(prefix + "mlp.norm.weight");
#else
    layer.ffn_norm = loader.dense(prefix + "mlp.norm.weight");
#endif
    layer.ffn = tyloi::models::load_dense_ffn<tyloi::models::Mlp>(loader, config, prefix + "mlp.");
    return layer;
}

void load(Model& model, const tyloi::weights::Loader& weights, LayerPlacementConfig& placement,
          int gpu_layers, bool expert_parallel, bool load_layers, bool load_input_output,
          const tyloi::ops::YarnScaling& yarn) {
    const auto& config = model.config;
    if (expert_parallel)
        throw std::runtime_error("--expert-parallel requires a model with routed experts");
    placement.prepare(config.num_hidden_layers);
    const int cpu_layers = gpu_layers < 0 ? 0 :
        static_cast<int>(std::max<int64_t>(config.num_hidden_layers - gpu_layers, 0));
    if (cpu_layers > 0 && (weights.placement.tensor.enabled() || placement.enabled()))
        throw std::runtime_error("--n-gpu-layers cannot be combined with tensor/expert/layer parallelism");
    if (model.plan.vision == CudaVisionAdapter::grid_vit && cpu_layers > 0)
        throw std::runtime_error("CUDA grid-Vision currently requires GPU-resident text layers");
    MfqCudaGuard guard(placement.primary_device());
    const auto make_rope = [&](::tyloi::Device device) {
        RopeCache rope(config.max_position_embeddings, config.rotary_dim, config.rope_base,
                         0, -1, device, true, yarn);

#ifdef TYLOINFER_NATIVE_CUDA
        return rope.view();
#else
        return rope;
#endif
    };
    model.rope = make_rope(::tyloi::Device(::tyloi::kCUDA, placement.primary_device()));
    if (cpu_layers > 0) model.cpu_rope = make_rope(::tyloi::Device(::tyloi::kCPU));
    if (placement.enabled())
        for (int device : placement.devices) {
            MfqCudaGuard device_guard(device);
            model.device_ropes.emplace(device, make_rope(::tyloi::Device(::tyloi::kCUDA, device)));
        }
    if (load_input_output) {
        const bool separate = !config.tie_word_embeddings && weights.has("model.output.weight");
        model.embed = weights.embedding("model.token_embedding.weight", separate);
#ifdef TYLOINFER_NATIVE_CUDA
        model.output_norm = weights.tensor("model.output_norm.weight");
#else
        model.output_norm = dynamic_cast<const ::tyloi::DenseTensor&>(*weights.tensor("model.output_norm.weight"));
#endif
        model.lm_head = separate ? weights.weight("model.output.weight", TensorParallelAxis::Output)
            : weights.placement.tensor.enabled()
                ? weights.weight("model.token_embedding.weight", TensorParallelAxis::Output) : model.embed;
        if (load_layers) {
            model.layers.reserve(config.num_hidden_layers);
            for (int index = 0; index < config.num_hidden_layers; ++index) {
                const int device = placement.device_for_layer(index);
                const bool cpu = index < cpu_layers;
                MfqCudaGuard layer_guard(device);
                const auto& type = model.metadata.layer_types.at(index);
                std::cerr << "loading layer " << index << ' ' << type << ' ' << (cpu ? "CPU" : "CUDA") << std::endl;
                model.layers.push_back(load_layer(weights.for_layer(device, cpu), config, index, type));
            }
        }
    } else {
        if (load_layers) throw std::invalid_argument("streamed model setup requires deferred blocks");
#ifdef TYLOINFER_NATIVE_CUDA
        model.output_norm = weights.tensor("model.output_norm.weight");
#else
        model.output_norm = dynamic_cast<const ::tyloi::DenseTensor&>(*weights.tensor("model.output_norm.weight"));
#endif
    }
}

State::State(const Model& model) : layers(model.layers.size()) {
    for (const auto& layer : model.layers) {
        const auto& a = layer.attention;
        layout_.push_back({layer.device, layer.cpu, a.sliding, a.kv_heads, a.attention_head_dim,
                           a.sliding ? a.attention_window : a.max_position_embeddings, a.kv_quantization});
    }
}
void State::preallocate() {
    if (!::tyloi::cuda::options::kv_vram_preallocated()) return;
    for (size_t i = 0; i < layers.size(); ++i) {
        const auto& l = layout_[i];
        if (l.cpu || layers[i].defined()) continue;
        MfqCudaGuard guard(l.device);
        layers[i] = tyloi::state::allocate_kv(
            1, l.heads, l.capacity, l.head_dim, l.sliding,
            ::tyloi::Device(::tyloi::kCUDA, l.device), ::tyloi::kBFloat16,
            l.quantization);
    }
}
void State::reset(int64_t next_batch) {
    cache_pos = decode_position_delta = 0;
    batch = next_batch;
    for (size_t i = 0; i < layers.size(); ++i)
        if (layers[i].defined() && layers[i].batch != batch) {
            MfqCudaGuard guard(layout_[i].device);
            layers[i] = {};
        }
}
std::vector<::tyloi::Tensor> State::storage() const {
    std::vector<::tyloi::Tensor> result;
    for (const auto& cache : layers) {
        if (cache.paged()) {
            if (cache.key_chunk_ptrs.defined()) result.push_back(cache.key_chunk_ptrs);
            if (cache.value_chunk_ptrs.defined()) result.push_back(cache.value_chunk_ptrs);
            if (cache.page_table.defined()) result.push_back(cache.page_table);
        } else {
            if (cache.physical_key.defined()) result.push_back(cache.physical_key);
            if (cache.physical_value.defined()) result.push_back(cache.physical_value);
        }
    }
    return result;
}
TextSessionStateKind State::text_session_state_kind() const {
    return layers.empty() ? TextSessionStateKind::Unsupported : TextSessionStateKind::FullAttention;
}
bool State::supports_text_session_state() const { return !layers.empty(); }
bool State::supports_paged_text_session_state() const {
    return !layers.empty() && std::none_of(layout_.begin(), layout_.end(), [](const auto& l) { return l.sliding; });
}
TextSessionState State::capture_text_session_state(const std::vector<int64_t>& tokens) const {
    if (layers.empty()) throw std::runtime_error("text session state is unsupported by this block layout");
    if (cache_pos <= 0 || static_cast<size_t>(cache_pos) != tokens.size())
        throw std::runtime_error("text session token count does not match the model cache");
    TextSessionState state;
    state.tokens = tokens;
    state.cache_pos = cache_pos;
    state.decode_position_delta = decode_position_delta;
    state.payload = std::vector<KvSnapshot>{};
    auto& saved = std::get<std::vector<KvSnapshot>>(state.payload);
    saved.reserve(layers.size());
    for (size_t i = 0; i < layers.size(); ++i) {
        MfqCudaGuard guard(layout_[i].device);
        saved.push_back(capture_kv(layers[i], cache_pos, state.bytes));
    }
    return state;
}
void State::restore_text_session_state(const TextSessionState& state) {
    const auto* saved = std::get_if<std::vector<KvSnapshot>>(&state.payload);
    if (layers.empty() || !saved || state.cache_pos <= 0 ||
        static_cast<size_t>(state.cache_pos) != state.tokens.size() || saved->size() != layers.size())
        throw CudaSessionStateError("full-attention text session state is incompatible");
    for (size_t i = 0; i < layers.size(); ++i) {
        MfqCudaGuard guard(layout_[i].device);
        restore_kv(layers[i], (*saved)[i], layout_[i].head_dim, ::tyloi::kBFloat16, layout_[i].quantization);
    }
    cache_pos = state.cache_pos;
    decode_position_delta = state.decode_position_delta;
    batch = 1;
}

namespace {
DenseTensor token_ids(Collectives& copies, int primary, DenseTensor ids) {
    tyloi::models::require_model(ids.dim() == 1 || ids.dim() == 2,
        "token IDs must have shape [tokens] or [batch,tokens]");
    if (ids.dim() == 1) ids = ids.unsqueeze(0);
    tyloi::models::require_model(ids.size(0) > 0 && ids.size(1) > 0, "token IDs must not be empty");
    return copies.copy(ids.to(::tyloi::kInt64), primary).contiguous();
}
}

#ifdef TYLOINFER_NATIVE_CUDA
::tyloi::Tensor embed(const Model& model, Executor& executor, Collectives& copies, int primary,
             const ::tyloi::Tensor& ids_handle, const Executor::Mode& mode) {
    auto ids = ids_handle.dense();
#else
DenseTensor embed(const Model& model, Executor& executor, Collectives& copies, int primary,
             DenseTensor ids, const Executor::Mode& mode) {
#endif
    auto normalized = token_ids(copies, primary, std::move(ids));
#ifdef TYLOINFER_NATIVE_CUDA
    auto scope = executor.enter(primary, mode.ffn);
    auto result = ::tyloi::ops::measure("model.embed", normalized.device(), [&] {
        return ::tyloi::ops::embedding(normalized, model.embed);
    });
    return ::tyloi::ops::contiguous(::tyloi::ops::cast(result, ::tyloi::kBFloat16));
#else
    return executor.profiler().measure("model.embed", [&] {
        return ::tyloi::cuda::quant_embedding_lookup(model.embed, std::move(normalized));
    }).to(::tyloi::kBFloat16).contiguous();
#endif
}
#ifdef TYLOINFER_NATIVE_CUDA
::tyloi::Tensor normalize(const Model& model, Executor& executor, const ::tyloi::Tensor& hidden_handle, int64_t batch, int64_t tokens) {
    auto hidden = hidden_handle.dense();
#else
DenseTensor normalize(const Model& model, Executor& executor, DenseTensor hidden, int64_t batch, int64_t tokens) {
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    auto scope = executor.enter(hidden.device().index, FfnMode{});
    return ::tyloi::ops::measure("model.output_norm", hidden.device(), [&] {
        return ::tyloi::ops::rms_norm(
            hidden.reshape({batch, tokens, model.metadata.hidden_size}), model.output_norm,
            model.metadata.rms_norm_eps, model.metadata.norm_weight_offset,
            ::tyloi::kBFloat16);
    });
#else
    return executor.profiler().measure("model.output_norm", [&] {
        return qwen_rms_norm_bf16(executor.attention_options().fused_norm,
            hidden.reshape({batch * tokens, model.metadata.hidden_size}),
            model.output_norm, model.metadata.rms_norm_eps, model.metadata.norm_weight_offset)
            .reshape({batch, tokens, model.metadata.hidden_size});
    });
#endif
}
#ifdef TYLOINFER_NATIVE_CUDA
::tyloi::Tensor logits(const Model& model, Executor& executor, const ::tyloi::Tensor& hidden_handle, bool,
              const Executor::Mode& mode) {
    auto hidden = hidden_handle.dense();
#else
DenseTensor logits(const Model& model, Executor& executor, DenseTensor hidden, bool,
              const Executor::Mode& mode) {
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    auto scope = executor.enter(hidden.device().index, mode.ffn);
    auto result = ::tyloi::ops::measure("model.lm_head", hidden.device(), [&] {
        return ::tyloi::ops::project(hidden, model.lm_head);
    });
    return ::tyloi::ops::contiguous(::tyloi::ops::cast(result, ::tyloi::kBFloat16));
#else
    return ::tyloi::cuda::logits(model.lm_head, std::move(hidden),
        executor.projector(mode.ffn), executor.profiler(), 0.0)
        .to(::tyloi::kBFloat16).contiguous();
#endif
}
#ifdef TYLOINFER_NATIVE_CUDA
::tyloi::Tensor forward(const Model& model, State& state, Executor& executor, Collectives& copies,
               const Executor::Mode& run_mode, tyloi::models::Input input) {
#else
DenseTensor forward(const Model& model, State& state, Executor& executor, Collectives& copies,
               const Executor::Mode& run_mode, tyloi::models::Input input) {
#endif
    auto mode = run_mode;
    mode.ffn.independent = true;
    MfqCudaGuard guard(mode.primary);
    auto ids = token_ids(copies, mode.primary, std::move(input.ids).dense());
#ifdef TYLOINFER_NATIVE_CUDA
    auto hidden = input.embeddings ? input.embeddings.dense()
        : embed(model, executor, copies, mode.primary, ids, mode).dense();
#else
    auto hidden = input.embeddings ? *input.embeddings
        : embed(model, executor, copies, mode.primary, ids, mode);
#endif
    const auto B = ids.size(0), T = ids.size(1);
    tyloi::models::require_model(hidden.dim() == 3 && hidden.size(0) == B && hidden.size(1) == T &&
        hidden.size(2) == model.metadata.hidden_size,
        "inputs_embeds shape must match [batch,tokens,hidden_size]");
    tyloi::models::require_model(input.confirmed == 0, "unsupported speculative backbone geometry");
    if (state.cache_pos == 0) state.reset(B);
    const auto options = ::tyloi::TensorOptions().dtype(::tyloi::kInt64).device(::tyloi::Device(::tyloi::kCUDA, mode.primary));
    auto physical = input.cache_positions
        ? copies.copy(
#ifdef TYLOINFER_NATIVE_CUDA
            input.cache_positions.dense().to(::tyloi::kInt64),
#else
            input.cache_positions->to(::tyloi::kInt64),
#endif
            mode.primary).contiguous()
        : ::tyloi::cuda::arange(state.cache_pos, state.cache_pos + T, options);
    auto positions = input.positions
        ? copies.copy(
#ifdef TYLOINFER_NATIVE_CUDA
            input.positions.dense().to(::tyloi::kInt64),
#else
            input.positions->to(::tyloi::kInt64),
#endif
            mode.primary).contiguous()
        : (state.decode_position_delta ? physical + state.decode_position_delta : physical);
    tyloi::models::require_model((physical.dim() == 1 && physical.numel() == T) ||
        (physical.dim() == 2 && physical.size(0) == B && physical.size(1) == T),
        "cache_positions must have shape [tokens] or [batch,tokens]");
    tyloi::models::require_model((positions.dim() == 1 && positions.size(0) == T) ||
        (positions.dim() == 2 && positions.size(0) == B && positions.size(1) == T),
        "position_ids must have shape [tokens] or [batch,tokens]");
    hidden = copies.copy(std::move(hidden), mode.primary).to(::tyloi::kBFloat16).contiguous();
    const auto trace = [&] {
        if (input.trace) input.trace->push_back(copies.copy(hidden, mode.primary).to(::tyloi::kFloat32).clone());
    };
    trace();
    for (size_t i = 0; i < model.layers.size(); ++i) {
        const auto& layer = model.layers[i];
        const auto& attention = layer.attention;
        MfqCudaGuard layer_guard(layer.device);
        const auto local = [&](DenseTensor value) {
            return layer.cpu ? value.to(::tyloi::kCPU).contiguous() : copies.copy(value, layer.device).contiguous();
        };
        hidden = local(std::move(hidden));
        AttentionStep step{local(positions), {}, local(physical), state.cache_pos,
                           input.kv_length, input.attention_parts};
#ifdef TYLOINFER_NATIVE_CUDA
        if (input.lengths) step.sequence_lengths = local(input.lengths.dense());
#else
        if (input.lengths) step.sequence_lengths = local(*input.lengths);
#endif
        auto& cache = state.layers[i];
        const auto capacity = attention.sliding ? attention.attention_window :
            (mode.kv_capacity > 0 ? std::max(state.cache_pos + T, mode.kv_capacity)
                                  : attention.max_position_embeddings);
        if (!cache.defined())
            cache = tyloi::state::allocate_kv(
                B, attention.kv_heads, capacity, attention.attention_head_dim,
                attention.sliding, hidden.device(), ::tyloi::kBFloat16,
                attention.kv_quantization);
        cache.quantization = attention.kv_quantization;
        const auto* rope = layer.cpu ? &model.cpu_rope : &model.rope;
        if (!layer.cpu && !model.device_ropes.empty()) rope = &model.device_ropes.at(layer.device);
        auto* layer_trace = mode.trace_layer == attention.layer ? mode.trace : nullptr;
#ifdef TYLOINFER_NATIVE_CUDA
        auto scope = executor.enter(layer.device, mode.ffn);
        auto output = attention.forward(::tyloi::Tensor(std::move(hidden)), step, cache, *rope, layer_trace);
        hidden = ::tyloi::models::full_ffn(
            layer.ffn, layer.ffn_norm, output[0], output[1],
            attention.rms_norm_eps, attention.norm_weight_offset,
            attention.official_bf16, attention.attention_output_gate, layer_trace).dense();
#else
        auto output = attention.forward(executor.projector(mode.ffn), executor.profiler(),
            executor.attention_options(), std::move(hidden), step, cache,
            executor.attention_scratch(2 * i), *rope, layer_trace);
        hidden = ::tyloi::models::full_ffn(
            layer.ffn, layer.ffn_norm, std::move(output[0]), std::move(output[1]),
            attention.rms_norm_eps, attention.norm_weight_offset,
            attention.official_bf16, attention.attention_output_gate, layer_trace).dense();
#endif
        trace();
    }
    if (!input.positions || input.advance) state.cache_pos += T;
    hidden = copies.copy(std::move(hidden), mode.primary);
    auto finalized = normalize(model, executor, hidden, B, T);
    if (input.raw_hidden) *input.raw_hidden = std::move(hidden);
    return finalized;
}
} // namespace tyloi::cuda::minicpmo45::text
