#include "engine/backends/cuda/kernels/mfq_cuda_moe_ops.h"
#include "model.h"
#include "mtp.h"
#include "load.h"
#include "forward.h"
#include "engine/backends/cuda/native/collectives.h"
#include "engine/backends/cuda/native/scope.h"
#include <algorithm>
#include "engine/models/common/load.h"
#include "engine/weights/loader.h"
#include "engine/models/session/paged.h"
#include "engine/ops.h"

namespace tyloi::cuda::glm5_next {
using Config = tyloi::models::glm5_next::Config;
DenseTensor FeedForward::dense(const DenseTensor& source) const {
    DenseTensor hidden;
    {
        auto g = ::tyloi::ops::project(source, gate).dense();
        auto u = ::tyloi::ops::project(source, up).dense();
        g = ::tyloi::cuda::clamp_max(g, limit);
        u = ::tyloi::cuda::clamp(u, -limit, limit);
        hidden = (g * ::tyloi::cuda::sigmoid(g)) * u;
    }
    return ::tyloi::ops::project(hidden, down).dense();
}

DenseTensor FeedForward::forward(const DenseTensor& x) const {
    if (!router) return dense(x);
    auto source = x.reshape({-1, hidden}).to(::tyloi::kFloat16);
    auto logits = ::tyloi::ops::project(source.to(::tyloi::kFloat32), router).dense()
        .to(::tyloi::kFloat32).contiguous();
    auto selected = ::tyloi::ops::topk(logits, topk, routing, router_bias);
    auto shared_output = dense(source);
    DenseTensor routed;
    {
        DenseTensor activation;
        {
            auto gu = ::tyloi::ops::project(source, routed_gate_up[0], selected.ids).dense();
            if (routed_gate_up.size() == 2) {
                auto routed_up = ::tyloi::ops::project(source, routed_gate_up[1], selected.ids).dense();
                gu = ::tyloi::cuda::cat({gu, routed_up}, -1);
            }
            auto g = ::tyloi::cuda::clamp_max(gu.narrow(-1, 0, intermediate), limit);
            auto u = ::tyloi::cuda::clamp(gu.narrow(-1, intermediate, intermediate), -limit, limit);
            activation = (g * ::tyloi::cuda::sigmoid(g)) * u;
        }
        auto pairs = ::tyloi::ops::project(activation, routed_down, selected.ids).dense();
        activation = {};
        auto reduced = ::tyloi::cuda::zeros({source.size(0), hidden},
                                            source.options().dtype(::tyloi::kFloat32));
        for (int64_t r = 0; r < topk; ++r)
            reduced = reduced + pairs.select(1, r).to(::tyloi::kFloat32) *
                                selected.weights.dense().select(1, r).unsqueeze(-1);
        routed = reduced.to(pairs.scalar_type());
    }
    return (routed + shared_output).reshape(x.sizes());
}

struct LayerLoader : ::tyloi::weights::Loader {
    using KdaWeights = glm5_next::KdaWeights;
    using MlaWeights = glm5_next::MlaWeights;

    static Mhc mhc(Parameter function, Parameter base, Parameter scale) {
        return {std::move(function), std::move(base), std::move(scale)};
    }
#ifdef TYLOINFER_NATIVE_CUDA
    static Parameter concat(const std::vector<Parameter> &parts) {
        std::vector<DenseTensor> dense;
        dense.reserve(parts.size());
        for (const auto& part : parts) dense.push_back(part.dense());
        return std::make_shared<DenseTensor>(::tyloi::cuda::cat(dense, 0));
    }
#else
    static DenseTensor concat(const std::vector<DenseTensor> &parts) { return ::tyloi::cuda::cat(parts, 0); }
#endif
    static auto kda(KdaWeights weights, const Config &c) {
        return std::make_unique<Kda>(std::move(weights), c.kda_heads, c.kda_width, c.kda_kernel,
                                     c.lower_bound, c.eps);
    }
    static auto mla(MlaWeights weights, const Config &c) {
        MlaConfig geometry{c.heads, c.nope, c.latent, c.value_width, c.index_heads, c.index_width,
                           c.pool, c.budget, c.maximum, c.tail, c.eps};
        return std::make_unique<SparseMla>(std::move(weights), geometry);
    }
    static FeedForward gated_mlp(Linear gate, Linear up, Linear down, double limit) {
        FeedForward result;
        result.gate=std::move(gate);result.up=std::move(up);result.down=std::move(down);
        result.limit=limit;
        return result;
    }
    static FeedForward moe(std::vector<Linear> gate_up, Linear down, Linear router,
                           FeedForward shared, Parameter bias, const Config &c) {
        shared.routed_gate_up=std::move(gate_up);shared.routed_down=std::move(down);
        shared.router=std::move(router);shared.router_bias=std::move(bias);
        shared.hidden=c.hidden;shared.topk=c.topk;shared.intermediate=c.moe_intermediate;
        shared.routing=c.routing();
        return shared;
    }
};

static DenseTensor execute(const Layer& layer, LayerState& activity,
                      const DenseTensor &x, int64_t position, int64_t confirmed = 0) {
    const auto& [config, attention_hc, ffn_hc, attention_norm, ffn_norm, ffn, kda, mla, device] = layer;
    if (mla)
        TYLOINFER_RUNTIME_CHECK(std::get<MlaState>(activity).position() == position,
                               "GLM MLA/model cache positions diverged");
    auto attention_mix = attention_hc.pre(x, config);
    auto normalized = rms_norm(attention_mix[2], attention_norm, config.eps);
    auto branch = kda ? kda->forward(std::get<KdaState>(activity), std::move(normalized), true, confirmed)
                      : mla->forward(std::get<MlaState>(activity), std::move(normalized), true);
    auto hidden = mfq_glm5_next::mhc_post(branch, x, attention_mix[0], attention_mix[1]);
    auto ffn_mix = ffn_hc.pre(hidden, config);
    normalized = rms_norm(ffn_mix[2], ffn_norm, config.eps);
    branch = ffn.forward(normalized);
    return mfq_glm5_next::mhc_post(branch, hidden, ffn_mix[0], ffn_mix[1]);
}

Layer load_layer(const ::tyloi::weights::Loader& weights, const Config& config, int index) {
    MfqCudaGuard guard(weights.placement.device);
    Layer layer;
    layer.device = weights.placement.device;
    LayerLoader loader{weights};
    loader.role = "glm5_next";
    tyloi::models::glm5_next::load_layer_weights(layer, loader, config, index);
    return layer;
}

Model read(std::shared_ptr<const tyloi::ModelSource> source, std::string_view definition,
           int64_t context) {
    validate_model_source(*source);
    Model model;
    model.source = std::move(source);
    model.graph = tyloi::models::graph(*model.source);
    model.plan = cuda_model_plan(model.graph);
    tyloi::models::require_model(model.graph.backbone == "glm5_next",
                               "model backbone does not match the requested causal model");
    tyloi::models::glm5_next::load_config(definition, model.config, model.metadata);
    tyloi::models::require_model(model.config.layers == model.graph.topology.text_layers,
                               "model graph/config text-layer topology mismatch");
    if (context > 0) {
        tyloi::models::require_model(context <= model.metadata.max_position_embeddings,
                                   "--ctx-size exceeds max_position_embeddings");
        model.config.maximum = context;
        model.metadata.max_position_embeddings = context;
    }
    return model;
}

void load(Model& model, const ::tyloi::weights::Loader& weights,
          LayerPlacementConfig& placement, int gpu_layers, bool load_layers,
          bool load_input_output) {
    validate_native_placement(weights.placement.tensor, placement, gpu_layers);
    placement.prepare(model.config.layers);
    MfqCudaGuard guard(placement.primary_device());
    if (load_input_output) {
        const bool separate = !model.config.tied_embeddings && weights.has("model.output.weight");
        model.embed = weights.embedding("model.token_embedding.weight", separate);
#ifdef TYLOINFER_NATIVE_CUDA
        model.output_norm = weights.tensor("model.output_norm.weight");
#else
        model.output_norm = weights.tensor("model.output_norm.weight").dense();
#endif
        model.lm_head = separate ? weights.weight("model.output.weight", TensorParallelAxis::Output)
                                 : model.embed;
        if (load_layers) {
            model.layers.reserve(static_cast<size_t>(model.config.layers));
            for (int index = 0; index < model.config.layers; ++index)
                model.layers.push_back(load_layer(weights.for_layer(placement.device_for_layer(index), false),
                                                  model.config, index));
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

State::State(const Model& model) {
    layers.reserve(model.layers.size());
    for (const auto& layer : model.layers) {
        if (layer.kda) layers.emplace_back(std::in_place_type<KdaState>);
        else layers.emplace_back(std::in_place_type<MlaState>, model.config.maximum,
                                 model.config.latent, model.config.index_width);
    }
}

void State::reset(int64_t next_batch) {
    cache_pos = decode_position_delta = 0;
    speculative_start = -1;
    speculative_confirmed = 0;
    batch = next_batch;
    for (auto& layer : layers)
        std::visit([](auto& activity) { activity.reset(); }, layer);
}

void State::commit_speculative() {
    tyloi::models::require_model(speculative_start >= 0, "no speculative transaction");
    for (auto& layer : layers)
        if (auto* kda = std::get_if<KdaState>(&layer)) kda->commit();
    speculative_start = -1;
    speculative_confirmed = 0;
}

TextSessionState State::capture_text_session_state(const std::vector<int64_t>&) const {
    throw std::runtime_error("text session state is unsupported by this model");
}
void State::restore_text_session_state(const TextSessionState&) {
    throw CudaSessionStateError("text session state is unsupported by this model");
}

static DenseTensor normalize_ids(Collectives& copies, int primary, DenseTensor ids) {
    tyloi::models::require_model(ids.dim() == 1 || ids.dim() == 2,
                               "token IDs must have shape [tokens] or [batch,tokens]");
    if (ids.dim() == 1) ids = ids.unsqueeze(0);
    tyloi::models::require_model(ids.size(0) > 0 && ids.size(1) > 0, "token IDs must not be empty");
    return copies.copy(ids.to(::tyloi::kInt64), primary).contiguous();
}

#ifdef TYLOINFER_NATIVE_CUDA
::tyloi::Tensor embed(const Model& model, Executor& executor, Collectives& copies, int primary,
             const ::tyloi::Tensor& input_ids, const Executor::Mode& mode) {
    auto ids = input_ids.dense();
    auto normalized = normalize_ids(copies, primary, std::move(ids));
    return ::tyloi::ops::measure("model.embed", normalized.device(), [&] {
        return ::tyloi::ops::embedding(normalized, model.embed);
    });
#else
DenseTensor embed(const Model& model, Executor& executor, Collectives& copies, int primary,
             DenseTensor ids, const Executor::Mode& mode) {
    auto normalized = normalize_ids(copies, primary, std::move(ids));
    return executor.profiler().measure("model.embed", [&] {
        return quant_embedding_lookup(model.embed, std::move(normalized));
    });
#endif
}

#ifdef TYLOINFER_NATIVE_CUDA
::tyloi::Tensor normalize(const Model& model, Executor&, const ::tyloi::Tensor& input_hidden, int64_t, int64_t) {
    auto hidden = input_hidden.dense();
#else
DenseTensor normalize(const Model& model, Executor&, DenseTensor hidden, int64_t, int64_t) {
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    return rms_norm(hidden.mean(2), model.output_norm, model.metadata.rms_norm_eps);
#else
    return rms_norm(hidden.mean(2), model.output_norm, model.metadata.rms_norm_eps);
#endif
}

#ifdef TYLOINFER_NATIVE_CUDA
::tyloi::Tensor softcap(const Model&, Executor&, const ::tyloi::Tensor& value) { return value; }
#else
DenseTensor softcap(const Model&, Executor&, DenseTensor value) { return value; }
#endif

#ifdef TYLOINFER_NATIVE_CUDA
::tyloi::Tensor logits(const Model& model, Executor& executor, const ::tyloi::Tensor& input_hidden, bool,
              const Executor::Mode& mode) {
    return ::tyloi::ops::project(input_hidden, model.lm_head);
#else
DenseTensor logits(const Model& model, Executor& executor, DenseTensor hidden, bool,
              const Executor::Mode& mode) {
    return executor.projector(mode.ffn).run(model.lm_head, hidden);
#endif
}

bool supports_speculation(const Model& model) noexcept { return !model.layers.empty(); }

#ifdef TYLOINFER_NATIVE_CUDA
::tyloi::Tensor forward(const Model& model, State& state, Executor& executor, Collectives& copies,
               const Executor::Mode& mode, tyloi::models::Input input) {
#else
DenseTensor forward(const Model& model, State& state, Executor& executor, Collectives& copies,
               const Executor::Mode& mode, tyloi::models::Input input) {
#endif
    MfqCudaGuard guard(mode.primary);
    auto ids = normalize_ids(copies, mode.primary, std::move(input.ids).dense());
    auto embeddings = input.embeddings.defined() ? std::move(input.embeddings).dense()
#ifdef TYLOINFER_NATIVE_CUDA
        : embed(model, executor, copies, mode.primary, ids, mode).dense();
#else
        : embed(model, executor, copies, mode.primary, ids, mode);
#endif
    const auto B = ids.size(0), T = ids.size(1);
    tyloi::models::require_model(embeddings.dim() == 3 && embeddings.size(0) == B &&
        embeddings.size(1) == T && embeddings.size(2) == model.metadata.hidden_size,
        "inputs_embeds shape must match [batch,tokens,hidden_size]");
    tyloi::models::require_model(T > 0 && state.cache_pos + T <= model.metadata.max_position_embeddings &&
        !input.positions.defined() && !input.cache_positions.defined(),
        "GLM Flash-Next currently requires contiguous causal cache positions");
    tyloi::models::require_model(state.speculative_start < 0,
        "commit or roll back the pending speculative pass before forwarding");
    tyloi::models::require_model(input.confirmed >= 0 &&
        (input.confirmed == 0 || (input.confirmed < T && B == 1 && state.cache_pos > 0 &&
                                  supports_speculation(model))),
        "unsupported speculative backbone geometry");
    if (input.confirmed > 0) {
        state.speculative_start = state.cache_pos;
        state.speculative_confirmed = input.confirmed;
    }
    if (state.cache_pos == 0) state.reset(B);
    auto hidden = copies.copy(std::move(embeddings), mode.primary).contiguous()
        .to(::tyloi::kFloat16).unsqueeze(2).expand({B, T, model.config.streams, model.config.hidden})
        .contiguous();
    const auto trace = [&](const DenseTensor& value) {
        if (input.trace) input.trace->push_back(copies.copy(value, mode.primary).to(::tyloi::kFloat32).clone());
    };
    trace(hidden);
    for (size_t index = 0; index < model.layers.size(); ++index) {
        const auto& layer = model.layers[index];
        MfqCudaGuard layer_guard(layer.device);
        hidden = execute(layer, state.layers[index],
                         copies.copy(std::move(hidden), layer.device), state.cache_pos, input.confirmed);
        trace(hidden);
    }
    state.cache_pos += T;
    hidden = copies.copy(std::move(hidden), mode.primary);
    auto finalized = normalize(model, executor, std::move(hidden), B, T);
    if (input.raw_hidden) *input.raw_hidden = finalized;
    return finalized;
}

void replay(const Model& model, State& state, Executor&, int64_t accepted) {
    tyloi::models::require_model(state.speculative_start >= 0 && accepted >= 0, "no speculative transaction");
    const auto keep = state.speculative_start + state.speculative_confirmed + accepted;
    for (size_t index = 0; index < state.layers.size(); ++index) {
        MfqCudaGuard guard(model.layers[index].device);
        if (auto* kda = std::get_if<KdaState>(&state.layers[index])) kda->rollback();
        else std::get<MlaState>(state.layers[index]).truncate(keep);
    }
    state.cache_pos = keep;
    state.speculative_start = -1;
    state.speculative_confirmed = 0;
}


Mtp::Mtp(::tyloi::Tensor embedding, int primary)
    : embedding_(std::move(embedding)), primary_(primary) {}
Mtp::~Mtp() = default;
Mtp::Mtp(Mtp&&) noexcept = default;

std::optional<Mtp> Mtp::load_if_present(const ::tyloi::weights::Loader& weights,
                                                      ::tyloi::Tensor embedding,
                                                      const tyloi::models::glm5_next::Config &main) {
    Mtp result{std::move(embedding), weights.placement.device};
    LayerLoader loader{weights};
    loader.role = "glm5_next";
    if (!tyloi::models::glm5_next::load_predictor(result, loader, main))
        return std::nullopt;
    return result;
}

void Mtp::reset(tyloi::models::MtpState& value, int64_t next_batch) const {
    auto& state = this->state<PredictorState>(value);
    TYLOINFER_RUNTIME_CHECK(next_batch > 0, "GLM5-Next MTP batch must be positive");
    for (auto &layer : state.states)
        layer.reset();
    std::fill(state.lengths.begin(), state.lengths.end(), 0);
    state.batch = next_batch;
}

std::pair<DenseTensor, DenseTensor> Mtp::evaluate(tyloi::models::MtpState& value,
                                       const DenseTensor &hidden,
                                       const DenseTensor &ids, int64_t depth, bool cache,
                                       const DenseTensor &supplied_positions,
                                       const DenseTensor &supplied_embeddings) const {
    auto& state = this->state<PredictorState>(value);
#ifdef TYLOINFER_NATIVE_CUDA
    TYLOINFER_RUNTIME_CHECK(ids.dim() == 2 && ids.size(0) > 0 && ids.size(1) > 0 &&
                          hidden.dim() == 3 && hidden.size(0) == ids.size(0) &&
                          hidden.size(1) == ids.size(1) && hidden.size(2) == hidden_norm.numel(),
                      "GLM5-Next MTP input geometry mismatch");
#else
    TYLOINFER_RUNTIME_CHECK(ids.dim() == 2 && ids.size(0) > 0 && ids.size(1) > 0 &&
                          hidden.dim() == 3 && hidden.size(0) == ids.size(0) &&
                          hidden.size(1) == ids.size(1) && hidden.size(2) == hidden_norm.numel(),
                      "GLM5-Next MTP input geometry mismatch");
#endif
    const auto b = ids.size(0), t = ids.size(1);
    auto dtype = ::tyloi::kFloat16;
    const auto capacity = config.maximum;
    const auto count = int64_t(state.lengths.size());
    tyloi::models::require_model(count > 0 && b > 0 && t > 0, "invalid predictor geometry");
    const auto layer = (depth % count + count) % count;
    if (cache && state.batch && state.batch != b)
        reset(state, b);
    const auto start = cache ? state.lengths[layer] : 0;
    tyloi::models::require_model(start >= 0 && t <= capacity - start,
                                 "predictor history exceeds context capacity");
    auto embeds = supplied_embeddings;
    if (!embeds.defined()) {
        const auto device = ::tyloi::Device(::tyloi::kCUDA, primary_);
        auto normalized = ::tyloi::ops::contiguous(::tyloi::ops::copy(::tyloi::ops::cast(ids, ::tyloi::kInt64), device));
        embeds = ::tyloi::ops::measure("model.embed", device, [&] {
            return ::tyloi::ops::embedding(normalized, embedding_);
        }).dense();
    }
    TYLOINFER_RUNTIME_CHECK(embeds.sizes().vec() ==
                          std::vector<int64_t>({b, t, config.hidden}),
                      "GLM5-Next MTP embedding shape mismatch");
    DenseTensor pos;
    {
        auto current = supplied_positions.defined()
                           ? supplied_positions
                           : ::tyloi::cuda::arange(start, start + t,
                                        ids.options().dtype(::tyloi::kInt32));
        TYLOINFER_RUNTIME_CHECK((current.dim() == 1 || current.dim() == 2) &&
                              current.size(-1) == t &&
                              (current.dim() == 1 || current.size(0) == b),
                          "GLM MTP positions require [T] or [B,T]");
        current = current.to(::tyloi::kInt32);
        pos = std::move(current);
    }
    DenseTensor x;
    {
        DenseTensor e;
        {
            auto mask = (pos == 0).reshape({pos.dim() == 1 ? 1 : b, t, 1});
            auto masked = ::tyloi::cuda::where(mask, ::tyloi::cuda::zeros_like(embeds), embeds);
#ifdef TYLOINFER_NATIVE_CUDA
            e = rms_norm(masked, embedding_norm, config.eps);
#else
            e = rms_norm(masked, embedding_norm, config.eps);
#endif
        }
        // Release prior-stage tensors before later projections allocate.
        embeds = {};
        DenseTensor streams;
#ifdef TYLOINFER_NATIVE_CUDA
        streams = rms_norm(hidden, hidden_norm, config.eps);
#else
        streams = rms_norm(hidden, hidden_norm, config.eps);
#endif
        x = ::tyloi::ops::project(::tyloi::cuda::cat({e, streams}, -1), fusion).dense();
    }
    DenseTensor multi;
    {
        auto &block = layers[layer];
        dtype = x.scalar_type() == ::tyloi::kFloat32 ? ::tyloi::kFloat32 : ::tyloi::kFloat16;
        auto normalized = rms_norm(x, block.attention_norm, config.eps);
        auto branch = block.attention->forward(state.states[layer], normalized, cache);
        x = x.to(::tyloi::kFloat32) + branch.to(::tyloi::kFloat32);
        normalized = rms_norm(x, block.ffn_norm, config.eps).to(dtype);
        branch = block.ffn.forward(normalized);
        multi = x.to(::tyloi::kFloat32) + branch.to(::tyloi::kFloat32);
    }
    x = {};
    DenseTensor result;
#ifdef TYLOINFER_NATIVE_CUDA
    result = rms_norm(multi, output_norm, config.eps).to(dtype);
#else
    result = rms_norm(multi, output_norm, config.eps).to(dtype);
#endif
    if (cache) {
        state.batch = b;
        state.lengths[layer] = start + t;
    }
    return {std::move(result), std::move(multi)};
}

tyloi::models::MtpStep Mtp::step(tyloi::models::MtpState& value,
                                  const ::tyloi::Tensor& hidden, const ::tyloi::Tensor& ids,
                                  const ::tyloi::Tensor& positions) const {
    if (positions.defined())
        throw std::runtime_error("this CUDA MTP predictor does not accept explicit positions");
    auto result = evaluate(value, hidden.dense(),
                           ids.dense());
    return {std::make_shared<DenseTensor>(std::move(result.first)),
            std::make_shared<DenseTensor>(std::move(result.second))};
}

void Mtp::trim_cache_to(tyloi::models::MtpState& value, int64_t position) const {
    auto& state = this->state<PredictorState>(value);
    TYLOINFER_RUNTIME_CHECK(position >= 0 && position <= state.position(),
                      "GLM5-Next MTP cache trim position is invalid");
    constexpr size_t index = 0;
    if (index < layers.size() && layers[index].attention)
        state.states[index].truncate(position);
    state.lengths[index] = position;
}

bool Mtp::teacher_forced_prompt_prime() const noexcept { return false; }

bool Mtp::target_bootstrap_decode() const noexcept { return true; }

bool Mtp::preserve_output_dtype() const noexcept { return true; }

} // namespace tyloi::cuda::glm5_next
