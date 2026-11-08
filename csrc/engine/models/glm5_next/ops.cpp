#include "engine/backends/cuda/kernels/mfq_cuda_moe_ops.h"
#include "model.h"
#include "mtp.h"
#include "load.h"
#include "forward.h"
#include "engine/backends/cuda/native/collectives.h"
#include "engine/backends/cuda/native/scope.h"
#include <algorithm>
#include <span>
#include "engine/models/common/load.h"
#include "engine/models/session/paged.h"
#include "engine/ops.h"

namespace tyloi::cuda::glm5_next {
using Config = tyloi::models::glm5_next::Config;
using attention_ops::Transform;

static Transform headwise(Linear projection,int64_t heads,int64_t output) {
    return [projection=std::move(projection),heads,output](
            const Projector& project, const DenseTensor& x) {
        TYLOINFER_RUNTIME_CHECK(x.dim()==4 && x.size(2)==heads,"GLM head-wise projection shape mismatch");
        const auto b=x.size(0),t=x.size(1),rows=b*t*heads;
        auto ids=::tyloi::cuda::arange(rows,x.options().dtype(::tyloi::kInt32)).remainder(heads).reshape({rows,1});
        return project.run(*projection,x.reshape({rows,x.size(-1)}),ids).reshape({b,t,heads,output});
    };
}

static Transform dense_ffn(Linear gate, Linear up, Linear down, double limit) {
    return [gate, up, down, limit](const Projector& project, const DenseTensor &x) {
        DenseTensor hidden;
        {
            auto g = project.run(*gate, x);
            auto u = project.run(*up, x);
            g = ::tyloi::cuda::clamp_max(g, limit);
            u = ::tyloi::cuda::clamp(u, -limit, limit);
            hidden = (g * ::tyloi::cuda::sigmoid(g)) * u;
        }
        return project.run(*down, hidden);
    };
}

static Transform glm_moe(std::vector<Linear> gate_up, Linear down,
                      Linear router, Transform shared, Parameter bias, const Config &c) {
    return [gate_up, down, router, shared, bias, c](const Projector& project,
                                                    const DenseTensor &x) {
        auto source = x.reshape({-1, c.hidden}).to(::tyloi::kFloat16);
        const auto routing = c.routing();
        auto selected = moe_topk_cuda(
            project.run(*router, source.to(::tyloi::kFloat32)).to(::tyloi::kFloat32).contiguous(),
            c.topk, routing.activation == tyloi::ops::RouterActivation::sigmoid,
            routing.activation == tyloi::ops::RouterActivation::sqrt_softplus,
            routing.normalize, routing.delayed_softmax,
#ifdef TYLOINFER_NATIVE_CUDA
            dynamic_cast<const DenseTensor&>(*bias),
#else
            bias,
#endif
            1e-20, routing.scale);
        auto shared_output = shared(project, source);
        DenseTensor routed;
        {
            DenseTensor hidden;
            {
                auto gu = project.run(*gate_up[0], source, selected[0]);
                if (gate_up.size() == 2) {
                    auto up = project.run(*gate_up[1], source, selected[0]);
                    gu = ::tyloi::cuda::cat({gu, up}, -1);
                }
                auto gate = ::tyloi::cuda::clamp_max(gu.narrow(-1, 0, c.moe_intermediate), c.swiglu_limit);
                auto up = ::tyloi::cuda::clamp(gu.narrow(-1, c.moe_intermediate, c.moe_intermediate),
                                                -c.swiglu_limit, c.swiglu_limit);
                hidden = (gate * ::tyloi::cuda::sigmoid(gate)) * up;
            }
            auto pairs = project.run(*down, hidden, selected[0]);
            hidden = {};
            auto reduced = ::tyloi::cuda::zeros({source.size(0), c.hidden},
                                                source.options().dtype(::tyloi::kFloat32));
            for (int64_t r = 0; r < c.topk; ++r)
                reduced = reduced + pairs.select(1, r).to(::tyloi::kFloat32) *
                                    selected[1].select(1, r).unsqueeze(-1);
            routed = reduced.to(pairs.scalar_type());
        }
        return (routed + shared_output).reshape(x.sizes());
    };
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
        for (const auto& part : parts) dense.push_back(dynamic_cast<const DenseTensor&>(*part));
        return std::make_shared<DenseTensor>(::tyloi::cuda::cat(dense, 0));
    }
#else
    static DenseTensor concat(const std::vector<DenseTensor> &parts) { return ::tyloi::cuda::cat(parts, 0); }
#endif
    static auto headwise(Linear projection, int64_t heads, int64_t output) {
        return glm5_next::headwise(std::move(projection), heads, output);
    }
    static auto kda(KdaWeights weights, const Config &c) {
        return std::make_unique<Kda>(std::move(weights), c.kda_heads, c.kda_width, c.kda_kernel,
                                     c.lower_bound, c.eps);
    }
    static auto mla(MlaWeights weights, const Config &c) {
        MlaConfig geometry{c.heads, c.nope, c.latent, c.value_width, c.index_heads, c.index_width,
                           c.pool, c.budget, c.maximum, c.tail, c.eps};
        return std::make_unique<SparseMla>(std::move(weights), geometry);
    }
    static auto gated_mlp(Linear gate, Linear up, Linear down, double limit) {
        return dense_ffn(std::move(gate), std::move(up), std::move(down), limit);
    }
    static auto moe(std::vector<Linear> gate_up, Linear down, Linear router,
                    Transform shared, Parameter bias, const Config &c) {
        return glm_moe(std::move(gate_up), std::move(down), std::move(router), std::move(shared),
                       std::move(bias), c);
    }
};

static DenseTensor execute(const Layer& layer, LayerState& activity, const Projector& project,
                      const DenseTensor &x, int64_t position, int64_t confirmed = 0) {
    const auto& [config, attention_hc, ffn_hc, attention_norm, ffn_norm, ffn, kda, mla, device] = layer;
    if (mla)
        TYLOINFER_RUNTIME_CHECK(std::get<MlaState>(activity).position() == position,
                               "GLM MLA/model cache positions diverged");
    auto attention_mix = attention_hc.pre(x, config);
    auto normalized = rms_norm(attention_mix[2], attention_norm, config.eps);
    auto branch = kda ? kda->forward(project, std::get<KdaState>(activity), std::move(normalized), true, confirmed)
                      : mla->forward(project, std::get<MlaState>(activity), std::move(normalized), true);
    auto hidden = mfq_glm5_next::mhc_post(branch, x, attention_mix[0], attention_mix[1]);
    auto ffn_mix = ffn_hc.pre(hidden, config);
    normalized = rms_norm(ffn_mix[2], ffn_norm, config.eps);
    branch = ffn(project, normalized);
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
        model.output_norm = dynamic_cast<const DenseTensor&>(*weights.tensor("model.output_norm.weight"));
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
        model.output_norm = dynamic_cast<const DenseTensor&>(*weights.tensor("model.output_norm.weight"));
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
             ::tyloi::Tensor input_ids, const Executor::Mode& mode) {
    auto ids = input_ids.dense();
#else
DenseTensor embed(const Model& model, Executor& executor, Collectives& copies, int primary,
             DenseTensor ids, const Executor::Mode& mode) {
#endif
    const Task task{task::Embed{0, 0, model.embed}, -1, {}, "model.embed"};
    Inputs inputs;
    inputs.values = {normalize_ids(copies, primary, std::move(ids))};
    return executor.run(std::span<const Task>{&task, 1}, std::move(inputs), mode);
}

#ifdef TYLOINFER_NATIVE_CUDA
::tyloi::Tensor normalize(const Model& model, Executor&, ::tyloi::Tensor input_hidden, int64_t, int64_t) {
    auto hidden = input_hidden.dense();
#else
DenseTensor normalize(const Model& model, Executor&, DenseTensor hidden, int64_t, int64_t) {
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    return rms_norm(hidden.mean(2), *model.output_norm, model.metadata.rms_norm_eps);
#else
    return rms_norm(hidden.mean(2), model.output_norm, model.metadata.rms_norm_eps);
#endif
}

#ifdef TYLOINFER_NATIVE_CUDA
::tyloi::Tensor softcap(const Model&, Executor&, ::tyloi::Tensor value) { return value; }
#else
DenseTensor softcap(const Model&, Executor&, DenseTensor value) { return value; }
#endif

#ifdef TYLOINFER_NATIVE_CUDA
::tyloi::Tensor logits(const Model& model, Executor& executor, ::tyloi::Tensor input_hidden, bool,
              const Executor::Mode& mode) {
    auto hidden = input_hidden.dense();
#else
DenseTensor logits(const Model& model, Executor& executor, DenseTensor hidden, bool,
              const Executor::Mode& mode) {
#endif
    return executor.projector(mode.ffn).run(model.lm_head, hidden);
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
        !input.positions.has_value() && !input.cache_positions.has_value(),
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
    auto project = executor.projector(mode.ffn);
    for (size_t index = 0; index < model.layers.size(); ++index) {
        const auto& layer = model.layers[index];
        MfqCudaGuard layer_guard(layer.device);
        hidden = execute(layer, state.layers[index], project,
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


Mtp::Mtp(Projector project, Collectives& copies, CudaProfiler& profiler,
                        std::shared_ptr<const ::tyloi::Tensor> embedding, int primary)
    : project_(std::move(project)), copies_(copies), profiler_(profiler),
      embedding_(std::move(embedding)), primary_(primary) {}
Mtp::~Mtp() = default;
Mtp::Mtp(Mtp&&) noexcept = default;

std::optional<Mtp> Mtp::load_if_present(const ::tyloi::weights::Loader& weights,
                                                      Projector projector, Collectives& copies,
                                                      CudaProfiler& profiler, std::shared_ptr<const ::tyloi::Tensor> embedding,
                                                      const tyloi::models::glm5_next::Config &main) {
    Mtp result{std::move(projector), copies, profiler, std::move(embedding),
                       weights.placement.device};
    LayerLoader loader{weights};
    loader.role = "glm5_next";
    if (!tyloi::models::glm5_next::load_predictor(result, loader, main))
        return std::nullopt;
    result.states.reserve(result.layers.size());
    for (size_t i = 0; i < result.layers.size(); ++i)
        result.states.emplace_back(main.maximum, main.latent, main.index_width);
    return result;
}

void Mtp::reset(int64_t next_batch) {
    TYLOINFER_RUNTIME_CHECK(next_batch > 0, "GLM5-Next MTP batch must be positive");
    for (auto &state : states)
        state.reset();
    std::fill(lengths.begin(), lengths.end(), 0);
    batch = next_batch;
}

std::pair<DenseTensor, DenseTensor> Mtp::evaluate(const DenseTensor &hidden,
                                       const DenseTensor &ids, int64_t depth, bool cache,
                                       const DenseTensor &supplied_positions,
                                       const DenseTensor &supplied_embeddings) {
#ifdef TYLOINFER_NATIVE_CUDA
    TYLOINFER_RUNTIME_CHECK(ids.dim() == 2 && ids.size(0) > 0 && ids.size(1) > 0 &&
                          hidden.dim() == 3 && hidden.size(0) == ids.size(0) &&
                          hidden.size(1) == ids.size(1) && hidden.size(2) == hidden_norm->numel(),
                      "GLM5-Next MTP input geometry mismatch");
#else
    TYLOINFER_RUNTIME_CHECK(ids.dim() == 2 && ids.size(0) > 0 && ids.size(1) > 0 &&
                          hidden.dim() == 3 && hidden.size(0) == ids.size(0) &&
                          hidden.size(1) == ids.size(1) && hidden.size(2) == hidden_norm.numel(),
                      "GLM5-Next MTP input geometry mismatch");
#endif
    const auto b = ids.size(0), t = ids.size(1);
    auto dtype = ::tyloi::kFloat16;
    return tyloi::models::projected_predictor(
        *this, b, t, depth, cache, [&](int64_t next) { reset(next); },
        [&] {
            auto embeds =
                supplied_embeddings.defined() ? supplied_embeddings : tyloi::cuda::embed(*embedding_, ids, copies_, profiler_, primary_);
            TYLOINFER_RUNTIME_CHECK(embeds.sizes().vec() ==
                                  std::vector<int64_t>({b, t, config.hidden}),
                              "GLM5-Next MTP embedding shape mismatch");
            return embeds;
        },
        [&](tyloi::models::PredictorCursor cursor) {
            auto current = supplied_positions.defined()
                               ? supplied_positions
                               : ::tyloi::cuda::arange(cursor.start, cursor.start + t,
                                            ids.options().dtype(::tyloi::kInt32));
            TYLOINFER_RUNTIME_CHECK((current.dim() == 1 || current.dim() == 2) &&
                                  current.size(-1) == t &&
                                  (current.dim() == 1 || current.size(0) == b),
                              "GLM MTP positions require [T] or [B,T]");
            current = current.to(::tyloi::kInt32);
            return current;
        },
        [&](DenseTensor embeds, const auto &current) {
            auto mask = (current == 0).reshape({current.dim() == 1 ? 1 : b, t, 1});
            auto masked = ::tyloi::cuda::where(mask, ::tyloi::cuda::zeros_like(embeds), embeds);
#ifdef TYLOINFER_NATIVE_CUDA
            return rms_norm(masked, *embedding_norm, config.eps);
#else
            return rms_norm(masked, embedding_norm, config.eps);
#endif
        },
#ifdef TYLOINFER_NATIVE_CUDA
        [&] { return rms_norm(hidden, *hidden_norm, config.eps); },
#else
        [&] { return rms_norm(hidden, hidden_norm, config.eps); },
#endif
        [&](DenseTensor e, DenseTensor streams) { return project_.run(*fusion, ::tyloi::cuda::cat({e, streams}, -1)); },
        [&](DenseTensor x, int64_t layer, const auto &pos) {
            auto &block = layers[layer];
            dtype = x.scalar_type() == ::tyloi::kFloat32 ? ::tyloi::kFloat32 : ::tyloi::kFloat16;
            auto normalized = rms_norm(x, block.attention_norm, config.eps);
            auto branch = block.attention->forward(project_, states[layer], normalized, cache);
            x = x.to(::tyloi::kFloat32) + branch.to(::tyloi::kFloat32);
            normalized = rms_norm(x, block.ffn_norm, config.eps).to(dtype);
            branch = block.ffn(project_, normalized);
            return x.to(::tyloi::kFloat32) + branch.to(::tyloi::kFloat32);
        },
#ifdef TYLOINFER_NATIVE_CUDA
        [&](const DenseTensor &multi) { return rms_norm(multi, *output_norm, config.eps).to(dtype); },
#else
        [&](const DenseTensor &multi) { return rms_norm(multi, output_norm, config.eps).to(dtype); },
#endif
        [&](int64_t layer, const auto &pos) {});
}

tyloi::models::MtpStep Mtp::step(const ::tyloi::Tensor& hidden, const ::tyloi::Tensor& ids,
                                  const ::tyloi::Tensor* positions) {
    if (positions && positions->defined())
        throw std::runtime_error("this CUDA MTP predictor does not accept explicit positions");
    auto result = evaluate(dynamic_cast<const DenseTensor&>(hidden),
                           dynamic_cast<const DenseTensor&>(ids));
    return {std::make_shared<DenseTensor>(std::move(result.first)),
            std::make_shared<DenseTensor>(std::move(result.second))};
}

int64_t Mtp::cache_position() const noexcept {
    return lengths.empty() ? 0 : lengths.front();
}

void Mtp::trim_cache_to(int64_t position) {
    TYLOINFER_RUNTIME_CHECK(position >= 0 && position <= cache_position(),
                      "GLM5-Next MTP cache trim position is invalid");
    constexpr size_t index = 0;
    if (index < layers.size() && layers[index].attention)
        states[index].truncate(position);
    lengths[index] = position;
}

bool Mtp::teacher_forced_prompt_prime() const noexcept { return false; }

bool Mtp::target_bootstrap_decode() const noexcept { return true; }

bool Mtp::preserve_output_dtype() const noexcept { return true; }

} // namespace tyloi::cuda::glm5_next
