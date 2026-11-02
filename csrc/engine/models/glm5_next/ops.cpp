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
#include "engine/models/common/session/paged.h"
#include "engine/ops.h"
#include "engine/models/common/transformer_layer.h"

namespace mfq::cuda::glm5_next {
using Config = mfq::models::glm5_next::Config;
using attention_ops::Transform;

static Transform headwise(weight_loader::Routed projection,int64_t heads,int64_t output) {
    return [projection=std::move(projection),heads,output](
            const Projector& project, const Tensor& x) {
        TYLOINFER_RUNTIME_CHECK(x.dim()==4 && x.size(2)==heads,"GLM head-wise projection shape mismatch");
        const auto b=x.size(0),t=x.size(1),rows=b*t*heads;
        auto ids=tb::arange(rows,x.options().dtype(tb::kInt32)).remainder(heads).reshape({rows,1});
        return projection(project,x.reshape({rows,x.size(-1)}),ids).reshape({b,t,heads,output});
    };
}

static Transform dense_ffn(Linear gate, Linear up, Linear down, double limit) {
    return [gate, up, down, limit](const Projector& project, const Tensor &x) {
        auto unfused = [](const auto &...) { return std::optional<Tensor>{}; };
        return mfq::ops::gated_mlp(
            x, false, limit, unfused, unfused,
            [&](Tensor input) {
                return std::array<Tensor, 2>{project.run(*gate, input), project.run(*up, input)};
            },
            [](Tensor g, Tensor u, mfq::ops::GatedActivation, double bound) {
                g = tb::clamp_max(g, bound);
                u = tb::clamp(u, -bound, bound);
                return (g * tb::sigmoid(g)) * u;
            },
            [&](Tensor hidden) { return project.run(*down, hidden); }, unfused);
    };
}

static Transform glm_moe(weight_loader::Routed gate_up, weight_loader::Routed down,
                      Linear router, Transform shared, Tensor bias, const Config &c) {
    return [gate_up, down, router, shared, bias, c](const Projector& project,
                                                    const Tensor &x) {
        auto source = x.reshape({-1, c.hidden}).to(tb::kFloat16);
        const auto routing = c.routing();
        return mfq::ops::mixture_of_experts(
            [&] {
                return project.run(*router, source.to(tb::kFloat32)).to(tb::kFloat32).contiguous();
            },
            [&](Tensor logits) {
                return moe_topk_cuda(
                    logits, c.topk, routing.activation == mfq::ops::RouterActivation::sigmoid,
                    routing.activation == mfq::ops::RouterActivation::sqrt_softplus,
                    routing.normalize, routing.delayed_softmax, bias, 1e-20, routing.scale);
            },
            [&](const auto &) { return shared(project, source); },
            [&](const auto &selected) {
                return mfq::ops::routed_experts(
                    [&] { return gate_up(project, source, selected[0]); },
                    [&](Tensor gu) {
                        auto gate =
                            tb::clamp_max(gu.narrow(-1, 0, c.moe_intermediate), c.swiglu_limit);
                        auto up = tb::clamp(gu.narrow(-1, c.moe_intermediate, c.moe_intermediate),
                                            -c.swiglu_limit, c.swiglu_limit);
                        return (gate * tb::sigmoid(gate)) * up;
                    },
                    [&](Tensor hidden) { return down(project, hidden, selected[0]); },
                    [&](Tensor pairs) {
                        auto reduced = tb::zeros({source.size(0), c.hidden},
                                                 source.options().dtype(tb::kFloat32));
                        for (int64_t r = 0; r < c.topk; ++r)
                            reduced = reduced + pairs.select(1, r).to(tb::kFloat32) *
                                                    selected[1].select(1, r).unsqueeze(-1);
                        return reduced.to(pairs.scalar_type());
                    });
            },
            [&](Tensor routed, Tensor shared_output, const auto &) {
                return (routed + shared_output).reshape(x.sizes());
            });
    };
}

struct LayerLoader : weight_loader::Loader {
    using KdaWeights = glm5_next::KdaWeights;
    using MlaWeights = glm5_next::MlaWeights;

    static Mhc mhc(Tensor function, Tensor base, Tensor scale) {
        return {std::move(function), std::move(base), std::move(scale)};
    }
    static Tensor concat(const std::vector<Tensor> &parts) { return tb::cat(parts, 0); }
    static auto headwise(weight_loader::Routed projection, int64_t heads, int64_t output) {
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
    static auto moe(weight_loader::Routed gate_up, weight_loader::Routed down, Linear router,
                    Transform shared, Tensor bias, const Config &c) {
        return glm_moe(std::move(gate_up), std::move(down), std::move(router), std::move(shared),
                       std::move(bias), c);
    }
};

static Tensor execute(const Layer& layer, LayerState& activity, const Projector& project,
                      const Tensor &x, int64_t position, int64_t confirmed = 0) {
    const auto& [config, attention_hc, ffn_hc, attention_norm, ffn_norm, ffn, kda, mla, device] = layer;
    if (mla)
        TYLOINFER_RUNTIME_CHECK(std::get<MlaState>(activity).position() == position,
                               "GLM MLA/model cache positions diverged");
    return mfq::models::glm5_next::decoder_layer(
        x, bool(kda), [&](const Tensor &hidden) { return attention_hc.pre(hidden, config); },
        [&](Tensor branch, int role) {
            return rms_norm(branch, role == 0 ? attention_norm : ffn_norm, config.eps);
        },
        [&](Tensor branch) { return kda->forward(project, std::get<KdaState>(activity), branch, true, confirmed); },
        [&](Tensor branch) { return mla->forward(project, std::get<MlaState>(activity), branch, true); },
        [](Tensor branch, const Tensor &residual, const auto &mix) {
            return mfq_glm5_next::mhc_post(branch, residual, mix[0], mix[1]);
        },
        [&](const Tensor &hidden) { return ffn_hc.pre(hidden, config); },
        [&](Tensor branch) { return ffn(project, branch); });
}

Layer load_layer(const weight_loader::Loader& weights, const Config& config, int index) {
    MfqCudaGuard guard(weights.placement.device);
    Layer layer;
    layer.device = weights.placement.device;
    LayerLoader loader{weights};
    loader.role = "glm5_next";
    mfq::models::glm5_next::load_layer_weights(layer, loader, config, index);
    return layer;
}

Model read(std::shared_ptr<const mfq::ModelSource> source, std::string_view definition,
           int64_t context) {
    validate_model_source(*source);
    Model model;
    model.source = std::move(source);
    model.graph = mfq::models::graph(*model.source);
    model.plan = cuda_model_plan(model.graph);
    mfq::models::require_model(model.graph.backbone == "glm5_next",
                               "model backbone does not match the requested causal model");
    mfq::models::glm5_next::load_config(definition, model.config, model.metadata);
    mfq::models::require_model(model.config.layers == model.graph.topology.text_layers,
                               "model graph/config text-layer topology mismatch");
    if (context > 0) {
        mfq::models::require_model(context <= model.metadata.max_position_embeddings,
                                   "--ctx-size exceeds max_position_embeddings");
        model.config.maximum = context;
        model.metadata.max_position_embeddings = context;
    }
    return model;
}

void load(Model& model, const weight_loader::Loader& weights,
          LayerPlacementConfig& placement, int gpu_layers, bool load_layers,
          bool load_input_output) {
    validate_native_placement(weights.placement.tensor, placement, gpu_layers);
    placement.prepare(model.config.layers);
    MfqCudaGuard guard(placement.primary_device());
    if (load_input_output) {
        const bool separate = !model.config.tied_embeddings && weights.has("model.output.weight");
        model.embed = weights.embedding("model.token_embedding.weight", separate);
        model.output_norm = weights.tensor("model.output_norm.weight");
        model.lm_head = separate ? weights.projection("model.output.weight", TensorParallelAxis::Output)
                                 : model.embed;
        if (load_layers) {
            model.layers.reserve(static_cast<size_t>(model.config.layers));
            for (int index = 0; index < model.config.layers; ++index)
                model.layers.push_back(load_layer(weights.for_layer(placement.device_for_layer(index), false),
                                                  model.config, index));
        }
    } else {
        if (load_layers) throw std::invalid_argument("streamed model setup requires deferred blocks");
        model.output_norm = weights.tensor("model.output_norm.weight");
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
    mfq::models::require_model(speculative_start >= 0, "no speculative transaction");
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

static Tensor normalize_ids(Collectives& copies, int primary, Tensor ids) {
    mfq::models::require_model(ids.dim() == 1 || ids.dim() == 2,
                               "token IDs must have shape [tokens] or [batch,tokens]");
    if (ids.dim() == 1) ids = ids.unsqueeze(0);
    mfq::models::require_model(ids.size(0) > 0 && ids.size(1) > 0, "token IDs must not be empty");
    return copies.copy(ids.to(tb::kInt64), primary).contiguous();
}

Tensor embed(const Model& model, Executor& executor, Collectives& copies, int primary,
             Tensor ids, const Executor::Mode& mode) {
    const Task task{task::Embed{0, 0, model.embed}, -1, {}, "model.embed"};
    Inputs inputs;
    inputs.values = {normalize_ids(copies, primary, std::move(ids))};
    return executor.run(std::span<const Task>{&task, 1}, std::move(inputs), mode);
}

Tensor normalize(const Model& model, Executor&, Tensor hidden, int64_t, int64_t) {
    return rms_norm(hidden.mean(2), model.output_norm, model.metadata.rms_norm_eps);
}

Tensor softcap(const Model&, Executor&, Tensor value) { return value; }

Tensor logits(const Model& model, Executor& executor, Tensor hidden, bool,
              const Executor::Mode& mode) {
    return executor.projector(mode.ffn).run(model.lm_head, hidden);
}

bool supports_speculation(const Model& model) noexcept { return !model.layers.empty(); }

Tensor forward(const Model& model, State& state, Executor& executor, Collectives& copies,
               const Executor::Mode& mode, Input input) {
    MfqCudaGuard guard(mode.primary);
    auto ids = normalize_ids(copies, mode.primary, std::move(input.ids));
    auto embeddings = input.embeddings.has_value() ? std::move(*input.embeddings)
        : embed(model, executor, copies, mode.primary, ids, mode);
    const auto B = ids.size(0), T = ids.size(1);
    mfq::models::require_model(embeddings.dim() == 3 && embeddings.size(0) == B &&
        embeddings.size(1) == T && embeddings.size(2) == model.metadata.hidden_size,
        "inputs_embeds shape must match [batch,tokens,hidden_size]");
    mfq::models::require_model(T > 0 && state.cache_pos + T <= model.metadata.max_position_embeddings &&
        !input.positions.has_value() && !input.cache_positions.has_value() && !input.attention_mask.has_value(),
        "GLM Flash-Next currently requires contiguous causal cache positions without an external mask");
    mfq::models::require_model(state.speculative_start < 0,
        "commit or roll back the pending speculative pass before forwarding");
    mfq::models::require_model(input.confirmed >= 0 &&
        (input.confirmed == 0 || (input.confirmed < T && B == 1 && state.cache_pos > 0 &&
                                  supports_speculation(model))),
        "unsupported speculative backbone geometry");
    if (input.confirmed > 0) {
        state.speculative_start = state.cache_pos;
        state.speculative_confirmed = input.confirmed;
    }
    if (state.cache_pos == 0) state.reset(B);
    auto hidden = copies.copy(std::move(embeddings), mode.primary).contiguous()
        .to(tb::kFloat16).unsqueeze(2).expand({B, T, model.config.streams, model.config.hidden})
        .contiguous();
    const auto trace = [&](const Tensor& value) {
        if (input.trace) input.trace->push_back(copies.copy(value, mode.primary).to(tb::kFloat32).clone());
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
    mfq::models::require_model(state.speculative_start >= 0 && accepted >= 0, "no speculative transaction");
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
                        QuantLinear embedding, int primary)
    : project_(std::move(project)), copies_(copies), profiler_(profiler),
      embedding_(std::move(embedding)), primary_(primary) {}
Mtp::~Mtp() = default;
Mtp::Mtp(Mtp&&) noexcept = default;

std::optional<Mtp> Mtp::load_if_present(const weight_loader::Loader& weights,
                                                      Projector projector, Collectives& copies,
                                                      CudaProfiler& profiler, QuantLinear embedding,
                                                      const mfq::models::glm5_next::Config &main) {
    Mtp result{std::move(projector), copies, profiler, std::move(embedding),
                       weights.placement.device};
    LayerLoader loader{weights};
    loader.role = "glm5_next";
    if (!mfq::models::glm5_next::load_predictor(result, loader, main))
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

std::pair<Tensor, Tensor> Mtp::evaluate(const Tensor &hidden,
                                       const Tensor &ids, int64_t depth, bool cache,
                                       const Tensor &supplied_positions,
                                       const Tensor &supplied_embeddings) {
    namespace tb = mfq_tensor_backend;
    TYLOINFER_RUNTIME_CHECK(ids.dim() == 2 && ids.size(0) > 0 && ids.size(1) > 0 &&
                          hidden.dim() == 3 && hidden.size(0) == ids.size(0) &&
                          hidden.size(1) == ids.size(1) &&
                          hidden.size(2) == hidden_norm.numel(),
                      "GLM5-Next MTP input geometry mismatch");
    const auto b = ids.size(0), t = ids.size(1);
    auto dtype = tb::kFloat16;
    return mfq::models::projected_predictor(
        *this, b, t, depth, cache, [&](int64_t next) { reset(next); },
        [&] {
            auto embeds =
                supplied_embeddings.defined() ? supplied_embeddings : mfq::cuda::embed(embedding_, ids, copies_, profiler_, primary_);
            TYLOINFER_RUNTIME_CHECK(embeds.sizes().vec() ==
                                  std::vector<int64_t>({b, t, config.hidden}),
                              "GLM5-Next MTP embedding shape mismatch");
            return embeds;
        },
        [&](mfq::models::PredictorCursor cursor) {
            auto current = supplied_positions.defined()
                               ? supplied_positions
                               : tb::arange(cursor.start, cursor.start + t,
                                            ids.options().dtype(tb::kInt32));
            TYLOINFER_RUNTIME_CHECK((current.dim() == 1 || current.dim() == 2) &&
                                  current.size(-1) == t &&
                                  (current.dim() == 1 || current.size(0) == b),
                              "GLM MTP positions require [T] or [B,T]");
            current = current.to(tb::kInt32);
            return current;
        },
        [&](Tensor embeds, const auto &current) {
            auto mask = (current == 0).reshape({current.dim() == 1 ? 1 : b, t, 1});
            auto masked = tb::where(mask, tb::zeros_like(embeds), embeds);
            return rms_norm(masked, embedding_norm, config.eps);
        },
        [&] { return rms_norm(hidden, hidden_norm, config.eps); },
        [&](Tensor e, Tensor streams) { return project_.run(*fusion, tb::cat({e, streams}, -1)); },
        [&](Tensor x, int64_t layer, const auto &pos) {
            auto &block = layers[layer];
            dtype = x.scalar_type() == tb::kFloat32 ? tb::kFloat32 : tb::kFloat16;
            return mfq::models::pre_norm_layer(
                std::move(x),
                [&](const Tensor &value, int stage) {
                    auto normalized = rms_norm(
                        value, stage == 0 ? block.attention_norm : block.ffn_norm, config.eps);
                    return stage == 0 ? normalized : normalized.to(dtype);
                },
                [&](Tensor value) {
                    return block.attention->forward(project_, states[layer], value, cache);
                },
                [&](Tensor value) { return block.ffn(project_, value); },
                [](Tensor residual, Tensor branch) {
                    return residual.to(tb::kFloat32) + branch.to(tb::kFloat32);
                });
        },
        [&](const Tensor &multi) { return rms_norm(multi, output_norm, config.eps).to(dtype); },
        [&](int64_t layer, const auto &pos) {});
}

Tensor Mtp::forward(Tensor hidden, Tensor ids) {
    return evaluate(hidden, ids).first;
}

MtpStep Mtp::step(Tensor hidden, Tensor ids) {
    auto result = evaluate(hidden, ids);
    return {std::move(result.first), std::move(result.second)};
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

} // namespace mfq::cuda::glm5_next
