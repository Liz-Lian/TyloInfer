#pragma once
#include "engine/backends/cuda/kernels/mfq_cuda_linear_attention_ops.h"
#include "engine/models/glm5_next/weights.h"
#include "engine/models/glm5_next/state.h"
#include "engine/models/metadata.h"
#include "engine/models/common/capabilities.h"
#include "engine/models/source.h"
#include "engine/backends/cuda/ops/projection/quant_linear.h"
#include "engine/models/common/attention_ops.h"
#include "engine/backends/cuda/kernels/glm5_next.h"
#include <array>

namespace tyloi::cuda::glm5_next {
using attention_ops::DenseTensor;
using attention_ops::Linear;
using attention_ops::Embedding;
using attention_ops::rms_norm;
using attention_ops::SequenceCache;
using attention_ops::select_pooled_blocks;

#ifdef TYLOINFER_NATIVE_CUDA
using Parameter = ::tyloi::Tensor;
#else
using Parameter = DenseTensor;
#endif

struct KdaWeights {
    Linear query, key, value, beta, gate_a, gate_b, output;
    Parameter conv, forget_a, forget_b, dt_bias, a_log, output_norm;
};

class Kda {
  public:
    Kda(KdaWeights weights, int64_t heads, int64_t width, int64_t kernel, double lower_bound,
        double eps)
        : w_(std::move(weights)), heads_(heads), width_(width), kernel_(kernel),
          lower_bound_(lower_bound), eps_(eps) {
        TYLOINFER_RUNTIME_CHECK(heads > 0 && width > 0 && kernel > 1 && std::isfinite(lower_bound) &&
                              std::isfinite(eps) && eps > 0,
                          "invalid GLM KDA configuration");
    }
    DenseTensor forward(KdaState& state, const DenseTensor &hidden, bool use_cache,
                   int64_t confirmed = 0) const {
        TYLOINFER_RUNTIME_CHECK(hidden.is_cuda() && hidden.dim() == 3 && hidden.size(0) > 0,
                          "recurrent attention requires nonempty [B,T,H] input");
        return tyloi::models::recurrent_window(
            hidden.size(1), use_cache, confirmed, state.pending(),
            [&](int64_t start, int64_t count) {
                return forward_chunk(state, hidden.narrow(1, start, count), use_cache);
            },
            [&] {
                state.rollback_conv = state.conv;
                state.rollback_recurrent = state.recurrent;
            },
            [&] { state.rollback(); },
            [](DenseTensor prefix, DenseTensor suffix) { return ::tyloi::cuda::cat({prefix, suffix}, 1); });
    }
    DenseTensor forward_chunk(KdaState& state, const DenseTensor &hidden, bool use_cache) const {
        const auto b = hidden.size(0), t = hidden.size(1), channels = heads_ * width_;
        const auto options = hidden.options().dtype(::tyloi::kFloat32);
        TYLOINFER_RUNTIME_CHECK(!use_cache || !state.conv.defined() ||
                              (state.conv.size(0) == b && state.conv.device() == hidden.device()),
                          "reset GLM KDA before changing batch/device");
        auto previous = use_cache && state.conv.defined()
                            ? state.conv
                            : ::tyloi::cuda::zeros({b, kernel_ - 1, 3 * channels}, options);
        return tyloi::models::gated_delta_attention(
            use_cache,
            [&] {
                auto projected = ::tyloi::cuda::cat({::tyloi::ops::project(hidden, w_.query).dense(), ::tyloi::ops::project(hidden, w_.key).dense(),
                                          ::tyloi::ops::project(hidden, w_.value).dense()},
                                         -1)
                                     .to(::tyloi::kFloat32);
                TYLOINFER_RUNTIME_CHECK(projected.sizes().vec() ==
                                      std::vector<int64_t>({b, t, 3 * channels}),
                                  "GLM KDA projection shape mismatch");
                return projected;
            },
            [&](DenseTensor projected) {
                auto joined = ::tyloi::cuda::cat({previous, projected}, 1).contiguous();
                auto convolved = ssm_conv_silu_cuda(joined,
#ifdef TYLOINFER_NATIVE_CUDA
                                                    w_.conv.dense().to(::tyloi::kFloat32).contiguous(),
#else
                                                    w_.conv.to(::tyloi::kFloat32).contiguous(),
#endif
                                                    ::tyloi::cuda::empty({0}, options), t);
                auto next_conv = joined.narrow(1, t, kernel_ - 1).contiguous();
                const auto heads = [&](int64_t begin) {
                    return convolved.narrow(-1, begin, channels)
                        .reshape({b, t, heads_, width_})
                        .permute({0, 2, 1, 3});
                };
                const auto normalize = [&](const DenseTensor &x) {
                    return (x / ::tyloi::cuda::clamp_min((x * x).sum(-1, true).sqrt(), 1e-6)).contiguous();
                };
                auto q = normalize(heads(0)), k = normalize(heads(channels)),
                     v = heads(2 * channels).contiguous();
                return std::array<DenseTensor, 4>{q, k, v, next_conv};
            },
            [&] {
                auto forget =
                    mfq_glm5_next::kda_forget_gate(hidden,
#ifdef TYLOINFER_NATIVE_CUDA
                                                   w_.forget_a.dense(),
                                                   w_.forget_b.dense(),
                                                   w_.dt_bias.dense(),
                                                   w_.a_log.dense(),
#else
                                                   w_.forget_a, w_.forget_b, w_.dt_bias, w_.a_log,
#endif
                                                   heads_, width_, lower_bound_)
                        .permute({0, 2, 1, 3})
                        .contiguous();
                auto beta = ::tyloi::cuda::sigmoid(::tyloi::ops::project(hidden, w_.beta).dense().to(::tyloi::kFloat32))
                                .transpose(1, 2)
                                .contiguous();
                return std::array<DenseTensor, 2>{forget, beta};
            },
            [&](const auto &convolution, const auto &gates) {
                auto q = convolution[0], k = convolution[1], v = convolution[2];
                auto forget = gates[0], beta = gates[1];
                DenseTensor attended, next_recurrent;
                auto initial = use_cache && state.recurrent.defined() ? state.recurrent : DenseTensor{};
                if (width_ == 32 || width_ == 64 || width_ == 128) {
                    auto result =
                        gdn_cuda(q, k, v, forget, beta,
                                 initial.defined() ? MfqOptional<DenseTensor>(initial) : mfq_nullopt);
                    attended = result[0];
                    next_recurrent = result[1];
                } else {
                    // Public model dimensions use the fused kernel. Keep the exact
                    // recurrence available for small independent architecture fixtures.
                    next_recurrent = initial.defined()
                                         ? initial
                                         : ::tyloi::cuda::zeros({b, heads_, width_, width_}, options);
                    std::vector<DenseTensor> steps;
                    for (int64_t i = 0; i < t; ++i) {
                        auto ki = k.select(2, i);
                        next_recurrent = next_recurrent * forget.select(2, i).exp().unsqueeze(-1);
                        auto predicted = (next_recurrent * ki.unsqueeze(-1)).sum(-2);
                        auto delta = (v.select(2, i) - predicted) * beta.select(2, i).unsqueeze(-1);
                        next_recurrent = next_recurrent + ki.unsqueeze(-1) * delta.unsqueeze(-2);
                        steps.push_back((next_recurrent *
                                         (q.select(2, i) / std::sqrt(double(width_))).unsqueeze(-1))
                                            .sum(-2)
                                            .unsqueeze(2));
                    }
                    attended = ::tyloi::cuda::cat(steps, 2);
                }
                return std::array<DenseTensor, 2>{attended, next_recurrent};
            },
            [&](DenseTensor attended) {
                auto gate = ::tyloi::ops::project(::tyloi::ops::project(hidden, w_.gate_a), w_.gate_b).dense()
                                .reshape({b, t, heads_, width_})
                                .permute({0, 2, 1, 3});
                return rms_norm(attended,
#ifdef TYLOINFER_NATIVE_CUDA
                                w_.output_norm,
#else
                                w_.output_norm,
#endif
                                eps_) *
                       ::tyloi::cuda::sigmoid(gate.to(::tyloi::kFloat32));
            },
            [&](DenseTensor normalized) {
                auto result = ::tyloi::ops::project(normalized.permute({0, 2, 1, 3})
                                                       .reshape({b, t, channels})
                                                       .to(hidden.scalar_type()), w_.output).dense();
                return result;
            },
            [&](const auto &convolution, const DenseTensor &next) {
                state.conv = convolution[3];
                state.recurrent = next;
            });
    }

  private:
    KdaWeights w_;
    int64_t heads_, width_, kernel_;
    double lower_bound_, eps_;
};

struct MlaWeights {
    Linear query_a, key_value_a, query_b, output, index_query, index_key, index_score;
    // Head-wise absorbed projections accept/return [B,T,heads,width].
    Linear embed_query, unembed_output;
    Parameter query_norm, latent_norm, index_norm, index_bias, index_gate, index_position;
};

struct MlaConfig {
    int64_t heads, nope, latent, value_width, index_heads, index_width, pool, budget, maximum;
    bool tail;
    double eps;
};

class SparseMla {
  public:
    SparseMla(MlaWeights weights, MlaConfig config)
        : w_(std::move(weights)), c_(config) {
        TYLOINFER_RUNTIME_CHECK(c_.heads > 0 && c_.nope > 0 && c_.value_width > 0 && c_.index_heads > 0 &&
                              c_.pool > 0 && c_.budget > 0 && c_.budget % c_.pool == 0,
                          "invalid Flash-Next MLA configuration");
    }
    DenseTensor forward(MlaState& state, const DenseTensor &hidden, bool use_cache) const {
        TYLOINFER_RUNTIME_CHECK(hidden.is_cuda() && hidden.dim() == 3 && hidden.size(0) > 0 &&
                              hidden.size(1) > 0,
                          "GLM MLA requires nonempty [B,T,H] input");
        const auto b = hidden.size(0), t = hidden.size(1), offset = use_cache ? state.position() : 0;
        TYLOINFER_RUNTIME_CHECK(offset == (use_cache ? state.index.position() : 0) && t <= c_.maximum - offset,
                          "GLM MLA cache position/capacity mismatch");
        struct Projection {
            DenseTensor rank, query, latent, index;
        };
        const double scale = 1.0 / std::sqrt(double(c_.nope));
        return tyloi::models::glm5_next::sparse_mla(
            offset + t, c_.budget, use_cache,
            [&] {
                auto qr = rms_norm(::tyloi::ops::project(hidden, w_.query_a).dense(),
#ifdef TYLOINFER_NATIVE_CUDA
                                   w_.query_norm,
#else
                                   w_.query_norm,
#endif
                                   c_.eps);
                auto query = ::tyloi::ops::project(qr, w_.query_b).dense().reshape({b, t, c_.heads, c_.nope});
                auto latent = rms_norm(::tyloi::ops::project(hidden, w_.key_value_a).dense().narrow(-1, 0, c_.latent),
#ifdef TYLOINFER_NATIVE_CUDA
                                       w_.latent_norm, c_.eps);
#else
                                       w_.latent_norm, c_.eps);
#endif
                // Metal's indexer LayerNorm explicitly casts the source/output to F16.
                auto ik = ::tyloi::ops::project(hidden, w_.index_key).dense().to(::tyloi::kFloat16).to(::tyloi::kFloat32);
                auto centered = ik - ik.mean(-1, true);
#ifdef TYLOINFER_NATIVE_CUDA
                ik = (centered * ::tyloi::cuda::rsqrt((centered * centered).mean(-1, true) + 1e-6) *
                          w_.index_norm.dense().to(::tyloi::kFloat32) +
                      w_.index_bias.dense().to(::tyloi::kFloat32))
                         .to(::tyloi::kFloat16);
                auto gate_dtype = hidden.scalar_type() == w_.index_gate.scalar_type()
                                      ? hidden.scalar_type()
                                      : ::tyloi::kFloat32;
                auto gates = ::tyloi::cuda::matmul(hidden.to(gate_dtype),
                    w_.index_gate.dense().to(gate_dtype).transpose(-1, -2));
#else
                ik = (centered * ::tyloi::cuda::rsqrt((centered * centered).mean(-1, true) + 1e-6) *
                          w_.index_norm.to(::tyloi::kFloat32) +
                      w_.index_bias.to(::tyloi::kFloat32))
                         .to(::tyloi::kFloat16);
                auto gate_dtype = hidden.scalar_type() == w_.index_gate.scalar_type()
                                      ? hidden.scalar_type()
                                      : ::tyloi::kFloat32;
                auto gates = ::tyloi::cuda::matmul(hidden.to(gate_dtype),
                                        w_.index_gate.to(gate_dtype).transpose(-1, -2));
#endif
                auto packed = ::tyloi::cuda::cat({ik.to(gates.scalar_type()), gates}, -1);
                return Projection{qr, query, latent, packed};
            },
            [&](Projection &p) {
                p.latent = state.latent.append(p.latent);
                p.index = state.index.append(p.index);
            },
            [&](const Projection &p) {
                TYLOINFER_RUNTIME_CHECK(p.query.dim()==4 && p.query.size(2)==c_.heads,"GLM head-wise projection shape mismatch");
                const auto batch=p.query.size(0), tokens=p.query.size(1), rows=batch*tokens*c_.heads;
                auto ids=::tyloi::cuda::arange(rows,p.query.options().dtype(::tyloi::kInt32)).remainder(c_.heads).reshape({rows,1});
                return ::tyloi::ops::project(p.query.reshape({rows,p.query.size(-1)}),w_.embed_query,ids).dense()
                    .reshape({batch,tokens,c_.heads,c_.latent}).permute({0,2,1,3}).to(::tyloi::kFloat32);
            },
            [&](const DenseTensor &query, const Projection &p) {
                return mfq_glm5_next::dense_mla_attention(query, p.latent, offset, scale);
            },
            [&](const Projection &p) {
                return mfq_glm5_next::kpool_states(
                    p.index.narrow(-1, 0, c_.index_width),
                    p.index.narrow(-1, c_.index_width, c_.index_width),
#ifdef TYLOINFER_NATIVE_CUDA
                    w_.index_position.dense(), c_.pool);
#else
                    w_.index_position, c_.pool);
#endif
            },
            [&](const Projection &p) {
                return ::tyloi::ops::project(p.rank, w_.index_query).dense()
                    .reshape({b, t, c_.index_heads, c_.index_width});
            },
            [&](DenseTensor query, DenseTensor pooled) {
                return mfq_glm5_next::kpool_scores(query, pooled,
                                                   ::tyloi::ops::project(hidden, w_.index_score).dense());
            },
            [&](DenseTensor scores) {
                return select_pooled_blocks(scores, offset, offset + t, c_.pool, c_.budget,
                                            c_.tail);
            },
            [&](const DenseTensor &query, const Projection &p, DenseTensor selected) {
                return mfq_glm5_next::sparse_mla_attention(query, p.latent, selected, scale);
            },
            [&](DenseTensor attended) {
                auto value=attended.to(::tyloi::kFloat16);
                TYLOINFER_RUNTIME_CHECK(value.dim()==4 && value.size(2)==c_.heads,"GLM head-wise projection shape mismatch");
                const auto batch=value.size(0), tokens=value.size(1), rows=batch*tokens*c_.heads;
                auto ids=::tyloi::cuda::arange(rows,value.options().dtype(::tyloi::kInt32)).remainder(c_.heads).reshape({rows,1});
                return ::tyloi::ops::project(value.reshape({rows,value.size(-1)}),w_.unembed_output,ids).dense()
                    .reshape({batch,tokens,c_.heads,c_.value_width});
            },
            [&](DenseTensor value) {
                return ::tyloi::ops::project(value.reshape({b, t, c_.heads * c_.value_width}), w_.output).dense();
            },
            [&] {
                state.latent.truncate(offset);
                state.index.truncate(offset);
            });
    }

  private:
    MlaWeights w_;
    MlaConfig c_;
};

struct Mhc {
    Parameter function, base, scale;
    std::vector<DenseTensor> pre(const DenseTensor& x, const tyloi::models::glm5_next::Config& c) const {
#ifdef TYLOINFER_NATIVE_CUDA
        return mfq_glm5_next::mhc_pre(x, function.dense(),
            base.dense(), scale.dense(),
            c.sinkhorn, c.hc_eps, c.eps);
#else
        return mfq_glm5_next::mhc_pre(x, function, base, scale, c.sinkhorn, c.hc_eps, c.eps);
#endif
    }
};

struct FeedForward {
    Linear gate, up, down;
    std::vector<Linear> routed_gate_up;
    Linear routed_down, router;
    Parameter router_bias;
    double limit = 0;
    int64_t hidden = 0, topk = 0, intermediate = 0;
    tyloi::ops::MoeRouting routing;

    DenseTensor dense(const DenseTensor& input) const;
    DenseTensor forward(const DenseTensor& input) const;
};

struct Layer {
    tyloi::models::glm5_next::Config config;
    Mhc attention_hc, ffn_hc;
    Parameter attention_norm, ffn_norm;
    FeedForward ffn;
    std::unique_ptr<Kda> kda;
    std::unique_ptr<SparseMla> mla;
    int device = 0;
};

struct Model {
    std::shared_ptr<const tyloi::ModelSource> source;
    tyloi::ModelGraph graph;
    CudaModelPlan plan;
    tyloi::models::glm5_next::Config config;
    tyloi::models::CausalLmMetadata metadata;
    ::tyloi::Tensor embed, lm_head;
    Parameter output_norm;
    std::vector<Layer> layers;
};
} // namespace tyloi::cuda::glm5_next
