#pragma once

#include "engine/backends/cuda/ops/projection/quant_linear.h"
#include "engine/weights/moe/experts.h"
#include "engine/ops.h"

#include <memory>

class MoeFfnPipeline;
struct TqMoeBanks;
struct MoeRoutePlan;

namespace mfq::cuda {

class Executor;
struct Mlp;
class Projector;

class Ffn {
public:
    struct Options {
        bool small_heterogeneous = true;
        bool delayed_readback = true;
        bool bundle_prefetch = true;
        bool swiglu_fusion = true;
        bool reduce_fusion = true;
    };

    struct Mode {
        bool materialized = false;
        bool unfused = false;
        bool serial = false;
        bool batch_serial = false;
        int tile = 0;
        int tiles = 1;
        bool reuse_input = true;
        bool projection_major = false;
    };

    QuantLinearGroup gate_up;
    QuantLinear down;
    std::unique_ptr<Ffn> important_neurons;
    bool geglu = false;
    // Dense Ffn and shared/routed experts share the explicit activation label.
    mfq::ops::MlpActivation activation = mfq::ops::MlpActivation::silu_gated;
    bool is_moe = false;
    bool moe_split_gate_up = false;
    // Routed experts without a gate projection use the activation above.
    bool moe_ungated_experts = false;
    Experts moe_gate_up;
    Experts moe_gate;
    Experts moe_up;
    Experts moe_down;
    // Ordinary TQ row tables retain their own cache and never enter MFE pools.
    std::shared_ptr<TqMoeBanks> moe_tq;
    std::shared_ptr<MixedMoeRuntime> cpu_moe_gate_up;
    std::shared_ptr<MixedMoeRuntime> cpu_moe_gate;
    std::shared_ptr<MixedMoeRuntime> cpu_moe_up;
    std::shared_ptr<MixedMoeRuntime> cpu_moe_down;
    // Dense (F32/F16/BF16) routed expert banks: [experts,out,in].
    mfq_tensor_backend::Tensor moe_dense_gate;
    mfq_tensor_backend::Tensor moe_dense_up;
    mfq_tensor_backend::Tensor moe_dense_down;
    // Optional latent projections around the routed branch only.
    QuantLinear moe_latent_in;
    QuantLinear moe_latent_out;
    mfq_tensor_backend::Tensor moe_router;
    std::shared_ptr<const QuantLinear> moe_router_projection;
    mfq_tensor_backend::Tensor moe_shared_gate;
    mfq_tensor_backend::Tensor moe_router_bias;
    mfq_tensor_backend::Tensor moe_hash_ids;
    std::unique_ptr<Ffn> shared;
    int moe_top_k = 0;
    bool moe_use_sigmoid = false;
    bool moe_use_sqrt_softplus = false;
    bool moe_normalize = false;
    bool moe_delayed_softmax = true;
    bool moe_shared_ungated = false;
    double moe_router_scale = 1.0;
    double swiglu_limit = 0.0;
    int moe_layer = -1;
    std::weak_ptr<MoeFfnPipeline> moe_pipeline;

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    mfq_tensor_backend::Tensor forward(
        mfq_tensor_backend::Tensor input, const Mode& mode,
        MfqOptional<mfq_tensor_backend::Tensor> input_ids = mfq_nullopt,
        MfqOptional<mfq_tensor_backend::Tensor> router_bias = mfq_nullopt) const;
    void prefetch(
        const mfq_tensor_backend::Tensor& input, const Mode& mode,
        bool enabled, MfqOptional<mfq_tensor_backend::Tensor> input_ids = mfq_nullopt) const;
#else
    mfq_tensor_backend::Tensor forward(
        mfq_tensor_backend::Tensor input, Executor& executor, const Mode& mode,
        MfqOptional<mfq_tensor_backend::Tensor> input_ids = mfq_nullopt,
        MfqOptional<mfq_tensor_backend::Tensor> router_bias = mfq_nullopt) const;
    void prefetch(
        const mfq_tensor_backend::Tensor& input, Executor& executor, const Mode& mode,
        bool enabled, MfqOptional<mfq_tensor_backend::Tensor> input_ids = mfq_nullopt) const;
#endif

private:
    mfq_tensor_backend::Tensor forward_impl(
        mfq_tensor_backend::Tensor input, Executor& executor, const Mode& mode,
        const Projector& project, MfqOptional<mfq_tensor_backend::Tensor> input_ids,
        bool allow_important_neurons,
        MfqOptional<mfq_tensor_backend::Tensor> router_bias) const;
    mfq_tensor_backend::Tensor forward_experts(
        mfq_tensor_backend::Tensor input, Executor& executor, const Mode& mode,
        const Projector& project, const MoeRoutePlan& route,
        mfq_tensor_backend::Tensor route_weights) const;
};

Mlp dense(const Ffn& ffn);
bool uses_expert_cache(const Ffn& ffn);
bool expert_parallel(const Ffn& ffn);

} // namespace mfq::cuda

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
namespace mfq::cuda::ops {
mfq::Tensor mlp(mfq::Tensor input, const mfq::weights::Weights& gate_up,
                const mfq::weights::Weights& down, mfq::ops::MlpActivation activation,
                double swiglu_limit);
mfq::ops::SelectedExperts topk(mfq::Tensor logits, std::int64_t count, const mfq::ops::MoeRouting& routing);
mfq::Tensor expert_scale(mfq::Tensor weights, mfq::Tensor ids, const mfq::Tensor& scales);
mfq::Tensor weighted_reduce(mfq::Tensor pairs, mfq::Tensor weights);
mfq::ops::RoutedGeglu routed_geglu(mfq::Tensor input, mfq::Tensor ids,
    const mfq::weights::Weights& gate_up, const mfq::weights::Weights& down, bool trace);
}
#endif
