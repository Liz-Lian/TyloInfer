#pragma once

#include "engine/backends/cuda/ops/projection/quant_linear.h"
#include "engine/weights/moe/experts.h"
#include "engine/ops.h"

#include <memory>

class MoeFfnPipeline;
struct MoeRoutePlan;

namespace tyloi::cuda {

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
        bool independent = false;
    };

    std::vector<std::shared_ptr<const ::tyloi::Tensor>> gate_up;
    std::shared_ptr<const ::tyloi::Tensor> down;
    std::unique_ptr<Ffn> important_neurons;
    bool geglu = false;
    // Dense Ffn and shared/routed experts share the explicit activation label.
    tyloi::ops::MlpActivation activation = tyloi::ops::MlpActivation::silu_gated;
    bool is_moe = false;
    bool moe_split_gate_up = false;
    // Routed experts without a gate projection use the activation above.
    bool moe_ungated_experts = false;
    std::shared_ptr<const ::tyloi::Tensor> moe_gate_up;
    std::shared_ptr<const ::tyloi::Tensor> moe_gate;
    std::shared_ptr<const ::tyloi::Tensor> moe_up;
    std::shared_ptr<const ::tyloi::Tensor> moe_down;
    // Optional latent projections around the routed branch only.
    std::shared_ptr<const ::tyloi::Tensor> moe_latent_in;
    std::shared_ptr<const ::tyloi::Tensor> moe_latent_out;
    std::shared_ptr<const ::tyloi::Tensor> moe_router;
    std::shared_ptr<const ::tyloi::Tensor> moe_shared_gate;
    std::shared_ptr<const ::tyloi::Tensor> moe_router_bias;
    std::shared_ptr<const ::tyloi::Tensor> moe_hash_ids;
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

#ifdef TYLOINFER_NATIVE_CUDA
    ::tyloi::DenseTensor forward(
        ::tyloi::DenseTensor input,
        MfqOptional<::tyloi::DenseTensor> input_ids = mfq_nullopt,
        MfqOptional<::tyloi::DenseTensor> router_bias = mfq_nullopt) const;
    void prefetch(
        const ::tyloi::DenseTensor& input,
        MfqOptional<::tyloi::DenseTensor> input_ids = mfq_nullopt) const;
#else
    ::tyloi::DenseTensor forward(
        ::tyloi::DenseTensor input, Executor& executor, const Mode& mode,
        MfqOptional<::tyloi::DenseTensor> input_ids = mfq_nullopt,
        MfqOptional<::tyloi::DenseTensor> router_bias = mfq_nullopt) const;
    void prefetch(
        const ::tyloi::DenseTensor& input, Executor& executor, const Mode& mode,
        bool enabled, MfqOptional<::tyloi::DenseTensor> input_ids = mfq_nullopt) const;
#endif

private:
    ::tyloi::DenseTensor forward_impl(
        ::tyloi::DenseTensor input, Executor& executor, const Mode& mode,
        const Projector& project, MfqOptional<::tyloi::DenseTensor> input_ids,
        bool allow_important_neurons,
        MfqOptional<::tyloi::DenseTensor> router_bias) const;
    ::tyloi::DenseTensor forward_experts(
        ::tyloi::DenseTensor input, Executor& executor, const Mode& mode,
        const Projector& project, const MoeRoutePlan& route,
        ::tyloi::DenseTensor route_weights) const;
};

Mlp dense(const Ffn& ffn, const Projector& project);
std::shared_ptr<const ::tyloi::MfeTensor> cpu_expert(const std::shared_ptr<const ::tyloi::Tensor>& weight);
bool uses_expert_cache(const Ffn& ffn);
bool expert_parallel(const Ffn& ffn);

} // namespace tyloi::cuda

#ifdef TYLOINFER_NATIVE_CUDA
namespace tyloi::cuda::ops {
tyloi::DenseTensor mlp(tyloi::DenseTensor input, std::span<const std::shared_ptr<const tyloi::Tensor>> gate_up,
                const ::tyloi::Tensor& down, tyloi::ops::MlpActivation activation,
                double swiglu_limit);
tyloi::ops::SelectedExperts topk(tyloi::DenseTensor logits, std::int64_t count, const tyloi::ops::MoeRouting& routing);
tyloi::DenseTensor expert_scale(tyloi::DenseTensor weights, tyloi::DenseTensor ids, const tyloi::DenseTensor& scales);
tyloi::DenseTensor weighted_reduce(tyloi::DenseTensor pairs, tyloi::DenseTensor weights);
tyloi::ops::RoutedGeglu routed_geglu(tyloi::DenseTensor input, tyloi::DenseTensor ids,
    const ::tyloi::Tensor& gate_up, const ::tyloi::Tensor& down, bool trace);
}
#endif
