#include "routed.h"
#include "engine/backends/cuda/ops/ffn/moe.h"
#include "engine/backends/cuda/ops/ffn/mlp.h"
#include "engine/backends/cuda/ops/ffn/tq.h"
#include "engine/backends/cuda/ops/projection/quant_linear.h"
#include "engine/backends/cuda/ops/projection/projection.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/kernels/mfq_cuda_activation_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_moe_ops.h"
#include "engine/weights/encoding/mfe.h"
#include "engine/weights/moe/experts.h"
#include "engine/weights/tq/routed.h"

#include <climits>
#include <optional>
#include <stdexcept>
#include <utility>

#ifdef TYLOINFER_NATIVE_CUDA
namespace tyloi::cuda::ops {
tyloi::ops::SelectedExperts topk(tyloi::DenseTensor logits, std::int64_t count,
                               const tyloi::ops::MoeRouting& routing, const tyloi::DenseTensor* bias) {
    MfqCudaGuard guard(logits.device().index);
    MfqOptional<DenseTensor> selection_bias = mfq_nullopt;
    if (bias) selection_bias = *bias;
    auto result = moe_topk_cuda(logits.dense().contiguous(), count,
        routing.activation == tyloi::ops::RouterActivation::sigmoid,
        routing.activation == tyloi::ops::RouterActivation::sqrt_softplus,
        routing.normalize, routing.delayed_softmax, selection_bias, 1e-20, routing.scale);
    return {std::make_shared<DenseTensor>(result.at(0)), std::make_shared<DenseTensor>(result.at(1))};
}
tyloi::DenseTensor expert_scale(tyloi::DenseTensor weights, tyloi::DenseTensor ids, const tyloi::DenseTensor& scales) {
    MfqCudaGuard guard(weights.device().index);
    return moe_apply_expert_scale_cuda(DenseTensor(weights), DenseTensor(ids), DenseTensor(scales));
}
tyloi::DenseTensor weighted_reduce(tyloi::DenseTensor pairs, tyloi::DenseTensor weights) {
    MfqCudaGuard guard(pairs.device().index);
    return moe_weighted_reduce_cuda(DenseTensor(pairs), DenseTensor(weights));
}
tyloi::ops::RoutedGeglu routed_geglu(tyloi::DenseTensor input, tyloi::DenseTensor ids,
    const ::tyloi::Tensor& gate_bank, const ::tyloi::Tensor& down_bank, bool trace) {
    const auto* dense_gate = dynamic_cast<const DenseTensor*>(&gate_bank);
    const auto* dense_down = dynamic_cast<const DenseTensor*>(&down_bank);
    const auto* tq_gate = dynamic_cast<const TqBank*>(&gate_bank);
    const auto* tq_down = dynamic_cast<const TqBank*>(&down_bank);
#if TYLOINFER_ENABLE_MFQ_COMPAT
    const auto* gate = dynamic_cast<const MfeTensor*>(&gate_bank);
    const auto* down = dynamic_cast<const MfeTensor*>(&down_bank);
    if (gate) ::tyloi::weights::check_placement(
        ::tyloi::weights::expert_placement(*gate), input.device());
    if (down) ::tyloi::weights::check_placement(
        ::tyloi::weights::expert_placement(*down), input.device());
#else
    constexpr const MfeTensor* gate = nullptr;
    constexpr const MfeTensor* down = nullptr;
#endif
    if ((!dense_gate && !tq_gate && !gate) || (!dense_down && !tq_down && !down))
        throw std::invalid_argument("unsupported routed GEGLU expert tensor layout");
    for (const auto* bank : {dense_gate, dense_down}) {
        if (!bank) continue;
        if (bank->is_encoded() || !bank->is_contiguous() || bank->device() != input.device() ||
            (bank->scalar_type() != kFloat16 && bank->scalar_type() != kBFloat16 &&
             bank->scalar_type() != kFloat32))
            throw std::invalid_argument("dense expert bank must be contiguous CUDA F16/BF16/F32 on the input device");
    }
    for (const auto* bank : {tq_gate, tq_down}) {
        if (bank && (!bank->defined() || !bank->slices.empty()))
            throw std::invalid_argument("routed GEGLU does not support TQ expert-parallel slices");
    }
    if (gate_bank.size(0) > INT_MAX || gate_bank.size(1) > INT_MAX ||
        gate_bank.size(2) > INT_MAX || down_bank.size(1) > INT_MAX ||
        ids.size(0) > INT_MAX / ids.size(1) ||
        ((tq_gate || tq_down) && ids.size(1) > 65535))
        throw std::invalid_argument("routed GEGLU geometry exceeds CUDA limits");
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if ((gate && gate->shards.empty() && gate->partial_experts) ||
        (down && down->shards.empty() && down->partial_experts))
        throw std::invalid_argument("routed GEGLU requires complete expert banks");
#endif
    MfqCudaGuard guard(input.device().index);
    auto& executor = Executor::current();
    const auto project = executor.projector(Executor::current_ffn());
    auto& profiler = executor.profiler();
    input = input.contiguous();
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if ((gate && !gate->shards.empty()) || (down && !down->shards.empty())) {
        if (project.primary_device() != input.device().index)
            throw std::invalid_argument("expert-parallel projection must reduce to the input device");
    }
    if (gate && down) {
        auto route = profiler.measure("gemma.route_map", [&] {
            return build_moe_route_plan(DenseTensor(ids), gate->n_experts);
        });
        const bool prefetched =
            ::tyloi::cuda::prefetch_cached_moe_projection_bundle(*gate, *down, route);
        auto prefetch_down = [&] {
            if (!prefetched) ::tyloi::cuda::prefetch_experts(*down, route);
        };
        tyloi::ops::RoutedGeglu result;
        if (!trace && ::tyloi::weights::supports_geglu(*gate)) {
            auto hidden = profiler.measure("gemma.moe_gate_up_geglu", [&] {
                return project.glu(*gate, DenseTensor(input), route, true);
            });
            prefetch_down();
            result.pairs = std::make_shared<DenseTensor>(profiler.measure("gemma.moe_down", [&] {
                return project.run(*down, hidden, route);
            }));
            return result;
        }
        auto packed = profiler.measure("gemma.moe_gate_up", [&] {
            return project.run(*gate, DenseTensor(input), route);
        });
        prefetch_down();
        if (trace) result.gate_up = std::make_shared<DenseTensor>(packed.to(::tyloi::kFloat32).contiguous().clone());
        if (!trace && packed.size(0) <= 4) {
            result.pairs = std::make_shared<DenseTensor>(profiler.measure("gemma.moe_geglu_down", [&] {
                return project.gated(*down, packed, route, true);
            }));
            return result;
        }
        auto hidden = profiler.measure("gemma.moe_geglu", [&] { return moe_geglu_split_cuda(packed); });
        if (trace) result.hidden = std::make_shared<DenseTensor>(hidden.to(::tyloi::kFloat32).contiguous().clone());
        result.pairs = std::make_shared<DenseTensor>(profiler.measure("gemma.moe_down", [&] { return project.run(*down, hidden, route); }));
        return result;
    }
#endif
    tyloi::ops::RoutedGeglu result;
    if (tq_gate || tq_down) {
        const auto host_ids = ids.contiguous().cpu().to(kInt64);
        const auto* selected = host_ids.data_ptr<std::int64_t>();
        for (std::int64_t i = 0; i < host_ids.numel(); ++i)
            if (selected[i] < 0 || selected[i] >= gate_bank.size(0))
                throw std::out_of_range("routed GEGLU expert ID is out of bounds");
    }
    DenseTensor packed;
#if TYLOINFER_ENABLE_MFQ_COMPAT
    std::optional<MoeRoutePlan> route;
    if (gate || down) route.emplace(profiler.measure("gemma.route_map", [&] {
        return build_moe_route_plan(DenseTensor(ids), static_cast<int>(gate_bank.size(0)));
    }));
    if (gate) packed = project.run(*gate, input, *route);
    else
#endif
    if (dense_gate) packed = dense_routed_projection(*dense_gate, input, ids);
    else packed = tq_ops::routed_projection(*tq_gate, input, ids);
    if (trace) result.gate_up = std::make_shared<DenseTensor>(packed.to(kFloat32).contiguous().clone());
    auto hidden = moe_geglu_split_cuda(packed.to(kFloat16).contiguous());
    if (trace) result.hidden = std::make_shared<DenseTensor>(hidden.to(kFloat32).contiguous().clone());
    DenseTensor pairs;
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (down) pairs = project.run(*down, hidden, *route);
    else
#endif
    if (dense_down) pairs = dense_routed_projection(*dense_down, hidden, ids);
    else pairs = tq_ops::routed_projection(*tq_down, hidden, ids);
    result.pairs = std::make_shared<DenseTensor>(std::move(pairs));
    return result;
}

} // namespace tyloi::cuda::ops

#endif
