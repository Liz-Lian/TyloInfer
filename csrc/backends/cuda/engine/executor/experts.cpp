#include "experts.h"
#include "execution.h"
#include "options.h"

#if TYLOINFER_ENABLE_MFQ_COMPAT
#include "engine/weights/moe/cached_source_internal.h"
#endif

#include <stdexcept>
#include <utility>

namespace mfq::cuda {

using Tensor = mfq_tensor_backend::Tensor;

MoeOptions moe_options(const CudaExecutionConfig& config, bool prefill_off, bool pool_only) {
    return {config.moe_prefill_mma && !prefill_off, config.moe_prefill_mma_min_tokens,
            config.moe_nvq_heterogeneous_decode && !pool_only, config.moe_nvq_active_decode,
            config.moe_nint_heterogeneous_decode && !pool_only,
            runtime_options::nint_route_row_workspace(), runtime_options::nint_direct_pool(),
            runtime_options::nint_group_dot(), runtime_options::nint_route_hint(),
            runtime_options::nint_route_warps()};
}

bool Experts::supports_clamped_swiglu() const {
    if (expert_parallel()) {
        return std::all_of(expert_parallel_shards.begin(), expert_parallel_shards.end(),
                          [](const Shard& shard) {
                              return shard.weight && shard.weight->supports_clamped_swiglu();
                          });
    }
#if TYLOINFER_ENABLE_MFQ_COMPAT
    return !pools.empty() || (resident && resident->supports_clamped_swiglu()) ||
           (cached_source && cached_source->supports_clamped_swiglu());
#else
    return !pools.empty();
#endif
}

void prefetch_experts(const Experts& weight, const MoeRoutePlan& route, bool begin) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (weight.cached_source) {
        if (begin) weight.cached_source->begin_prefetch(route);
        else weight.cached_source->prefetch(route);
    }
#endif
}

#if TYLOINFER_ENABLE_MFQ_COMPAT
namespace {

Mfe nint(const Experts& weight) {
    return {weight.n_experts, weight.out_per_expert, weight.neuron_len,
            weight.partial_experts, weight.pools};
}

template <typename Forward>
Tensor sharded(CudaExecutionContext& execution, const Experts& weight, Tensor input,
               const MoeRoutePlan& route, Forward&& forward) {
    const auto count = weight.expert_parallel_shards.size();
    std::vector<Tensor> outputs(count);
    for (size_t launch = 0; launch < count; ++launch) {
        const auto index = parallel_index(execution.config.tasks.projection.peer_first, launch, count);
        const auto& shard = weight.expert_parallel_shards[index];
        if (!shard.weight)
            throw std::runtime_error("expert-parallel MoE shard is missing");
        MfqCudaGuard guard(shard.device);
        auto local = input.defined() ? execution.collectives.copy(input, shard.device) : input;
        const auto& local_route = moe_route_to_device(execution.collectives, route, shard.device);
        outputs[index] = forward(*shard.weight, std::move(local), local_route);
    }
    return execution.collectives.sum(std::move(outputs), model_parallel_primary_device(execution),
                                     execution.config.tasks.projection.reduce_to_primary,
                                     execution.config.tasks.projection.fp16_reduce);
}

} // namespace
#endif

Tensor run_experts(CudaExecutionContext& execution, const Experts& weight, Tensor input,
                   const MoeRoutePlan& route, bool prequantized, int epilogue) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (weight.expert_parallel()) {
        if (prequantized)
            throw std::runtime_error("prequantized MoE activation reuse is unavailable with expert parallelism");
        return sharded(execution, weight, input, route,
            [&](const Experts& shard, Tensor local, const MoeRoutePlan& local_route) {
                return run_experts(execution, shard, local, local_route, false, epilogue);
            });
    }
    if (epilogue < 0 || epilogue > 2 || (epilogue != 0 && weight.out_per_expert % 2 != 0))
        throw std::runtime_error("invalid MFE NINT projection epilogue");
    if (weight.cached_source || weight.resident) {
        if (epilogue != 0)
            throw std::runtime_error(prequantized
                ? "mixed prequantized MFE GLU epilogue is unavailable"
                : "mixed MFE GLU epilogue must use its native dispatch");
        const auto options = moe_options(execution.config, execution.force_moe_prefill_mma_off,
                                         execution.force_moe_pool_path);
        if (weight.cached_source) {
            if (prequantized)
                return weight.cached_source->forward_prequantized(
                    options, execution.kl_mmq, execution.config.moe_nvq_heterogeneous, input, route);
            return weight.cached_source->forward(
                options, execution.kl_mmq, execution.config.moe_nvq_heterogeneous, input, route);
        }
        return weight.resident->forward(options, execution.kl_mmq, input, route, prequantized);
    }
    return nint(weight).forward(*weight.scratch, input, route, prequantized, epilogue);
#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}

Tensor glu_experts(CudaExecutionContext& execution, const Experts& weight, Tensor input,
                   const MoeRoutePlan& route, bool gelu) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (weight.expert_parallel())
        return sharded(execution, weight, input, route,
            [&](const Experts& shard, Tensor local, const MoeRoutePlan& local_route) {
                return glu_experts(execution, shard, local, local_route, gelu);
            });
    if (weight.cached_source || weight.resident) {
        const auto options = moe_options(execution.config, execution.force_moe_prefill_mma_off,
                                         execution.force_moe_pool_path);
        if (weight.cached_source)
            return weight.cached_source->forward_glu_output(
                options, execution.kl_mmq, execution.config.moe_nvq_heterogeneous, input, route, gelu);
        return weight.resident->forward_glu_output(options, execution.kl_mmq, input, route, gelu);
    }
    return run_experts(execution, weight, input, route, false, gelu ? 2 : 1);
#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}

Tensor gated_experts(CudaExecutionContext& execution, const Experts& weight, Tensor input,
                     const MoeRoutePlan& route, bool gelu) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (weight.expert_parallel())
        return sharded(execution, weight, input, route,
            [&](const Experts& shard, Tensor local, const MoeRoutePlan& local_route) {
                return gated_experts(execution, shard, local, local_route, gelu);
            });
    if (weight.cached_source) {
        const auto options = moe_options(execution.config, execution.force_moe_prefill_mma_off,
                                         execution.force_moe_pool_path);
        return weight.cached_source->forward_glu(
            options, execution.kl_mmq, execution.config.moe_nvq_heterogeneous, input, route, gelu);
    }
    if (!input.is_cuda() || !input.is_contiguous() ||
        input.scalar_type() != mfq_tensor_backend::kFloat16 ||
        input.dim() != 3 || input.size(2) != 2 * weight.neuron_len)
        throw std::runtime_error("fused MFE GLU input has an unsupported layout");
    auto hidden = gelu ? moe_geglu_split_cuda(input) : moe_swiglu_split_cuda(input);
    return run_experts(execution, weight, hidden, route);
#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}

Tensor clamped_experts(CudaExecutionContext& execution, const Experts& weight, Tensor input,
                       const MoeRoutePlan& route, double limit) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (weight.expert_parallel())
        return sharded(execution, weight, input, route,
            [&](const Experts& shard, Tensor local, const MoeRoutePlan& local_route) {
                return clamped_experts(execution, shard, local, local_route, limit);
            });
    if (weight.cached_source && weight.cached_source->supports_clamped_swiglu()) {
        const auto options = moe_options(execution.config, execution.force_moe_prefill_mma_off,
                                         execution.force_moe_pool_path);
        return weight.cached_source->forward_clamped_swiglu(
            options, execution.kl_mmq, execution.config.moe_nvq_heterogeneous, input, route, limit);
    }
    if (weight.resident && weight.resident->supports_clamped_swiglu()) {
        const auto options = moe_options(execution.config, execution.force_moe_prefill_mma_off,
                                         execution.force_moe_pool_path);
        return weight.resident->forward_clamped_swiglu(options, execution.kl_mmq, input, route, limit);
    }
    return nint(weight).clamped(*weight.scratch, input, route, limit);
#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}

} // namespace mfq::cuda
