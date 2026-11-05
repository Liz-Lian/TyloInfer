#include "projection.h"
#include "engine/weights/moe/experts.h"
#include "engine/backends/cuda/executor/tq_moe.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/backends/cuda/native/collectives.h"

#if TYLOINFER_ENABLE_MFQ_COMPAT
#include "engine/backends/cuda/ops/ffn/mfe.h"
#include "engine/weights/moe/cached_source_internal.h"
#endif

#include <stdexcept>
#include <utility>

namespace tyloi::cuda {


#if TYLOINFER_ENABLE_MFQ_COMPAT
namespace {

Mfe nint(const Experts& weight) {
    return {weight.n_experts, weight.out_per_expert, weight.neuron_len,
            weight.partial_experts, weight.pools};
}

std::shared_ptr<const void> mixed_source(const MfeTensor& weight) {
    if (weight.nint_dispatch) return weight.nint_dispatch;
    if (weight.nvq_dispatch) return weight.nvq_dispatch;
    if (!weight.pools.empty() && weight.pools.front().expert_local.storage())
        return weight.pools.front().expert_local.storage();
    throw std::runtime_error("mixed MFE has no backing identity");
}

template <typename Forward>
DenseTensor sharded_experts(Collectives& collectives, int primary, const Projector::Options& options,
                       const Experts& weight, DenseTensor input, const MoeRoutePlan& route,
                       Forward&& forward) {
    const auto count = weight.expert_parallel_shards.size();
    std::vector<DenseTensor> outputs(count);
    for (size_t launch = 0; launch < count; ++launch) {
        const auto index = parallel_index(options.peer_first, launch, count);
        const auto& shard = weight.expert_parallel_shards[index];
        if (!shard.weight)
            throw std::runtime_error("expert-parallel MoE shard is missing");
        MfqCudaGuard guard(shard.device);
        auto local = input.defined() ? collectives.copy(input, shard.device) : input;
        const auto& local_route = moe_route_to_device(collectives, route, shard.device);
        outputs[index] = forward(*shard.weight, std::move(local), local_route);
    }
    return collectives.sum(std::move(outputs), primary,
                           options.reduce_to_primary, options.fp16_reduce);
}

} // namespace
#endif

DenseTensor Projector::run(const Experts& weight, DenseTensor input,
                      const MoeRoutePlan& route, bool prequantized, int epilogue) const {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (weight.expert_parallel()) {
        if (prequantized)
            throw std::runtime_error("prequantized MoE activation reuse is unavailable with expert parallelism");
        return sharded_experts(collectives_, primary_, options_, weight, input, route,
            [&](const Experts& shard, DenseTensor local, const MoeRoutePlan& local_route) {
                return run(shard, local, local_route, false, epilogue);
            });
    }
    if (epilogue < 0 || epilogue > 2 || (epilogue != 0 && weight.out_per_expert % 2 != 0))
        throw std::runtime_error("invalid MFE NINT projection epilogue");
    if (weight.cached_source || weight.resident) {
        if (epilogue != 0)
            throw std::runtime_error(prequantized
                ? "mixed prequantized MFE GLU epilogue is unavailable"
                : "mixed MFE GLU epilogue must use its native dispatch");
        const auto options = runtime_options::moe(options_.moe);
        if (weight.cached_source) {
            auto scratch = moe_scratch_.get(weight.cached_source, input);
            if (prequantized)
                return weight.cached_source->forward_prequantized(
                    *scratch, options, mmq_, options.heterogeneous, input, route);
            return weight.cached_source->forward(
                *scratch, options, mmq_, options.heterogeneous, input, route);
        }
        return tyloi::cuda::mixed_project(*weight.resident, *moe_scratch_.get(weight.resident, input), options, mmq_, input, route, prequantized);
    }
    return nint(weight).forward(mfe_scratch_, input, route, prequantized, epilogue);
#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}

DenseTensor Projector::glu(const Experts& weight, DenseTensor input,
                      const MoeRoutePlan& route, bool gelu) const {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (weight.expert_parallel())
        return sharded_experts(collectives_, primary_, options_, weight, input, route,
            [&](const Experts& shard, DenseTensor local, const MoeRoutePlan& local_route) {
                return glu(shard, local, local_route, gelu);
            });
    if (weight.cached_source || weight.resident) {
        const auto options = runtime_options::moe(options_.moe);
        if (weight.cached_source)
            return weight.cached_source->forward_glu_output(
                *moe_scratch_.get(weight.cached_source, input), options, mmq_, options.heterogeneous, input, route, gelu);
        return tyloi::cuda::mixed_glu(*weight.resident, *moe_scratch_.get(weight.resident, input), options, mmq_, input, route, gelu);
    }
    return run(weight, input, route, false, gelu ? 2 : 1);
#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}

DenseTensor Projector::gated(const Experts& weight, DenseTensor input,
                        const MoeRoutePlan& route, bool gelu) const {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (weight.expert_parallel())
        return sharded_experts(collectives_, primary_, options_, weight, input, route,
            [&](const Experts& shard, DenseTensor local, const MoeRoutePlan& local_route) {
                return gated(shard, local, local_route, gelu);
            });
    if (weight.cached_source) {
        const auto options = runtime_options::moe(options_.moe);
        return weight.cached_source->forward_glu(
            *moe_scratch_.get(weight.cached_source, input), options, mmq_, options.heterogeneous, input, route, gelu);
    }
    if (!input.is_cuda() || !input.is_contiguous() ||
        input.scalar_type() != ::tyloi::kFloat16 ||
        input.dim() != 3 || input.size(2) != 2 * weight.neuron_len)
        throw std::runtime_error("fused MFE GLU input has an unsupported layout");
    auto hidden = gelu ? moe_geglu_split_cuda(input) : moe_swiglu_split_cuda(input);
    return run(weight, hidden, route);
#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}

DenseTensor Projector::clamped(const Experts& weight, DenseTensor input,
                          const MoeRoutePlan& route, double limit) const {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (weight.expert_parallel())
        return sharded_experts(collectives_, primary_, options_, weight, input, route,
            [&](const Experts& shard, DenseTensor local, const MoeRoutePlan& local_route) {
                return clamped(shard, local, local_route, limit);
            });
    if (weight.cached_source && weight.cached_source->supports_clamped_swiglu()) {
        const auto options = runtime_options::moe(options_.moe);
        return weight.cached_source->forward_clamped_swiglu(
            *moe_scratch_.get(weight.cached_source, input), options, mmq_, options.heterogeneous, input, route, limit);
    }
    if (weight.resident && tyloi::cuda::supports_clamp(*weight.resident)) {
        const auto options = runtime_options::moe(options_.moe);
        return tyloi::cuda::mixed_clamp(*weight.resident, *moe_scratch_.get(weight.resident, input), options, mmq_, input, route, limit);
    }
    return nint(weight).clamped(mfe_scratch_, input, route, limit);
#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}

DenseTensor Projector::run(const MfeTensor& weight, DenseTensor input,
                           const MoeRoutePlan& route, int epilogue) const {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (!weight.shards.empty()) {
        std::vector<DenseTensor> outputs(weight.shards.size());
        for (size_t launch = 0; launch < weight.shards.size(); ++launch) {
            const auto index = parallel_index(options_.peer_first, launch, weight.shards.size());
            const auto& shard = weight.shards[index];
            if (!shard.weight) throw std::runtime_error("expert-parallel MoE shard is missing");
            MfqCudaGuard guard(shard.device);
            auto local = input.defined() ? collectives_.copy(input, shard.device) : input;
            const auto& local_route = moe_route_to_device(collectives_, route, shard.device);
            outputs[index] = run(*shard.weight, std::move(local), local_route, epilogue);
        }
        return collectives_.sum(std::move(outputs), primary_, options_.reduce_to_primary, options_.fp16_reduce);
    }
    if (epilogue < 0 || epilogue > 2 || (epilogue && weight.out_per_expert % 2))
        throw std::runtime_error("invalid MFE NINT projection epilogue");
    if (weight.compact)
        return Mfe{weight.n_experts, weight.out_per_expert, weight.neuron_len,
            weight.partial_experts, weight.compact}.forward(mfe_scratch_, input, route, false, epilogue);
    if (epilogue) throw std::runtime_error("mixed MFE GLU epilogue must use its native dispatch");
    const auto options = runtime_options::moe(options_.moe);
    if (weight.cached_source)
        return weight.cached_source->forward(*moe_scratch_.get(weight.cached_source, input),
            options, mmq_, options.heterogeneous, input, route);
    return mixed_project(weight, *moe_scratch_.get(mixed_source(weight), input), options, mmq_, input, route);
#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}

DenseTensor Projector::glu(const MfeTensor& weight, DenseTensor input,
                           const MoeRoutePlan& route, bool gelu) const {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (!weight.shards.empty()) {
        std::vector<DenseTensor> outputs(weight.shards.size());
        for (size_t launch = 0; launch < weight.shards.size(); ++launch) {
            const auto index = parallel_index(options_.peer_first, launch, weight.shards.size());
            const auto& shard = weight.shards[index];
            if (!shard.weight) throw std::runtime_error("expert-parallel MoE shard is missing");
            MfqCudaGuard guard(shard.device);
            auto local = input.defined() ? collectives_.copy(input, shard.device) : input;
            const auto& local_route = moe_route_to_device(collectives_, route, shard.device);
            outputs[index] = glu(*shard.weight, std::move(local), local_route, gelu);
        }
        return collectives_.sum(std::move(outputs), primary_, options_.reduce_to_primary, options_.fp16_reduce);
    }
    if (weight.compact) return run(weight, input, route, gelu ? 2 : 1);
    const auto options = runtime_options::moe(options_.moe);
    if (weight.cached_source)
        return weight.cached_source->forward_glu_output(*moe_scratch_.get(weight.cached_source, input),
            options, mmq_, options.heterogeneous, input, route, gelu);
    return mixed_glu(weight, *moe_scratch_.get(mixed_source(weight), input), options, mmq_, input, route, gelu);
#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}

DenseTensor Projector::gated(const MfeTensor& weight, DenseTensor input,
                             const MoeRoutePlan& route, bool gelu) const {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (!weight.shards.empty()) {
        std::vector<DenseTensor> outputs(weight.shards.size());
        for (size_t launch = 0; launch < weight.shards.size(); ++launch) {
            const auto index = parallel_index(options_.peer_first, launch, weight.shards.size());
            const auto& shard = weight.shards[index];
            if (!shard.weight) throw std::runtime_error("expert-parallel MoE shard is missing");
            MfqCudaGuard guard(shard.device);
            auto local = input.defined() ? collectives_.copy(input, shard.device) : input;
            const auto& local_route = moe_route_to_device(collectives_, route, shard.device);
            outputs[index] = gated(*shard.weight, std::move(local), local_route, gelu);
        }
        return collectives_.sum(std::move(outputs), primary_, options_.reduce_to_primary, options_.fp16_reduce);
    }
    if (weight.cached_source) {
        const auto options = runtime_options::moe(options_.moe);
        return weight.cached_source->forward_glu(*moe_scratch_.get(weight.cached_source, input),
            options, mmq_, options.heterogeneous, input, route, gelu);
    }
    if (!input.is_cuda() || !input.is_contiguous() || input.scalar_type() != ::tyloi::kFloat16 ||
        input.dim() != 3 || input.size(2) != 2 * weight.neuron_len)
        throw std::runtime_error("fused MFE GLU input has an unsupported layout");
    auto hidden = gelu ? moe_geglu_split_cuda(input) : moe_swiglu_split_cuda(input);
    return run(weight, hidden, route);
#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}

DenseTensor Projector::run(const TqMoeBanks& banks, DenseTensor input, DenseTensor ids,
                      DenseTensor weights, float limit) const {
    if (banks.slices.empty())
        return banks.forward(tq_, std::move(input), std::move(ids), std::move(weights),
                             limit, &profiler_);
    if (!banks.gate)
        throw std::invalid_argument("TQ expert parallelism requires a Gate bank");
    std::vector<DenseTensor> partials(banks.slices.size());
    for (size_t launch = 0; launch < banks.slices.size(); ++launch) {
        const auto index = parallel_index(options_.peer_first, launch, banks.slices.size());
        const auto& slice = banks.slices[index];
        const MfqCudaGuard guard(slice.device);
        auto local_input = collectives_.copy(input, slice.device);
        auto local_ids = collectives_.copy(ids, slice.device);
        auto local_weights = collectives_.copy(weights, slice.device);
        partials[index] = banks.forward(tq_, std::move(local_input), std::move(local_ids),
            std::move(local_weights), limit, &profiler_, slice.begin, slice.end);
    }
    return collectives_.sum(std::move(partials), primary_,
                            options_.reduce_to_primary, options_.fp16_reduce);
}

} // namespace tyloi::cuda
