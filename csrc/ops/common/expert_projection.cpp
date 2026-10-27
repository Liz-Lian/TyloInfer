#include "ops/common/include/projection.h"
#include "engine/weights/moe/experts.h"
#include "engine/executor/tq_moe.h"
#include "engine/executor/options.h"
#include "native/cuda/collectives.h"

#if TYLOINFER_ENABLE_MFQ_COMPAT
#include "engine/weights/moe/cached_source_internal.h"
#endif

#include <stdexcept>
#include <utility>

namespace mfq::cuda {

using Tensor = mfq_tensor_backend::Tensor;

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
Tensor sharded_experts(Collectives& collectives, int primary, const Projector::Options& options,
                       const Experts& weight, Tensor input, const MoeRoutePlan& route,
                       Forward&& forward) {
    const auto count = weight.expert_parallel_shards.size();
    std::vector<Tensor> outputs(count);
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

Tensor Projector::run(const Experts& weight, Tensor input,
                      const MoeRoutePlan& route, bool prequantized, int epilogue) const {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (weight.expert_parallel()) {
        if (prequantized)
            throw std::runtime_error("prequantized MoE activation reuse is unavailable with expert parallelism");
        return sharded_experts(collectives_, primary_, options_, weight, input, route,
            [&](const Experts& shard, Tensor local, const MoeRoutePlan& local_route) {
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
            if (prequantized)
                return weight.cached_source->forward_prequantized(
                    options, mmq_, options.heterogeneous, input, route);
            return weight.cached_source->forward(
                options, mmq_, options.heterogeneous, input, route);
        }
        return weight.resident->forward(options, mmq_, input, route, prequantized);
    }
    return nint(weight).forward(*weight.scratch, input, route, prequantized, epilogue);
#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}

Tensor Projector::glu(const Experts& weight, Tensor input,
                      const MoeRoutePlan& route, bool gelu) const {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (weight.expert_parallel())
        return sharded_experts(collectives_, primary_, options_, weight, input, route,
            [&](const Experts& shard, Tensor local, const MoeRoutePlan& local_route) {
                return glu(shard, local, local_route, gelu);
            });
    if (weight.cached_source || weight.resident) {
        const auto options = runtime_options::moe(options_.moe);
        if (weight.cached_source)
            return weight.cached_source->forward_glu_output(
                options, mmq_, options.heterogeneous, input, route, gelu);
        return weight.resident->forward_glu_output(options, mmq_, input, route, gelu);
    }
    return run(weight, input, route, false, gelu ? 2 : 1);
#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}

Tensor Projector::gated(const Experts& weight, Tensor input,
                        const MoeRoutePlan& route, bool gelu) const {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (weight.expert_parallel())
        return sharded_experts(collectives_, primary_, options_, weight, input, route,
            [&](const Experts& shard, Tensor local, const MoeRoutePlan& local_route) {
                return gated(shard, local, local_route, gelu);
            });
    if (weight.cached_source) {
        const auto options = runtime_options::moe(options_.moe);
        return weight.cached_source->forward_glu(
            options, mmq_, options.heterogeneous, input, route, gelu);
    }
    if (!input.is_cuda() || !input.is_contiguous() ||
        input.scalar_type() != mfq_tensor_backend::kFloat16 ||
        input.dim() != 3 || input.size(2) != 2 * weight.neuron_len)
        throw std::runtime_error("fused MFE GLU input has an unsupported layout");
    auto hidden = gelu ? moe_geglu_split_cuda(input) : moe_swiglu_split_cuda(input);
    return run(weight, hidden, route);
#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}

Tensor Projector::clamped(const Experts& weight, Tensor input,
                          const MoeRoutePlan& route, double limit) const {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (weight.expert_parallel())
        return sharded_experts(collectives_, primary_, options_, weight, input, route,
            [&](const Experts& shard, Tensor local, const MoeRoutePlan& local_route) {
                return clamped(shard, local, local_route, limit);
            });
    if (weight.cached_source && weight.cached_source->supports_clamped_swiglu()) {
        const auto options = runtime_options::moe(options_.moe);
        return weight.cached_source->forward_clamped_swiglu(
            options, mmq_, options.heterogeneous, input, route, limit);
    }
    if (weight.resident && weight.resident->supports_clamped_swiglu()) {
        const auto options = runtime_options::moe(options_.moe);
        return weight.resident->forward_clamped_swiglu(options, mmq_, input, route, limit);
    }
    return nint(weight).clamped(*weight.scratch, input, route, limit);
#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}

Tensor Projector::run(const TqMoeBanks& banks, Tensor input, Tensor ids,
                      Tensor weights, float limit) const {
    if (banks.slices.empty())
        return banks.forward(tq_, std::move(input), std::move(ids), std::move(weights),
                             limit, &profiler_);
    if (!banks.gate)
        throw std::invalid_argument("TQ expert parallelism requires a Gate bank");
    std::vector<Tensor> partials(banks.slices.size());
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

} // namespace mfq::cuda
