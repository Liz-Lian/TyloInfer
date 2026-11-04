#include "experts.h"
#include "engine/weights/encoding/format_features.h"

#if TYLOINFER_ENABLE_MFQ_COMPAT
#include "cached_source_internal.h"
#include "engine/backends/cuda/ops/ffn/moe.h"
#endif

#include <algorithm>

namespace tyloi::cuda {

weights::Placement Experts::placement(std::size_t index) const {
    if (index >= size()) throw std::out_of_range("expert index out of range");
    weights::Placement result;
    if (expert_parallel()) {
        for (const auto& shard : expert_parallel_shards)
            result.devices.emplace_back(DeviceType::cuda, shard.device);
        return result;
    }
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (cached_source) return cached_source->placement();
#endif
    // Every buffer the pool requires, not one representative per format.
    const auto add_payload = [&](const ::tyloi::Tensor& weight) {
        for (const auto* buffer : weight.buffers()) {
            if (!buffer || !buffer->defined()) continue;
            if (!buffer->is_cuda())
                throw std::invalid_argument("expert weight is not prepared for CUDA execution");
            const Device device{DeviceType::cuda, buffer->get_device()};
            if (std::find(result.devices.begin(), result.devices.end(), device) == result.devices.end())
                result.devices.push_back(device);
        }
    };
    if (pools) {
        for (const auto& pool : *pools) add_payload(pool.weight);
    }
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (resident) {
        for (const auto& pool : resident->pools)
            if (const auto* weight = pool.payload()) add_payload(*weight);
    }
#endif
    if (result.devices.empty() && !result.caller)
        throw std::invalid_argument("expert weight is not prepared");
    return result;
}

bool Experts::supports_clamped_swiglu() const {
    if (expert_parallel()) {
        return std::all_of(expert_parallel_shards.begin(), expert_parallel_shards.end(),
                          [](const Shard& shard) {
                              return shard.weight && shard.weight->supports_clamped_swiglu();
                          });
    }
#if TYLOINFER_ENABLE_MFQ_COMPAT
    return (pools && !pools->empty()) || (resident && resident->supports_clamped_swiglu()) ||
           (cached_source && cached_source->supports_clamped_swiglu());
#else
    return pools && !pools->empty();
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

} // namespace tyloi::cuda
