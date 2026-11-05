#include "experts.h"
#include "engine/weights/encoding/mfe.h"
#include "engine/weights/moe/prepare.h"
#include "engine/weights/encoding/format_features.h"

#if TYLOINFER_ENABLE_MFQ_COMPAT
#include "cached_source_internal.h"
#include "engine/backends/cuda/ops/ffn/moe.h"
#endif

#include <algorithm>

namespace tyloi::cuda {

Experts expert_runtime(const std::shared_ptr<const ::tyloi::Tensor>& value) {
    if (!value) return {};
    auto weight = std::dynamic_pointer_cast<const ::tyloi::MfeTensor>(value);
    if (!weight) throw std::invalid_argument("expected MFE expert tensor");
    Experts result;
    result.n_experts = weight->n_experts;
    result.out_per_expert = weight->out_per_expert;
    result.neuron_len = weight->neuron_len;
    result.partial_experts = weight->partial_experts;
    result.pools = weight->compact;
    result.cached_source = weight->cached_source;
    for (const auto& shard : weight->shards) {
        if (!shard.weight) throw std::runtime_error("expert-parallel MoE shard is missing");
        result.expert_parallel_shards.push_back({shard.device, shard.expert_begin, shard.expert_end,
            std::make_shared<Experts>(expert_runtime(shard.weight))});
    }
    if (result.pools) {
        result.unified_nint_projection = true;
        result.activation_workspace_domain = 1;
        for (const auto& pool : *result.pools) {
            MoeActivationGeometry geometry{static_cast<int>(pool.weight.ng), static_cast<int>(pool.weight.gs), 0, 0};
            if (std::find(result.activation_geometries.begin(), result.activation_geometries.end(), geometry) == result.activation_geometries.end())
                result.activation_geometries.push_back(geometry);
        }
    } else if (result.cached_source || !weight->pools.empty()) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
        const auto& metadata = result.cached_source ? result.cached_source->source_tensor() : *weight;
        result.unified_nint_projection = metadata.nint_only();
        result.activation_geometries = moe_activation_geometry(metadata.pools);
        result.activation_workspace_domain = 2;
        if (!result.cached_source) result.resident = weight;
#endif
    }
    return result;
}

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
    return (pools && !pools->empty()) || (resident && supports_clamp(*resident)) ||
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
