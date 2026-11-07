#include "experts.h"
#include "engine/weights/encoding/mfe.h"
#include "engine/weights/moe/prepare.h"
#include "engine/weights/encoding/format_features.h"

#if TYLOINFER_ENABLE_MFQ_COMPAT
#include "cached_source_internal.h"
#include "engine/backends/cuda/ops/ffn/moe.h"
#endif

#include <algorithm>
#include <stdexcept>

namespace tyloi::cuda {

bool supports_projection_glu_epilogue(const MfeTensor& weight) {
    if (!weight.shards.empty()) {
        return std::all_of(weight.shards.begin(), weight.shards.end(), [](const auto& shard) {
            if (!shard.weight) throw std::runtime_error("expert-parallel MoE shard is missing");
            return supports_projection_glu_epilogue(*shard.weight);
        });
    }
    if (weight.compact) return true;
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (weight.cached_source) return weight.cached_source->source_tensor().nint_only();
    if (!weight.pools.empty()) return weight.nint_only();
#endif
    return false;
}

static bool supports_prequantized_input(const MfeTensor& weight) {
    if (!weight.shards.empty()) return false;
    if ((weight.compact && !weight.compact->empty()) || weight.cached_source) return true;
#if TYLOINFER_ENABLE_MFQ_COMPAT
    return !weight.compact && !weight.pools.empty();
#else
    return false;
#endif
}

bool can_reuse_activation_for(const MfeTensor& producer, const MfeTensor& consumer) {
    if (!supports_prequantized_input(producer) || !supports_prequantized_input(consumer) ||
        producer.neuron_len != consumer.neuron_len || bool(producer.compact) != bool(consumer.compact))
        return false;
    const bool shared = (producer.cached_source && producer.cached_source == consumer.cached_source) ||
        (!producer.cached_source && !consumer.cached_source && !producer.compact &&
         !consumer.compact && &producer == &consumer) ||
        (!producer.cached_source && !consumer.cached_source && producer.compact &&
         producer.compact == consumer.compact);
    if (!shared) return false;
    const auto geometry = [](const MfeTensor& weight) {
        std::vector<MoeActivationGeometry> result;
        if (weight.compact) {
            for (const auto& pool : *weight.compact) {
                MoeActivationGeometry value{static_cast<int>(pool.weight.ng), static_cast<int>(pool.weight.gs), 0, 0};
                if (std::find(result.begin(), result.end(), value) == result.end()) result.push_back(value);
            }
        }
#if TYLOINFER_ENABLE_MFQ_COMPAT
        else if (weight.cached_source || !weight.pools.empty()) {
            result = moe_activation_geometry(weight.cached_source ? weight.cached_source->source_tensor().pools : weight.pools);
        }
#endif
        return result;
    };
    const auto available = geometry(producer);
    const auto required = geometry(consumer);
    return !available.empty() && !required.empty() &&
        std::all_of(required.begin(), required.end(), [&](const auto& value) {
            return std::find(available.begin(), available.end(), value) != available.end();
        });
}

bool supports_clamped_swiglu(const MfeTensor& weight) {
    if (!weight.shards.empty()) {
        return std::all_of(weight.shards.begin(), weight.shards.end(), [](const auto& shard) {
            if (!shard.weight) throw std::runtime_error("expert-parallel MoE shard is missing");
            return supports_clamped_swiglu(*shard.weight);
        });
    }
#if TYLOINFER_ENABLE_MFQ_COMPAT
    return (weight.compact && !weight.compact->empty()) ||
        (!weight.compact && !weight.cached_source && !weight.pools.empty() && supports_clamp(weight)) ||
        (weight.cached_source && weight.cached_source->supports_clamped_swiglu());
#else
    return weight.compact && !weight.compact->empty();
#endif
}

void prefetch_experts(const MfeTensor& weight, const MoeRoutePlan& route, bool begin) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (weight.cached_source) {
        if (begin) weight.cached_source->begin_prefetch(route);
        else weight.cached_source->prefetch(route);
    }
#endif
}

} // namespace tyloi::cuda
