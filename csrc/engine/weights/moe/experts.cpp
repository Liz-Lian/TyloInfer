#include "experts.h"
#include "engine/weights/encoding/format_features.h"

#if TYLOINFER_ENABLE_MFQ_COMPAT
#include "cached_source_internal.h"
#endif

#include <algorithm>

namespace mfq::cuda {

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

} // namespace mfq::cuda
