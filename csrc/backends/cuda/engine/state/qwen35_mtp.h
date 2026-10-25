#pragma once

#include "engine/state/kv_cache.h"

#include <vector>

namespace mfq::cuda::qwen35 {

struct MtpState {
    int64_t position = 0;
    std::vector<KVCache> kv;
};

} // namespace mfq::cuda::qwen35
