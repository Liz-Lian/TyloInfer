#pragma once

#include "engine/state/kv_cache.h"
#include "engine/state/recurrent.h"
#include "ops/attention/linear.h"

#include <cstdint>
#include <variant>

namespace mfq::cuda::qwen35 {

// Linear attention keeps the projected rows until a speculative pass has
// either been committed or replayed from its confirmed prefix.
struct LinearState {
    Recurrent mixer;
    LinearInputs projected;
    std::int64_t speculative_ffn_batches = 0;
    std::int64_t speculative_projection_batches = 0;
};

using State = std::variant<KVCache, LinearState>;

} // namespace mfq::cuda::qwen35
