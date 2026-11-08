#pragma once

#include "engine/tensor.h"
#include "engine/requests/result.h"

#include <cstdint>

namespace tyloi::models {

struct BatchSample {
    ::tyloi::Tensor next;
    int64_t token = 0;
    MfqPrefillTiming timing;
};

} // namespace tyloi::models
