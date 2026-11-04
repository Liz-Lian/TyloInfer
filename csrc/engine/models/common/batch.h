#pragma once

#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/requests/result.h"

#include <cstdint>

namespace tyloi::cuda {

struct BatchSample {
    ::tyloi::Tensor next;
    int64_t token = 0;
    MfqPrefillTiming timing;
};

} // namespace tyloi::cuda
