#pragma once

#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/requests/result.h"

#include <cstdint>

namespace mfq::cuda {

struct BatchSample {
    mfq_tensor_backend::Tensor next;
    int64_t token = 0;
    MfqPrefillTiming timing;
};

} // namespace mfq::cuda
