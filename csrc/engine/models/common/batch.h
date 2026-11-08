#pragma once

#include "engine/tensor.h"
#ifndef TYLOINFER_NATIVE_CUDA
#include "engine/backends/cuda/native/tensor/backend.h"
#endif
#include "engine/requests/result.h"

#include <cstdint>

namespace tyloi::cuda {

struct BatchSample {
#ifdef TYLOINFER_NATIVE_CUDA
    ::tyloi::Tensor next;
#else
    ::tyloi::DenseTensor next;
#endif
    int64_t token = 0;
    MfqPrefillTiming timing;
};

} // namespace tyloi::cuda
