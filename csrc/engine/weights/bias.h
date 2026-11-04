#pragma once

#include "engine/backends/cuda/ops/projection/quant_linear.h"

#include <cstdint>

namespace tyloi::cuda {

struct BiasWeight {
    QuantLinear weight;
    ::tyloi::Tensor bias;
};

BiasWeight bias_rows(const BiasWeight& value, int64_t begin, int64_t count);

} // namespace tyloi::cuda
