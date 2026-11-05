#pragma once

#include "engine/tensor.h"

#include <cstdint>
#include <memory>

namespace tyloi::weights {

struct BiasWeight {
    std::shared_ptr<const ::tyloi::Tensor> weight;
    std::shared_ptr<const ::tyloi::Tensor> bias;
};

BiasWeight bias_rows(const BiasWeight& value, int64_t begin, int64_t count);

} // namespace tyloi::weights
