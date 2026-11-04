#pragma once

#include "engine/backends/cuda/ops/projection/tq.h"

#include <cstdint>
#include <memory>

class TqCache;

// A routed projection owns only its source identity and cache owner. The
// cache owns compressed images and their final-consumer lifetime.
struct TqRoutedWeight {
    std::shared_ptr<TqCache> cache;
    std::uint64_t source_id = 0;
    std::int64_t experts = 0, out = 0, input = 0;

    ::tyloi::DenseTensor forward(
        ::tyloi::DenseTensor input_value,
        ::tyloi::DenseTensor ids) const;
};
