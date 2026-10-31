#pragma once

#include "ops/tq/include/tq.h"

#include <cstdint>
#include <memory>

class TqCache;

// A routed projection owns only its source identity and cache owner. The
// cache owns compressed images and their final-consumer lifetime.
struct TqRoutedWeight {
    std::shared_ptr<TqCache> cache;
    std::uint64_t source_id = 0;
    std::int64_t experts = 0, out = 0, input = 0;

    mfq_tensor_backend::Tensor forward(
        mfq_tensor_backend::Tensor input_value,
        mfq_tensor_backend::Tensor ids) const;
};
