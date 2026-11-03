#pragma once

#include "engine/tensor.h"

#include <cmath>
#include <cstdint>

namespace mfq::state {

struct KvQuantization {
    double bits = 0;
    bool enabled() const noexcept { return bits != 0; }
    int key_bits() const noexcept { return int(std::floor(bits)); }
    int value_bits() const noexcept { return int(std::ceil(bits)); }
};

struct KvView {
    Tensor visible_key, visible_value;
    Tensor physical_key, physical_value;
    Tensor key_chunk_ptrs, value_chunk_ptrs, page_table;
    KvQuantization quantization;
    std::int64_t page_size = 0;
    std::int64_t pages_per_chunk = 0;
    std::int64_t heads = 0;
    bool ring = false;
    std::int64_t batch = 0;
    std::int64_t head_dim = 0;
    ScalarType dtype = ScalarType::float16;

    bool paged() const noexcept { return page_size > 0; }
};

} // namespace mfq::state
