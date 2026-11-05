#pragma once

#include "engine/tensor.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace tyloi::state {

struct KvQuantization {
    double bits = 0;
    bool enabled() const noexcept { return bits != 0; }
    int key_bits() const noexcept { return int(std::floor(bits)); }
    int value_bits() const noexcept { return int(std::ceil(bits)); }
};

struct KvView {
    std::shared_ptr<Tensor> visible_key, visible_value;
    std::shared_ptr<Tensor> physical_key, physical_value;
    std::shared_ptr<Tensor> key_chunk_ptrs, value_chunk_ptrs, page_table;
    KvQuantization quantization;
    std::int64_t page_size = 0;
    std::int64_t pages_per_chunk = 0;
    std::int64_t heads = 0;
    bool ring = false;
    std::int64_t batch = 0;
    std::int64_t head_dim = 0;
    ScalarType dtype = ScalarType::float16;

    bool paged() const noexcept { return page_size > 0; }
    bool defined() const noexcept {
        return paged()
            ? key_chunk_ptrs && key_chunk_ptrs->defined() && value_chunk_ptrs && value_chunk_ptrs->defined() && page_table && page_table->defined()
            : physical_key && physical_key->defined() && physical_value && physical_value->defined();
    }
};

std::size_t kv_bytes(std::int64_t batch, std::int64_t heads, std::int64_t capacity,
                     std::int64_t head_dim, bool ring, ScalarType dtype,
                     KvQuantization settings = {});

KvView allocate_kv(std::int64_t batch, std::int64_t heads, std::int64_t capacity,
                   std::int64_t head_dim, bool ring, Device device, ScalarType dtype,
                   KvQuantization settings = {});

} // namespace tyloi::state
