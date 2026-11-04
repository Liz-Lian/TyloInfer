#pragma once

#include "backend.h"

#include <cstdint>

void paged_kv_cache_write_cuda(
    ::tyloi::DenseTensor k_chunk_ptrs,
    ::tyloi::DenseTensor v_chunk_ptrs,
    ::tyloi::DenseTensor page_table,
    ::tyloi::DenseTensor k,
    ::tyloi::DenseTensor v,
    ::tyloi::DenseTensor positions,
    int64_t page_size,
    int64_t pages_per_chunk);

::tyloi::DenseTensor attention_paged_cache_decode_cuda(
    ::tyloi::DenseTensor q,
    ::tyloi::DenseTensor k_chunk_ptrs,
    ::tyloi::DenseTensor v_chunk_ptrs,
    ::tyloi::DenseTensor page_table,
    ::tyloi::DenseTensor seq_len,
    double scale,
    int64_t page_size,
    int64_t pages_per_chunk,
    int64_t kv_heads,
    ::tyloi::DenseTensor partial_o,
    ::tyloi::DenseTensor partial_m,
    ::tyloi::DenseTensor partial_l,
    int64_t parts,
    bool dynamic_parts);

void paged_kv_cache_gather_cuda(
    ::tyloi::DenseTensor k_chunk_ptrs,
    ::tyloi::DenseTensor v_chunk_ptrs,
    ::tyloi::DenseTensor page_table,
    ::tyloi::DenseTensor k,
    ::tyloi::DenseTensor v,
    int64_t page_size,
    int64_t pages_per_chunk);
