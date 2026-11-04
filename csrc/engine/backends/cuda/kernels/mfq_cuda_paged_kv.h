#pragma once

#include "backend.h"

#include <cstdint>

void paged_kv_cache_write_cuda(
    ::tyloi::Tensor k_chunk_ptrs,
    ::tyloi::Tensor v_chunk_ptrs,
    ::tyloi::Tensor page_table,
    ::tyloi::Tensor k,
    ::tyloi::Tensor v,
    ::tyloi::Tensor positions,
    int64_t page_size,
    int64_t pages_per_chunk);

::tyloi::Tensor attention_paged_cache_decode_cuda(
    ::tyloi::Tensor q,
    ::tyloi::Tensor k_chunk_ptrs,
    ::tyloi::Tensor v_chunk_ptrs,
    ::tyloi::Tensor page_table,
    ::tyloi::Tensor seq_len,
    double scale,
    int64_t page_size,
    int64_t pages_per_chunk,
    int64_t kv_heads,
    ::tyloi::Tensor partial_o,
    ::tyloi::Tensor partial_m,
    ::tyloi::Tensor partial_l,
    int64_t parts,
    bool dynamic_parts);

void paged_kv_cache_gather_cuda(
    ::tyloi::Tensor k_chunk_ptrs,
    ::tyloi::Tensor v_chunk_ptrs,
    ::tyloi::Tensor page_table,
    ::tyloi::Tensor k,
    ::tyloi::Tensor v,
    int64_t page_size,
    int64_t pages_per_chunk);
