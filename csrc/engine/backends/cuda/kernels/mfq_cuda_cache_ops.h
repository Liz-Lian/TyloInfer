#pragma once

#include "backend.h"

#include <cstdint>
#include <vector>

::tyloi::DenseTensor glm_dsa_cache_write_cuda(
    ::tyloi::DenseTensor cache, ::tyloi::DenseTensor values, ::tyloi::DenseTensor positions);
std::vector<::tyloi::DenseTensor> kv_cache_write_cuda(::tyloi::DenseTensor k_cache, ::tyloi::DenseTensor v_cache,
                                               ::tyloi::DenseTensor k, ::tyloi::DenseTensor v, ::tyloi::DenseTensor positions);
std::vector<::tyloi::DenseTensor> kv_cache_write_ring_cuda(
    ::tyloi::DenseTensor k_cache, ::tyloi::DenseTensor v_cache,
    ::tyloi::DenseTensor k, ::tyloi::DenseTensor v, int64_t position_start);
std::vector<::tyloi::DenseTensor> kv_cache_write_ring_positions_cuda(
    ::tyloi::DenseTensor k_cache, ::tyloi::DenseTensor v_cache,
    ::tyloi::DenseTensor k, ::tyloi::DenseTensor v, ::tyloi::DenseTensor positions);
