#pragma once

#include "backend.h"

#include <cstdint>
#include <vector>

::tyloi::Tensor glm_dsa_cache_write_cuda(
    ::tyloi::Tensor cache, ::tyloi::Tensor values, ::tyloi::Tensor positions);
std::vector<::tyloi::Tensor> kv_cache_write_cuda(::tyloi::Tensor k_cache, ::tyloi::Tensor v_cache,
                                               ::tyloi::Tensor k, ::tyloi::Tensor v, ::tyloi::Tensor positions);
std::vector<::tyloi::Tensor> kv_cache_write_ring_cuda(
    ::tyloi::Tensor k_cache, ::tyloi::Tensor v_cache,
    ::tyloi::Tensor k, ::tyloi::Tensor v, int64_t position_start);
std::vector<::tyloi::Tensor> kv_cache_write_ring_positions_cuda(
    ::tyloi::Tensor k_cache, ::tyloi::Tensor v_cache,
    ::tyloi::Tensor k, ::tyloi::Tensor v, ::tyloi::Tensor positions);
