#pragma once

#include "backend.h"

#include <cstdint>
#include <vector>

std::vector<::tyloi::DenseTensor> dsv4_hc_pre_cuda(
    ::tyloi::DenseTensor x,
    ::tyloi::DenseTensor mixes,
    ::tyloi::DenseTensor scale,
    ::tyloi::DenseTensor base,
    std::int64_t iterations,
    double eps);

std::vector<::tyloi::DenseTensor> dsv4_hc_coefficients_cuda(
    ::tyloi::DenseTensor mixes,
    ::tyloi::DenseTensor scale,
    ::tyloi::DenseTensor base,
    std::int64_t iterations,
    double eps);

::tyloi::DenseTensor dsv4_hc_post_cuda(
    ::tyloi::DenseTensor x,
    ::tyloi::DenseTensor residual,
    ::tyloi::DenseTensor post,
    ::tyloi::DenseTensor combination);

// CPU sinkhorn reference used to validate the fused CUDA hc kernels.
std::vector<::tyloi::DenseTensor> dsv4_hc_split_sinkhorn(
    ::tyloi::DenseTensor mixes,
    ::tyloi::DenseTensor scale,
    ::tyloi::DenseTensor base,
    std::int64_t hc_mult,
    std::int64_t iterations,
    double eps);
