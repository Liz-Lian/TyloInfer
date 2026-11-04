#pragma once

#include "backend.h"

#include <cstdint>
#include <vector>

std::vector<::tyloi::Tensor> dsv4_hc_pre_cuda(
    ::tyloi::Tensor x,
    ::tyloi::Tensor mixes,
    ::tyloi::Tensor scale,
    ::tyloi::Tensor base,
    std::int64_t iterations,
    double eps);

std::vector<::tyloi::Tensor> dsv4_hc_coefficients_cuda(
    ::tyloi::Tensor mixes,
    ::tyloi::Tensor scale,
    ::tyloi::Tensor base,
    std::int64_t iterations,
    double eps);

::tyloi::Tensor dsv4_hc_post_cuda(
    ::tyloi::Tensor x,
    ::tyloi::Tensor residual,
    ::tyloi::Tensor post,
    ::tyloi::Tensor combination);
