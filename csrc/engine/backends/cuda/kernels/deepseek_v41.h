#pragma once

#include "backend.h"
#include <vector>

::tyloi::Tensor deepseek_v41_engram_rows_cuda(
    ::tyloi::Tensor packed, ::tyloi::Tensor inverse, int64_t width);

::tyloi::Tensor deepseek_v41_engram_gate_add_cuda(
    ::tyloi::Tensor hidden, ::tyloi::Tensor projected,
    ::tyloi::Tensor query_key_weight, double epsilon);

std::vector<::tyloi::Tensor> deepseek_v41_hc_prepare_cuda(
    ::tyloi::Tensor residual,
    ::tyloi::Tensor previous_pre,
    ::tyloi::Tensor norm,
    double epsilon);

::tyloi::Tensor deepseek_v41_hc_input_cuda(
    ::tyloi::Tensor residual,
    ::tyloi::Tensor previous_pre,
    ::tyloi::Tensor norm,
    double epsilon);

::tyloi::Tensor deepseek_v41_mxfp8_e4m3_sim_cuda(
    ::tyloi::Tensor input);

::tyloi::Tensor deepseek_v41_mxfp4_e4m3_scale_sim_cuda(
    ::tyloi::Tensor input);

::tyloi::Tensor deepseek_v41_hc_project_cuda(
    ::tyloi::Tensor residual, ::tyloi::Tensor previous,
    ::tyloi::Tensor function);

std::vector<::tyloi::Tensor> deepseek_v41_hc_finalize_cuda(
    ::tyloi::Tensor residual, ::tyloi::Tensor previous,
    ::tyloi::Tensor norm, ::tyloi::Tensor partials,
    ::tyloi::Tensor scale, ::tyloi::Tensor base,
    int64_t iterations, double rms_epsilon, double hc_epsilon);
