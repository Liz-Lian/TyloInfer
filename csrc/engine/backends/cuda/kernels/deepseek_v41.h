#pragma once

#include "backend.h"
#include <vector>

::tyloi::DenseTensor deepseek_v41_engram_rows_cuda(
    ::tyloi::DenseTensor packed, ::tyloi::DenseTensor inverse, int64_t width);

::tyloi::DenseTensor deepseek_v41_engram_gate_add_cuda(
    ::tyloi::DenseTensor hidden, ::tyloi::DenseTensor projected,
    ::tyloi::DenseTensor query_key_weight, double epsilon);

std::vector<::tyloi::DenseTensor> deepseek_v41_hc_prepare_cuda(
    ::tyloi::DenseTensor residual,
    ::tyloi::DenseTensor previous_pre,
    ::tyloi::DenseTensor norm,
    double epsilon);

::tyloi::DenseTensor deepseek_v41_hc_input_cuda(
    ::tyloi::DenseTensor residual,
    ::tyloi::DenseTensor previous_pre,
    ::tyloi::DenseTensor norm,
    double epsilon);

::tyloi::DenseTensor deepseek_v41_mxfp8_e4m3_sim_cuda(
    ::tyloi::DenseTensor input);

::tyloi::DenseTensor deepseek_v41_mxfp4_e4m3_scale_sim_cuda(
    ::tyloi::DenseTensor input);

::tyloi::DenseTensor deepseek_v41_hc_project_cuda(
    ::tyloi::DenseTensor residual, ::tyloi::DenseTensor previous,
    ::tyloi::DenseTensor function);

std::vector<::tyloi::DenseTensor> deepseek_v41_hc_finalize_cuda(
    ::tyloi::DenseTensor residual, ::tyloi::DenseTensor previous,
    ::tyloi::DenseTensor norm, ::tyloi::DenseTensor partials,
    ::tyloi::DenseTensor scale, ::tyloi::DenseTensor base,
    int64_t iterations, double rms_epsilon, double hc_epsilon);
