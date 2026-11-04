#pragma once
#include "engine/weights/nint.h"
#include <functional>
#include <vector>

// Replace the existing F32 [column, stream] buffer with an equal-size
// [column/4, stream, item] buffer. Its rank explicitly identifies the layout.
::tyloi::Tensor prepare_gr_dense_vector_right(
    const ::tyloi::Tensor& right);

// Cache only the effective subgroup scale/min; packed weight bytes stay intact.
::tyloi::Tensor prepare_gr_up_group_coefficients(const NintWeight& weight);
::tyloi::Tensor prepare_gr_up_packed_metadata(const NintWeight& weight);

// Fused norm, Down/injection, and Up/mix residual stages. Markers instrument
// the same stages used by ordinary decode without replacing their arithmetic.
std::vector<::tyloi::Tensor> gated_residual_two_stage_cuda(
    const ::tyloi::Tensor& branch,
    const ::tyloi::Tensor& residual,
    const ::tyloi::Tensor& prior_injection,
    const ::tyloi::Tensor& norm,
    const NintWeight& down,const NintWeight& up,const NintWeight* injection,
    int64_t streams,double eps,bool after,
    const ::tyloi::Tensor& dense_injection={},
    const ::tyloi::Tensor& prepared_injection_right={},
    const std::function<void(const char*)>& marker={},
    const ::tyloi::Tensor& up_group_coefficients={});
