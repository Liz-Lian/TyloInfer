#pragma once
#include "engine/weights/nint.h"
#ifdef TYLOINFER_NATIVE_CUDA
#include "residual_trace.h"
#else
#include <functional>
#endif
#include <vector>

// Replace the existing F32 [column, stream] buffer with an equal-size
// [column/4, stream, item] buffer. Its rank explicitly identifies the layout.
::tyloi::DenseTensor prepare_gr_dense_vector_right(
    const ::tyloi::DenseTensor& right);

// Cache only the effective subgroup scale/min; packed weight bytes stay intact.
::tyloi::DenseTensor prepare_gr_up_group_coefficients(const tyloi::NintTensor& weight);
::tyloi::DenseTensor prepare_gr_up_packed_metadata(const tyloi::NintTensor& weight);

// Fused norm, Down/injection, and Up/mix residual stages. Markers instrument
// the same stages used by ordinary decode without replacing their arithmetic.
std::vector<::tyloi::DenseTensor> gated_residual_two_stage_cuda(
    const ::tyloi::DenseTensor& branch,
    const ::tyloi::DenseTensor& residual,
    const ::tyloi::DenseTensor& prior_injection,
    const ::tyloi::DenseTensor& norm,
    const tyloi::NintTensor& down,const tyloi::NintTensor& up,const tyloi::NintTensor* injection,
    int64_t streams,double eps,bool after,
    const ::tyloi::DenseTensor& dense_injection={},
    const ::tyloi::DenseTensor& prepared_injection_right={},
#ifdef TYLOINFER_NATIVE_CUDA
    ResidualTraceData* trace=nullptr,
#else
    const std::function<void(const char*)>& marker={},
#endif
    const ::tyloi::DenseTensor& up_group_coefficients={});
