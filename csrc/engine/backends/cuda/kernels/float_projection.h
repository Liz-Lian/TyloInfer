#pragma once
#include "backend.h"

struct NintWeight;
int nint_float_projection_resident_blocks(int device);

::tyloi::Tensor try_gr_bf16_down_cuda(
    const ::tyloi::Tensor& input,const ::tyloi::Tensor& weight,int64_t streams);
::tyloi::Tensor try_gr_bf16_up_mix_cuda(
    const ::tyloi::Tensor& input,const ::tyloi::Tensor& normalized,
    const ::tyloi::Tensor& weight,int64_t streams);

// Direct packed affine projection, converting each activation to FP32 at load.
// No activation quantization, half output, or dense weight expansion.
::tyloi::Tensor nint_float_projection_cuda(
    const NintWeight&, const ::tyloi::Tensor&,bool parallel_groups=true,bool native_input=true,bool fixed_groups=true);
::tyloi::Tensor nint_float_projection_activation_cuda(
    const NintWeight&, const ::tyloi::Tensor&,int64_t streams,bool injection,
    bool parallel_groups=true,bool native_input=true,bool fixed_groups=true);
::tyloi::Tensor nint_float_projection_mix_cuda(
    const NintWeight&, const ::tyloi::Tensor&,
    const ::tyloi::Tensor& normalized,int64_t streams,bool native_input=true,bool fixed_groups=true,bool compact=false);
