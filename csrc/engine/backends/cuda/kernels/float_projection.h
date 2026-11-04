#pragma once
#include "backend.h"

#include "engine/weights/encoding/nint.h"
int nint_float_projection_resident_blocks(int device);

::tyloi::DenseTensor try_gr_bf16_down_cuda(
    const ::tyloi::DenseTensor& input,const ::tyloi::DenseTensor& weight,int64_t streams);
::tyloi::DenseTensor try_gr_bf16_up_mix_cuda(
    const ::tyloi::DenseTensor& input,const ::tyloi::DenseTensor& normalized,
    const ::tyloi::DenseTensor& weight,int64_t streams);

// Direct packed affine projection, converting each activation to FP32 at load.
// No activation quantization, half output, or dense weight expansion.
::tyloi::DenseTensor nint_float_projection_cuda(
    const tyloi::NintTensor&, const ::tyloi::DenseTensor&,bool parallel_groups=true,bool native_input=true,bool fixed_groups=true);
::tyloi::DenseTensor nint_float_projection_activation_cuda(
    const tyloi::NintTensor&, const ::tyloi::DenseTensor&,int64_t streams,bool injection,
    bool parallel_groups=true,bool native_input=true,bool fixed_groups=true);
::tyloi::DenseTensor nint_float_projection_mix_cuda(
    const tyloi::NintTensor&, const ::tyloi::DenseTensor&,
    const ::tyloi::DenseTensor& normalized,int64_t streams,bool native_input=true,bool fixed_groups=true,bool compact=false);
