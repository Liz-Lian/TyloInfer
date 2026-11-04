#pragma once
#include "engine/weights/nint.h"
bool gr_prefill_blas_selected(const tyloi::NintTensor&,const ::tyloi::DenseTensor&);
::tyloi::DenseTensor gr_prefill_blas_fp32(const tyloi::NintTensor&,const ::tyloi::DenseTensor&);
::tyloi::DenseTensor gr_prefill_blas_mix(const tyloi::NintTensor&,const ::tyloi::DenseTensor&,
    const ::tyloi::DenseTensor& normalized,int64_t streams);
void prepare_gr_prefill_blas_runtime();
