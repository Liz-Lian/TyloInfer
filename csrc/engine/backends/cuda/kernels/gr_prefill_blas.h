#pragma once
#include "engine/weights/nint.h"
bool gr_prefill_blas_selected(const NintWeight&,const ::tyloi::Tensor&);
::tyloi::Tensor gr_prefill_blas_fp32(const NintWeight&,const ::tyloi::Tensor&);
::tyloi::Tensor gr_prefill_blas_mix(const NintWeight&,const ::tyloi::Tensor&,
    const ::tyloi::Tensor& normalized,int64_t streams);
void prepare_gr_prefill_blas_runtime();
