#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>
namespace mfq::cuda::gguf {
// GR residual update, exact grouped RMS, and the native Q8_1 activation.
// Each hidden group contains complete Q8 blocks. F32 updated storage must
// be distinct from residual; null branch/injection leaves residual unchanged.
void prepare_residual_input(const float* residual,const void* branch,bool half_branch,
    const void* injection,bool half_injection,const float* norm,
    std::int64_t tokens,std::int64_t hidden,std::int64_t streams,float eps,
    float* updated,float* normalized,void* native_q8,cudaStream_t);
}
