#pragma once
#include <cuda_fp16.h>
#include <cstdint>

__device__ __forceinline__ float gdn_qkv_value(const void* input,std::int64_t index,bool fp32) {
    return fp32?static_cast<const float*>(input)[index]:__half2float(static_cast<const __half*>(input)[index]);
}
