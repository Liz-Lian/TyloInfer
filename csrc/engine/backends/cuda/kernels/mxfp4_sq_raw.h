#pragma once
#include "numeric_type.h"
#include "engine/weights/encoding/mxfp4_sq_blob.h"
#include <cuda_runtime_api.h>
#include <cstdint>

namespace tyloi::cuda::mxfp4_sq_raw {
/** Decode packed MXFP4-SQ rows to F16 or F32. */
void dequant(const std::uint8_t*,const std::uint8_t*,const std::int32_t*,const std::int32_t*,void*,sq::Layout,bool,cudaStream_t);
/** Apply the original packed projection or tensor-core projection. */
void matmul(const std::uint8_t*,const std::uint8_t*,const std::int32_t*,const std::int32_t*,const void*,void*,sq::Layout,int,NumericType,bool,cudaStream_t);
/** Apply routed MXFP4-SQ into caller-owned output. */
void routed(const std::uint8_t*,const std::uint8_t*,const std::int32_t*,const std::int32_t*,const void*,void*,const std::int32_t*,const std::int32_t*,sq::Layout,int,int,int,int,int,bool,cudaStream_t);
}
