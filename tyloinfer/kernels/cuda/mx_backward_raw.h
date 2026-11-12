#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>
namespace mfq_jit_mx {
/** Select the existing short-matrix backward kernel family. */
enum class Backward { fp4, block, scalar, vec4, mma8, mma16 };
/** Launch the selected packed backward kernel using caller-owned partial storage. */
void backward(Backward kind,const std::uint8_t* values,const std::uint8_t* scales,
    const void* gradient,void* output,float* partials,int rows,int outputs,int width,cudaStream_t stream);
}
