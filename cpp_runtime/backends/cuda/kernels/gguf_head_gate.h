#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace mfq::cuda::gguf {
// Contiguous [B,T,H*D] activations and strided [B,T,H,D] gates.
// Preserves the ordinary sigmoid/SiLU and multiplication dtype rounding.
void quantize_gated_head_input(const void* input,const void* gate,bool half_input,
    int64_t batch,int64_t tokens,int64_t heads,int64_t width,
    int64_t gate_batch_stride,int64_t gate_token_stride,int64_t gate_head_stride,
    bool silu,bool aligned_m1,void* output,cudaStream_t stream);
}
