#pragma once
#include "gguf_quad_matvec.h"

namespace mfq::cuda::gguf {
// Small verification/concurrent batches reuse each decoded weight fragment.
// Input is native Q8_1, or a position-major Q8 packet for IQ2/IQ3. The output
// is [tokens, rows]. The compressed four-row storage stays unchanged.
bool supports_quad_small_m(ggml_type type,std::int64_t tokens) noexcept;
void matvec_quad_small_m(const QuadProjection&,std::int64_t columns,const void* input,
    std::int64_t tokens,bool position_major,bool half_output,cudaStream_t stream);
}
