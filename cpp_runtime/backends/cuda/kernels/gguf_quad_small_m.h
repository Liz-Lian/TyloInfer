#pragma once
#include "gguf_quad_matvec.h"

namespace mfq::cuda::gguf {
// Small verification/concurrent batches reuse each decoded weight fragment.
// Input is native Q8_1, or the position-major packet selected by its producer. The output
// is [tokens, rows]. The compressed four-row storage stays unchanged.
bool supports_quad_small_m(ggml_type type,std::int64_t tokens) noexcept;
void matvec_quad_small_m(const QuadProjection&,std::int64_t columns,const void* input,
    std::int64_t tokens,bool position_major,bool half_output,cudaStream_t stream);
// Independent projections share one launch grid while retaining their own
// formats, row views and output tensors. The position-major flags describe
// packets already emitted by the shared activation quantizer.
void matvec_quad_group_small_m(const QuadProjection*,std::size_t count,std::int64_t columns,
    const void* native,const void* interleaved,const std::uint8_t* position_major,
    std::int64_t tokens,bool half_output,cudaStream_t stream);
// A format-paired candidate for two projections, with the same activation ABI.
void matvec_quad_pair_small_m(const QuadProjection*,std::size_t count,std::int64_t columns,
    const void* native,const void* interleaved,const std::uint8_t* position_major,
    std::int64_t tokens,bool half_output,cudaStream_t stream);
}
