#pragma once
#include "gguf.h"

namespace mfq::cuda::gguf {
struct QuadProjection {
    DenseProjection projection{};
    std::int64_t storage_rows=0;
    std::int64_t row_begin=0;
    bool quad=false;
};
// Encoded row views start at a four-row boundary. Unaligned concatenations
// retain canonical GGUF storage; canonical row gathering has no such limit.
void matvec_quad_m1(const QuadProjection&,std::int64_t columns,const void* input,
    bool aligned_input,bool half_output,const void* residual,cudaStream_t stream);
void matvec_quad_group_m1(const QuadProjection*,std::size_t count,std::int64_t columns,
    const void* input,bool aligned_input,bool half_output,cudaStream_t stream);
}
