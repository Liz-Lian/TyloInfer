#pragma once
#include "tyloinfer/gguf_quant.h"
#include <cuda_runtime_api.h>
#include <cstdint>

namespace mfq::cuda::gguf {
// Lossless compute layout. The quantization format and encoded coefficients
// remain unchanged. Complete four-row groups use interleaved planes; the
// final one to three rows remain canonical. There is no padding or expansion.
// Packing may use a temporary source, but retains no permanent native copy.
bool supports_quad_storage(ggml_type type) noexcept;
std::uint64_t quad_storage_bytes(ggml_type type,std::int64_t rows,std::int64_t columns);
void pack_quad_storage(ggml_type type,const void* native,std::uint64_t stride,
    std::int64_t rows,std::int64_t columns,void* packed,cudaStream_t stream);
// Restore canonical GGUF rows for consumers such as prefill. Optional indices
// select rows relative to the supplied view. Invalid indices produce zeros.
void unpack_quad_storage(ggml_type type,const void* packed,std::int64_t storage_rows,
    std::int64_t columns,std::int64_t row_begin,std::int64_t rows,
    const std::int64_t* indices,std::int64_t count,void* native,
    std::uint64_t stride,cudaStream_t stream);
}
