#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>

namespace mfq::cuda::gguf {
// Normative GGUF type IDs. These operators consume original 128-value blocks;
// no expanded weights or state-sized lookup cache is required.
enum class CompactFormat : int {
    PTQ1_0 = 143, TQ2_T = 144, TQK6 = 145, TQK7 = 146,
    TQP5 = 147, TQP6 = 148, TQP7 = 149, TQK5 = 150, TQK4 = 151
};
enum class CompactInput { Float16, Float32, Q8_1 };
std::uint64_t compact_row_bytes(CompactFormat format, std::int64_t columns);
// Rotation, if required by model metadata, must already have been applied.
// Input is token-major contiguous; output is token-major FP32 or FP16.
void compact_project(CompactFormat format, const void* packed, std::uint64_t row_stride,
    std::int64_t rows, std::int64_t columns, const void* input, CompactInput input_type,
    std::int64_t tokens, void* output, bool half_output, cudaStream_t stream);
void compact_dequantize(CompactFormat format, const void* packed, std::uint64_t row_stride,
    std::int64_t rows, std::int64_t columns, void* output, bool half_output, cudaStream_t stream);
struct RouteGroup;
// Original-codebook trellis projections, with adjacent states decoded together
// and grouped inputs staged once per CTA. A second bank can share the launch.
bool try_compact_routed_project(CompactFormat format,const std::uint64_t* addresses,
    std::uint64_t row_stride,std::int64_t experts,std::int64_t rows,std::int64_t columns,
    const std::int64_t* ids,const RouteGroup* groups,std::int64_t tokens,std::int64_t routes,
    const void* input,bool routed_input,float* output,cudaStream_t stream,bool preserve_missing=false,
    const std::uint64_t* second_addresses=nullptr,std::uint64_t second_stride=0,float* second_output=nullptr,
    bool allow_shared_lookup=true,bool allow_decoded_input=true,bool allow_narrow_rows=true);
}
