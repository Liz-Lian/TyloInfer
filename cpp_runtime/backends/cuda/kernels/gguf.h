#pragma once

#include "ggml.h"
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>

namespace mfq::cuda::gguf {
// Match the FP32 projection path's arithmetic and its final FP16 conversion,
// without materializing separate activation/result conversion buffers.
void quantize_input_f16(const void*,std::int64_t tokens,std::int64_t columns,void*,cudaStream_t);
void matvec_q8_f16(ggml_type,const void*,std::uint64_t stride,std::int64_t rows,std::int64_t columns,
    const void* input,std::int64_t tokens,void* output,cudaStream_t);
// Position-major activation words make each 2/4-token vector contiguous.
// Weight storage, quantization and the projection reduction order are unchanged.
bool supports_interleaved_activation(ggml_type,std::int64_t tokens) noexcept;
std::uint64_t interleaved_activation_bytes(std::int64_t tokens,std::int64_t columns);
// native_output may request the original Q8_1 layout for other group members.
void quantize_input_interleaved(const void* input,bool half,std::int64_t tokens,std::int64_t columns,
    void* output,void* native_output,cudaStream_t);
void matvec_interleaved(ggml_type,const void*,std::uint64_t stride,std::int64_t rows,std::int64_t columns,
    const void* input,std::int64_t tokens,void* output,bool half_output,cudaStream_t);
struct DenseProjection {
    const void* packed;
    std::uint64_t stride;
    std::int64_t rows;
    void* output;
    ggml_type type;
};
// Independent dense matrices keep their formats and outputs. A common graph
// node schedules their row tiles and consumes the already-quantized input.
void matvec_group(const DenseProjection*,std::size_t count,std::int64_t columns,
    const void* native_input,const void* interleaved_input,std::int64_t tokens,
    bool half_output,cudaStream_t);
struct RouteGroup;
bool supports_shared_routed_decode(ggml_type type) noexcept;
// Returns false when the format or actual shared-memory requirement needs
// separate projections. Pointer tables and row strides remain independent.
bool try_gate_up_matvec_q8(ggml_type type,const std::uint64_t* gate_addresses,const std::uint64_t* up_addresses,
    std::uint64_t gate_stride,std::uint64_t up_stride,std::int64_t experts,std::int64_t rows,std::int64_t columns,
    const std::int64_t* ids,const RouteGroup* groups,std::int64_t tokens,std::int64_t routes,
    const void* input,float* gate,float* up,cudaStream_t stream,bool preserve_missing=false);
bool try_gate_up_hidden_q8(ggml_type type,const std::uint64_t* gate_addresses,const std::uint64_t* up_addresses,
    std::uint64_t gate_stride,std::uint64_t up_stride,std::int64_t experts,std::int64_t rows,std::int64_t columns,
    const std::int64_t* ids,const RouteGroup* groups,std::int64_t tokens,std::int64_t routes,const void* input,
    void* hidden,bool half_output,float limit,cudaStream_t stream);
void publish_cpu_input(const float* input,float* mapped,std::int64_t count,cudaStream_t stream);
void snapshot_direct_plan(const void* mapped,const std::uint32_t* cancelled,void* snapshot,
    std::uint64_t capacity,const std::int32_t* cpu_slots,std::int32_t* device_slots,
    std::int64_t slots,const RouteGroup* host_groups,RouteGroup* device_groups,
    std::int64_t group_capacity,cudaStream_t stream);
// Preserve the native per-expert FP16 rounding before summation. Optional
// mapped CPU values replace their GPU slots in the same reduction kernel.
void reduce_moe_output(const float* pairs,const float* weights,std::int64_t tokens,
    std::int64_t routes,std::int64_t columns,bool half_output,void* output,
    const float* cpu_values,const std::int32_t* cpu_slots,const std::uint32_t* cpu_ready,cudaStream_t stream,
    std::uint64_t* cpu_waited_until=nullptr);
struct StagedRow {
    std::uint64_t offset;
    std::int32_t type;
    std::int32_t reserved = 0;
};
struct GemmJob {
    const void* packed;
    const std::int64_t* slots;
    std::int64_t count;
};
struct RouteGroup {
    std::int64_t expert;
    std::int32_t slots[8];
    int count;
};
// Allocate slots+1 entries. Entry zero holds the number of compact groups.
void prepare_route_groups(const std::int64_t* ids,std::int64_t slots,int tokens,RouteGroup* groups,cudaStream_t stream);
// Snapshot only selected pointers; the host table is stable after its doorbell.
void snapshot_routed_addresses(const std::uint64_t* mapped,std::uint64_t* device,
    const std::int64_t* ids,std::int64_t slots,std::int64_t experts,cudaStream_t stream,
    const std::uint32_t* admission_ready=nullptr);
void grouped_routed_matvec_q8(ggml_type type,const std::uint64_t* expert_addresses,
    std::uint64_t row_bytes,std::int64_t experts,std::int64_t rows,std::int64_t columns,
    const std::int64_t* ids,const RouteGroup* groups,std::int64_t tokens,std::int64_t routes,
    const void* input,bool routed_input,float* output,cudaStream_t stream,bool preserve_missing=false);
struct UploadChunk {
    std::uint64_t destination, offset, bytes, reserved = 0;
};
void scatter_upload(const void* packet, int chunks, std::uint64_t maximum_bytes, cudaStream_t stream);
bool grouped_f16_available(int device);
void grouped_matmul_f16(ggml_type type, const GemmJob* jobs, int count, std::int64_t max_tokens,
    std::uint64_t row_bytes, std::int64_t rows, std::int64_t columns,
    const void* input, std::int64_t routes, bool routed_input, void* output, cudaStream_t stream);
void grouped_swiglu_f16(const std::int64_t* slots, std::int64_t count, std::int64_t width,
    const void* gate, const void* up, void* hidden, float limit, cudaStream_t stream);
void grouped_decode_f16(ggml_type type,const GemmJob* jobs,int experts,
    std::uint64_t row_bytes,std::int64_t rows,std::int64_t columns,void* output,cudaStream_t stream,
    std::int64_t batch_stride = 0);
void gate_up_swiglu_f16(const void* gate_up,std::int64_t tokens,std::int64_t width,
    void* hidden,float limit,cudaStream_t stream);
// The ordinary half-rounded SwiGLU followed by Q8 packing, without a half
// hidden tensor between them. Width must contain complete Q8 blocks.
void gate_up_swiglu_q8_f16(const void* gate_up,std::int64_t tokens,std::int64_t width,
    void* quantized,bool position_major,cudaStream_t stream);
void grouped_gather_f16(const GemmJob* jobs,int experts,std::int64_t tokens,std::int64_t columns,
    const void* input,std::int64_t routes,void* output,cudaStream_t stream);
void grouped_scatter_f16(const GemmJob* jobs,int experts,std::int64_t tokens,std::int64_t columns,
    const void* input,void* output,cudaStream_t stream);
// Power-of-two row scaling keeps FP32 activation/weight exponent range while
// using FP16 tensor-core operands and FP32 accumulation/output for prefill.
void pack_scaled_f16(const float* input,void* output,std::int32_t* exponents,
    std::int64_t rows,std::int64_t columns,cudaStream_t stream);
void pack_scaled_f32(const float* input,float* output,std::int32_t* exponents,
    std::int64_t rows,std::int64_t columns,cudaStream_t stream);
void rescale_f32(float* output,const std::int32_t* input_exponents,
    std::int64_t tokens,std::int64_t rows,cudaStream_t stream);
void gemm_scaled_f16(const void* input,const void* weight,const std::int32_t* input_exponents,
    const std::int32_t* weight_exponents,float* output,std::int64_t tokens,std::int64_t rows,
    std::int64_t columns,void* blas_handle,cudaStream_t stream);
void staged_rows(const void* packet, std::int64_t rows, std::int64_t columns,
    void* output, bool half_output, cudaStream_t stream);
void dequantize_rows(ggml_type type, const void* packed, std::uint64_t row_bytes,
    std::int64_t rows, std::int64_t columns, void* output, bool half_output,
    cudaStream_t stream);
void embedding_rows(ggml_type type, const void* packed, std::uint64_t row_bytes,
    std::int64_t table_rows, std::int64_t columns, const std::int64_t* ids,
    std::int64_t count, float* output, cudaStream_t stream);
void matvec(ggml_type type, const void* packed, std::uint64_t row_bytes,
    std::int64_t rows, std::int64_t columns, const float* input,
    std::int64_t tokens, float* output, cudaStream_t stream);
std::uint64_t q8_row_bytes(std::int64_t columns);
void quantize_input(const float* input, std::int64_t tokens, std::int64_t columns,
    void* packed, cudaStream_t stream);
void quantize_moe_hidden(const float* gate,const float* up,std::int64_t rows,std::int64_t columns,
    bool half_output,float limit,void* packed,cudaStream_t stream);
void matvec_q8(ggml_type type, const void* packed, std::uint64_t row_bytes,
    std::int64_t rows, std::int64_t columns, const void* input,
    std::int64_t tokens, float* output, cudaStream_t stream);
void routed_matvec(ggml_type type, const std::uint64_t* expert_addresses,
    std::uint64_t row_bytes, std::int64_t experts, std::int64_t rows, std::int64_t columns,
    const std::int64_t* ids, std::int64_t tokens, std::int64_t routes,
    const float* input, bool routed_input, float* output, cudaStream_t stream);
void routed_matvec_q8(ggml_type type, const std::uint64_t* expert_addresses,
    std::uint64_t row_bytes, std::int64_t experts, std::int64_t rows, std::int64_t columns,
    const std::int64_t* ids, std::int64_t tokens, std::int64_t routes,
    const void* input, bool routed_input, float* output, cudaStream_t stream,bool preserve_missing=false);
}
