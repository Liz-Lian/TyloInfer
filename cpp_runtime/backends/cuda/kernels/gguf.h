#pragma once

#include "ggml.h"
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace mfq::cuda::gguf {
// Match the FP32 projection path's arithmetic and its final FP16 conversion,
// without materializing separate activation/result conversion buffers.
void quantize_input_f16(const void*,std::int64_t tokens,std::int64_t columns,void*,cudaStream_t);
// Sigmoid/SiLU, multiplication and Q8 packing with the ordinary dtype rounding.
void quantize_gated_input(const void* up,const void* gate,bool half,std::int64_t tokens,
    std::int64_t columns,bool silu,void* output,cudaStream_t);
// Gated Q8 values directly in the small-window position-major layout.
void quantize_gated_input_interleaved(const void* up,const void* gate,bool half,std::int64_t tokens,
    std::int64_t columns,bool silu,void* output,cudaStream_t);
void quantize_gated_head_input_interleaved(const void* input,const void* gate,bool half,
    std::int64_t batch,std::int64_t tokens,std::int64_t heads,std::int64_t width,
    std::int64_t batch_stride,std::int64_t token_stride,std::int64_t head_stride,
    bool silu,void* output,cudaStream_t);
// Materialized prefill activation, preserving the activation's dtype boundary.
void materialize_gated_input(const void* up,const void* gate,bool half,std::int64_t count,
    bool silu,void* output,cudaStream_t);
// Ordinary RMSNorm with an F32 result and its Q8_1 packet in one pass.
// Complete 32-value rows; input may be FP16 or FP32.
void quantize_rms_input(const void* input,bool half_input,const float* weight,
    std::int64_t rows,std::int64_t width,float eps,float offset,float* normalized,
    void* output,cudaStream_t);
// A small mixed-format group can request both layouts from the same reduction.
// native_output may be null when every projection consumes position-major Q8.
void quantize_rms_input_interleaved(const void* input,bool half_input,const float* weight,
    std::int64_t rows,std::int64_t width,float eps,float offset,float* normalized,
    void* output,void* native_output,cudaStream_t);
// FP16 residual + FP16-rounded projection, FP16 RMS result, and Q8_1.
// Complete Q8 rows; residual/output pointers must be half2 aligned.
void quantize_acc_rms_input(const void* residual,const void* projection,bool projection_float,
    const float* weight,std::int64_t rows,std::int64_t width,float eps,float offset,
    void* sum,void* normalized,void* output,cudaStream_t);
void quantize_acc_rms_input_interleaved(const void* residual,const void* projection,bool projection_float,
    const float* weight,std::int64_t rows,std::int64_t width,float eps,float offset,
    void* sum,void* normalized,void* output,void* native_output,cudaStream_t);
// F32 grouped RMSNorm + SiLU gate + Q8_1, from [B,H,T,D] to [B,T,H*D].
// D must contain complete 32-value Q8 blocks. Gate rows may be strided.
void quantize_rms_gated_input(const float* input,const float* gate,const float* weight,
    std::int64_t batch,std::int64_t heads,std::int64_t tokens,std::int64_t width,
    std::int64_t gate_batch_stride,std::int64_t gate_token_stride,float eps,
    void* output,cudaStream_t);
// Grouped RMS input with the model's SiLU or Sigmoid gate.
void quantize_rms_gated_input(const float* input,const float* gate,const float* weight,
    std::int64_t batch,std::int64_t heads,std::int64_t tokens,std::int64_t width,
    std::int64_t gate_batch_stride,std::int64_t gate_token_stride,float eps,bool silu,
    void* output,cudaStream_t);
void quantize_rms_gated_input_interleaved(const float* input,const float* gate,const float* weight,
    std::int64_t batch,std::int64_t heads,std::int64_t tokens,std::int64_t width,
    std::int64_t gate_batch_stride,std::int64_t gate_token_stride,float eps,bool silu,
    void* output,cudaStream_t);
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
// Internal single-token packet with aligned coordinate and half2 scale planes.
// These APIs never reinterpret a native Q8_1 packet as an aligned packet.
bool supports_m1_aligned(ggml_type type) noexcept;
bool prefers_m1_aligned(ggml_type type) noexcept;
void quantize_input_m1_aligned(const void* input,bool half_input,std::int64_t columns,
    void* output,cudaStream_t);
void quantize_gated_input_m1_aligned(const void* up,const void* gate,bool half,
    std::int64_t tokens,std::int64_t columns,bool silu,void* output,cudaStream_t);
void quantize_rms_input_m1_aligned(const void* input,bool half_input,const float* weight,
    std::int64_t rows,std::int64_t width,float eps,float offset,float* normalized,void* output,cudaStream_t);
void quantize_acc_rms_input_m1_aligned(const void* residual,const void* projection,bool projection_float,
    const float* weight,std::int64_t rows,std::int64_t width,float eps,float offset,
    void* sum,void* normalized,void* output,cudaStream_t);
void matvec_m1_aligned(ggml_type,const void*,std::uint64_t stride,std::int64_t rows,
    std::int64_t columns,const void* input,void* output,bool half_output,cudaStream_t);
void matvec_group_m1_aligned(const DenseProjection*,std::size_t count,std::int64_t columns,
    const void* input,bool half_output,cudaStream_t);
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
// All mapped inputs are published by ready before the snapshot. Address
// tables are optional; CPU slots and route groups retain independent limits.
struct StreamPlanSnapshot {
    const std::uint32_t* ready=nullptr;
    const void* mapped=nullptr;
    const std::uint32_t* cancelled=nullptr;
    void* snapshot=nullptr;
    std::uint64_t capacity=0;
    const std::int32_t* cpu_slots=nullptr;
    std::int32_t* device_slots=nullptr;
    std::int64_t slots=0;
    const RouteGroup* host_groups=nullptr;
    RouteGroup* device_groups=nullptr;
    std::int64_t group_capacity=0;
    const std::uint64_t* mapped_addresses=nullptr;
    std::uint64_t* addresses=nullptr;
    const std::int64_t* ids=nullptr;
    std::int64_t selected_slots=0,experts=0;
};
void snapshot_stream_plan(const StreamPlanSnapshot&,cudaStream_t stream);
// A stable, GPU-readable residency table can prove that no selected expert
// needs streaming. No CPU slots may be active. The caller still serves the
// CPU task before reusing its mapped packet, preserving route learning.
struct ResidentStreamPlan {
    const std::uint64_t* addresses=nullptr;
    const std::uint32_t* ready=nullptr;
    std::uint32_t* gate_up_done=nullptr;
    std::uint32_t* down_done=nullptr;
};
void snapshot_stream_plan(const StreamPlanSnapshot&,const ResidentStreamPlan&,cudaStream_t stream);
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
// Write a prefill tile directly into the final row-strided output. Optional
// power-of-two scales preserve the FP32 activation path without an output copy.
void prefill_gemm_f16(const void* input,const void* weight,void* output,bool half_output,
    const std::int32_t* input_exponents,const std::int32_t* weight_exponents,
    std::int64_t tokens,std::int64_t rows,std::int64_t columns,std::int64_t output_stride,
    void* blas_handle,cudaStream_t stream);
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
// Normalize only FP32 operands; FP16 operands use llama.cpp's direct GEMM mode.
void pack_scaled_f16_unit(const float* input, void* output, std::int32_t* exponents,
    std::int64_t rows, std::int64_t columns, cudaStream_t stream);
bool rms_prefill_available(std::int64_t rows,std::int64_t width,int device);
void pack_rms_prefill(const void* input,ggml_type type,const float* weight,
    std::int64_t rows,std::int64_t width,float eps,float offset,float* normalized,
    void* output,std::int32_t* exponents,cudaStream_t stream);
void prefill_gemm_f16_accum(const void* input, const void* weight, void* temporary, void* output,
    bool half_output, const std::int32_t* input_exponents, const std::int32_t* weight_exponents,
    std::int64_t tokens, std::int64_t rows, std::int64_t columns, std::int64_t output_stride,
    void* blas_handle, cudaStream_t stream);
struct PrefillResidualGemmPlan {
    std::shared_ptr<void> implementation;
    std::size_t workspace_bytes=0;
    explicit operator bool()const{return bool(implementation);}
};
PrefillResidualGemmPlan prepare_prefill_residual_gemm(const void* input,const void* weight,
    const void* residual,void* output,std::int64_t tokens,std::int64_t rows,std::int64_t columns,
    std::int64_t output_stride,cudaStream_t stream);
// Adds the immutable residual in the GEMM epilogue. Caller-owned scratch is
// charged by the ordinary tensor allocator. An empty plan uses GEMM plus add.
void prefill_gemm_f16_residual(const void* input,const void* weight,const void* residual,
    void* output,std::int64_t tokens,std::int64_t rows,std::int64_t columns,
    std::int64_t output_stride,const PrefillResidualGemmPlan& plan,void* workspace,
    std::size_t workspace_bytes,void* blas_handle,cudaStream_t stream);
// Immutable per-row scales are tiny; decoded weights remain temporary tiles.
void prefill_row_exponents(ggml_type type, const void* packed, std::uint64_t stride,
    std::int64_t rows, std::int64_t columns, std::int32_t* output, cudaStream_t stream);
void dequantize_rows_scaled(ggml_type type, const void* packed, std::uint64_t stride,
    std::int64_t rows, std::int64_t columns, const std::int32_t* exponents, void* output, cudaStream_t stream);
}

namespace mfq::cuda::gguf {
bool rms_gated_prefill_available(int64_t columns,int device);
void pack_rms_gated_prefill(const float* input,const float* gate,const float* norm,
    int64_t batch,int64_t heads,int64_t tokens,int64_t width,int64_t gate_batch_stride,
    int64_t gate_token_stride,float eps,bool silu,void* output,int32_t* exponents,cudaStream_t stream);
void pack_rms_scaled_gate_prefill(const float* input,const void* gate,const float* norm,
    const int32_t* gate_rows,const int32_t* gate_columns,
    int64_t batch,int64_t heads,int64_t tokens,int64_t width,
    int64_t gate_batch_stride,int64_t gate_token_stride,float eps,bool silu,
    void* output,int32_t* exponents,cudaStream_t stream);
}

namespace mfq::cuda::gguf {
void restore_scaled_f16(const void* input,void* output,bool half_output,
    const int32_t* row_exponents,const int32_t* column_exponents,
    int64_t rows,int64_t columns,int64_t stride,cudaStream_t stream);
void acc_rms_scaled_f16(const void* product,const void* residual,const float* norm,
    const int32_t* row_exponents,const int32_t* column_exponents,int64_t rows,int64_t columns,
    float eps,float offset,void* sum,void* output,cudaStream_t stream);
}
