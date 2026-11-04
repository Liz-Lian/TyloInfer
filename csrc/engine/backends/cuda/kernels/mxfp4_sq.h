#pragma once
#include "backend.h"
#include "engine/weights/encoding/mxfp4_sq_blob.h"

// Parse the self-describing blob on CPU before upload.  These capture-safe
// entry points consume loader-expanded row metadata without GPU-to-CPU reads.
::tyloi::DenseTensor mxfp4_sq_dequant_cuda(
    ::tyloi::DenseTensor blob,
    ::tyloi::DenseTensor row_q,
    ::tyloi::DenseTensor row_symbol_byte_offsets,
    ::tyloi::DenseTensor row_auxiliary,
    std::int64_t bits, std::int64_t outputs,
    std::int64_t width, std::int64_t base,
    std::int64_t q_sum, std::int64_t sq4_rows, bool fp32);
::tyloi::DenseTensor mxfp4_sq_matmul_cuda(
    ::tyloi::DenseTensor blob,
    ::tyloi::DenseTensor row_q,
    ::tyloi::DenseTensor row_symbol_byte_offsets,
    ::tyloi::DenseTensor row_auxiliary,
    ::tyloi::DenseTensor input,
    std::int64_t bits, std::int64_t outputs,
    std::int64_t width, std::int64_t base,
    std::int64_t q_sum, std::int64_t sq4_rows);
void mxfp4_sq_moe_matmul_cuda(
    ::tyloi::DenseTensor blob,
    ::tyloi::DenseTensor row_q,
    ::tyloi::DenseTensor row_symbol_byte_offsets,
    ::tyloi::DenseTensor row_auxiliary,
    ::tyloi::DenseTensor input,
    ::tyloi::DenseTensor expert_ids,
    ::tyloi::DenseTensor expert_local,
    std::int64_t bits,
    std::int64_t n_experts,
    std::int64_t local_experts,
    std::int64_t out_per_expert,
    std::int64_t width,
    std::int64_t base,
    std::int64_t q_sum,
    std::int64_t sq4_rows,
    ::tyloi::DenseTensor output);
::tyloi::DenseTensor mxfp4_sq_backward_input_cuda(
    ::tyloi::DenseTensor blob,
    ::tyloi::DenseTensor row_q,
    ::tyloi::DenseTensor row_symbol_byte_offsets,
    ::tyloi::DenseTensor row_auxiliary,
    ::tyloi::DenseTensor output_gradient,
    std::int64_t bits, std::int64_t outputs, std::int64_t width,
    std::int64_t base, std::int64_t q_sum, std::int64_t sq4_rows);
