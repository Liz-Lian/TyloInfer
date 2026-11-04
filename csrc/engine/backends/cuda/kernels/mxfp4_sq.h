#pragma once
#include "backend.h"
#include "engine/weights/encoding/mxfp4_sq_blob.h"

// Parse the self-describing blob on CPU before upload.  These capture-safe
// entry points consume loader-expanded row metadata without GPU-to-CPU reads.
::tyloi::Tensor mxfp4_sq_dequant_cuda(
    ::tyloi::Tensor blob,
    ::tyloi::Tensor row_q,
    ::tyloi::Tensor row_symbol_byte_offsets,
    ::tyloi::Tensor row_auxiliary,
    std::int64_t bits, std::int64_t outputs,
    std::int64_t width, std::int64_t base,
    std::int64_t q_sum, std::int64_t sq4_rows, bool fp32);
::tyloi::Tensor mxfp4_sq_matmul_cuda(
    ::tyloi::Tensor blob,
    ::tyloi::Tensor row_q,
    ::tyloi::Tensor row_symbol_byte_offsets,
    ::tyloi::Tensor row_auxiliary,
    ::tyloi::Tensor input,
    std::int64_t bits, std::int64_t outputs,
    std::int64_t width, std::int64_t base,
    std::int64_t q_sum, std::int64_t sq4_rows);
void mxfp4_sq_moe_matmul_cuda(
    ::tyloi::Tensor blob,
    ::tyloi::Tensor row_q,
    ::tyloi::Tensor row_symbol_byte_offsets,
    ::tyloi::Tensor row_auxiliary,
    ::tyloi::Tensor input,
    ::tyloi::Tensor expert_ids,
    ::tyloi::Tensor expert_local,
    std::int64_t bits,
    std::int64_t n_experts,
    std::int64_t local_experts,
    std::int64_t out_per_expert,
    std::int64_t width,
    std::int64_t base,
    std::int64_t q_sum,
    std::int64_t sq4_rows,
    ::tyloi::Tensor output);
::tyloi::Tensor mxfp4_sq_backward_input_cuda(
    ::tyloi::Tensor blob,
    ::tyloi::Tensor row_q,
    ::tyloi::Tensor row_symbol_byte_offsets,
    ::tyloi::Tensor row_auxiliary,
    ::tyloi::Tensor output_gradient,
    std::int64_t bits, std::int64_t outputs, std::int64_t width,
    std::int64_t base, std::int64_t q_sum, std::int64_t sq4_rows);
