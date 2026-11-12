#pragma once
#include <torch/extension.h>

// Parse the self-describing blob on CPU before upload.  These capture-safe
// entry points consume loader-expanded row metadata without GPU-to-CPU reads.
/** Decode packed weights. */
torch::Tensor mxfp4_sq_dequant_cuda(
    torch::Tensor blob,
    torch::Tensor row_q,
    torch::Tensor row_symbol_byte_offsets,
    torch::Tensor row_auxiliary,
    std::int64_t bits, std::int64_t outputs,
    std::int64_t width, std::int64_t base,
    std::int64_t q_sum, std::int64_t sq4_rows, bool fp32);
/** Apply a packed projection. */
torch::Tensor mxfp4_sq_matmul_cuda(
    torch::Tensor blob,
    torch::Tensor row_q,
    torch::Tensor row_symbol_byte_offsets,
    torch::Tensor row_auxiliary,
    torch::Tensor input,
    std::int64_t bits, std::int64_t outputs,
    std::int64_t width, std::int64_t base,
    std::int64_t q_sum, std::int64_t sq4_rows);
/** Apply routed packed weights to the supplied output. */
void mxfp4_sq_moe_matmul_cuda(
    torch::Tensor blob,
    torch::Tensor row_q,
    torch::Tensor row_symbol_byte_offsets,
    torch::Tensor row_auxiliary,
    torch::Tensor input,
    torch::Tensor expert_ids,
    torch::Tensor expert_local,
    std::int64_t bits,
    std::int64_t n_experts,
    std::int64_t local_experts,
    std::int64_t out_per_expert,
    std::int64_t width,
    std::int64_t base,
    std::int64_t q_sum,
    std::int64_t sq4_rows,
    torch::Tensor output);
/** Compute the packed projection input gradient. */
torch::Tensor mxfp4_sq_backward_input_cuda(
    torch::Tensor blob,
    torch::Tensor row_q,
    torch::Tensor row_symbol_byte_offsets,
    torch::Tensor row_auxiliary,
    torch::Tensor output_gradient,
    std::int64_t bits, std::int64_t outputs, std::int64_t width,
    std::int64_t base, std::int64_t q_sum, std::int64_t sq4_rows);
