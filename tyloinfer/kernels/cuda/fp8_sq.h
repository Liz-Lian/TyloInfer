#pragma once

#include <cstdint>

#include <torch/extension.h>

// The CPU loader validates the self-describing wire payload and expands the
// compact three-bit q descriptor into row_q/row_symbol_byte_offsets.  CUDA
// never reads metadata back from the device and never keeps a full decoded
// weight resident.
/** Decode packed weights. */
torch::Tensor mxfp8_sq_dequant_cuda(
    torch::Tensor blob,
    torch::Tensor row_q,
    torch::Tensor row_symbol_byte_offsets,
    std::int64_t outputs, std::int64_t width,
    std::int64_t block_rows, std::int64_t block_columns,
    std::int64_t scale_rows, std::int64_t scale_columns,
    std::int64_t palettes_offset, std::int64_t symbols_offset,
    std::int64_t scales_offset, bool fp32);

/** Decode packed weights. */
torch::Tensor fp8_128_sq_dequant_cuda(
    torch::Tensor blob,
    torch::Tensor row_q,
    torch::Tensor row_symbol_byte_offsets,
    std::int64_t outputs, std::int64_t width,
    std::int64_t scale_kind,
    std::int64_t palettes_offset, std::int64_t symbols_offset,
    std::int64_t scales_offset, bool fp32);

/** Apply a packed projection. */
torch::Tensor mxfp8_sq_matmul_cuda(
    torch::Tensor blob,
    torch::Tensor row_q,
    torch::Tensor row_symbol_byte_offsets,
    torch::Tensor input,
    std::int64_t outputs, std::int64_t width,
    std::int64_t block_rows, std::int64_t block_columns,
    std::int64_t scale_rows, std::int64_t scale_columns,
    std::int64_t palettes_offset, std::int64_t symbols_offset,
    std::int64_t scales_offset);

/** Apply a packed projection. */
torch::Tensor fp8_128_sq_matmul_cuda(
    torch::Tensor blob,
    torch::Tensor row_q,
    torch::Tensor row_symbol_byte_offsets,
    torch::Tensor input,
    std::int64_t outputs, std::int64_t width,
    std::int64_t scale_kind,
    std::int64_t palettes_offset, std::int64_t symbols_offset,
    std::int64_t scales_offset);

/** Compute the packed projection input gradient. */
torch::Tensor mxfp8_sq_backward_input_cuda(
    torch::Tensor blob,
    torch::Tensor row_q,
    torch::Tensor row_symbol_byte_offsets,
    torch::Tensor output_gradient,
    std::int64_t outputs, std::int64_t width,
    std::int64_t block_rows, std::int64_t block_columns,
    std::int64_t scale_rows, std::int64_t scale_columns,
    std::int64_t palettes_offset, std::int64_t symbols_offset,
    std::int64_t scales_offset);

/** Compute the packed projection input gradient. */
torch::Tensor fp8_128_sq_backward_input_cuda(
    torch::Tensor blob,
    torch::Tensor row_q,
    torch::Tensor row_symbol_byte_offsets,
    torch::Tensor output_gradient,
    std::int64_t outputs, std::int64_t width,
    std::int64_t scale_kind,
    std::int64_t palettes_offset, std::int64_t symbols_offset,
    std::int64_t scales_offset);

/** Apply routed packed weights to the supplied output. */
void mxfp8_sq_moe_matmul_cuda(
    torch::Tensor blob,
    torch::Tensor row_q,
    torch::Tensor row_symbol_byte_offsets,
    torch::Tensor input,
    torch::Tensor expert_ids,
    torch::Tensor expert_local,
    std::int64_t n_experts, std::int64_t local_experts,
    std::int64_t out_per_expert, std::int64_t width,
    std::int64_t block_rows, std::int64_t block_columns,
    std::int64_t scale_rows, std::int64_t scale_columns,
    std::int64_t palettes_offset, std::int64_t symbols_offset,
    std::int64_t scales_offset,
    torch::Tensor output);

/** Apply routed packed weights to the supplied output. */
void fp8_128_sq_moe_matmul_cuda(
    torch::Tensor blob,
    torch::Tensor row_q,
    torch::Tensor row_symbol_byte_offsets,
    torch::Tensor input,
    torch::Tensor expert_ids,
    torch::Tensor expert_local,
    std::int64_t n_experts, std::int64_t local_experts,
    std::int64_t out_per_expert, std::int64_t width,
    std::int64_t scale_kind,
    std::int64_t palettes_offset, std::int64_t symbols_offset,
    std::int64_t scales_offset,
    torch::Tensor output);
