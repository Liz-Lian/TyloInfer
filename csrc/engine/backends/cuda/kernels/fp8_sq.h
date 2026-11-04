#pragma once

#include <cstdint>

#include "backend.h"
#include "engine/weights/encoding/fp8_sq_blob.h"

// The CPU loader validates the self-describing wire payload and expands the
// compact three-bit q descriptor into row_q/row_symbol_byte_offsets.  CUDA
// never reads metadata back from the device and never keeps a full decoded
// weight resident.
::tyloi::DenseTensor mxfp8_sq_dequant_cuda(
    ::tyloi::DenseTensor blob,
    ::tyloi::DenseTensor row_q,
    ::tyloi::DenseTensor row_symbol_byte_offsets,
    std::int64_t outputs, std::int64_t width,
    std::int64_t block_rows, std::int64_t block_columns,
    std::int64_t scale_rows, std::int64_t scale_columns,
    std::int64_t palettes_offset, std::int64_t symbols_offset,
    std::int64_t scales_offset, bool fp32);

::tyloi::DenseTensor fp8_128_sq_dequant_cuda(
    ::tyloi::DenseTensor blob,
    ::tyloi::DenseTensor row_q,
    ::tyloi::DenseTensor row_symbol_byte_offsets,
    std::int64_t outputs, std::int64_t width,
    std::int64_t scale_kind,
    std::int64_t palettes_offset, std::int64_t symbols_offset,
    std::int64_t scales_offset, bool fp32);

::tyloi::DenseTensor mxfp8_sq_matmul_cuda(
    ::tyloi::DenseTensor blob,
    ::tyloi::DenseTensor row_q,
    ::tyloi::DenseTensor row_symbol_byte_offsets,
    ::tyloi::DenseTensor input,
    std::int64_t outputs, std::int64_t width,
    std::int64_t block_rows, std::int64_t block_columns,
    std::int64_t scale_rows, std::int64_t scale_columns,
    std::int64_t palettes_offset, std::int64_t symbols_offset,
    std::int64_t scales_offset);

::tyloi::DenseTensor fp8_128_sq_matmul_cuda(
    ::tyloi::DenseTensor blob,
    ::tyloi::DenseTensor row_q,
    ::tyloi::DenseTensor row_symbol_byte_offsets,
    ::tyloi::DenseTensor input,
    std::int64_t outputs, std::int64_t width,
    std::int64_t scale_kind,
    std::int64_t palettes_offset, std::int64_t symbols_offset,
    std::int64_t scales_offset);

::tyloi::DenseTensor mxfp8_sq_backward_input_cuda(
    ::tyloi::DenseTensor blob,
    ::tyloi::DenseTensor row_q,
    ::tyloi::DenseTensor row_symbol_byte_offsets,
    ::tyloi::DenseTensor output_gradient,
    std::int64_t outputs, std::int64_t width,
    std::int64_t block_rows, std::int64_t block_columns,
    std::int64_t scale_rows, std::int64_t scale_columns,
    std::int64_t palettes_offset, std::int64_t symbols_offset,
    std::int64_t scales_offset);

::tyloi::DenseTensor fp8_128_sq_backward_input_cuda(
    ::tyloi::DenseTensor blob,
    ::tyloi::DenseTensor row_q,
    ::tyloi::DenseTensor row_symbol_byte_offsets,
    ::tyloi::DenseTensor output_gradient,
    std::int64_t outputs, std::int64_t width,
    std::int64_t scale_kind,
    std::int64_t palettes_offset, std::int64_t symbols_offset,
    std::int64_t scales_offset);

void mxfp8_sq_moe_matmul_cuda(
    ::tyloi::DenseTensor blob,
    ::tyloi::DenseTensor row_q,
    ::tyloi::DenseTensor row_symbol_byte_offsets,
    ::tyloi::DenseTensor input,
    ::tyloi::DenseTensor expert_ids,
    ::tyloi::DenseTensor expert_local,
    std::int64_t n_experts, std::int64_t local_experts,
    std::int64_t out_per_expert, std::int64_t width,
    std::int64_t block_rows, std::int64_t block_columns,
    std::int64_t scale_rows, std::int64_t scale_columns,
    std::int64_t palettes_offset, std::int64_t symbols_offset,
    std::int64_t scales_offset,
    ::tyloi::DenseTensor output);

void fp8_128_sq_moe_matmul_cuda(
    ::tyloi::DenseTensor blob,
    ::tyloi::DenseTensor row_q,
    ::tyloi::DenseTensor row_symbol_byte_offsets,
    ::tyloi::DenseTensor input,
    ::tyloi::DenseTensor expert_ids,
    ::tyloi::DenseTensor expert_local,
    std::int64_t n_experts, std::int64_t local_experts,
    std::int64_t out_per_expert, std::int64_t width,
    std::int64_t scale_kind,
    std::int64_t palettes_offset, std::int64_t symbols_offset,
    std::int64_t scales_offset,
    ::tyloi::DenseTensor output);
