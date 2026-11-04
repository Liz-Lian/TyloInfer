#pragma once

#include "backend.h"

#include <cstdint>
#include <vector>

::tyloi::DenseTensor nint_matmul_online_f16_cuda(
    ::tyloi::DenseTensor bitstream,::tyloi::DenseTensor row_q_bits,
    ::tyloi::DenseTensor row_q_bit_offsets,::tyloi::DenseTensor subgroup_scale,
    ::tyloi::DenseTensor subgroup_minimum,::tyloi::DenseTensor neuron_scale,
    ::tyloi::DenseTensor neuron_minimum,::tyloi::DenseTensor input,int64_t group_size);

void nint_moe_grouped_matmul_hetero_cuda(
    ::tyloi::DenseTensor weight_ptrs,::tyloi::DenseTensor pool_params,
    ::tyloi::DenseTensor expert_pool,::tyloi::DenseTensor expert_local,
    ::tyloi::DenseTensor route_ids,::tyloi::DenseTensor output,
    int64_t input_width,bool routed_input);

::tyloi::DenseTensor nepq_hadamard_input_cuda(
    ::tyloi::DenseTensor input, ::tyloi::DenseTensor signs, int64_t block_size);
// Selected packed rows; descriptors use the common NintRowBatch protocol.
::tyloi::DenseTensor nint_selected_rows_cuda(
    ::tyloi::DenseTensor packed, ::tyloi::DenseTensor descriptors,
    int64_t width);
void nint_selected_rows_into_cuda(
    ::tyloi::DenseTensor packed, ::tyloi::DenseTensor descriptors,
    ::tyloi::DenseTensor inverse, ::tyloi::DenseTensor output);
std::vector<::tyloi::DenseTensor> nint8_one_quantize_reconstruct_cuda(
    ::tyloi::DenseTensor x);
::tyloi::DenseTensor nint8_zero_mmq_f16_packed_cuda(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor scale, ::tyloi::DenseTensor x,
    int64_t neuron_len);
::tyloi::DenseTensor nint8_zero_mmq_f32_packed_cuda(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor scale, ::tyloi::DenseTensor x,
    int64_t neuron_len);
// Packed GEMV accepts contiguous Half/Float activations (and gates). Float
// loads preserve the existing Half rounding inside the quantizer; output is
// always Half, with no intermediate converted activation tensor.
::tyloi::DenseTensor nint_matmul_ws_cuda(
    ::tyloi::DenseTensor q_packed, ::tyloi::DenseTensor row_q_bits,
    ::tyloi::DenseTensor row_q_bit_offsets,
    ::tyloi::DenseTensor sub_scale, ::tyloi::DenseTensor sub_min,
    ::tyloi::DenseTensor neuron_scale, ::tyloi::DenseTensor neuron_min,
    ::tyloi::DenseTensor x, int64_t gs,
    ::tyloi::DenseTensor qx, ::tyloi::DenseTensor xscale);
// Caller proves every row is q8 with a 32-bit-aligned row bit offset.
// Misaligned storage or a non-four-element group uses the generic kernel.
::tyloi::DenseTensor nint_matmul_q8_ws_cuda(
    ::tyloi::DenseTensor q_packed, ::tyloi::DenseTensor row_q_bits,
    ::tyloi::DenseTensor row_q_bit_offsets,
    ::tyloi::DenseTensor sub_scale, ::tyloi::DenseTensor sub_min,
    ::tyloi::DenseTensor neuron_scale, ::tyloi::DenseTensor neuron_min,
    ::tyloi::DenseTensor x, int64_t gs,
    ::tyloi::DenseTensor qx, ::tyloi::DenseTensor xscale);
::tyloi::DenseTensor nint_matmul_input_mul_ws_cuda(
    ::tyloi::DenseTensor q_packed, ::tyloi::DenseTensor row_q_bits,
    ::tyloi::DenseTensor row_q_bit_offsets,
    ::tyloi::DenseTensor sub_scale, ::tyloi::DenseTensor sub_min,
    ::tyloi::DenseTensor neuron_scale, ::tyloi::DenseTensor neuron_min,
    ::tyloi::DenseTensor x, ::tyloi::DenseTensor gate,
    int64_t activation_mode, int64_t gs,
    ::tyloi::DenseTensor qx, ::tyloi::DenseTensor xscale);
// Same all-row q8 proof and storage fallback as nint_matmul_q8_ws_cuda.
::tyloi::DenseTensor nint_matmul_input_mul_q8_ws_cuda(
    ::tyloi::DenseTensor q_packed, ::tyloi::DenseTensor row_q_bits,
    ::tyloi::DenseTensor row_q_bit_offsets,
    ::tyloi::DenseTensor sub_scale, ::tyloi::DenseTensor sub_min,
    ::tyloi::DenseTensor neuron_scale, ::tyloi::DenseTensor neuron_min,
    ::tyloi::DenseTensor x, ::tyloi::DenseTensor gate,
    int64_t activation_mode, int64_t gs,
    ::tyloi::DenseTensor qx, ::tyloi::DenseTensor xscale);
::tyloi::DenseTensor nint8_zero_gemv_ws_cuda(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor scale, ::tyloi::DenseTensor x,
    ::tyloi::DenseTensor qx, ::tyloi::DenseTensor xscale);
::tyloi::DenseTensor nint8_zero_dequant_cuda(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor scale, int64_t neuron_len);
::tyloi::DenseTensor nint8_zero_dequant_f32_cuda(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor scale, int64_t neuron_len);
::tyloi::DenseTensor nint_decode_cuda(
    ::tyloi::DenseTensor q_packed, ::tyloi::DenseTensor row_q_bits,
    ::tyloi::DenseTensor row_q_bit_offsets,
    ::tyloi::DenseTensor sub_scale, ::tyloi::DenseTensor sub_min,
    ::tyloi::DenseTensor neuron_scale, ::tyloi::DenseTensor neuron_min,
    int64_t neuron_len, int64_t gs);
::tyloi::DenseTensor nint_decode_f32_cuda(
    ::tyloi::DenseTensor q_packed, ::tyloi::DenseTensor row_q_bits,
    ::tyloi::DenseTensor row_q_bit_offsets,
    ::tyloi::DenseTensor sub_scale, ::tyloi::DenseTensor sub_min,
    ::tyloi::DenseTensor neuron_scale, ::tyloi::DenseTensor neuron_min,
    int64_t neuron_len, int64_t gs);
::tyloi::DenseTensor nint_cublas_gemm_nt_f32acc_cuda(::tyloi::DenseTensor x, ::tyloi::DenseTensor w);
::tyloi::DenseTensor mxfp8_dequant_cuda(
    ::tyloi::DenseTensor values, ::tyloi::DenseTensor scales);
::tyloi::DenseTensor mxfp8_small_m_cuda(
    ::tyloi::DenseTensor values, ::tyloi::DenseTensor scales, ::tyloi::DenseTensor x);
::tyloi::DenseTensor mxfp8_matmul_f16_cuda(
    ::tyloi::DenseTensor values, ::tyloi::DenseTensor scales, ::tyloi::DenseTensor input);
::tyloi::DenseTensor mxfp8_small_m_f32_cuda(
    ::tyloi::DenseTensor values, ::tyloi::DenseTensor scales, ::tyloi::DenseTensor x);
::tyloi::DenseTensor mxfp8_gemm_f32_cuda(
    ::tyloi::DenseTensor values, ::tyloi::DenseTensor scales, ::tyloi::DenseTensor x);
::tyloi::DenseTensor mxfp8_groupwise_small_m_cuda(
    ::tyloi::DenseTensor values, ::tyloi::DenseTensor scales,
    ::tyloi::DenseTensor x, int64_t groups);
::tyloi::DenseTensor mxfp8_groupwise_small_m_f32_cuda(
    ::tyloi::DenseTensor values, ::tyloi::DenseTensor scales,
    ::tyloi::DenseTensor x, int64_t groups);
::tyloi::DenseTensor mxfp4_dequant_cuda(
    ::tyloi::DenseTensor values, ::tyloi::DenseTensor scales);
::tyloi::DenseTensor mxfp4_matmul_f16_cuda(
    ::tyloi::DenseTensor values, ::tyloi::DenseTensor scales, ::tyloi::DenseTensor input);
::tyloi::DenseTensor nvq_dequant_cuda(
    ::tyloi::DenseTensor indices, ::tyloi::DenseTensor aux, ::tyloi::DenseTensor sub_scale,
    ::tyloi::DenseTensor neuron_scale, ::tyloi::DenseTensor codebook,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode);
::tyloi::DenseTensor nepq_dequant_cuda(
    ::tyloi::DenseTensor indices, ::tyloi::DenseTensor aux, ::tyloi::DenseTensor sub_scale,
    ::tyloi::DenseTensor neuron_scale, ::tyloi::DenseTensor table_pool,
    ::tyloi::DenseTensor bank_ids, int64_t neuron_len,
    int64_t sub_bits, int64_t format);
::tyloi::DenseTensor nepq_sparse_residual_dequant_cuda(
    ::tyloi::DenseTensor dictionary, ::tyloi::DenseTensor first, ::tyloi::DenseTensor second,
    int64_t position_bits, int64_t block_vectors, ::tyloi::DenseTensor weight);
::tyloi::DenseTensor nvq_gemm_f16_cuda(
    ::tyloi::DenseTensor indices, ::tyloi::DenseTensor aux, ::tyloi::DenseTensor sub_scale,
    ::tyloi::DenseTensor neuron_scale, ::tyloi::DenseTensor codebook, ::tyloi::DenseTensor x,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode);
::tyloi::DenseTensor nvq_gemv_ws_cuda(
    ::tyloi::DenseTensor indices, ::tyloi::DenseTensor aux, ::tyloi::DenseTensor sub_scale,
    ::tyloi::DenseTensor neuron_scale, ::tyloi::DenseTensor codebook, ::tyloi::DenseTensor x,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode,
    ::tyloi::DenseTensor qx, ::tyloi::DenseTensor xscale);
::tyloi::DenseTensor nvq_gemv_qx_ws_cuda(
    ::tyloi::DenseTensor indices, ::tyloi::DenseTensor aux, ::tyloi::DenseTensor sub_scale,
    ::tyloi::DenseTensor neuron_scale, ::tyloi::DenseTensor codebook,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode,
    ::tyloi::DenseTensor qx, ::tyloi::DenseTensor xscale);
::tyloi::DenseTensor nvq_gemv_qx_residual_ws_cuda(
    ::tyloi::DenseTensor indices, ::tyloi::DenseTensor aux, ::tyloi::DenseTensor sub_scale,
    ::tyloi::DenseTensor neuron_scale, ::tyloi::DenseTensor codebook,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format,
    int64_t sign_mode, ::tyloi::DenseTensor qx, ::tyloi::DenseTensor xscale,
    ::tyloi::DenseTensor residual);
::tyloi::DenseTensor nvq_gemv_multi2_ws_cuda(
    ::tyloi::DenseTensor first_indices, ::tyloi::DenseTensor first_aux, ::tyloi::DenseTensor first_sub_scale,
    ::tyloi::DenseTensor first_neuron_scale, ::tyloi::DenseTensor first_codebook,
    ::tyloi::DenseTensor second_indices, ::tyloi::DenseTensor second_aux, ::tyloi::DenseTensor second_sub_scale,
    ::tyloi::DenseTensor second_neuron_scale, ::tyloi::DenseTensor second_codebook,
    ::tyloi::DenseTensor x, int64_t neuron_len, int64_t gs,
    int64_t first_sub_bits, int64_t first_format, int64_t first_sign_mode,
    int64_t second_sub_bits, int64_t second_format, int64_t second_sign_mode,
    ::tyloi::DenseTensor qx, ::tyloi::DenseTensor xscale);
::tyloi::DenseTensor nvq_gemv_swiglu_ws_cuda(
    ::tyloi::DenseTensor gate_indices, ::tyloi::DenseTensor gate_aux, ::tyloi::DenseTensor gate_sub_scale,
    ::tyloi::DenseTensor gate_neuron_scale, ::tyloi::DenseTensor gate_codebook,
    ::tyloi::DenseTensor up_indices, ::tyloi::DenseTensor up_aux, ::tyloi::DenseTensor up_sub_scale,
    ::tyloi::DenseTensor up_neuron_scale, ::tyloi::DenseTensor up_codebook,
    ::tyloi::DenseTensor x, int64_t neuron_len, int64_t gs,
    int64_t gate_sub_bits, int64_t gate_format, int64_t gate_sign_mode,
    int64_t up_sub_bits, int64_t up_format, int64_t up_sign_mode,
    ::tyloi::DenseTensor qx, ::tyloi::DenseTensor xscale);
void nvq_ffn_swiglu_quant_ws_cuda(
    ::tyloi::DenseTensor gate_indices, ::tyloi::DenseTensor gate_aux, ::tyloi::DenseTensor gate_sub_scale,
    ::tyloi::DenseTensor gate_neuron_scale, ::tyloi::DenseTensor gate_codebook,
    ::tyloi::DenseTensor up_indices, ::tyloi::DenseTensor up_aux, ::tyloi::DenseTensor up_sub_scale,
    ::tyloi::DenseTensor up_neuron_scale, ::tyloi::DenseTensor up_codebook,
    ::tyloi::DenseTensor x, int64_t neuron_len, int64_t gs,
    int64_t gate_sub_bits, int64_t gate_format, int64_t gate_sign_mode,
    int64_t up_sub_bits, int64_t up_format, int64_t up_sign_mode, int64_t down_gs,
    ::tyloi::DenseTensor input_qx, ::tyloi::DenseTensor input_xscale,
    ::tyloi::DenseTensor output_qx, ::tyloi::DenseTensor output_xscale, ::tyloi::DenseTensor swiglu_scratch);
::tyloi::DenseTensor nvq_gemv_gate_ws_cuda(
    ::tyloi::DenseTensor indices, ::tyloi::DenseTensor aux, ::tyloi::DenseTensor sub_scale,
    ::tyloi::DenseTensor neuron_scale, ::tyloi::DenseTensor codebook, ::tyloi::DenseTensor x, ::tyloi::DenseTensor gate,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode,
    int64_t mode, ::tyloi::DenseTensor qx, ::tyloi::DenseTensor xscale);
::tyloi::DenseTensor nvq_mmq_ws_cuda(
    ::tyloi::DenseTensor indices, ::tyloi::DenseTensor aux, ::tyloi::DenseTensor sub_scale,
    ::tyloi::DenseTensor neuron_scale, ::tyloi::DenseTensor codebook, ::tyloi::DenseTensor x,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode,
    ::tyloi::DenseTensor qx, ::tyloi::DenseTensor xscale);
::tyloi::DenseTensor nvq_mmq_gate_ws_cuda(
    ::tyloi::DenseTensor indices, ::tyloi::DenseTensor aux, ::tyloi::DenseTensor sub_scale,
    ::tyloi::DenseTensor neuron_scale, ::tyloi::DenseTensor codebook, ::tyloi::DenseTensor x, ::tyloi::DenseTensor gate,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode,
    int64_t mode, ::tyloi::DenseTensor qx, ::tyloi::DenseTensor xscale);
