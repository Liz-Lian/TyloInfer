#pragma once

#include "backend.h"

#include <cstdint>
#include <vector>

::tyloi::Tensor nint_matmul_online_f16_cuda(
    ::tyloi::Tensor bitstream,::tyloi::Tensor row_q_bits,
    ::tyloi::Tensor row_q_bit_offsets,::tyloi::Tensor subgroup_scale,
    ::tyloi::Tensor subgroup_minimum,::tyloi::Tensor neuron_scale,
    ::tyloi::Tensor neuron_minimum,::tyloi::Tensor input,int64_t group_size);

void nint_moe_grouped_matmul_hetero_cuda(
    ::tyloi::Tensor weight_ptrs,::tyloi::Tensor pool_params,
    ::tyloi::Tensor expert_pool,::tyloi::Tensor expert_local,
    ::tyloi::Tensor route_ids,::tyloi::Tensor output,
    int64_t input_width,bool routed_input);

::tyloi::Tensor nepq_hadamard_input_cuda(
    ::tyloi::Tensor input, ::tyloi::Tensor signs, int64_t block_size);
// Selected packed rows; descriptors use the common NintRowBatch protocol.
::tyloi::Tensor nint_selected_rows_cuda(
    ::tyloi::Tensor packed, ::tyloi::Tensor descriptors,
    int64_t width);
void nint_selected_rows_into_cuda(
    ::tyloi::Tensor packed, ::tyloi::Tensor descriptors,
    ::tyloi::Tensor inverse, ::tyloi::Tensor output);
std::vector<::tyloi::Tensor> nint8_one_quantize_reconstruct_cuda(
    ::tyloi::Tensor x);
::tyloi::Tensor nint8_zero_mmq_f16_packed_cuda(
    ::tyloi::Tensor q, ::tyloi::Tensor scale, ::tyloi::Tensor x,
    int64_t neuron_len);
::tyloi::Tensor nint8_zero_mmq_f32_packed_cuda(
    ::tyloi::Tensor q, ::tyloi::Tensor scale, ::tyloi::Tensor x,
    int64_t neuron_len);
// Packed GEMV accepts contiguous Half/Float activations (and gates). Float
// loads preserve the existing Half rounding inside the quantizer; output is
// always Half, with no intermediate converted activation tensor.
::tyloi::Tensor nint_matmul_ws_cuda(
    ::tyloi::Tensor q_packed, ::tyloi::Tensor row_q_bits,
    ::tyloi::Tensor row_q_bit_offsets,
    ::tyloi::Tensor sub_scale, ::tyloi::Tensor sub_min,
    ::tyloi::Tensor neuron_scale, ::tyloi::Tensor neuron_min,
    ::tyloi::Tensor x, int64_t gs,
    ::tyloi::Tensor qx, ::tyloi::Tensor xscale);
// Caller proves every row is q8 with a 32-bit-aligned row bit offset.
// Misaligned storage or a non-four-element group uses the generic kernel.
::tyloi::Tensor nint_matmul_q8_ws_cuda(
    ::tyloi::Tensor q_packed, ::tyloi::Tensor row_q_bits,
    ::tyloi::Tensor row_q_bit_offsets,
    ::tyloi::Tensor sub_scale, ::tyloi::Tensor sub_min,
    ::tyloi::Tensor neuron_scale, ::tyloi::Tensor neuron_min,
    ::tyloi::Tensor x, int64_t gs,
    ::tyloi::Tensor qx, ::tyloi::Tensor xscale);
::tyloi::Tensor nint_matmul_input_mul_ws_cuda(
    ::tyloi::Tensor q_packed, ::tyloi::Tensor row_q_bits,
    ::tyloi::Tensor row_q_bit_offsets,
    ::tyloi::Tensor sub_scale, ::tyloi::Tensor sub_min,
    ::tyloi::Tensor neuron_scale, ::tyloi::Tensor neuron_min,
    ::tyloi::Tensor x, ::tyloi::Tensor gate,
    int64_t activation_mode, int64_t gs,
    ::tyloi::Tensor qx, ::tyloi::Tensor xscale);
// Same all-row q8 proof and storage fallback as nint_matmul_q8_ws_cuda.
::tyloi::Tensor nint_matmul_input_mul_q8_ws_cuda(
    ::tyloi::Tensor q_packed, ::tyloi::Tensor row_q_bits,
    ::tyloi::Tensor row_q_bit_offsets,
    ::tyloi::Tensor sub_scale, ::tyloi::Tensor sub_min,
    ::tyloi::Tensor neuron_scale, ::tyloi::Tensor neuron_min,
    ::tyloi::Tensor x, ::tyloi::Tensor gate,
    int64_t activation_mode, int64_t gs,
    ::tyloi::Tensor qx, ::tyloi::Tensor xscale);
::tyloi::Tensor nint8_zero_gemv_ws_cuda(
    ::tyloi::Tensor q, ::tyloi::Tensor scale, ::tyloi::Tensor x,
    ::tyloi::Tensor qx, ::tyloi::Tensor xscale);
::tyloi::Tensor nint8_zero_dequant_cuda(
    ::tyloi::Tensor q, ::tyloi::Tensor scale, int64_t neuron_len);
::tyloi::Tensor nint8_zero_dequant_f32_cuda(
    ::tyloi::Tensor q, ::tyloi::Tensor scale, int64_t neuron_len);
::tyloi::Tensor nint_decode_cuda(
    ::tyloi::Tensor q_packed, ::tyloi::Tensor row_q_bits,
    ::tyloi::Tensor row_q_bit_offsets,
    ::tyloi::Tensor sub_scale, ::tyloi::Tensor sub_min,
    ::tyloi::Tensor neuron_scale, ::tyloi::Tensor neuron_min,
    int64_t neuron_len, int64_t gs);
::tyloi::Tensor nint_decode_f32_cuda(
    ::tyloi::Tensor q_packed, ::tyloi::Tensor row_q_bits,
    ::tyloi::Tensor row_q_bit_offsets,
    ::tyloi::Tensor sub_scale, ::tyloi::Tensor sub_min,
    ::tyloi::Tensor neuron_scale, ::tyloi::Tensor neuron_min,
    int64_t neuron_len, int64_t gs);
::tyloi::Tensor nint_cublas_gemm_nt_f32acc_cuda(::tyloi::Tensor x, ::tyloi::Tensor w);
::tyloi::Tensor mxfp8_dequant_cuda(
    ::tyloi::Tensor values, ::tyloi::Tensor scales);
::tyloi::Tensor mxfp8_small_m_cuda(
    ::tyloi::Tensor values, ::tyloi::Tensor scales, ::tyloi::Tensor x);
::tyloi::Tensor mxfp8_matmul_f16_cuda(
    ::tyloi::Tensor values, ::tyloi::Tensor scales, ::tyloi::Tensor input);
::tyloi::Tensor mxfp8_small_m_f32_cuda(
    ::tyloi::Tensor values, ::tyloi::Tensor scales, ::tyloi::Tensor x);
::tyloi::Tensor mxfp8_gemm_f32_cuda(
    ::tyloi::Tensor values, ::tyloi::Tensor scales, ::tyloi::Tensor x);
::tyloi::Tensor mxfp8_groupwise_small_m_cuda(
    ::tyloi::Tensor values, ::tyloi::Tensor scales,
    ::tyloi::Tensor x, int64_t groups);
::tyloi::Tensor mxfp8_groupwise_small_m_f32_cuda(
    ::tyloi::Tensor values, ::tyloi::Tensor scales,
    ::tyloi::Tensor x, int64_t groups);
::tyloi::Tensor mxfp4_dequant_cuda(
    ::tyloi::Tensor values, ::tyloi::Tensor scales);
::tyloi::Tensor mxfp4_matmul_f16_cuda(
    ::tyloi::Tensor values, ::tyloi::Tensor scales, ::tyloi::Tensor input);
::tyloi::Tensor nvq_dequant_cuda(
    ::tyloi::Tensor indices, ::tyloi::Tensor aux, ::tyloi::Tensor sub_scale,
    ::tyloi::Tensor neuron_scale, ::tyloi::Tensor codebook,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode);
::tyloi::Tensor nepq_dequant_cuda(
    ::tyloi::Tensor indices, ::tyloi::Tensor aux, ::tyloi::Tensor sub_scale,
    ::tyloi::Tensor neuron_scale, ::tyloi::Tensor table_pool,
    ::tyloi::Tensor bank_ids, int64_t neuron_len,
    int64_t sub_bits, int64_t format);
::tyloi::Tensor nepq_sparse_residual_dequant_cuda(
    ::tyloi::Tensor dictionary, ::tyloi::Tensor first, ::tyloi::Tensor second,
    int64_t position_bits, int64_t block_vectors, ::tyloi::Tensor weight);
::tyloi::Tensor nvq_gemm_f16_cuda(
    ::tyloi::Tensor indices, ::tyloi::Tensor aux, ::tyloi::Tensor sub_scale,
    ::tyloi::Tensor neuron_scale, ::tyloi::Tensor codebook, ::tyloi::Tensor x,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode);
::tyloi::Tensor nvq_gemv_ws_cuda(
    ::tyloi::Tensor indices, ::tyloi::Tensor aux, ::tyloi::Tensor sub_scale,
    ::tyloi::Tensor neuron_scale, ::tyloi::Tensor codebook, ::tyloi::Tensor x,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode,
    ::tyloi::Tensor qx, ::tyloi::Tensor xscale);
::tyloi::Tensor nvq_gemv_qx_ws_cuda(
    ::tyloi::Tensor indices, ::tyloi::Tensor aux, ::tyloi::Tensor sub_scale,
    ::tyloi::Tensor neuron_scale, ::tyloi::Tensor codebook,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode,
    ::tyloi::Tensor qx, ::tyloi::Tensor xscale);
::tyloi::Tensor nvq_gemv_qx_residual_ws_cuda(
    ::tyloi::Tensor indices, ::tyloi::Tensor aux, ::tyloi::Tensor sub_scale,
    ::tyloi::Tensor neuron_scale, ::tyloi::Tensor codebook,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format,
    int64_t sign_mode, ::tyloi::Tensor qx, ::tyloi::Tensor xscale,
    ::tyloi::Tensor residual);
::tyloi::Tensor nvq_gemv_multi2_ws_cuda(
    ::tyloi::Tensor first_indices, ::tyloi::Tensor first_aux, ::tyloi::Tensor first_sub_scale,
    ::tyloi::Tensor first_neuron_scale, ::tyloi::Tensor first_codebook,
    ::tyloi::Tensor second_indices, ::tyloi::Tensor second_aux, ::tyloi::Tensor second_sub_scale,
    ::tyloi::Tensor second_neuron_scale, ::tyloi::Tensor second_codebook,
    ::tyloi::Tensor x, int64_t neuron_len, int64_t gs,
    int64_t first_sub_bits, int64_t first_format, int64_t first_sign_mode,
    int64_t second_sub_bits, int64_t second_format, int64_t second_sign_mode,
    ::tyloi::Tensor qx, ::tyloi::Tensor xscale);
::tyloi::Tensor nvq_gemv_swiglu_ws_cuda(
    ::tyloi::Tensor gate_indices, ::tyloi::Tensor gate_aux, ::tyloi::Tensor gate_sub_scale,
    ::tyloi::Tensor gate_neuron_scale, ::tyloi::Tensor gate_codebook,
    ::tyloi::Tensor up_indices, ::tyloi::Tensor up_aux, ::tyloi::Tensor up_sub_scale,
    ::tyloi::Tensor up_neuron_scale, ::tyloi::Tensor up_codebook,
    ::tyloi::Tensor x, int64_t neuron_len, int64_t gs,
    int64_t gate_sub_bits, int64_t gate_format, int64_t gate_sign_mode,
    int64_t up_sub_bits, int64_t up_format, int64_t up_sign_mode,
    ::tyloi::Tensor qx, ::tyloi::Tensor xscale);
void nvq_ffn_swiglu_quant_ws_cuda(
    ::tyloi::Tensor gate_indices, ::tyloi::Tensor gate_aux, ::tyloi::Tensor gate_sub_scale,
    ::tyloi::Tensor gate_neuron_scale, ::tyloi::Tensor gate_codebook,
    ::tyloi::Tensor up_indices, ::tyloi::Tensor up_aux, ::tyloi::Tensor up_sub_scale,
    ::tyloi::Tensor up_neuron_scale, ::tyloi::Tensor up_codebook,
    ::tyloi::Tensor x, int64_t neuron_len, int64_t gs,
    int64_t gate_sub_bits, int64_t gate_format, int64_t gate_sign_mode,
    int64_t up_sub_bits, int64_t up_format, int64_t up_sign_mode, int64_t down_gs,
    ::tyloi::Tensor input_qx, ::tyloi::Tensor input_xscale,
    ::tyloi::Tensor output_qx, ::tyloi::Tensor output_xscale, ::tyloi::Tensor swiglu_scratch);
::tyloi::Tensor nvq_gemv_gate_ws_cuda(
    ::tyloi::Tensor indices, ::tyloi::Tensor aux, ::tyloi::Tensor sub_scale,
    ::tyloi::Tensor neuron_scale, ::tyloi::Tensor codebook, ::tyloi::Tensor x, ::tyloi::Tensor gate,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode,
    int64_t mode, ::tyloi::Tensor qx, ::tyloi::Tensor xscale);
::tyloi::Tensor nvq_mmq_ws_cuda(
    ::tyloi::Tensor indices, ::tyloi::Tensor aux, ::tyloi::Tensor sub_scale,
    ::tyloi::Tensor neuron_scale, ::tyloi::Tensor codebook, ::tyloi::Tensor x,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode,
    ::tyloi::Tensor qx, ::tyloi::Tensor xscale);
::tyloi::Tensor nvq_mmq_gate_ws_cuda(
    ::tyloi::Tensor indices, ::tyloi::Tensor aux, ::tyloi::Tensor sub_scale,
    ::tyloi::Tensor neuron_scale, ::tyloi::Tensor codebook, ::tyloi::Tensor x, ::tyloi::Tensor gate,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode,
    int64_t mode, ::tyloi::Tensor qx, ::tyloi::Tensor xscale);
