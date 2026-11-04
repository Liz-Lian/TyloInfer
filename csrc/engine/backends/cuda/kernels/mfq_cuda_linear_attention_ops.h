#pragma once

#include "backend.h"

#include <cstdint>
#include <vector>

std::vector<::tyloi::Tensor> linear_gate_beta_cuda(
    ::tyloi::Tensor alpha, ::tyloi::Tensor beta, ::tyloi::Tensor dt_bias, ::tyloi::Tensor a_log,
    bool stable_softplus = false);
std::vector<::tyloi::Tensor> gdn_cuda(::tyloi::Tensor q, ::tyloi::Tensor k, ::tyloi::Tensor v,
                                    ::tyloi::Tensor g, ::tyloi::Tensor beta, MfqOptional<::tyloi::Tensor> state);
std::vector<::tyloi::Tensor> gdn_transposed_cuda(
    ::tyloi::Tensor q, ::tyloi::Tensor k, ::tyloi::Tensor v,
    ::tyloi::Tensor g, ::tyloi::Tensor beta, MfqOptional<::tyloi::Tensor> state);
std::vector<::tyloi::Tensor> linear_conv_qkv_gate_short_cuda(
    ::tyloi::Tensor state,::tyloi::Tensor qkv,
    ::tyloi::Tensor weight,::tyloi::Tensor alpha,
    ::tyloi::Tensor beta,::tyloi::Tensor dt_bias,
    ::tyloi::Tensor a_log,int64_t nk,int64_t nv,int64_t width,double eps);
std::vector<::tyloi::Tensor> gdn_inplace_cuda(::tyloi::Tensor q, ::tyloi::Tensor k, ::tyloi::Tensor v,
                                             ::tyloi::Tensor g, ::tyloi::Tensor beta, ::tyloi::Tensor state);
std::vector<::tyloi::Tensor> gdn_inplace_transposed_cuda(
    ::tyloi::Tensor q, ::tyloi::Tensor k, ::tyloi::Tensor v,
    ::tyloi::Tensor g, ::tyloi::Tensor beta, ::tyloi::Tensor state);
std::vector<::tyloi::Tensor> gdn_inplace_tiled_cuda(
    ::tyloi::Tensor q, ::tyloi::Tensor k, ::tyloi::Tensor v,
    ::tyloi::Tensor g, ::tyloi::Tensor beta, ::tyloi::Tensor state);
std::vector<::tyloi::Tensor> gdn_inplace_transposed_tiled_cuda(
    ::tyloi::Tensor q, ::tyloi::Tensor k, ::tyloi::Tensor v,
    ::tyloi::Tensor g, ::tyloi::Tensor beta, ::tyloi::Tensor state);
::tyloi::Tensor ssm_conv_silu_cuda(::tyloi::Tensor conv_input, ::tyloi::Tensor weight, ::tyloi::Tensor bias, int64_t n_tokens);
::tyloi::Tensor ssm_conv_silu_decode_cuda(::tyloi::Tensor state, ::tyloi::Tensor x, ::tyloi::Tensor weight, ::tyloi::Tensor bias);
std::vector<::tyloi::Tensor> linear_conv_qkv_decode_cuda(
    ::tyloi::Tensor state, ::tyloi::Tensor qk, ::tyloi::Tensor v, ::tyloi::Tensor weight, ::tyloi::Tensor bias,
    int64_t nk, int64_t nv, int64_t dk, int64_t dv, double eps);
std::vector<::tyloi::Tensor> linear_conv_qkv_prefill_cuda(
    ::tyloi::Tensor state, ::tyloi::Tensor qk, ::tyloi::Tensor v, ::tyloi::Tensor weight, ::tyloi::Tensor bias,
    int64_t nk, int64_t nv, int64_t dk, int64_t dv, double eps);
std::vector<::tyloi::Tensor> linear_conv_qkv_gate_decode_cuda(
    ::tyloi::Tensor state, ::tyloi::Tensor qk, ::tyloi::Tensor v,
    ::tyloi::Tensor weight, ::tyloi::Tensor alpha, ::tyloi::Tensor beta,
    ::tyloi::Tensor dt_bias, ::tyloi::Tensor a_log,
    int64_t nk, int64_t nv, int64_t dk, int64_t dv, double eps);

// Workspace counters start at zero and reset after each completed decode.
// With inplace_state, every CTA owns disjoint recurrent columns and returns the
// updated input storage. The caller must exclusively own that recurrent state.
// Optional CUDA int64 [B,3] addresses select independently owned convolution,
// recurrent states and counters. This requires inplace_state; convolution is committed
// after every core CTA completes, and the returned state aliases row zero.
std::vector<::tyloi::Tensor> gdn_decode_core_cuda(
    ::tyloi::Tensor qkv, ::tyloi::Tensor output_gate,
    ::tyloi::Tensor alpha, ::tyloi::Tensor beta,
    ::tyloi::Tensor convolution_state, ::tyloi::Tensor recurrent_state,
    ::tyloi::Tensor convolution_weight, ::tyloi::Tensor dt_bias,
    ::tyloi::Tensor a_log, ::tyloi::Tensor norm_weight,
    int64_t key_heads, int64_t value_heads, int64_t width,
    double convolution_eps, double norm_eps, bool silu_gate, bool output_half,
    bool transposed_state,
    ::tyloi::Tensor workspace={},bool inplace_state=false,
    ::tyloi::Tensor state_addresses={});
// Write speculative prefix snapshots with the producing core; no state-copy launches.
// Batch one speculative sequence (2..8 rows, width128/taps4). The recurrence
// stays in registers; prefix snapshots retain the existing rollback contract.
// Recurrent state is updated in place. Returned convolution state is separate
// and must be committed only after the window kernel completes on its stream.
std::vector<::tyloi::Tensor> gdn_decode_window_cuda(
    const ::tyloi::Tensor& qkv,const ::tyloi::Tensor& output_gate,
    const ::tyloi::Tensor& alpha,const ::tyloi::Tensor& beta,
    const ::tyloi::Tensor& convolution_state,const ::tyloi::Tensor& recurrent_state,
    const ::tyloi::Tensor& convolution_weight,const ::tyloi::Tensor& dt_bias,
    const ::tyloi::Tensor& a_log,const ::tyloi::Tensor& norm_weight,
    int key_heads,int value_heads,int confirmed,double convolution_eps,double norm_eps,
    bool silu_gate,bool output_half,bool transposed_state,
    const std::vector<::tyloi::Tensor>& checkpoint_convolution,
    const std::vector<::tyloi::Tensor>& checkpoint_recurrent);

std::vector<::tyloi::Tensor> gdn_decode_core_checkpoint_cuda(
    ::tyloi::Tensor qkv, ::tyloi::Tensor output_gate,
    ::tyloi::Tensor alpha, ::tyloi::Tensor beta,
    ::tyloi::Tensor convolution_state, ::tyloi::Tensor recurrent_state,
    ::tyloi::Tensor convolution_weight, ::tyloi::Tensor dt_bias,
    ::tyloi::Tensor a_log, ::tyloi::Tensor norm_weight,
    int64_t key_heads, int64_t value_heads, int64_t width,
    double convolution_eps, double norm_eps, bool silu_gate, bool output_half,
    bool transposed_state,
    ::tyloi::Tensor workspace,bool inplace_state,
    ::tyloi::Tensor checkpoint_convolution,
    ::tyloi::Tensor checkpoint_recurrent);
