#pragma once

#include "backend.h"

#include <cstdint>
#include <vector>

std::vector<::tyloi::DenseTensor> linear_gate_beta_cuda(
    ::tyloi::DenseTensor alpha, ::tyloi::DenseTensor beta, ::tyloi::DenseTensor dt_bias, ::tyloi::DenseTensor a_log,
    bool stable_softplus = false);
std::vector<::tyloi::DenseTensor> gdn_cuda(::tyloi::DenseTensor q, ::tyloi::DenseTensor k, ::tyloi::DenseTensor v,
                                    ::tyloi::DenseTensor g, ::tyloi::DenseTensor beta, MfqOptional<::tyloi::DenseTensor> state);
std::vector<::tyloi::DenseTensor> gdn_transposed_cuda(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor k, ::tyloi::DenseTensor v,
    ::tyloi::DenseTensor g, ::tyloi::DenseTensor beta, MfqOptional<::tyloi::DenseTensor> state);
std::vector<::tyloi::DenseTensor> linear_conv_qkv_gate_short_cuda(
    ::tyloi::DenseTensor state,::tyloi::DenseTensor qkv,
    ::tyloi::DenseTensor weight,::tyloi::DenseTensor alpha,
    ::tyloi::DenseTensor beta,::tyloi::DenseTensor dt_bias,
    ::tyloi::DenseTensor a_log,int64_t nk,int64_t nv,int64_t width,double eps);
std::vector<::tyloi::DenseTensor> gdn_inplace_cuda(::tyloi::DenseTensor q, ::tyloi::DenseTensor k, ::tyloi::DenseTensor v,
                                             ::tyloi::DenseTensor g, ::tyloi::DenseTensor beta, ::tyloi::DenseTensor state);
std::vector<::tyloi::DenseTensor> gdn_inplace_transposed_cuda(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor k, ::tyloi::DenseTensor v,
    ::tyloi::DenseTensor g, ::tyloi::DenseTensor beta, ::tyloi::DenseTensor state);
std::vector<::tyloi::DenseTensor> gdn_inplace_tiled_cuda(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor k, ::tyloi::DenseTensor v,
    ::tyloi::DenseTensor g, ::tyloi::DenseTensor beta, ::tyloi::DenseTensor state);
std::vector<::tyloi::DenseTensor> gdn_inplace_transposed_tiled_cuda(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor k, ::tyloi::DenseTensor v,
    ::tyloi::DenseTensor g, ::tyloi::DenseTensor beta, ::tyloi::DenseTensor state);
::tyloi::DenseTensor ssm_conv_silu_cuda(::tyloi::DenseTensor conv_input, ::tyloi::DenseTensor weight, ::tyloi::DenseTensor bias, int64_t n_tokens);
::tyloi::DenseTensor ssm_conv_silu_decode_cuda(::tyloi::DenseTensor state, ::tyloi::DenseTensor x, ::tyloi::DenseTensor weight, ::tyloi::DenseTensor bias);
std::vector<::tyloi::DenseTensor> linear_conv_qkv_decode_cuda(
    ::tyloi::DenseTensor state, ::tyloi::DenseTensor qk, ::tyloi::DenseTensor v, ::tyloi::DenseTensor weight, ::tyloi::DenseTensor bias,
    int64_t nk, int64_t nv, int64_t dk, int64_t dv, double eps);
std::vector<::tyloi::DenseTensor> linear_conv_qkv_prefill_cuda(
    ::tyloi::DenseTensor state, ::tyloi::DenseTensor qk, ::tyloi::DenseTensor v, ::tyloi::DenseTensor weight, ::tyloi::DenseTensor bias,
    int64_t nk, int64_t nv, int64_t dk, int64_t dv, double eps);
std::vector<::tyloi::DenseTensor> linear_conv_qkv_gate_decode_cuda(
    ::tyloi::DenseTensor state, ::tyloi::DenseTensor qk, ::tyloi::DenseTensor v,
    ::tyloi::DenseTensor weight, ::tyloi::DenseTensor alpha, ::tyloi::DenseTensor beta,
    ::tyloi::DenseTensor dt_bias, ::tyloi::DenseTensor a_log,
    int64_t nk, int64_t nv, int64_t dk, int64_t dv, double eps);

// Workspace counters start at zero and reset after each completed decode.
// With inplace_state, every CTA owns disjoint recurrent columns and returns the
// updated input storage. The caller must exclusively own that recurrent state.
// Optional CUDA int64 [B,3] addresses select independently owned convolution,
// recurrent states and counters. This requires inplace_state; convolution is committed
// after every core CTA completes, and the returned state aliases row zero.
std::vector<::tyloi::DenseTensor> gdn_decode_core_cuda(
    ::tyloi::DenseTensor qkv, ::tyloi::DenseTensor output_gate,
    ::tyloi::DenseTensor alpha, ::tyloi::DenseTensor beta,
    ::tyloi::DenseTensor convolution_state, ::tyloi::DenseTensor recurrent_state,
    ::tyloi::DenseTensor convolution_weight, ::tyloi::DenseTensor dt_bias,
    ::tyloi::DenseTensor a_log, ::tyloi::DenseTensor norm_weight,
    int64_t key_heads, int64_t value_heads, int64_t width,
    double convolution_eps, double norm_eps, bool silu_gate, bool output_half,
    bool transposed_state,
    ::tyloi::DenseTensor workspace={},bool inplace_state=false,
    ::tyloi::DenseTensor state_addresses={});
// Write speculative prefix snapshots with the producing core; no state-copy launches.
// Batch one speculative sequence (2..8 rows, width128/taps4). The recurrence
// stays in registers; prefix snapshots retain the existing rollback contract.
// Recurrent state is updated in place. Returned convolution state is separate
// and must be committed only after the window kernel completes on its stream.
std::vector<::tyloi::DenseTensor> gdn_decode_window_cuda(
    const ::tyloi::DenseTensor& qkv,const ::tyloi::DenseTensor& output_gate,
    const ::tyloi::DenseTensor& alpha,const ::tyloi::DenseTensor& beta,
    const ::tyloi::DenseTensor& convolution_state,const ::tyloi::DenseTensor& recurrent_state,
    const ::tyloi::DenseTensor& convolution_weight,const ::tyloi::DenseTensor& dt_bias,
    const ::tyloi::DenseTensor& a_log,const ::tyloi::DenseTensor& norm_weight,
    int key_heads,int value_heads,int confirmed,double convolution_eps,double norm_eps,
    bool silu_gate,bool output_half,bool transposed_state,
    const std::vector<::tyloi::DenseTensor>& checkpoint_convolution,
    const std::vector<::tyloi::DenseTensor>& checkpoint_recurrent);

std::vector<::tyloi::DenseTensor> gdn_decode_core_checkpoint_cuda(
    ::tyloi::DenseTensor qkv, ::tyloi::DenseTensor output_gate,
    ::tyloi::DenseTensor alpha, ::tyloi::DenseTensor beta,
    ::tyloi::DenseTensor convolution_state, ::tyloi::DenseTensor recurrent_state,
    ::tyloi::DenseTensor convolution_weight, ::tyloi::DenseTensor dt_bias,
    ::tyloi::DenseTensor a_log, ::tyloi::DenseTensor norm_weight,
    int64_t key_heads, int64_t value_heads, int64_t width,
    double convolution_eps, double norm_eps, bool silu_gate, bool output_half,
    bool transposed_state,
    ::tyloi::DenseTensor workspace,bool inplace_state,
    ::tyloi::DenseTensor checkpoint_convolution,
    ::tyloi::DenseTensor checkpoint_recurrent);
