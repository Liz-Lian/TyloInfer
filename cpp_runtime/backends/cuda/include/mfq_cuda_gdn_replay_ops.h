#pragma once
#include "mfq_tensor_backend.h"
#include <cstdint>
#include <vector>
struct GdnReplayLayer {
    const void* qkv;const void* alpha;const void* beta;
    float* convolution;float* state;float* next_convolution;
    const float* weight;const float* bias;const float* a_log;
    std::int64_t qkv_stride,alpha_stride,alpha_head_stride,beta_stride,beta_head_stride;
    int key_heads,value_heads,channels,tokens,weight_layout;
    int transposed,tiled_heads,qkv_float,gate_float;
    float convolution_eps;
    // Optional [tokens,value_heads,257]: normalized key, rank-one delta,
    // decay. These are computed by verification, without changing state.
    const float* updates=nullptr;
};
// The descriptors and next-convolution buffers persist across windows. State
// commits run across layers, followed by one convolution-history copy.
void gdn_replay_commit_layers_cuda(const mfq_tensor_backend::Tensor& descriptors,
    int layers,int maximum_heads,int maximum_channels,int keep,bool recorded_updates=false);
// Read a proposed window without changing committed state, or replay only
// its accepted prefix to commit. No full per-prefix state snapshots.
std::vector<mfq_tensor_backend::Tensor> gdn_replay_window_cuda(
    const mfq_tensor_backend::Tensor& qkv,const mfq_tensor_backend::Tensor& output_gate,
    const mfq_tensor_backend::Tensor& alpha,const mfq_tensor_backend::Tensor& beta,
    const mfq_tensor_backend::Tensor& convolution_state,const mfq_tensor_backend::Tensor& recurrent_state,
    const mfq_tensor_backend::Tensor& convolution_weight,const mfq_tensor_backend::Tensor& dt_bias,
    const mfq_tensor_backend::Tensor& a_log,const mfq_tensor_backend::Tensor& norm_weight,
    int key_heads,int value_heads,double convolution_eps,double norm_eps,
    bool silu_gate,bool output_half,bool transposed_state,bool commit,bool tiled_heads=false,bool record_updates=false,bool quantize_output=false,bool position_major=false);
