#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>

namespace tyloi::cuda::ssm_conv_raw {
/** Storage and geometry for the original convolution launchers. */
struct Args {
    float* state = nullptr;
    const float* input = nullptr;
    const void* qk = nullptr;
    const void* v = nullptr;
    const float* weight = nullptr;
    const float* bias = nullptr;
    const void* alpha = nullptr;
    const void* beta = nullptr;
    const float* dt_bias = nullptr;
    const float* a_log = nullptr;
    float* q = nullptr;
    float* k = nullptr;
    float* output = nullptr;
    float* next = nullptr;
    float* decay = nullptr;
    float* beta_out = nullptr;
    int batch = 0, tokens = 0, channels = 0, kernel = 0;
    int nk = 0, nv = 0, dk = 0, dv = 0, layout = 0, has_bias = 0;
    float eps = 0;
    bool half_gate = false;
    int64_t q_stride0 = 0, q_stride1 = 0;
    int64_t a_stride0 = 0, a_stride1 = 0, b_stride0 = 0, b_stride1 = 0;
};
/** Run the F32 causal convolution and SiLU. */
void prefill(const Args&, cudaStream_t);
/** Run the F32 decode convolution and update state in place. */
void decode(const Args&, cudaStream_t);
/** Run fused QKV decode and update state in place. */
void qkv_decode(const Args&, cudaStream_t);
/** Run fused QKV and gate decode and update state in place. */
void qkv_gate_decode(const Args&, cudaStream_t);
/** Run fused QKV prefill and write a new state. */
void qkv_prefill(const Args&, cudaStream_t);
/** Run short packed QKV and gate prefill and write a new state. */
void qkv_gate_short(const Args&, cudaStream_t);
}
