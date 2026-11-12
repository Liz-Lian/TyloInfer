// Fused causal depthwise SSM conv + SiLU.
// conv_input [B, K - 1 + T, C], weight [C,1,K], [C,K], or [K,C], output [B,T,C].

#include <cuda_runtime.h>
#include "ssm_conv_raw.h"
#include <cuda_fp16.h>
#include <stdexcept>
#include <string>


#include "reduce.cuh"

constexpr int SSM_CONV_BD = 256;

__device__ inline float silu_f32(float x)
{
    return x / (1.0f + expf(-x));
}

__global__ void ssm_conv_silu_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int B,
    int T,
    int C,
    int K,
    int weight_layout,
    int has_bias)
{
    int b = blockIdx.x;
    int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= B || c >= C) {
        return;
    }
    for (int t = 0; t < T; ++t) {
        float sum = 0.0f;
        for (int j = 0; j < K; ++j) {
            float xv = x[((size_t)b * (T + K - 1) + t + j) * C + c];
            float wv = weight_layout == 0 ? w[(size_t)c * K + j] : w[(size_t)j * C + c];
            sum += xv * wv;
        }
        if (has_bias) {
            sum += bias[c];
        }
        out[((size_t)b * T + t) * C + c] = silu_f32(sum);
    }
}



__global__ void ssm_conv_silu_decode_kernel(
    float* __restrict__ state,
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int B,
    int C,
    int K,
    int weight_layout,
    int has_bias)
{
    int b = blockIdx.x;
    int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= B || c >= C) {
        return;
    }
    size_t state_base = ((size_t)b * (size_t)(K - 1)) * (size_t)C + (size_t)c;
    float sum = 0.0f;
    for (int j = 0; j < K - 1; ++j) {
        float xv = state[state_base + (size_t)j * (size_t)C];
        float wv = weight_layout == 0 ? w[(size_t)c * K + j] : w[(size_t)j * C + c];
        sum += xv * wv;
    }
    float x_cur = x[(size_t)b * (size_t)C + (size_t)c];
    float wv = weight_layout == 0 ? w[(size_t)c * K + (K - 1)] : w[(size_t)(K - 1) * C + c];
    sum += x_cur * wv;
    if (has_bias) {
        sum += bias[c];
    }
    out[(size_t)b * (size_t)C + (size_t)c] = silu_f32(sum);
    for (int j = 0; j < K - 2; ++j) {
        state[state_base + (size_t)j * (size_t)C] = state[state_base + (size_t)(j + 1) * (size_t)C];
    }
    state[state_base + (size_t)(K - 2) * (size_t)C] = x_cur;
}



__device__ inline float ssm_decode_cur(
    const __half* __restrict__ qk,
    const __half* __restrict__ v,
    int b,
    int c,
    int qkC,
    int vsz)
{
    if (c < qkC) {
        return __half2float(qk[(size_t)b * (size_t)qkC + (size_t)c]);
    }
    return __half2float(v[(size_t)b * (size_t)vsz + (size_t)(c - qkC)]);
}

template <int BD>
__device__ __forceinline__ void ssm_conv_qk_norm_decode(
    float* __restrict__ state,
    const __half* __restrict__ qk,
    const __half* __restrict__ v,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ q,
    float* __restrict__ k,
    int B,
    int nk,
    int nv,
    int dk,
    int dv,
    int K,
    int weight_layout,
    int has_bias,
    float eps, int row)
{
    int b = row / (2 * nk);
    int rem = row - b * 2 * nk;
    int which = rem / nk;
    int h = rem - which * nk;
    int tid = threadIdx.x;
    int qkC = 2 * nk * dk;
    int vsz = nv * dv;
    int C = qkC + nv * dv;
    int base_c = (which * nk + h) * dk;

    __shared__ float vals[BD];
    __shared__ float sums[BD];
    float val = 0.0f;
    if (tid < dk) {
        int c = base_c + tid;
        size_t state_base = ((size_t)b * (size_t)(K - 1)) * (size_t)C + (size_t)c;
        float sum = 0.0f;
        for (int j = 0; j < K - 1; ++j) {
            float xv = state[state_base + (size_t)j * (size_t)C];
            float wv = weight_layout == 0 ? w[(size_t)c * K + j] : w[(size_t)j * C + c];
            sum += xv * wv;
        }
        float x_cur = ssm_decode_cur(qk, v, b, c, qkC, vsz);
        float wv = weight_layout == 0 ? w[(size_t)c * K + (K - 1)] : w[(size_t)(K - 1) * C + c];
        sum += x_cur * wv;
        if (has_bias) {
            sum += bias[c];
        }
        val = silu_f32(sum);
        vals[tid] = val;
        for (int j = 0; j < K - 2; ++j) {
            state[state_base + (size_t)j * (size_t)C] = state[state_base + (size_t)(j + 1) * (size_t)C];
        }
        state[state_base + (size_t)(K - 2) * (size_t)C] = x_cur;
    }
    float ssq = (tid < dk) ? val * val : 0.0f;
    sums[tid] = ssq;
    __syncthreads();
    for (int stride = BD / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sums[tid] += sums[tid + stride];
        }
        __syncthreads();
    }
    float inv = 1.0f / fmaxf(sqrtf(sums[0]), eps);
    if (tid < dk) {
        float outv = vals[tid] * inv;
        float* dst = which == 0 ? q : k;
        dst[((size_t)b * nk + h) * dk + tid] = outv;
    }
}

__device__ __forceinline__ void ssm_conv_v_decode(
    float* __restrict__ state,
    const __half* __restrict__ qk,
    const __half* __restrict__ v_in,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ v_out,
    int B,
    int nk,
    int nv,
    int dk,
    int dv,
    int K,
    int weight_layout,
    int has_bias, int b, int idx)
{
    int vsz = nv * dv;
    if (b >= B || idx >= vsz) {
        return;
    }
    int qkC = 2 * nk * dk;
    int C = qkC + vsz;
    int c = qkC + idx;
    size_t state_base = ((size_t)b * (size_t)(K - 1)) * (size_t)C + (size_t)c;
    float sum = 0.0f;
    for (int j = 0; j < K - 1; ++j) {
        float xv = state[state_base + (size_t)j * (size_t)C];
        float wv = weight_layout == 0 ? w[(size_t)c * K + j] : w[(size_t)j * C + c];
        sum += xv * wv;
    }
    float x_cur = __half2float(v_in[(size_t)b * (size_t)vsz + (size_t)idx]);
    float wv = weight_layout == 0 ? w[(size_t)c * K + (K - 1)] : w[(size_t)(K - 1) * C + c];
    sum += x_cur * wv;
    if (has_bias) {
        sum += bias[c];
    }
    v_out[(size_t)b * (size_t)vsz + (size_t)idx] = silu_f32(sum);
    for (int j = 0; j < K - 2; ++j) {
        state[state_base + (size_t)j * (size_t)C] = state[state_base + (size_t)(j + 1) * (size_t)C];
    }
    state[state_base + (size_t)(K - 2) * (size_t)C] = x_cur;
}

template<int BD>
__global__ void ssm_conv_qk_norm_decode_kernel(float* state,const __half* qk,const __half* v,
    const float* w,const float* bias,float* q,float* k,int B,int nk,int nv,int dk,int dv,
    int K,int layout,int has_bias,float eps) {
    ssm_conv_qk_norm_decode<BD>(state,qk,v,w,bias,q,k,B,nk,nv,dk,dv,K,layout,has_bias,eps,blockIdx.x);
}
__global__ void ssm_conv_v_decode_kernel(float* state,const __half* qk,const __half* v,
    const float* w,const float* bias,float* output,int B,int nk,int nv,int dk,int dv,
    int K,int layout,int has_bias) {
    ssm_conv_v_decode(state,qk,v,w,bias,output,B,nk,nv,dk,dv,K,layout,has_bias,
        blockIdx.x,blockIdx.y*blockDim.x+threadIdx.x);
}

template<class Gate>
__global__ void ssm_conv_qkv_gate_decode_kernel(float* state,const __half* qk,const __half* v,
    const float* weight,const Gate* alpha,const Gate* beta,const float* dt_bias,const float* a_log,
    float* q,float* k,float* output,float* decay,float* beta_out,
    int B,int nk,int nv,int dk,int dv,int K,int layout,float eps,
    int64_t as0,int64_t as2,int64_t bs0,int64_t bs2) {
    const int v_blocks=(nv*dv+255)/256;
    if(blockIdx.x<unsigned(2*nk)) {
        ssm_conv_qk_norm_decode<256>(state,qk,v,weight,nullptr,q,k,B,nk,nv,dk,dv,K,layout,0,
            eps,blockIdx.y*2*nk+blockIdx.x);
    } else if(blockIdx.x<unsigned(2*nk+v_blocks)) {
        ssm_conv_v_decode(state,qk,v,weight,nullptr,output,B,nk,nv,dk,dv,K,layout,0,
            blockIdx.y,(blockIdx.x-2*nk)*256+threadIdx.x);
    } else {
        const int head=(blockIdx.x-2*nk-v_blocks)*256+threadIdx.x,b=blockIdx.y;
        if(head<nv) {
            const float a=static_cast<float>(alpha[int64_t(b)*as0+int64_t(head)*as2])+dt_bias[head];
            const float sp=fmaxf(a,0.0f)+log1pf(expf(-fabsf(a)));
            decay[int64_t(b)*nv+head]=sp*-expf(a_log[head]);
            const float bv=static_cast<float>(beta[int64_t(b)*bs0+int64_t(head)*bs2]);
            beta_out[int64_t(b)*nv+head]=1.0f/(1.0f+expf(-bv));
        }
    }
}





template <int BD, bool Packed = false>
__global__ void linear_conv_qk_norm_prefill_kernel(
    const float* __restrict__ state,
    const __half* __restrict__ qk,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ q,
    float* __restrict__ k,
    int B, int T, int nk, int dk, int C, int K,
    int weight_layout, int has_bias, float eps,int64_t ps0=0,int64_t ps1=0)
{
    int row = blockIdx.x;
    int h = row % nk;
    int r = row / nk;
    int which = r & 1;
    r >>= 1;
    int t = r % T;
    int b = r / T;
    int tid = threadIdx.x;
    int qkC = 2 * nk * dk;
    int c = (which * nk + h) * dk + tid;
    float val = 0.0f;
    if (tid < dk) {
        float sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (j >= K) break;
            int token = t + j - (K - 1);
            float xv;
            if (token < 0) {
                xv = state[((size_t)b * (K - 1) + (token + K - 1)) * C + c];
            } else {
                xv = __half2float(qk[Packed && ps1?int64_t(b)*ps0+int64_t(token)*ps1+c:
                    ((size_t)b * T + token) * (Packed ? C : qkC) + c]);
            }
            float wv = weight_layout == 0 ? w[(size_t)c * K + j] : w[(size_t)j * C + c];
            sum += xv * wv;
        }
        if (has_bias) sum += bias[c];
        val = silu_f32(sum);
    }
    float ssq = block_sum<BD / 32>(tid < dk ? val * val : 0.0f);
    float inv = 1.0f / fmaxf(sqrtf(ssq), eps);
    if (tid < dk) {
        float* dst = which == 0 ? q : k;
        dst[(((size_t)b * nk + h) * T + t) * dk + tid] = val * inv;
    }
}

template <bool Packed = false, class Gate = float>
__global__ void linear_conv_v_prefill_kernel(
    const float* __restrict__ state,
    const __half* __restrict__ v_in,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ v_out,
    int B, int T, int nk, int nv, int dk, int dv, int C, int K,
    int weight_layout, int has_bias,
    const Gate* alpha = nullptr, const Gate* beta = nullptr,
    const float* dt_bias = nullptr, const float* a_log = nullptr,
    float* gate_out = nullptr, float* beta_out = nullptr,
    int64_t ps0=0,int64_t ps1=0,int64_t as0=0,int64_t as1=0,int64_t bs0=0,int64_t bs1=0)
{
    int bt = blockIdx.x;
    int b = bt / T;
    int t = bt - b * T;
    int vi = blockIdx.y * blockDim.x + threadIdx.x;
    int vsz = nv * dv;
    if (b >= B || vi >= vsz) return;
    int qkC = 2 * nk * dk;
    int c = qkC + vi;
    float sum = 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        if (j >= K) break;
        int token = t + j - (K - 1);
        float xv;
        if (token < 0) {
            xv = state[((size_t)b * (K - 1) + (token + K - 1)) * C + c];
        } else {
            xv = __half2float(v_in[Packed && ps1?int64_t(b)*ps0+int64_t(token)*ps1+vi:
                ((size_t)b * T + token) * (Packed ? C : vsz) + vi]);
        }
        float wv = weight_layout == 0 ? w[(size_t)c * K + j] : w[(size_t)j * C + c];
        sum += xv * wv;
    }
    if (has_bias) sum += bias[c];
    int h = vi / dv;
    int d = vi - h * dv;
    v_out[(((size_t)b * nv + h) * T + t) * dv + d] = silu_f32(sum);
    if constexpr(Packed) {
        if(d == 0) {
            const float a = static_cast<float>(alpha[as1?int64_t(b)*as0+int64_t(t)*as1+h:(size_t)bt * nv + h]) + dt_bias[h];
            const float sp = fmaxf(a,0.f) + log1pf(expf(-fabsf(a)));
            const auto index = ((size_t)b * nv + h) * T + t;
            gate_out[index] = sp * -expf(a_log[h]);
            const float raw = static_cast<float>(beta[bs1?int64_t(b)*bs0+int64_t(t)*bs1+h:(size_t)bt * nv + h]);
            beta_out[index] = 1.f / (1.f + expf(-raw));
        }
    }
}

template <bool Packed = false>
__device__ __forceinline__ float linear_prefill_input(
    const __half* qk, const __half* v, int b, int t, int c,
    int T, int qkC, int vsz,int64_t ps0=0,int64_t ps1=0)
{
    if (c < qkC) return __half2float(qk[Packed && ps1?int64_t(b)*ps0+int64_t(t)*ps1+c:
        ((size_t)b * T + t) * (Packed ? qkC+vsz : qkC) + c]);
    return __half2float(v[Packed && ps1?int64_t(b)*ps0+int64_t(t)*ps1+(c-qkC):
        ((size_t)b * T + t) * (Packed ? qkC+vsz : vsz) + (c - qkC)]);
}

template <bool Packed = false>
__global__ void linear_conv_state_prefill_kernel(
    const float* __restrict__ old_state,
    const __half* __restrict__ qk,
    const __half* __restrict__ v,
    float* __restrict__ new_state,
    int B, int T, int qkC, int vsz, int C, int K,int64_t ps0=0,int64_t ps1=0)
{
    size_t total = (size_t)B * (K - 1) * C;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (size_t)gridDim.x * blockDim.x) {
        int c = (int)(idx % C);
        size_t br = idx / C;
        int j = (int)(br % (K - 1));
        int b = (int)(br / (K - 1));
        int combined = T + j;
        if (combined < K - 1) {
            new_state[idx] = old_state[((size_t)b * (K - 1) + combined) * C + c];
        } else {
            int token = combined - (K - 1);
            new_state[idx] = linear_prefill_input<Packed>(qk, v, b, token, c, T, qkC, vsz,ps0,ps1);
        }
    }
}



// Short verification windows keep the projected Q/K/V together. Their V
// convolution also prepares the recurrent gates, with the original FP order.


namespace tyloi::cuda::ssm_conv_raw {
void prefill(const Args& a, cudaStream_t s) {
    dim3 blocks(a.batch,(a.channels+255)/256);
    ssm_conv_silu_kernel<<<blocks,256,0,s>>>(a.input,a.weight,a.bias,a.output,a.batch,a.tokens,a.channels,a.kernel,a.layout,a.has_bias);
}
void decode(const Args& a, cudaStream_t s) {
    dim3 blocks(a.batch,(a.channels+255)/256);
    ssm_conv_silu_decode_kernel<<<blocks,256,0,s>>>(a.state,a.input,a.weight,a.bias,a.output,a.batch,a.channels,a.kernel,a.layout,a.has_bias);
}
void qkv_decode(const Args& a, cudaStream_t s) {
    auto qk=static_cast<const __half*>(a.qk); auto v=static_cast<const __half*>(a.v);
    ssm_conv_qk_norm_decode_kernel<256><<<a.batch*2*a.nk,256,0,s>>>(a.state,qk,v,a.weight,a.bias,a.q,a.k,a.batch,a.nk,a.nv,a.dk,a.dv,a.kernel,a.layout,a.has_bias,a.eps);
    dim3 blocks(a.batch,(a.nv*a.dv+255)/256);
    ssm_conv_v_decode_kernel<<<blocks,256,0,s>>>(a.state,qk,v,a.weight,a.bias,a.output,a.batch,a.nk,a.nv,a.dk,a.dv,a.kernel,a.layout,a.has_bias);
}
template<class Gate> void gate_decode(const Args& a, cudaStream_t s) {
    dim3 grid(unsigned(2*a.nk+(a.nv*a.dv+255)/256+(a.nv+255)/256),unsigned(a.batch));
    ssm_conv_qkv_gate_decode_kernel<Gate><<<grid,256,0,s>>>(a.state,static_cast<const __half*>(a.qk),static_cast<const __half*>(a.v),a.weight,
        static_cast<const Gate*>(a.alpha),static_cast<const Gate*>(a.beta),a.dt_bias,a.a_log,a.q,a.k,a.output,a.decay,a.beta_out,
        a.batch,a.nk,a.nv,a.dk,a.dv,a.kernel,a.layout,a.eps,a.a_stride0,a.a_stride1,a.b_stride0,a.b_stride1);
}
void qkv_gate_decode(const Args& a, cudaStream_t s) {
    if(a.half_gate) gate_decode<__half>(a,s); else gate_decode<float>(a,s);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("GDN decode launch: ") + cudaGetErrorString(error));
}
void qkv_prefill(const Args& a, cudaStream_t s) {
    auto qk=static_cast<const __half*>(a.qk); auto v=static_cast<const __half*>(a.v);
    const int qkc=2*a.nk*a.dk, vsz=a.nv*a.dv;
    linear_conv_qk_norm_prefill_kernel<256><<<a.batch*a.tokens*2*a.nk,256,0,s>>>(a.state,qk,a.weight,a.bias,a.q,a.k,a.batch,a.tokens,a.nk,a.dk,a.channels,a.kernel,a.layout,a.has_bias,a.eps);
    linear_conv_v_prefill_kernel<><<<dim3(a.batch*a.tokens,(vsz+255)/256),256,0,s>>>(a.state,v,a.weight,a.bias,a.output,a.batch,a.tokens,a.nk,a.nv,a.dk,a.dv,a.channels,a.kernel,a.layout,a.has_bias);
    const int n=a.batch*(a.kernel-1)*a.channels;
    linear_conv_state_prefill_kernel<><<<(n+255)/256,256,0,s>>>(a.state,qk,v,a.next,a.batch,a.tokens,qkc,vsz,a.channels,a.kernel);
}
template<class Gate> void short_gate(const Args& a,cudaStream_t s,const __half* projected,int qkc,int vsz) {
    linear_conv_v_prefill_kernel<true,Gate><<<dim3(a.batch*a.tokens,(vsz+255)/256),256,0,s>>>(a.state,projected+qkc,a.weight,nullptr,a.output,a.batch,a.tokens,a.nk,a.nv,a.dk,a.dv,a.channels,a.kernel,a.layout,0,
        static_cast<const Gate*>(a.alpha),static_cast<const Gate*>(a.beta),a.dt_bias,a.a_log,a.decay,a.beta_out,a.q_stride0,a.q_stride1,a.a_stride0,a.a_stride1,a.b_stride0,a.b_stride1);
}
void qkv_gate_short(const Args& a,cudaStream_t s) {
    auto projected=static_cast<const __half*>(a.qk); const int qkc=2*a.nk*a.dk, vsz=a.nv*a.dv;
    linear_conv_qk_norm_prefill_kernel<256,true><<<a.batch*a.tokens*2*a.nk,256,0,s>>>(a.state,projected,a.weight,nullptr,a.q,a.k,a.batch,a.tokens,a.nk,a.dk,a.channels,a.kernel,a.layout,0,a.eps,a.q_stride0,a.q_stride1);
    if(a.half_gate) short_gate<__half>(a,s,projected,qkc,vsz); else short_gate<float>(a,s,projected,qkc,vsz);
    const int n=a.batch*(a.kernel-1)*a.channels;
    linear_conv_state_prefill_kernel<true><<<(n+255)/256,256,0,s>>>(a.state,projected,projected+qkc,a.next,a.batch,a.tokens,qkc,vsz,a.channels,a.kernel,a.q_stride0,a.q_stride1);
}
}
