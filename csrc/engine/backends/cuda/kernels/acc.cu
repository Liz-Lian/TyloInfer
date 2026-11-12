// Residual add a + b (ggml acc.cu: ggml_acc). fp16/fp32/bf16, any shape flattened.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "acc_raw.h"
#include <cuda_fp16.h>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "reduce.cuh"

template <typename scalar_t>
__global__ void acc_kernel(const scalar_t* __restrict__ a, const scalar_t* __restrict__ b,
                           scalar_t* __restrict__ out, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        out[i] = a[i] + b[i];
    }
}

__global__ void acc_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ out,
    int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += gridDim.x * blockDim.x) {
        out[i] = __float2bfloat16_rn(
            __bfloat162float(a[i]) + __bfloat162float(b[i]));
    }
}



constexpr int ACC_RMS_BD = 256;

template <int BD>
__global__ void acc_rms_norm_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    const float* __restrict__ weight,
    __nv_bfloat16* __restrict__ sum_out,
    __nv_bfloat16* __restrict__ norm_out,
    int N, int D, float eps, float weight_offset)
{
    const int row = blockIdx.x;
    if (row >= N) {
        return;
    }
    const size_t row_offset = (size_t)row * D;
    float square_sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += BD) {
        const __nv_bfloat16 stored = __float2bfloat16_rn(
            __bfloat162float(a[row_offset + i]) +
            __bfloat162float(b[row_offset + i]));
        sum_out[row_offset + i] = stored;
        const float value = __bfloat162float(stored);
        square_sum += value * value;
    }
    square_sum = block_sum<BD / 32>(square_sum);
    const float inverse = rsqrtf(square_sum / (float)D + eps);
    for (int i = threadIdx.x; i < D; i += BD) {
        const __nv_bfloat16 normalized = __float2bfloat16_rn(
            __bfloat162float(sum_out[row_offset + i]) * inverse);
        __nv_bfloat16 scale = __float2bfloat16_rn(weight[i]);
        if (weight_offset != 0.0f) {
            scale = __float2bfloat16_rn(
                __bfloat162float(scale) + weight_offset);
        }
        norm_out[row_offset + i] = __float2bfloat16_rn(
            __bfloat162float(normalized) * __bfloat162float(scale));
    }
}

template <typename scalar_t, typename norm_t, int BD, bool REUSE_STORED_SUM>
__global__ void acc_rms_norm_kernel(const scalar_t* __restrict__ a,
                                    const scalar_t* __restrict__ b,
                                    const float* __restrict__ w,
                                    scalar_t* __restrict__ sum_out,
                                    norm_t* __restrict__ norm_out,
                                    int N, int D, float eps, float weight_offset)
{
    int row = blockIdx.x;
    if (row >= N) {
        return;
    }

    const scalar_t* ar = a + (size_t)row * D;
    const scalar_t* br = b + (size_t)row * D;
    scalar_t* sr = sum_out + (size_t)row * D;
    norm_t* nr = norm_out + (size_t)row * D;
    int tid = threadIdx.x;

    float ssq = 0.0f;
    for (int i = tid; i < D; i += BD) {
        float sf = (float)ar[i] + (float)br[i];
        scalar_t stored = (scalar_t)sf;
        sr[i] = stored;
        float sn = (float)stored;
        ssq += sn * sn;
    }
    ssq = block_sum<BD / 32>(ssq);

    float rinv = rsqrtf(ssq / (float)D + eps);
    for (int i = tid; i < D; i += BD) {
        float sn;
        if constexpr (REUSE_STORED_SUM) {
            sn = (float)sr[i];
        } else {
            float sf = (float)ar[i] + (float)br[i];
            scalar_t stored = (scalar_t)sf;
            sn = (float)stored;
        }
        nr[i] = (norm_t)(sn * rinv * (w[i] + weight_offset));
    }
}







template <int BD>
__global__ void gemma4_attn_residual_pre_norms_f16_kernel(
    const __half* __restrict__ residual,
    const __half* __restrict__ attn,
    const float* __restrict__ attn_post_weight,
    const float* __restrict__ dense_pre_weight,
    const float* __restrict__ router_weight,
    const float* __restrict__ moe_pre_weight,
    __half* __restrict__ residual_out,
    __half* __restrict__ dense_out,
    float* __restrict__ router_out,
    __half* __restrict__ moe_out,
    int N, int D, float eps)
{
    const int row = blockIdx.x;
    if (row >= N) {
        return;
    }

    extern __shared__ unsigned char shared_bytes[];
    auto* shared_x = reinterpret_cast<__half*>(shared_bytes);
    const size_t row_offset = (size_t)row * D;
    const __half* rr = residual + row_offset;
    const __half* ar = attn + row_offset;
    __half* xr = residual_out + row_offset;
    __half* dr = dense_out + row_offset;
    float* router_r = router_out + row_offset;
    __half* mr = moe_out + row_offset;

    float attn_ssq = 0.0f;
    for (int i = threadIdx.x; i < D; i += BD) {
        const float value = (float)ar[i];
        attn_ssq += value * value;
    }
    attn_ssq = block_sum<BD / 32>(attn_ssq);
    const float attn_rinv = rsqrtf(attn_ssq / (float)D + eps);

    float x_ssq = 0.0f;
    for (int i = threadIdx.x; i < D; i += BD) {
        const __half attn_norm = (__half)(
            (float)ar[i] * attn_rinv * attn_post_weight[i]);
        const __half x_value = (__half)((float)rr[i] + (float)attn_norm);
        xr[i] = x_value;
        shared_x[i] = x_value;
        const float value = (float)x_value;
        x_ssq += value * value;
    }
    x_ssq = block_sum<BD / 32>(x_ssq);
    const float x_rinv = rsqrtf(x_ssq / (float)D + eps);

    for (int i = threadIdx.x; i < D; i += BD) {
        const float value = (float)shared_x[i] * x_rinv;
        dr[i] = (__half)(value * dense_pre_weight[i]);
        router_r[i] = value * router_weight[i];
        mr[i] = (__half)(value * moe_pre_weight[i]);
    }
}



template <int BD>
__global__ void gemma4_ffn_merge_f16_kernel(
    const __half* __restrict__ dense,
    const __half* __restrict__ moe,
    const __half* __restrict__ residual,
    const float* __restrict__ dense_post_weight,
    const float* __restrict__ moe_post_weight,
    const float* __restrict__ final_post_weight,
    const __half* __restrict__ layer_scale,
    __half* __restrict__ out,
    int N, int D, float eps)
{
    const int row = blockIdx.x;
    if (row >= N) {
        return;
    }

    extern __shared__ unsigned char shared_bytes[];
    auto* combined = reinterpret_cast<__half*>(shared_bytes);
    const size_t row_offset = (size_t)row * D;
    const __half* dr = dense + row_offset;
    const __half* mr = moe + row_offset;
    const __half* rr = residual + row_offset;
    __half* orow = out + row_offset;

    float dense_ssq = 0.0f;
    float moe_ssq = 0.0f;
    for (int i = threadIdx.x; i < D; i += BD) {
        const float dense_value = (float)dr[i];
        const float moe_value = (float)mr[i];
        dense_ssq += dense_value * dense_value;
        moe_ssq += moe_value * moe_value;
    }
    dense_ssq = block_sum<BD / 32>(dense_ssq);
    moe_ssq = block_sum<BD / 32>(moe_ssq);
    const float dense_rinv = rsqrtf(dense_ssq / (float)D + eps);
    const float moe_rinv = rsqrtf(moe_ssq / (float)D + eps);

    float combined_ssq = 0.0f;
    for (int i = threadIdx.x; i < D; i += BD) {
        const __half dense_norm = (__half)(
            (float)dr[i] * dense_rinv * dense_post_weight[i]);
        const __half moe_norm = (__half)(
            (float)mr[i] * moe_rinv * moe_post_weight[i]);
        const __half value = (__half)((float)dense_norm + (float)moe_norm);
        combined[i] = value;
        const float value_f = (float)value;
        combined_ssq += value_f * value_f;
    }
    combined_ssq = block_sum<BD / 32>(combined_ssq);
    const float combined_rinv = rsqrtf(combined_ssq / (float)D + eps);
    const float scale = (float)layer_scale[0];

    for (int i = threadIdx.x; i < D; i += BD) {
        const __half post = (__half)(
            (float)combined[i] * combined_rinv * final_post_weight[i]);
        const __half residual_sum = (__half)((float)rr[i] + (float)post);
        orow[i] = (__half)((float)residual_sum * scale);
    }
}



__global__ void decode_graph_commit_kernel(const int64_t* __restrict__ next,
                                           int64_t* __restrict__ generated,
                                           int64_t* __restrict__ step,
                                           int64_t* __restrict__ input,
                                           int64_t* __restrict__ pos,
                                           int64_t* __restrict__ len)
{
    int64_t idx = step[0];
    int64_t tok = next[0];
    generated[idx] = tok;
    input[0] = tok;
    pos[0] += 1;
    len[0] += 1;
    step[0] = idx + 1;
}
namespace tyloi::cuda::acc_raw {
void add(const void* a, const void* b, void* out, int n, NumericType type, cudaStream_t stream) {
    constexpr int BD = 256;
    if (type == NumericType::bf16)
        acc_bf16_kernel<<<(n + BD - 1) / BD, BD, 0, stream>>>(static_cast<const __nv_bfloat16*>(a), static_cast<const __nv_bfloat16*>(b), static_cast<__nv_bfloat16*>(out), n);
    else if (type == NumericType::f16)
        acc_kernel<__half><<<(n + BD - 1) / BD, BD, 0, stream>>>(static_cast<const __half*>(a), static_cast<const __half*>(b), static_cast<__half*>(out), n);
    else if (type == NumericType::f32)
        acc_kernel<float><<<(n + BD - 1) / BD, BD, 0, stream>>>(static_cast<const float*>(a), static_cast<const float*>(b), static_cast<float*>(out), n);
    else throw std::invalid_argument("unsupported add dtype");
}
void rms(const void* a, const void* b, const float* w, void* sum, void* norm,
    int N, int D, float eps, float offset, NumericType input, NumericType output, cudaStream_t stream) {
    if (input == NumericType::bf16 && output == NumericType::bf16)
        acc_rms_norm_bf16_kernel<ACC_RMS_BD><<<N, ACC_RMS_BD, 0, stream>>>(static_cast<const __nv_bfloat16*>(a), static_cast<const __nv_bfloat16*>(b), w, static_cast<__nv_bfloat16*>(sum), static_cast<__nv_bfloat16*>(norm), N, D, eps, offset);
    else if (input == NumericType::f16 && output == NumericType::f16)
        acc_rms_norm_kernel<__half, __half, ACC_RMS_BD, false><<<N, ACC_RMS_BD, 0, stream>>>(static_cast<const __half*>(a), static_cast<const __half*>(b), w, static_cast<__half*>(sum), static_cast<__half*>(norm), N, D, eps, offset);
    else if (input == NumericType::f16 && output == NumericType::f32)
        acc_rms_norm_kernel<__half, float, ACC_RMS_BD, false><<<N, ACC_RMS_BD, 0, stream>>>(static_cast<const __half*>(a), static_cast<const __half*>(b), w, static_cast<__half*>(sum), static_cast<float*>(norm), N, D, eps, offset);
    else if (input == NumericType::f32 && output == NumericType::f32)
        acc_rms_norm_kernel<float, float, ACC_RMS_BD, true><<<N, ACC_RMS_BD, 0, stream>>>(static_cast<const float*>(a), static_cast<const float*>(b), w, static_cast<float*>(sum), static_cast<float*>(norm), N, D, eps, offset);
    else throw std::invalid_argument("unsupported add RMS dtype");
}
void gemma_pre(const void* residual, const void* attn, const float* post, const float* dense,
    const float* router, const float* moe, void* out, void* dense_out, float* router_out,
    void* moe_out, int N, int D, float eps, cudaStream_t stream) {
    gemma4_attn_residual_pre_norms_f16_kernel<ACC_RMS_BD><<<N, ACC_RMS_BD, std::size_t(D)*sizeof(__half), stream>>>(
        static_cast<const __half*>(residual), static_cast<const __half*>(attn), post, dense, router, moe,
        static_cast<__half*>(out), static_cast<__half*>(dense_out), router_out, static_cast<__half*>(moe_out), N, D, eps);
}
void gemma_merge(const void* dense, const void* moe, const void* residual, const float* dense_w,
    const float* moe_w, const float* post_w, const void* scale, void* out, int N, int D, float eps, cudaStream_t stream) {
    gemma4_ffn_merge_f16_kernel<ACC_RMS_BD><<<N, ACC_RMS_BD, std::size_t(D)*sizeof(__half), stream>>>(
        static_cast<const __half*>(dense), static_cast<const __half*>(moe), static_cast<const __half*>(residual),
        dense_w, moe_w, post_w, static_cast<const __half*>(scale), static_cast<__half*>(out), N, D, eps);
}
void commit(const std::int64_t* next, std::int64_t* generated, std::int64_t* step,
    std::int64_t* input, std::int64_t* pos, std::int64_t* len, cudaStream_t stream) {
    decode_graph_commit_kernel<<<1, 1, 0, stream>>>(next, generated, step, input, pos, len);
}
} // namespace tyloi::cuda::acc_raw
