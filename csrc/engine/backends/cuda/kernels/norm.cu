
// RMSNorm / L2 Norm (ggml norm.cu: ggml_rms_norm / ggml_l2_norm).
// One block normalizes one row (last dim D). fp32. x viewed as [N, D] (caller flattens).

#include <cuda_runtime.h>
#include "norm_raw.h"
#include <cuda_fp16.h>
#include <stdexcept>
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>
#include <cstdlib>

#include "reduce.cuh"

constexpr int NORM_BD = 256;

template<class Gate,class Output,int Threads,bool WarpTail=false>
__global__ void gdn_rms_norm_gate_kernel(const float* attended,const Gate* projection,
    const float* weight,Output* output,int heads,int tokens,int width,double eps,
    bool silu_gate,bool double_sigmoid,int64_t gate_batch_stride,int64_t gate_token_stride) {
    const int64_t row=blockIdx.x,base=row*width;
    const int64_t batch=row/(int64_t(heads)*tokens),head=(row/tokens)%heads,token=row%tokens;
    const int64_t destination=((batch*tokens+token)*heads+head)*width;
    float sum=0;
    for(int column=threadIdx.x;column<width;column+=Threads)
        sum=__fadd_rn(sum,__fmul_rn(attended[base+column],attended[base+column]));
    __shared__ float partial[Threads];partial[threadIdx.x]=sum;__syncthreads();
    if constexpr(WarpTail) {
        for(int stride=Threads/2;stride>=32;stride/=2) {
            if(threadIdx.x<stride)partial[threadIdx.x]=__fadd_rn(partial[threadIdx.x],partial[threadIdx.x+stride]);
            __syncthreads();
        }
        if(threadIdx.x<32) {
            float value=partial[threadIdx.x];
            #pragma unroll
            for(int stride=16;stride>0;stride/=2) {
                const float other=__shfl_down_sync(0xffffffffu,value,stride);
                if(threadIdx.x<stride)value=__fadd_rn(value,other);
            }
            if(threadIdx.x==0)partial[0]=value;
        }
        __syncthreads();
    }else {
        for(int stride=Threads/2;stride>0;stride/=2) {
            if(threadIdx.x<stride)partial[threadIdx.x]=__fadd_rn(partial[threadIdx.x],partial[threadIdx.x+stride]);
            __syncthreads();
        }
    }
    const float mean=__fdiv_rn(partial[0],static_cast<float>(width));
    const float variance=static_cast<float>(static_cast<double>(mean)+eps);
    const float inverse=__fdiv_rn(1.0f,__fsqrt_rn(variance));
    for(int column=threadIdx.x;column<width;column+=Threads) {
        const float z=static_cast<float>(projection[batch*gate_batch_stride+token*gate_token_stride+head*width+column]);
        const float sigmoid=double_sigmoid
            ? static_cast<float>(1.0/(1.0+::exp(-static_cast<double>(z))))
            : __fdiv_rn(1.0f,__fadd_rn(1.0f,::expf(-z)));
        const float gate=silu_gate?__fmul_rn(z,sigmoid):sigmoid;
        const float scaled=__fmul_rn(__fmul_rn(attended[base+column],inverse),weight[column]);
        output[destination+column]=static_cast<Output>(__fmul_rn(scaled,gate));
    }
}

template<class Value,int Threads>
__global__ void grouped_rms_norm_kernel(const Value* input,const float* weight,Value* output,
    int width,int weight_rows,float eps,float weight_offset) {
    const int64_t base=int64_t(blockIdx.x)*width;
    const float* scale=weight+int64_t(blockIdx.x%weight_rows)*width;
    float sum=0;
    for(int column=threadIdx.x;column<width;column+=Threads) {
        const float value=static_cast<float>(input[base+column]);
        sum=__fadd_rn(sum,__fmul_rn(value,value));
    }
    __shared__ float partial[Threads];partial[threadIdx.x]=sum;__syncthreads();
    for(int stride=Threads/2;stride>0;stride/=2) {
        if(threadIdx.x<stride)partial[threadIdx.x]=__fadd_rn(partial[threadIdx.x],partial[threadIdx.x+stride]);
        __syncthreads();
    }
    const float mean=__fdiv_rn(partial[0],static_cast<float>(width));
    const float inverse=__fdiv_rn(1.0f,__fsqrt_rn(__fadd_rn(mean,eps)));
    for(int column=threadIdx.x;column<width;column+=Threads) {
        const float normalized=__fmul_rn(static_cast<float>(input[base+column]),inverse);
        output[base+column]=static_cast<Value>(__fmul_rn(normalized,__fadd_rn(scale[column],weight_offset)));
    }
}

template <int BD>
__global__ void qwen_rms_norm_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    const float* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    int N, int D, float eps, float weight_offset)
{
    const int row = blockIdx.x;
    if (row >= N) {
        return;
    }
    const size_t row_offset = (size_t)row * D;
    float square_sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += BD) {
        const float value = __bfloat162float(input[row_offset + i]);
        square_sum += value * value;
    }
    square_sum = block_sum<BD / 32>(square_sum);
    const float inverse = rsqrtf(square_sum / (float)D + eps);
    for (int i = threadIdx.x; i < D; i += BD) {
        const __nv_bfloat16 normalized = __float2bfloat16_rn(
            __bfloat162float(input[row_offset + i]) * inverse);
        __nv_bfloat16 scale = __float2bfloat16_rn(weight[i]);
        if (weight_offset != 0.0f) {
            scale = __float2bfloat16_rn(
                __bfloat162float(scale) + weight_offset);
        }
        output[row_offset + i] = __float2bfloat16_rn(
            __bfloat162float(scale) * __bfloat162float(normalized));
    }
}

__global__ void qwen_rms_norm_pair_bf16_square_kernel(
    const __nv_bfloat16* __restrict__ first,
    const __nv_bfloat16* __restrict__ second,
    float* __restrict__ squared,
    int64_t first_count,
    int64_t total_count)
{
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < total_count; i += (int64_t)blockDim.x * gridDim.x) {
        const float value = i < first_count
            ? __bfloat162float(first[i])
            : __bfloat162float(second[i - first_count]);
        squared[i] = value * value;
    }
}

__global__ void qwen_rms_norm_pair_bf16_finalize_kernel(
    const __nv_bfloat16* __restrict__ first,
    const __nv_bfloat16* __restrict__ second,
    const float* __restrict__ mean,
    const float* __restrict__ first_weight,
    const float* __restrict__ second_weight,
    __nv_bfloat16* __restrict__ first_output,
    __nv_bfloat16* __restrict__ second_output,
    int64_t first_count,
    int64_t total_count,
    int D,
    float eps,
    float weight_offset)
{
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < total_count; i += (int64_t)blockDim.x * gridDim.x) {
        const bool use_second = i >= first_count;
        const int64_t local_i = use_second ? i - first_count : i;
        const __nv_bfloat16 value = use_second ? second[local_i] : first[local_i];
        const float* weight = use_second ? second_weight : first_weight;
        __nv_bfloat16* output = use_second ? second_output : first_output;
        const float inverse = rsqrtf(mean[i / D] + eps);
        const __nv_bfloat16 normalized = __float2bfloat16_rn(
            __bfloat162float(value) * inverse);
        __nv_bfloat16 scale = __float2bfloat16_rn(weight[local_i % D]);
        if (weight_offset != 0.0f) {
            scale = __float2bfloat16_rn(
                __bfloat162float(scale) + weight_offset);
        }
        output[local_i] = __float2bfloat16_rn(
            __bfloat162float(scale) * __bfloat162float(normalized));
    }
}

__device__ __forceinline__ __nv_bfloat16 qwen_rms_norm_bf16_value(
    __nv_bfloat16 value, float weight, float inverse, float weight_offset)
{
    const __nv_bfloat16 normalized = __float2bfloat16_rn(
        __bfloat162float(value) * inverse);
    __nv_bfloat16 scale = __float2bfloat16_rn(weight);
    if (weight_offset != 0.0f) {
        scale = __float2bfloat16_rn(
            __bfloat162float(scale) + weight_offset);
    }
    return __float2bfloat16_rn(
        __bfloat162float(scale) * __bfloat162float(normalized));
}

__global__ void minicpm_qk_norm_rope_cache_write_bf16_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ q_weight,
    const float* __restrict__ k_weight,
    const int64_t* __restrict__ rope_pos,
    const int64_t* __restrict__ write_pos,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    int B, int Hq, int Hk, int D, int half,
    int table_len, int max_seq, float eps, float weight_offset)
{
    const int q_row = blockIdx.x;
    if (q_row >= B * Hq) return;
    const int hq = q_row % Hq;
    const int b = q_row / Hq;
    const int tid = threadIdx.x;
    int64_t p = rope_pos[0];
    p = p < 0 ? 0 : (p >= table_len ? table_len - 1 : p);
    const int64_t wp = write_pos[0];
    const size_t q_offset = (size_t)q_row * D;
    const float q_value = __bfloat162float(q[q_offset + tid]);
    const float q_square_sum = block_sum<4>(q_value * q_value);
    const float q_inverse = rsqrtf(q_square_sum / (float)D + eps);

    const int k_row = b * Hk + (hq < Hk ? hq : 0);
    const size_t k_offset = (size_t)k_row * D;
    const float k_value = hq < Hk
        ? __bfloat162float(k[k_offset + tid]) : 0.0f;
    const float k_square_sum = block_sum<4>(k_value * k_value);
    const float k_inverse = rsqrtf(k_square_sum / (float)D + eps);

    if (tid < half) {
        const __nv_bfloat16 cs = __float2bfloat16_rn(
            cos[(size_t)p * half + tid]);
        const __nv_bfloat16 sn = __float2bfloat16_rn(
            sin[(size_t)p * half + tid]);
        const __nv_bfloat16 x0 = qwen_rms_norm_bf16_value(
            q[q_offset + tid], q_weight[tid], q_inverse, weight_offset);
        const __nv_bfloat16 x1 = qwen_rms_norm_bf16_value(
            q[q_offset + tid + half], q_weight[tid + half],
            q_inverse, weight_offset);
        const __nv_bfloat16 x0c = __float2bfloat16_rn(
            __bfloat162float(x0) * __bfloat162float(cs));
        const __nv_bfloat16 x1s = __float2bfloat16_rn(
            __bfloat162float(x1) * __bfloat162float(sn));
        const __nv_bfloat16 x1c = __float2bfloat16_rn(
            __bfloat162float(x1) * __bfloat162float(cs));
        const __nv_bfloat16 x0s = __float2bfloat16_rn(
            __bfloat162float(x0) * __bfloat162float(sn));
        q_out[q_offset + tid] = __float2bfloat16_rn(
            __bfloat162float(x0c) - __bfloat162float(x1s));
        q_out[q_offset + tid + half] = __float2bfloat16_rn(
            __bfloat162float(x1c) + __bfloat162float(x0s));
    }

    if (hq < Hk && wp >= 0 && wp < max_seq) {
        const size_t src = k_offset;
        const size_t dst = ((size_t)k_row * max_seq + wp) * D;
        if (tid < half) {
            const __nv_bfloat16 cs = __float2bfloat16_rn(
                cos[(size_t)p * half + tid]);
            const __nv_bfloat16 sn = __float2bfloat16_rn(
                sin[(size_t)p * half + tid]);
            const __nv_bfloat16 x0 = qwen_rms_norm_bf16_value(
                k[src + tid], k_weight[tid], k_inverse, weight_offset);
            const __nv_bfloat16 x1 = qwen_rms_norm_bf16_value(
                k[src + tid + half], k_weight[tid + half],
                k_inverse, weight_offset);
            const __nv_bfloat16 x0c = __float2bfloat16_rn(
                __bfloat162float(x0) * __bfloat162float(cs));
            const __nv_bfloat16 x1s = __float2bfloat16_rn(
                __bfloat162float(x1) * __bfloat162float(sn));
            const __nv_bfloat16 x1c = __float2bfloat16_rn(
                __bfloat162float(x1) * __bfloat162float(cs));
            const __nv_bfloat16 x0s = __float2bfloat16_rn(
                __bfloat162float(x0) * __bfloat162float(sn));
            k_cache[dst + tid] = __float2bfloat16_rn(
                __bfloat162float(x0c) - __bfloat162float(x1s));
            k_cache[dst + tid + half] = __float2bfloat16_rn(
                __bfloat162float(x1c) + __bfloat162float(x0s));
        }
        if (tid < D) {
            v_cache[dst + tid] = v[src + tid];
        }
    }
}

template <int BD>
__global__ void rms_norm_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                float* __restrict__ out, int N, int D, float eps, float weight_offset)
{
    int row = blockIdx.x;
    if (row >= N) {
        return;
    }
    const float* xr = x + (size_t)row * D;
    float* or_ = out + (size_t)row * D;
    int tid = threadIdx.x;

    float ssq = 0.0f;
    for (int i = tid; i < D; i += BD) {
        float xi = xr[i];
        ssq += xi * xi;
    }
    ssq = block_sum<BD / 32>(ssq);

    float rinv = rsqrtf(ssq / (float)D + eps);
    for (int i = tid; i < D; i += BD) {
        or_[i] = xr[i] * rinv * (w[i] + weight_offset);
    }
}

template <int BD>
__global__ void rms_norm_f16_kernel(const __half* __restrict__ x,
                                    const float* __restrict__ w,
                                    __half* __restrict__ out,
                                    int N, int D, float eps, float weight_offset)
{
    int row = blockIdx.x;
    if (row >= N) {
        return;
    }
    const __half* xr = x + (size_t)row * D;
    __half* or_ = out + (size_t)row * D;
    int tid = threadIdx.x;

    float ssq = 0.0f;
    for (int i = tid; i < D; i += BD) {
        float xi = (float)xr[i];
        ssq += xi * xi;
    }
    ssq = block_sum<BD / 32>(ssq);

    float rinv = rsqrtf(ssq / (float)D + eps);
    for (int i = tid; i < D; i += BD) {
        or_[i] = (__half)((float)xr[i] * rinv * (w[i] + weight_offset));
    }
}

template <int BD>
__global__ void rms_norm_f16x2_kernel(
    const half2* __restrict__ x,
    const float2* __restrict__ w,
    half2* __restrict__ out,
    int N, int pairs, float eps, float weight_offset)
{
    const int row = blockIdx.x;
    if (row >= N) {
        return;
    }
    const half2* input_row = x + (size_t)row * pairs;
    half2* output_row = out + (size_t)row * pairs;
    const int tid = threadIdx.x;

    float square_sum = 0.0f;
    for (int i = tid; i < pairs; i += BD) {
        const float2 value = __half22float2(input_row[i]);
        square_sum += value.x * value.x + value.y * value.y;
    }
    square_sum = block_sum<BD / 32>(square_sum);

    const float inverse = rsqrtf(square_sum / (float)(pairs * 2) + eps);
    for (int i = tid; i < pairs; i += BD) {
        const float2 value = __half22float2(input_row[i]);
        const float2 weight = w[i];
        output_row[i] = __floats2half2_rn(
            value.x * inverse * (weight.x + weight_offset),
            value.y * inverse * (weight.y + weight_offset));
    }
}

template <int BD>
__global__ void rms_norm_pair_f16_f32_kernel(
    const __half* __restrict__ first,
    const __half* __restrict__ second,
    const float* __restrict__ first_weight,
    const float* __restrict__ second_weight,
    float* __restrict__ first_out,
    float* __restrict__ second_out,
    int first_rows,
    int second_rows,
    int D,
    float eps,
    float weight_offset)
{
    const int combined_row = blockIdx.x;
    if (combined_row >= first_rows + second_rows) return;
    const bool use_second = combined_row >= first_rows;
    const int row = use_second ? combined_row - first_rows : combined_row;
    const __half* input = use_second ? second : first;
    const float* weight = use_second ? second_weight : first_weight;
    float* output = use_second ? second_out : first_out;
    const __half* input_row = input + (size_t)row * D;
    float* output_row = output + (size_t)row * D;
    const int tid = threadIdx.x;

    float ssq = 0.0f;
    for (int i = tid; i < D; i += BD) {
        const float value = (float)input_row[i];
        ssq += value * value;
    }
    ssq = block_sum<NORM_BD / 32>(ssq);

    const float inverse = rsqrtf(ssq / (float)D + eps);
    for (int i = tid; i < D; i += BD) {
        output_row[i] = (float)input_row[i] * inverse *
            (weight[i] + weight_offset);
    }
}

template <int BD>
__global__ void l2_norm_kernel(const float* __restrict__ x, float* __restrict__ out,
                               int N, int D, float eps)
{
    int row = blockIdx.x;
    if (row >= N) {
        return;
    }
    const float* xr = x + (size_t)row * D;
    float* or_ = out + (size_t)row * D;
    int tid = threadIdx.x;

    float ssq = 0.0f;
    for (int i = tid; i < D; i += BD) {
        float xi = xr[i];
        ssq += xi * xi;
    }
    ssq = block_sum<NORM_BD / 32>(ssq);

    float nrm = sqrtf(ssq);
    float denom = fmaxf(nrm, eps);   // matches F.normalize clamp_min(eps)
    float inv = 1.0f / denom;
    for (int i = tid; i < D; i += BD) {
        or_[i] = xr[i] * inv;
    }
}









// Grouped gated RMSNorm (the gated norm of state-space mixers).
//
// Every contiguous run of `group_size` columns inside one row is normalized on
// its own, then scaled by weight and rounded back to the input dtype:
//
//   hat = x * silu(gate)                        # gate applied first
//   y   = weight * hat / sqrt(mean(hat^2) + eps)
//
// The gate is applied *before* the mean square, so it changes the normalizer;
// applying it after normalization is a different function (the two differ by
// a factor that depends on the gate). Zamba2RMSNormGated and
// MambaRMSNormGated both implement this order, which is why mamba_ssm names
// the flag ``norm_before_gate=False``: the norm is not before the gate. The
// test pairs x and gate amplitudes 10x apart so a reversed order fails.
template <class Value>
__device__ __forceinline__ float gated_norm_value_as_float(Value value)
{
    if constexpr (std::is_same_v<Value, float>) {
        return value;
    } else if constexpr (std::is_same_v<Value, __half>) {
        return __half2float(value);
    } else {
        return __bfloat162float(value);
    }
}

template <class Value>
__device__ __forceinline__ Value gated_norm_value_from_float(float value)
{
    if constexpr (std::is_same_v<Value, float>) {
        return value;
    } else if constexpr (std::is_same_v<Value, __half>) {
        return __float2half_rn(value);
    } else {
        return __float2bfloat16_rn(value);
    }
}

template <class Value, class Gate>
__device__ __forceinline__ float gated_norm_gated_value(
    Value input, Gate gate)
{
    const float value = gated_norm_value_as_float(input);
    const float z = gated_norm_value_as_float(gate);
    return __fmul_rn(value, __fdiv_rn(z, __fadd_rn(1.0f, expf(-z))));
}

// One block normalizes one (row, group) pair; `width` is group_size and
// `groups_per_row` locates the weight slice of that group.
template <class Value, class Gate, int Threads>
__global__ void grouped_gated_rms_norm_kernel(
    const Value* __restrict__ input,
    const Gate* __restrict__ gate,
    const float* __restrict__ weight,
    Value* __restrict__ output,
    int width,
    int groups_per_row,
    float eps)
{
    const int64_t base = int64_t(blockIdx.x) * width;
    const float* scale = weight + int64_t(blockIdx.x % groups_per_row) * width;
    float square_sum = 0.0f;
    for (int column = threadIdx.x; column < width; column += Threads) {
        const float value =
            gated_norm_gated_value(input[base + column], gate[base + column]);
        square_sum = __fadd_rn(square_sum, __fmul_rn(value, value));
    }
    square_sum = block_sum<Threads / 32>(square_sum);
    const float mean = __fdiv_rn(square_sum, static_cast<float>(width));
    const float inverse = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(mean, eps)));
    for (int column = threadIdx.x; column < width; column += Threads) {
        const float value =
            gated_norm_gated_value(input[base + column], gate[base + column]);
        output[base + column] = gated_norm_value_from_float<Value>(
            __fmul_rn(__fmul_rn(value, inverse), scale[column]));
    }
}
namespace tyloi::cuda::norm_raw {
void rms(const void* x, const float* w, void* out, int N, int D, float eps, float offset, NumericType type, cudaStream_t stream) {
    if (type == NumericType::f32) {
        rms_norm_kernel<NORM_BD><<<N,NORM_BD,0,stream>>>(static_cast<const float*>(x),w,static_cast<float*>(out),N,D,eps,offset);
    } else if (type == NumericType::f16) {
        const auto a=reinterpret_cast<std::uintptr_t>(x)|reinterpret_cast<std::uintptr_t>(out);
        const auto b=reinterpret_cast<std::uintptr_t>(w);
        if (D>=4096 && !(D&1) && !(a&(alignof(half2)-1)) && !(b&(alignof(float2)-1)))
            rms_norm_f16x2_kernel<NORM_BD><<<N,NORM_BD,0,stream>>>(static_cast<const half2*>(x),reinterpret_cast<const float2*>(w),static_cast<half2*>(out),N,D/2,eps,offset);
        else rms_norm_f16_kernel<NORM_BD><<<N,NORM_BD,0,stream>>>(static_cast<const __half*>(x),w,static_cast<__half*>(out),N,D,eps,offset);
    } else throw std::invalid_argument("unsupported RMS dtype");
}
void l2(const float* x,float* out,int N,int D,float eps,cudaStream_t stream) {
    l2_norm_kernel<NORM_BD><<<N,NORM_BD,0,stream>>>(x,out,N,D,eps);
}
void gdn_gate(const float* x,const void* gate,const float* weight,void* out,int rows,int heads,int tokens,int width,double eps,bool silu,bool double_sigmoid,bool warp_tail,std::int64_t bs,std::int64_t ts,NumericType gt,NumericType ot,cudaStream_t stream) {
    const auto launch=[&]<class Gate,class Output,int Threads>() {
        const auto select=[&](auto tail) {
            gdn_rms_norm_gate_kernel<Gate,Output,Threads,decltype(tail)::value><<<static_cast<unsigned int>(rows),Threads,0,stream>>>(x,static_cast<const Gate*>(gate),weight,static_cast<Output*>(out),heads,tokens,width,eps,silu,double_sigmoid,bs,ts);
        };
        if(warp_tail)select(std::true_type{});else select(std::false_type{});
    };
    const auto dispatch_width=[&]<class Gate,class Output>() {
        if(width<=32)launch.template operator()<Gate,Output,32>();
        else if(width<=64)launch.template operator()<Gate,Output,64>();
        else if(width<=128)launch.template operator()<Gate,Output,128>();
        else launch.template operator()<Gate,Output,256>();
    };
    const auto dispatch_output=[&]<class Gate>() {
        if(ot==NumericType::f16)dispatch_width.template operator()<Gate,__half>();
        else if(ot==NumericType::f32)dispatch_width.template operator()<Gate,float>();
        else throw std::invalid_argument("invalid GDN output dtype");
    };
    if(gt==NumericType::f16)dispatch_output.template operator()<__half>();
    else if(gt==NumericType::f32)dispatch_output.template operator()<float>();
    else throw std::invalid_argument("invalid GDN gate dtype");
}
void grouped_gated(const void* x,const void* gate,const float* weight,void* out,std::int64_t blocks,int group,int groups,float eps,NumericType xt,NumericType gt,cudaStream_t stream) {
    if(!blocks)return;
    const auto launch=[&]<class Value,class Gate>() {
        grouped_gated_rms_norm_kernel<Value,Gate,NORM_BD><<<blocks,NORM_BD,0,stream>>>(static_cast<const Value*>(x),static_cast<const Gate*>(gate),weight,static_cast<Value*>(out),group,groups,eps);
    };
    const auto gates=[&]<class Value>() {
        if(gt==NumericType::f32)launch.template operator()<Value,float>();
        else if(gt==NumericType::f16)launch.template operator()<Value,__half>();
        else if(gt==NumericType::bf16)launch.template operator()<Value,__nv_bfloat16>();
        else throw std::invalid_argument("invalid gate dtype");
    };
    if(xt==NumericType::f32)gates.template operator()<float>();
    else if(xt==NumericType::f16)gates.template operator()<__half>();
    else if(xt==NumericType::bf16)gates.template operator()<__nv_bfloat16>();
    else throw std::invalid_argument("invalid input dtype");
}
void grouped(const void* x,const float* weight,void* out,int rows,int width,int weight_rows,float eps,float offset,NumericType type,cudaStream_t stream) {
    if(!rows)return;
    const auto launch=[&]<class Value,int Threads>() {
        grouped_rms_norm_kernel<Value,Threads><<<rows,Threads,0,stream>>>(static_cast<const Value*>(x),weight,static_cast<Value*>(out),width,weight_rows,eps,offset);
    };
    const auto dispatch=[&]<class Value>() {
        if(width<=32)launch.template operator()<Value,32>();
        else if(width<=64)launch.template operator()<Value,64>();
        else if(width<=128)launch.template operator()<Value,128>();
        else launch.template operator()<Value,256>();
    };
    if(type==NumericType::f16)dispatch.template operator()<__half>();
    else if(type==NumericType::f32)dispatch.template operator()<float>();
    else throw std::invalid_argument("invalid grouped RMS dtype");
}
void qwen_bf16(const void* x,const float* w,void* out,int N,int D,float eps,float offset,cudaStream_t stream) {
    qwen_rms_norm_bf16_kernel<NORM_BD><<<N,NORM_BD,0,stream>>>(static_cast<const __nv_bfloat16*>(x),w,static_cast<__nv_bfloat16*>(out),N,D,eps,offset);
}
void qwen_pair_square(const void* a,const void* b,float* out,std::int64_t first,std::int64_t total,int blocks,cudaStream_t stream) {
    qwen_rms_norm_pair_bf16_square_kernel<<<blocks,NORM_BD,0,stream>>>(static_cast<const __nv_bfloat16*>(a),static_cast<const __nv_bfloat16*>(b),out,first,total);
}
void qwen_pair_finish(const void* a,const void* b,const float* mean,const float* wa,const float* wb,void* oa,void* ob,std::int64_t first,std::int64_t total,int D,int blocks,float eps,float offset,cudaStream_t stream) {
    qwen_rms_norm_pair_bf16_finalize_kernel<<<blocks,NORM_BD,0,stream>>>(static_cast<const __nv_bfloat16*>(a),static_cast<const __nv_bfloat16*>(b),mean,wa,wb,static_cast<__nv_bfloat16*>(oa),static_cast<__nv_bfloat16*>(ob),first,total,D,eps,offset);
}
void minicpm_qk(const void* q,const void* k,const void* v,const float* qw,const float* kw,const std::int64_t* rp,const std::int64_t* wp,const float* cos,const float* sin,void* qo,void* kc,void* vc,int B,int table,int capacity,float eps,float offset,cudaStream_t stream) {
    minicpm_qk_norm_rope_cache_write_bf16_kernel<<<B*32,128,0,stream>>>(static_cast<const __nv_bfloat16*>(q),static_cast<const __nv_bfloat16*>(k),static_cast<const __nv_bfloat16*>(v),qw,kw,rp,wp,cos,sin,static_cast<__nv_bfloat16*>(qo),static_cast<__nv_bfloat16*>(kc),static_cast<__nv_bfloat16*>(vc),B,32,8,128,64,table,capacity,eps,offset);
}
void rms_pair_f16_f32(const void* a,const void* b,const float* wa,const float* wb,float* oa,float* ob,int na,int nb,int D,float eps,float offset,cudaStream_t stream) {
    rms_norm_pair_f16_f32_kernel<NORM_BD><<<na+nb,NORM_BD,0,stream>>>(static_cast<const __half*>(a),static_cast<const __half*>(b),wa,wb,oa,ob,na,nb,D,eps,offset);
}
} // namespace tyloi::cuda::norm_raw
