#include "rope_raw.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
namespace tyloi::cuda::rope_raw {
constexpr int ROPE_BD=256;
using RotaryProjectionLayout=Layout;
using RotaryNormalizedProjection=Projection;
using RotaryNormalizedGroup=Group;
__device__ int64_t rotary_position(const void* positions, bool wide, int64_t index) {
    return wide ? static_cast<const int64_t*>(positions)[index]
                : static_cast<const int32_t*>(positions)[index];
}

template<class Value>
__global__ void qsa_pool_rotary_cache_kernel(const Value* index,const float* weight,const void* positions,
        bool wide_positions,const int64_t* cache,const float* frequencies,const int32_t* selected_axes,Value* output,
        int tokens,int width,int pool,int maximum,int rotary_maximum,int pairs,int axes,float eps,
        int64_t row_stride,int64_t column_stride,PositionPolicy policy) {
    const int token=blockIdx.x,column=threadIdx.x;
    const int64_t block=cache[token]/pool,start=block*pool;
    // A verify batch can touch one block repeatedly. Only its last query writes
    // the pooled entry; all queries observe the same completed raw cache.
    if(token+1<tokens && cache[token+1]/pool==block)return;
    __shared__ float means[256],partial[256];
    float mean=0;
    if(column<width) {
        float sum=0;
        for(int i=0;i<pool;++i) {
            const int64_t row=min(max(int64_t(0),start+i),int64_t(maximum)-1);
            sum=__fadd_rn(sum,float(index[row*row_stride+column*column_stride]));
        }
        mean=float(Value(__fdiv_rn(sum,float(pool))));
    }
    means[column]=mean;partial[column]=__fadd_rn(0.f,__fmul_rn(mean,mean));
    __syncthreads();
    // Retain the normalized rotary kernel's reduction tree and intermediate
    // FP16 rounding, including the rounding before rotate-half.
    for(int stride=blockDim.x/2;stride>0;stride/=2) {
        if(column<stride)partial[column]=__fadd_rn(partial[column],partial[column+stride]);
        __syncthreads();
    }
    const float inverse=__fdiv_rn(1.f,__fsqrt_rn(__fadd_rn(__fdiv_rn(partial[0],float(width)),eps)));
    const auto normalized=[&](int col){return Value(__fmul_rn(__fmul_rn(means[col],inverse),__fadd_rn(weight[col],1.f)));};
    if(column>=width-pairs)return;
    if(column>=pairs){output[block*width+column+pairs]=normalized(column+pairs);return;}
    int axis=selected_axes[column];if(axis>=axes)axis=0;
    const int64_t raw=rotary_position(positions,wide_positions,int64_t(axis)*tokens+token)-cache[token]+start;
    int position;
    position=wide_positions && policy==PositionPolicy::native_double ? __double2int_rz(double(raw)) : int(raw);
    position=max(0,min(rotary_maximum-1,position));
    const float angle=__fmul_rn(float(position),frequencies[column]);
    const float cosine=float(::cos(double(angle))),sine=float(::sin(double(angle)));
    const float first=float(normalized(column)),second=float(normalized(column+pairs));
    output[block*width+column]=Value(__fsub_rn(__fmul_rn(first,cosine),__fmul_rn(second,sine)));
    output[block*width+column+pairs]=Value(__fadd_rn(__fmul_rn(second,cosine),__fmul_rn(first,sine)));
}

template<class Value>
__device__ __forceinline__ void rotary_normalized_row(
    const Value* input, const float* weight, const void* positions, bool wide_positions,
    const float* frequencies, const int32_t* selected_axes, Value* output,
    int heads, int tokens, int width, int pairs, int axes, int batches, int maximum,
    float eps, RotaryProjectionLayout layout, __half* key_cache,
    const void* cache_positions, bool wide_cache_positions, int64_t capacity,
    const void* projected_value, bool float_value, RotaryProjectionLayout value_layout,
    __half* value_cache, int64_t row, int active_threads, PositionPolicy policy) {
    const int64_t batch = row / (int64_t(heads) * tokens);
    const int head = int((row / tokens) % heads), token = int(row % tokens);
    const int64_t source = batch * layout.batch + token * layout.token + head * layout.head;
    float sum = 0;
    for (int column = threadIdx.x; column < width; column += active_threads) {
        const float x = float(input[source + int64_t(column) * layout.column]);
        sum = __fadd_rn(sum, __fmul_rn(x, x));
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = active_threads / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride)
            partial[threadIdx.x] = __fadd_rn(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    const float mean = __fdiv_rn(partial[0], float(width));
    const float inverse = __fdiv_rn(1.f, __fsqrt_rn(__fadd_rn(mean, eps)));
    const auto normalized = [&](int column) {
        const float scaled = __fmul_rn(float(input[source + int64_t(column) * layout.column]), inverse);
        return Value(__fmul_rn(scaled, __fadd_rn(weight[column], 1.f)));
    };
    int64_t cache_base = -1;
    if (key_cache) {
        const auto slot = rotary_position(cache_positions, wide_cache_positions, token);
        if (slot >= 0 && slot < capacity) cache_base = ((batch * heads + head) * capacity + slot) * width;
    }
    const auto store = [&](int column, Value value) {
        if (output) output[row * width + column] = value;
        if (cache_base >= 0) key_cache[cache_base + column] = __float2half_rn(float(value));
    };
    for (int column = threadIdx.x; column < width - pairs; column += active_threads) {
        if (column >= pairs) {
            store(column + pairs, normalized(column + pairs));
            continue;
        }
        int axis = selected_axes[column];
        if (axis >= axes) axis = 0;
        const int64_t position_index = (int64_t(axis) * batches + (batches == 1 ? 0 : batch)) * tokens + token;
        const int64_t raw_position = rotary_position(positions, wide_positions, position_index);
        int position = wide_positions && policy==PositionPolicy::native_double ? __double2int_rz(double(raw_position)) : static_cast<int32_t>(raw_position);
        position = max(0, min(maximum - 1, position));
        const float angle = __fmul_rn(float(position), frequencies[column]);
        const float cosine = float(::cos(double(angle))), sine = float(::sin(double(angle)));
        const float first = float(normalized(column)), second = float(normalized(column + pairs));
        store(column, Value(__fsub_rn(__fmul_rn(first, cosine), __fmul_rn(second, sine))));
        store(column + pairs, Value(__fadd_rn(__fmul_rn(second, cosine), __fmul_rn(first, sine))));
    }
    if (value_cache && cache_base >= 0) {
        const int64_t value_base = batch * value_layout.batch + token * value_layout.token + head * value_layout.head;
        for (int column = threadIdx.x; column < width; column += active_threads) {
            const int64_t at = value_base + int64_t(column) * value_layout.column;
            const float value = float_value ? static_cast<const float*>(projected_value)[at]
                : float(static_cast<const __half*>(projected_value)[at]);
            value_cache[cache_base + column] = __float2half_rn(value);
        }
    }
}

template<class Value>
__global__ void rotary_normalized_cached_kernel(
    const Value* input, const float* weight, const void* positions, bool wide_positions,
    const float* frequencies, const int32_t* selected_axes, Value* output,
    int heads, int tokens, int width, int pairs, int axes, int batches, int maximum,
    float eps, RotaryProjectionLayout layout, __half* key_cache,
    const void* cache_positions, bool wide_cache_positions, int64_t capacity,
    const void* projected_value, bool float_value, RotaryProjectionLayout value_layout,
    __half* value_cache, PositionPolicy policy) {
    rotary_normalized_row<Value>(input,weight,positions,wide_positions,frequencies,selected_axes,output,
        heads,tokens,width,pairs,axes,batches,maximum,eps,layout,key_cache,
        cache_positions,wide_cache_positions,capacity,projected_value,float_value,value_layout,value_cache,blockIdx.x,blockDim.x,policy);
}

template<class Value>
__global__ void rotary_normalized_grouped_kernel(RotaryNormalizedGroup group) {
    int64_t row=blockIdx.x;int selected=0;
    while(selected+1<group.count && row>=group.entries[selected].rows) {
        row-=group.entries[selected].rows;++selected;
    }
    const auto& p=group.entries[selected];
    rotary_normalized_row<Value>(static_cast<const Value*>(p.input),p.weight,p.positions,p.wide_positions,
        p.frequencies,p.selected_axes,static_cast<Value*>(p.output),p.heads,p.tokens,p.width,p.pairs,
        p.axes,p.batches,p.maximum,p.eps,p.layout,static_cast<__half*>(p.key_cache),p.cache_positions,p.wide_cache_positions,
        p.capacity,p.projected_value,p.float_value,p.value_layout,static_cast<__half*>(p.value_cache),row,p.threads,p.position_policy);
}
template<typename Value>
__global__ void rotary_embedding_cached_kernel(
    const Value* __restrict__ input,const int32_t* __restrict__ positions,
    const float* __restrict__ frequencies,const int32_t* __restrict__ selected_axes,
    Value* __restrict__ output,int64_t rows,int heads,int tokens,int width,int pairs,
    int axes,int batches,int maximum) {
    const int columns=width-pairs;
    const int64_t total=rows*columns;
    for(int64_t linear=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
        linear<total;linear+=int64_t(blockDim.x)*gridDim.x) {
        const int64_t row=linear/columns;
        const int column=int(linear-row*columns);
        const size_t offset=size_t(row)*width;
        if(column>=pairs) {
            output[offset+column+pairs]=input[offset+column+pairs];
            continue;
        }
        int axis=selected_axes[column];if(axis>=axes)axis=0;
        const int batch=batches==1?0:int(row/(int64_t(heads)*tokens));
        const int token=int(row%tokens);
        int position=positions[(size_t(axis)*batches+batch)*tokens+token];
        position=max(0,min(maximum-1,position));
        const float angle=__fmul_rn(float(position),frequencies[column]);
        // Match native tensor sin/cos promotion and every stored FP32 product.
        const float cosine=float(::cos(double(angle))),sine=float(::sin(double(angle)));
        const float first=float(input[offset+column]),second=float(input[offset+column+pairs]);
        const float a=__fsub_rn(__fmul_rn(first,cosine),__fmul_rn(second,sine));
        const float b=__fadd_rn(__fmul_rn(second,cosine),__fmul_rn(first,sine));
        output[offset+column]=Value(a);output[offset+column+pairs]=Value(b);
    }
}

__device__ int rope_axis_for_pair(int j, int s0, int s1, int s2)
{
    if (s0 <= 0 && s1 <= 0 && s2 <= 0) {
        return 0;
    }
    if (j < s0) {
        return 0;
    }
    if (j < s0 + s1) {
        return 1;
    }
    return 2;
}

__global__ void rope_kernel(const float* __restrict__ x, const float* __restrict__ pos,
                            float* __restrict__ out, int MT, int T, int D, int rotary_dim,
                            int pos_axes, int pos_batches,
                            int s0, int s1, int s2, float base)
{
    int mt = blockIdx.x;
    if (mt >= MT) {
        return;
    }
    int m = mt / T;
    int t = mt % T;
    int half = rotary_dim / 2;
    size_t base0 = ((size_t)m * T + t) * D;
    int tid = threadIdx.x;

    for (int i = tid; i < D; i += ROPE_BD) {
        out[base0 + i] = x[base0 + i];
    }
    __syncthreads();

    for (int j = tid; j < half; j += ROPE_BD) {
        float p;
        if (pos_batches > 0) {
            const int leading_rows = MT / T;
            const int rows_per_batch = leading_rows / pos_batches;
            const int batch = m / rows_per_batch;
            p = pos[(size_t)batch * T + t];
        } else {
            int axis = rope_axis_for_pair(j, s0, s1, s2);
            if (axis >= pos_axes) {
                axis = 0;
            }
            p = pos_axes == 1 ? pos[t] : pos[(size_t)axis * T + t];
        }
        float freq = powf(base, -2.0f * (float)j / (float)rotary_dim);
        float ang = p * freq;
        float cs = cosf(ang);
        float sn = sinf(ang);
        float x0 = x[base0 + j];
        float x1 = x[base0 + j + half];
        out[base0 + j] = x0 * cs - x1 * sn;
        out[base0 + j + half] = x1 * cs + x0 * sn;
    }
}

__global__ void rope_table_kernel(const float* __restrict__ x, const int64_t* __restrict__ pos,
                                  const float* __restrict__ cos, const float* __restrict__ sin,
                                  float* __restrict__ out, int MT, int T, int D, int rotary_dim,
                                  int table_len, int pos_axes, int pos_batches,
                                  int s0, int s1, int s2)
{
    int mt = blockIdx.x;
    if (mt >= MT) {
        return;
    }
    int m = mt / T;
    int t = mt % T;
    int half = rotary_dim / 2;
    size_t base0 = ((size_t)m * T + t) * D;
    int tid = threadIdx.x;

    for (int i = rotary_dim + tid; i < D; i += ROPE_BD) {
        out[base0 + i] = x[base0 + i];
    }

    for (int j = tid; j < half; j += ROPE_BD) {
        int64_t p;
        if (pos_batches > 0) {
            const int leading_rows = MT / T;
            const int rows_per_batch = leading_rows / pos_batches;
            const int batch = m / rows_per_batch;
            p = pos[(size_t)batch * T + t];
        } else {
            int axis = rope_axis_for_pair(j, s0, s1, s2);
            if (axis >= pos_axes) {
                axis = 0;
            }
            p = pos_axes == 1 ? pos[t] : pos[(size_t)axis * T + t];
        }
        if (p < 0) {
            p = 0;
        }
        if (p >= table_len) {
            p = table_len - 1;
        }
        float cs = cos[(size_t)p * half + j];
        float sn = sin[(size_t)p * half + j];
        float x0 = x[base0 + j];
        float x1 = x[base0 + j + half];
        out[base0 + j] = x0 * cs - x1 * sn;
        out[base0 + j + half] = x1 * cs + x0 * sn;
    }
}

__global__ void rope_table_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const int64_t* __restrict__ pos,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    __nv_bfloat16* __restrict__ out,
    int rows, int T, int D, int rotary_dim, int table_len)
{
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int t = row % T;
    const int half = rotary_dim / 2;
    const size_t offset = (size_t)row * D;
    int64_t p = pos[t];
    p = p < 0 ? 0 : (p >= table_len ? table_len - 1 : p);

    for (int i = rotary_dim + threadIdx.x; i < D; i += ROPE_BD) {
        out[offset + i] = x[offset + i];
    }

    for (int j = threadIdx.x; j < half; j += ROPE_BD) {
        const __nv_bfloat16 cs = __float2bfloat16_rn(
            cos[(size_t)p * half + j]);
        const __nv_bfloat16 sn = __float2bfloat16_rn(
            sin[(size_t)p * half + j]);
        const __nv_bfloat16 x0 = x[offset + j];
        const __nv_bfloat16 x1 = x[offset + j + half];
        const __nv_bfloat16 x0c = __float2bfloat16_rn(
            __bfloat162float(x0) * __bfloat162float(cs));
        const __nv_bfloat16 x1s = __float2bfloat16_rn(
            __bfloat162float(x1) * __bfloat162float(sn));
        const __nv_bfloat16 x1c = __float2bfloat16_rn(
            __bfloat162float(x1) * __bfloat162float(cs));
        const __nv_bfloat16 x0s = __float2bfloat16_rn(
            __bfloat162float(x0) * __bfloat162float(sn));
        out[offset + j] = __float2bfloat16_rn(
            __bfloat162float(x0c) - __bfloat162float(x1s));
        out[offset + j + half] = __float2bfloat16_rn(
            __bfloat162float(x1c) + __bfloat162float(x0s));
    }
}

__global__ void minicpm_bf16_rope_cache_write_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const int64_t* __restrict__ rope_pos,
    const int64_t* __restrict__ write_pos,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    int Hq, int Hk, int D, int half, int table_len, int max_seq)
{
    const int row = blockIdx.x;
    const int hq = row % Hq;
    const int b = row / Hq;
    const int tid = threadIdx.x;
    int64_t p = rope_pos[0];
    p = p < 0 ? 0 : (p >= table_len ? table_len - 1 : p);
    const int64_t wp = write_pos[0];

    const size_t q_offset = (size_t)row * D;
    if (tid < half) {
        const __nv_bfloat16 cs = __float2bfloat16_rn(
            cos[(size_t)p * half + tid]);
        const __nv_bfloat16 sn = __float2bfloat16_rn(
            sin[(size_t)p * half + tid]);
        const __nv_bfloat16 x0 = q[q_offset + tid];
        const __nv_bfloat16 x1 = q[q_offset + tid + half];
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
        const size_t src = ((size_t)b * Hk + hq) * D;
        const size_t dst = (((size_t)b * Hk + hq) * max_seq + wp) * D;
        if (tid < half) {
            const __nv_bfloat16 cs = __float2bfloat16_rn(
                cos[(size_t)p * half + tid]);
            const __nv_bfloat16 sn = __float2bfloat16_rn(
                sin[(size_t)p * half + tid]);
            const __nv_bfloat16 x0 = k[src + tid];
            const __nv_bfloat16 x1 = k[src + tid + half];
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

template<class V>
void pool_t(const void* x,const float* w,const void* positions,bool wide,const int64_t* cache,
    const float* freq,const int32_t* axes,void* output,int tokens,int width,int pool_size,int maximum,
    int rotary_maximum,int pairs,int axis_count,float eps,int64_t rs,int64_t cs,PositionPolicy policy,cudaStream_t stream) {
    const int threads=width<=32?32:width<=64?64:width<=128?128:256;
    qsa_pool_rotary_cache_kernel<<<unsigned(tokens),threads,0,stream>>>(static_cast<const V*>(x),w,positions,
        wide,cache,freq,axes,static_cast<V*>(output),tokens,width,pool_size,maximum,rotary_maximum,pairs,
        axis_count,eps,rs,cs,policy);
}
void pool(const void* x,const float* w,const void* positions,bool wide,const int64_t* cache,
    const float* freq,const int32_t* axes,void* output,int tokens,int width,int pool_size,int maximum,
    int rotary_maximum,int pairs,int axis_count,float eps,int64_t rs,int64_t cs,NumericType type,
    PositionPolicy policy,cudaStream_t stream) {
    if(type==NumericType::f16)pool_t<__half>(x,w,positions,wide,cache,freq,axes,output,tokens,width,pool_size,maximum,rotary_maximum,pairs,axis_count,eps,rs,cs,policy,stream);
    else pool_t<float>(x,w,positions,wide,cache,freq,axes,output,tokens,width,pool_size,maximum,rotary_maximum,pairs,axis_count,eps,rs,cs,policy,stream);
}
template<class V>
void normalized_t(const Projection& p,cudaStream_t stream) {
    rotary_normalized_cached_kernel<V><<<unsigned(p.rows),p.threads,0,stream>>>(
        static_cast<const V*>(p.input),p.weight,p.positions,p.wide_positions,p.frequencies,p.selected_axes,
        static_cast<V*>(p.output),p.heads,p.tokens,p.width,p.pairs,p.axes,p.batches,p.maximum,p.eps,p.layout,
        static_cast<__half*>(p.key_cache),p.cache_positions,p.wide_cache_positions,p.capacity,p.projected_value,
        p.float_value,p.value_layout,static_cast<__half*>(p.value_cache),p.position_policy);
}
void normalized(const Projection& p,NumericType type,cudaStream_t stream) {
    if(type==NumericType::f16)normalized_t<__half>(p,stream);else normalized_t<float>(p,stream);
}
void normalized_group(const Group& group,int64_t rows,int threads,NumericType type,cudaStream_t stream) {
    if(type==NumericType::f16)rotary_normalized_grouped_kernel<__half><<<unsigned(rows),threads,0,stream>>>(group);
    else rotary_normalized_grouped_kernel<float><<<unsigned(rows),threads,0,stream>>>(group);
}
template<class V>
void cached_t(const void* x,const int32_t* positions,const float* freq,const int32_t* axes,void* out,
    int64_t rows,int heads,int tokens,int width,int pairs,int axis_count,int batches,int maximum,cudaStream_t stream) {
    const int64_t total=rows*(width-pairs);
    rotary_embedding_cached_kernel<V><<<static_cast<unsigned>((total+127)/128),128,0,stream>>>(
        static_cast<const V*>(x),positions,freq,axes,static_cast<V*>(out),rows,heads,tokens,width,pairs,axis_count,batches,maximum);
}
void cached(const void* x,const int32_t* positions,const float* freq,const int32_t* axes,void* out,
    int64_t rows,int heads,int tokens,int width,int pairs,int axis_count,int batches,int maximum,NumericType type,cudaStream_t stream) {
    if(type==NumericType::f16)cached_t<__half>(x,positions,freq,axes,out,rows,heads,tokens,width,pairs,axis_count,batches,maximum,stream);
    else cached_t<float>(x,positions,freq,axes,out,rows,heads,tokens,width,pairs,axis_count,batches,maximum,stream);
}
void rope(const float* x,const float* pos,float* out,int rows,int tokens,int width,int rotary,
    int axes,int batches,int s0,int s1,int s2,float base,cudaStream_t stream) {
    rope_kernel<<<rows,ROPE_BD,0,stream>>>(x,pos,out,rows,tokens,width,rotary,axes,batches,s0,s1,s2,base);
}
void table(const float* x,const int64_t* pos,const float* cosine,const float* sine,float* out,
    int rows,int tokens,int width,int rotary,int length,int axes,int batches,int s0,int s1,int s2,cudaStream_t stream) {
    rope_table_kernel<<<rows,ROPE_BD,0,stream>>>(x,pos,cosine,sine,out,rows,tokens,width,rotary,length,axes,batches,s0,s1,s2);
}
void table_bf16(const void* x,const int64_t* pos,const float* cosine,const float* sine,void* out,
    int rows,int tokens,int width,int rotary,int length,cudaStream_t stream) {
    rope_table_bf16_kernel<<<rows,ROPE_BD,0,stream>>>(static_cast<const __nv_bfloat16*>(x),pos,cosine,sine,
        static_cast<__nv_bfloat16*>(out),rows,tokens,width,rotary,length);
}
void minicpm(const void* q,const void* k,const void* v,const int64_t* rope_pos,const int64_t* write_pos,
    const float* cosine,const float* sine,void* output,void* kc,void* vc,int batch,int hq,int hk,int length,int capacity,cudaStream_t stream) {
    minicpm_bf16_rope_cache_write_kernel<<<batch*hq,128,0,stream>>>(static_cast<const __nv_bfloat16*>(q),
        static_cast<const __nv_bfloat16*>(k),static_cast<const __nv_bfloat16*>(v),rope_pos,write_pos,cosine,sine,
        static_cast<__nv_bfloat16*>(output),static_cast<__nv_bfloat16*>(kc),static_cast<__nv_bfloat16*>(vc),
        hq,hk,128,64,length,capacity);
}
} // namespace tyloi::cuda::rope_raw
