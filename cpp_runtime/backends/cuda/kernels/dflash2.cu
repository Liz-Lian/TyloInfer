#include <math_constants.h>
#include "dflash2.h"
#include "kernel_dispatch_trace.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
#include <climits>
#include <type_traits>
#include <stdexcept>
#include "dflash2_q8.cuh"

namespace mfq::cuda::dflash2 {
namespace {
__device__ float rounded(float value) { return __bfloat162float(__float2bfloat16_rn(value)); }
__device__ float warp_sum(float value) {
    for (int offset = 16; offset; offset >>= 1) value += __shfl_xor_sync(0xffffffffu, value, offset);
    return value;
}
void check() {
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
__global__ void normalized(const float* input,const float* norm,float* output,int rows,int width,float epsilon) {
    const int row=blockIdx.x*4+threadIdx.y,lane=threadIdx.x;
    if(row>=rows)return;
    const auto offset=std::int64_t(row)*width;
    float sum=0;
    for(int column=lane;column<width;column+=32)sum+=input[offset+column]*input[offset+column];
    const float scale=rsqrtf(warp_sum(sum)/width+epsilon);
    for(int column=lane;column<width;column+=32)output[offset+column]=rounded(input[offset+column]*scale*norm[column]);
}
// Strata's block-wide RMS/planar-Q8 fusion, with runtime widths, padding and
// the adapter's BF16 activation rounding retained.
template<bool Quantized>
__global__ void normalized_block(const float* input,const float* weight,float* output,
    signed char* codes,float* scales,int rows,int width,int padded,float epsilon) {
    const int row=blockIdx.x,tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    const auto offset=int64_t(row)*width;const bool live=row<rows;
    __shared__ float reduction[8];float sum=0;
    if(live)for(int column=tid;column<width;column+=256)sum+=input[offset+column]*input[offset+column];
    sum=warp_sum(sum);if(!lane)reduction[warp]=sum;
    __syncthreads();sum=0;for(int i=0;i<8;++i)sum+=reduction[i];
    const float inverse=rsqrtf(sum/width+epsilon);
    for(int column=tid;column<(Quantized?padded:width);column+=256) {
        const float value=live && column<width?rounded(input[offset+column]*inverse*weight[column]):0.f;
        if(live && column<width)output[offset+column]=value;
        if constexpr(Quantized) {
            float maximum=fabsf(value);
            for(int shift=16;shift;shift>>=1)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,shift));
            const float scale=maximum>0.f?127.f/maximum:0.f;
            codes[int64_t(row)*padded+column]=static_cast<signed char>(__float2int_rn(value*scale));
            if(!lane)scales[int64_t(row)*(padded/32)+column/32]=maximum/127.f;
        }
    }
}

template<bool Joined>
__global__ void activate(const half* gate,const half* up,half* output,std::int64_t elements,std::int64_t width) {
    const auto i=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=elements)return;
    const auto at=Joined?i/width*(2*width)+i%width:i;
    const float g=__half2float(gate[at]),u=__half2float(Joined?gate[at+width]:up[i]);
    output[i]=__float2half_rn(rounded((g*(1.f/(1.f+expf(-g))))*u));
}
__device__ __forceinline__ float convolution_value(const float* x,const float* dynamic,const float* base,
        const float* residual,int row,int channel,int width,int group,bool finish) {
    const auto i=int64_t(row)*width+channel;
    const int groups=width/group;
    const int dynamic_offset = row * groups * 4 + int(finish) * groups * 2;
    const int base_offset = int(finish) * width * 2;
    float result = x[i] * (base[base_offset + channel] + dynamic[dynamic_offset + channel / group]);
    if (row) result += x[i - width] * (base[base_offset + width + channel] + dynamic[dynamic_offset + groups + channel / group]);
    if (finish) result += residual[i];
    return rounded(result);
}
__global__ void conv(const float* x,const float* dynamic,const float* base,const float* residual,
        float* output,int rows,int width,int group,bool finish) {
    const auto i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<int64_t(rows)*width)output[i]=convolution_value(x,dynamic,base,residual,int(i/width),int(i%width),width,group,finish);
}
__global__ void conv_q8(const float* x,const float* dynamic,const float* base,const float* residual,
        float* output,signed char* codes,float* scales,int rows,int width,int group,bool finish,int padded) {
    const int row=blockIdx.y,column=blockIdx.x*256+threadIdx.x;
    const bool live=row<rows && column<width;
    const float value=live?convolution_value(x,dynamic,base,residual,row,column,width,group,finish):0.f;
    if(live)output[int64_t(row)*width+column]=value;
    store_planar_q8(value,codes,scales,row,column,padded);
}
__global__ void make_rotary_table(const std::int64_t* positions,float* table,int rows,int dimension,float theta) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x,half=dimension/2;
    if(i>=rows*half)return;
    const int row=i/half,frequency=i%half;
    const double angle=double(positions[row])*pow(double(theta),-2.*frequency/dimension);
    double sine,cosine;sincos(angle,&sine,&cosine);
    table[row*dimension+frequency]=float(cosine);
    table[row*dimension+half+frequency]=float(sine);
}

template<bool Tabled>
__global__ void normalize_rotate(const float* x, const float* norm, const std::int64_t* positions,
    float* output, int count, int heads, int dimension, float theta, float epsilon,const float* table=nullptr) {
    const int item = blockIdx.x * 4 + threadIdx.y, lane = threadIdx.x;
    if (item >= count * heads) return;
    const auto* row = x + std::int64_t(item) * dimension;
    float square = 0.f;
    for (int d = lane; d < dimension; d += 32) square += row[d] * row[d];
    const float inverse = rsqrtf(warp_sum(square) / dimension + epsilon);
    for (int d = lane; d < dimension; d += 32) {
        const int half = dimension / 2, partner = d < half ? d + half : d - half;
        const int frequency = d < half ? d : d - half;
        float sine,cosine;
        if constexpr(Tabled) {
            cosine=table[(item/heads)*dimension+frequency];
            sine=table[(item/heads)*dimension+half+frequency];
        }else {
            const double angle = double(positions[item / heads]) * pow(double(theta), -2. * frequency / dimension);
            double s,c;sincos(angle,&s,&c);sine=float(s);cosine=float(c);
        }
        const float direct = rounded(row[d] * inverse * norm[d]);
        const float other = rounded(row[partner] * inverse * norm[partner]);
        output[std::int64_t(item) * dimension + d] = rounded(direct * float(cosine) +
            (d < half ? -other : other) * float(sine));
    }
}

template<class T> __device__ __forceinline__ float history_value(T value) {
    if constexpr(std::is_same_v<T,float>)return value;
    else return __uint_as_float(std::uint32_t(value)<<16);
}
template<class T> __device__ __forceinline__ float4 history_four(const T* input) {
    if constexpr(std::is_same_v<T,float>)return *reinterpret_cast<const float4*>(input);
    else {
        // Load four BF16 cells as one aligned 64-bit vector. Individual u16
        // fields otherwise generated extra loads and unpack instructions.
        const auto v=*reinterpret_cast<const uint2*>(input);
        return make_float4(__uint_as_float(v.x<<16),__uint_as_float(v.x&0xffff0000u),
            __uint_as_float(v.y<<16),__uint_as_float(v.y&0xffff0000u));
    }
}
template<class History>
__global__ void pack_context(const float* key,const float* value,const float* norm,
    const float* table,const std::int64_t* positions,History* ring_key,History* ring_value,
    int rows,int heads,int dimension,int window,float epsilon) {
    const int item=blockIdx.x*4+threadIdx.y,lane=threadIdx.x;
    if(item>=rows*heads)return;
    const auto* input=key+int64_t(item)*dimension;
    float square=0.f;for(int d=lane;d<dimension;d+=32)square+=input[d]*input[d];
    const float inverse=rsqrtf(warp_sum(square)/dimension+epsilon);
    const int row=item/heads,head=item%heads,half=dimension/2;
    const auto offset=((positions[row]%window)*heads+head)*dimension;
    for(int d=lane;d<dimension;d+=32) {
        const int partner=d<half?d+half:d-half,frequency=d<half?d:d-half;
        const float direct=rounded(input[d]*inverse*norm[d]);
        const float other=rounded(input[partner]*inverse*norm[partner]);
        const float rotated=rounded(direct*table[row*dimension+frequency]+
            (d<half?-other:other)*table[row*dimension+half+frequency]);
        if constexpr(std::is_same_v<History,float>) {
            ring_key[offset+d]=rotated;ring_value[offset+d]=value[int64_t(item)*dimension+d];
        }else {
            ring_key[offset+d]=std::uint16_t(__float_as_uint(rotated)>>16);
            ring_value[offset+d]=std::uint16_t(__float_as_uint(rounded(value[int64_t(item)*dimension+d]))>>16);
        }
    }
}
// Split the keys into 128-token tiles, then merge online-softmax partials.
// This follows Strata's DFlash2 attention design with runtime geometry.
template<int Values,class History=float>
__global__ void attend(const float* query, const History* ring_key, const History* ring_value,
    const float* block_key, const float* block_value, float* partial,
    int rows, int heads, int kv_heads, int dimension, int window, std::int64_t position,
    const std::int64_t* device_position) {
    if(device_position)position=device_position[0];
    const int item = blockIdx.x * 4 + threadIdx.y, lane = threadIdx.x, chunk = blockIdx.y;
    if (item >= rows * heads) return;
    const int row = item / heads, head = item % heads, kv = head / (heads / kv_heads);
    const auto begin = max(std::int64_t(0), position - window + 1);
    const int past = int(position - begin), total = past + rows;
    float maximum = -INFINITY, mass = 0, accumulator[Values]{};
    float q[Values];
    #pragma unroll
    for (int i = 0; i < Values; ++i) q[i] = lane + i * 32 < dimension ?
        query[std::int64_t(item) * dimension + lane + i * 32] : 0.f;
    for (int index = chunk * 128; index < min(total, (chunk + 1) * 128); ++index) {
        const auto logical = begin + index;
        if (index < past && logical < position + row - window + 1) continue;
        const auto offset = index < past ? ((logical % window) * kv_heads + kv) * dimension :
            (std::int64_t(index - past) * kv_heads + kv) * dimension;
        const auto key=[&](int d){return index<past?history_value(ring_key[offset+d]):block_key[offset+d];};
        const auto value=[&](int d){return index<past?history_value(ring_value[offset+d]):block_value[offset+d];};
        float score = 0;
        #pragma unroll
        for (int i = 0; i < Values; ++i) if (lane + i * 32 < dimension)
            score=__fmaf_rn(q[i],key(lane+i*32),score);
        score = warp_sum(score) * rsqrtf(float(dimension));
        const float next = fmaxf(maximum, score), alpha = __expf(maximum - next), probability = __expf(score - next);
        mass=__fmaf_rn(mass,alpha,probability); maximum = next;
        #pragma unroll
        for (int i = 0; i < Values; ++i) if (lane + i * 32 < dimension)
            accumulator[i]=__fmaf_rn(accumulator[i],alpha,__fmul_rn(probability,value(lane+i*32)));
    }
    auto* target = partial + (std::int64_t(chunk) * rows * heads + item) * (dimension + 2);
    if (!lane) { target[0] = maximum; target[1] = mass; }
    #pragma unroll
    for (int i = 0; i < Values; ++i) if (lane + i * 32 < dimension) target[2 + lane + i * 32] = accumulator[i];
}
#include "dflash2_attention_tiled.cuh"

template<int Values>
__global__ void merge(const float* partial, float* output, int count, int dimension, int chunks) {
    const int item = blockIdx.x * 4 + threadIdx.y, lane = threadIdx.x;
    if (item >= count) return;
    float maximum = -INFINITY, mass = 0, accumulator[Values]{};
    for (int chunk = 0; chunk < chunks; ++chunk)
        maximum = fmaxf(maximum, partial[(std::int64_t(chunk) * count + item) * (dimension + 2)]);
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const auto* source = partial + (std::int64_t(chunk) * count + item) * (dimension + 2);
        const float scale = source[1] > 0 ? __expf(source[0] - maximum) : 0;
        mass += source[1] * scale;
        #pragma unroll
        for (int i = 0; i < Values; ++i) if (lane + i * 32 < dimension)
            accumulator[i] += source[2 + lane + i * 32] * scale;
    }
    #pragma unroll
    for (int i = 0; i < Values; ++i) if (lane + i * 32 < dimension)
        output[std::int64_t(item) * dimension + lane + i * 32] = rounded(mass > 0 ? accumulator[i] / mass : 0.f);
}
// GPU chain scheduling follows ExLlamaV3 (MIT; ../../../third_party/exllamav3.LICENSE).
// Keep the predecessor chain on the GPU.
// Dot-product order and tie rules here match this backend's existing selector.
template<bool CompactSuccessor>
__global__ void select_greedy_walk(const std::int64_t* compact,int rows,int count,const std::int64_t* anchor,
        const std::uint16_t* predecessor,const std::uint16_t* successor,const float* hidden,
        std::int64_t* output,int rank,int vocabulary,bool pack_probability) {
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    __shared__ float scores[64];
    __shared__ std::int64_t previous;
    if(!threadIdx.x)previous=anchor[0];
    __syncthreads();
    for(int row=0;row<rows;++row) {
        for(int candidate=warp;candidate<count;candidate+=8) {
            const auto packed=std::uint64_t(compact[std::int64_t(row)*count+candidate]);
            const int token=int(packed&0xffffffffu);
            float edge=0.f,score=-CUDART_INF_F;
            if(token>=0 && token<vocabulary && previous>=0 && previous<vocabulary) {
                for(int d=lane;d<rank;d+=32)
                    edge+=(__uint_as_float(std::uint32_t(predecessor[previous*rank+d])<<16)*hidden[std::int64_t(row)*rank+d])*
                        __uint_as_float(std::uint32_t(successor[(CompactSuccessor?std::int64_t(row)*count+candidate:token)*rank+d])<<16);
                edge=warp_sum(edge);
                score=__uint_as_float(std::uint32_t(packed>>32))+edge;
            }
            if(!lane)scores[candidate]=score;
        }
        __syncthreads();
        if(!threadIdx.x) {
            int best=0;float score=-CUDART_INF_F;
            for(int candidate=0;candidate<count;++candidate) {
                const int token=int(std::uint64_t(compact[std::int64_t(row)*count+candidate])&0xffffffffu);
                const float value=scores[candidate];
                if(token>=0 && token<vocabulary && (value>score || (value==score && token<best))) {score=value;best=token;}
            }
            previous=best;
            if(pack_probability) {
                float mass=0.f;
                if(isfinite(score))for(int candidate=0;candidate<count;++candidate)
                    mass+=__expf(scores[candidate]-score);
                const float probability=mass>0.f?1.f/mass:0.f;
                output[row]=std::int64_t((std::uint64_t(__float_as_uint(probability))<<32)|std::uint32_t(best));
            } else output[row]=best;
        }
        __syncthreads();
    }
}
template<bool CompactSuccessor>
__global__ void select(const std::int64_t* compact, int count, const std::int64_t* previous,
    const std::uint16_t* predecessor, const std::uint16_t* successor, const float* hidden,
    float* output, int rank, int vocabulary) {
    const int candidate = blockIdx.x * 4 + threadIdx.y, lane = threadIdx.x;
    if (candidate >= count) return;
    const auto packed = std::uint64_t(compact[candidate]);
    const int token = int(packed & 0xffffffffu);
    const auto prev = previous[0];
    if (token < 0 || token >= vocabulary || prev < 0 || prev >= vocabulary) return;
    float edge = 0.f;
    for (int d = lane; d < rank; d += 32)
        edge += (__uint_as_float(std::uint32_t(predecessor[prev * rank + d]) << 16) * hidden[d]) *
            __uint_as_float(std::uint32_t(successor[std::int64_t(CompactSuccessor?candidate:token) * rank + d]) << 16);
    edge = warp_sum(edge);
    if (!lane) output[token] = __uint_as_float(std::uint32_t(packed >> 32)) + edge;
}
}
void rms_norm(const float* input,const float* weight,float* output,int rows,int width,float epsilon,cudaStream_t stream) {
    if(rows<=0 || width<=0)return;
    // Long rows amortize a full CTA; short vectors keep the compact warp path.
    if(width>=256)normalized_block<false><<<rows,256,0,stream>>>(input,weight,output,nullptr,nullptr,rows,width,width,epsilon);
    else normalized<<<(rows+3)/4,dim3(32,4),0,stream>>>(input,weight,output,rows,width,epsilon);
    check();
}
void rms_norm_q8(const float* input,const float* weight,float* output,void* codes,float* scales,
    int rows,int width,float epsilon,cudaStream_t stream) {
    if(rows<1 || rows>8 || width<=0 || width%32 || width>INT_MAX-511)
        throw std::invalid_argument("DFlash2 fused RMS/Q8 requires 1..8 rows and complete quant blocks");
    const int padded=(width+511)/512*512;
    normalized_block<true><<<8,256,0,stream>>>(input,weight,output,static_cast<signed char*>(codes),scales,rows,width,padded,epsilon);check();
    if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(stream,
        "dflash2_rms_q8",reinterpret_cast<const void*>(normalized_block<true>),
        "M="+std::to_string(rows)+" width="+std::to_string(width));
}
void swiglu(const void* gate,const void* up,void* output,std::int64_t elements,cudaStream_t stream) {
    if(elements<=0)return;
    activate<false><<<unsigned((elements+255)/256),256,0,stream>>>(static_cast<const half*>(gate),
        static_cast<const half*>(up),static_cast<half*>(output),elements,0);check();
}
void swiglu_joined(const void* gate_up,void* output,std::int64_t elements,std::int64_t width,cudaStream_t stream) {
    if(width<=0 || elements%width)throw std::invalid_argument("DFlash2 joined Gate/Up width differs");
    if(elements<=0)return;
    activate<true><<<unsigned((elements+255)/256),256,0,stream>>>(static_cast<const half*>(gate_up),nullptr,
        static_cast<half*>(output),elements,width);check();
}
void convolution(const float* input, const float* dynamic, const float* base, const float* residual,
    float* output, int rows, int width, int group, bool finish, cudaStream_t stream) {
    conv<<<unsigned((std::int64_t(rows) * width + 255) / 256), 256, 0, stream>>>(input, dynamic, base, residual, output, rows, width, group, finish); check();
}
void convolution_q8(const float* input,const float* dynamic,const float* base,const float* residual,
        float* output,void* codes,float* scales,int rows,int width,int group,bool finish,cudaStream_t stream) {
    if(rows<1 || rows>8 || width<=0 || width%32 || width>INT_MAX-511 || group<=0 || width%group || (finish && !residual))
        throw std::invalid_argument("DFlash2 fused convolution/Q8 geometry differs");
    const int padded=(width+511)/512*512;
    conv_q8<<<dim3(padded/256,8),256,0,stream>>>(input,dynamic,base,residual,output,
        static_cast<signed char*>(codes),scales,rows,width,group,finish,padded);check();
    if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(stream,
        "dflash2_convolution_q8",reinterpret_cast<const void*>(conv_q8),"M="+std::to_string(rows)+" width="+std::to_string(width));
}
void rotary_table(const std::int64_t* positions,float* table,int rows,int dimension,float theta,cudaStream_t stream) {
    if(rows<=0 || dimension<=0 || dimension%2)throw std::invalid_argument("DFlash2 rotary table geometry differs");
    make_rotary_table<<<(std::int64_t(rows)*(dimension/2)+255)/256,256,0,stream>>>(positions,table,rows,dimension,theta);check();
}
void norm_rope_tabled(const float* input,const float* norm,const float* table,float* output,
    int rows,int heads,int dimension,float epsilon,cudaStream_t stream) {
    if(rows<=0 || heads<=0 || dimension<=0 || dimension%2)throw std::invalid_argument("DFlash2 tabled RoPE geometry differs");
    normalize_rotate<true><<<(rows*heads+3)/4,dim3(32,4),0,stream>>>(input,norm,nullptr,output,rows,heads,dimension,0.f,epsilon,table);check();
}
void norm_rope(const float* input, const float* norm, const std::int64_t* positions,
    float* output, int rows, int heads, int dimension, float theta, float epsilon, cudaStream_t stream) {
    normalize_rotate<false><<<(rows * heads + 3) / 4, dim3(32, 4), 0, stream>>>(input, norm, positions, output, rows, heads, dimension, theta, epsilon); check();
}
void store_context(const float* key,const float* value,const float* norm,
    const float* table,const std::int64_t* positions,void* ring_key,void* ring_value,
    int rows,int heads,int dimension,int window,float epsilon,cudaStream_t stream,bool history_bf16) {
    if(rows<1 || heads<1 || dimension<1 || dimension%2 || window<rows || !std::isfinite(epsilon) || epsilon<=0)
        throw std::invalid_argument("DFlash2 context storage geometry differs");
    const auto launch=[&]<class History>() {
        pack_context<History><<<(rows*heads+3)/4,dim3(32,4),0,stream>>>(key,value,norm,table,positions,
            static_cast<History*>(ring_key),static_cast<History*>(ring_value),rows,heads,dimension,window,epsilon);
    };
    if(history_bf16)launch.template operator()<std::uint16_t>();else launch.template operator()<float>();check();
}
void attention(const float* query, const void* ring_key, const void* ring_value,
    const float* block_key, const float* block_value, float* partial, float* output,
    int rows, int heads, int kv_heads, int dimension, int window, std::int64_t position, cudaStream_t stream,
    const std::int64_t* device_position,bool history_bf16) {
    const int chunks = (window - 1 + rows + 127) / 128;
    const auto storage=[&]<class History>() {
    const auto* keys=static_cast<const History*>(ring_key);const auto* values=static_cast<const History*>(ring_value);
    const auto launch = [&]<int Values>() {
        if constexpr(Values==2 || Values==4) {
            constexpr int Width=Values*32;
            if(dimension==Width) {
            const int query_tiles=(rows*(heads/kv_heads)+31)/32;
            const auto kernel=attend_tiled<Width,History>;
            kernel<<<dim3(kv_heads*query_tiles,chunks),128,0,stream>>>(query,keys,values,block_key,block_value,
                partial,rows,heads,kv_heads,window,position,device_position);
            if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(stream,
                "dflash2_attention_tiled",reinterpret_cast<const void*>(kernel),
                "M="+std::to_string(rows)+" heads="+std::to_string(heads)+" kv_heads="+std::to_string(kv_heads)+
                " width="+std::to_string(Width)+" query_tiles="+std::to_string(query_tiles)+
                " kv_bits="+std::to_string(int(sizeof(History))*8));
            }else attend<Values,History><<<dim3((rows*heads+3)/4,chunks),dim3(32,4),0,stream>>>(
                query,keys,values,block_key,block_value,partial,rows,heads,kv_heads,
                dimension,window,position,device_position);
        }else {
            attend<Values,History><<<dim3((rows * heads + 3) / 4, chunks), dim3(32, 4), 0, stream>>>(query,keys,values, block_key, block_value,
                partial, rows, heads, kv_heads, dimension, window, position, device_position);
        }
        merge<Values><<<(rows * heads + 3) / 4, dim3(32, 4), 0, stream>>>(partial, output, rows * heads, dimension, chunks);
    };
    if (dimension <= 64) launch.template operator()<2>();
    else if (dimension <= 128) launch.template operator()<4>();
    else if (dimension <= 256) launch.template operator()<8>();
    else if (dimension <= 512) launch.template operator()<16>();
    else throw std::invalid_argument("DFlash2 attention head exceeds the CUDA tile capacity");
    };
    if(history_bf16)storage.template operator()<std::uint16_t>();else storage.template operator()<float>();
    check();
}
void selector_greedy_walk(const std::int64_t* compact,int rows,int candidates,const std::int64_t* previous,
        const std::uint16_t* predecessor,const std::uint16_t* successor,const float* hidden,
        std::int64_t* output,int rank,int vocabulary,cudaStream_t stream,bool successor_is_compact,bool pack_probability) {
    if(rows<1 || candidates<1 || candidates>64 || rank<1 || vocabulary<1)
        throw std::invalid_argument("DFlash2 greedy selector geometry differs");
    if(successor_is_compact)select_greedy_walk<true><<<1,256,0,stream>>>(compact,rows,candidates,previous,predecessor,successor,hidden,output,rank,vocabulary,pack_probability);
    else select_greedy_walk<false><<<1,256,0,stream>>>(compact,rows,candidates,previous,predecessor,successor,hidden,output,rank,vocabulary,pack_probability);
    check();
}
void selector_logits(const std::int64_t* compact, int candidates, const std::int64_t* previous,
    const std::uint16_t* predecessor, const std::uint16_t* successor, const float* hidden,
    float* output, int rank, int vocabulary, cudaStream_t stream,bool successor_is_compact) {
    const auto launch=[&]<bool Compact>() {
        select<Compact><<<(candidates + 3) / 4, dim3(32, 4), 0, stream>>>(compact, candidates, previous, predecessor, successor, hidden, output, rank, vocabulary);
    };
    if(successor_is_compact)launch.template operator()<true>();else launch.template operator()<false>();
    check();
}
}
