#include "gguf.h"
#include "gguf_compact.h"
#include "gguf_route_groups.h"
#include "tyloinfer/gguf_quant.h"
#include "ggml-cuda/common.cuh"
#include "gguf_decode.cuh"
#include "gguf_dot.cuh"
#include "warp_sums.cuh"
#include "gguf_routed_rows.cuh"
#include "gguf_routed_shared.cuh"
#include "gguf_quantize.cuh"
#include "gguf_gate_up.cuh"
#include "gguf_dense_rows.cuh"
#include "gguf_dense_interleaved.cuh"
#include "../runtime/moe_pipeline.h"
#include "kernel_dispatch_trace.h"
#include <algorithm>
#include <atomic>
#include <climits>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

namespace mfq::cuda::gguf {
namespace {
template<int Format> inline constexpr bool has_decoded_weight=requires {typename PreparedDot<Format>::DecodedWeight;};
constexpr int projection_reduction_warps(std::int64_t parts) {
    // Populate complete warps on short axes without issuing multiple passes
    // through the same row. Large reductions retain the four-warp tile.
    if(parts<64)return 1;
    const auto needed=(parts+31)/32;
    return int(needed<4?needed:4);
}
__global__ void copy_cpu_input(const float* input,float* mapped,std::int64_t count) {
    const auto i=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count)mapped[i]=input[i];
}
template<bool Half>
__global__ void reduce_expert_pairs(const float* pairs,const float* weights,std::int64_t tokens,
        std::int64_t routes,std::int64_t columns,void* output,const volatile float* cpu_values,
        const std::int32_t* cpu_slots) {
    for(auto i=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<tokens*columns;
            i+=std::int64_t(gridDim.x)*blockDim.x) {
        const auto token=i/columns,column=i%columns;
        float sum=0;
        for(std::int64_t route=0;route<routes;++route) {
            const auto slot=token*routes+route,index=slot*columns+column;
            float value=cpu_slots && cpu_slots[slot]?cpu_values[index]:pairs[index];
            if constexpr(Half)value=__half2float(__float2half_rn(value));
            sum+=weights[slot]*value;
        }
        if constexpr(Half)static_cast<half*>(output)[i]=__float2half_rn(sum);
        else static_cast<float*>(output)[i]=sum;
    }
}
__global__ void snapshot_direct_cpu_plan(const std::uint64_t* source,const std::uint32_t* cancelled,
        std::uint64_t* destination,std::uint64_t capacity,const volatile std::int32_t* cpu_slots,
        std::int32_t* device_slots,std::int64_t slots,const RouteGroup* host_groups,RouteGroup* device_groups,
        std::int64_t group_capacity) {
    __shared__ std::uint64_t count;
    __shared__ std::uint32_t abort;
    __shared__ int group_count;
    if(!threadIdx.x) {
        count=source[0];const bool empty=count==~std::uint64_t(0);if(empty)count=0;
        abort=*reinterpret_cast<const volatile std::uint32_t*>(cancelled) || count>capacity;
        destination[0]=abort?0:count;destination[1]=source[1];
        destination[2]=abort?1u:empty?2u:0u;destination[3]=abort || empty?0:source[3];
        group_count=host_groups?host_groups[0].count:0;
        if(group_count<0 || group_count>group_capacity)group_count=0;
        if(device_groups)device_groups[0].count=abort?0:group_count;
    }
    __syncthreads();
    // Each small host control word crosses PCIe once. The reduction reads only
    // this VRAM snapshot; its inner route loop must never read mapped flags.
    for(auto i=std::int64_t(threadIdx.x);i<slots;i+=blockDim.x)device_slots[i]=abort?0:cpu_slots[i];
    if(abort)return;
    if(host_groups && device_groups) {
        const auto* input=reinterpret_cast<const volatile std::uint32_t*>(host_groups+1);
        auto* output=reinterpret_cast<std::uint32_t*>(device_groups+1);
        for(auto i=std::int64_t(threadIdx.x);i<std::int64_t(group_count)*sizeof(RouteGroup)/4;i+=blockDim.x)output[i]=input[i];
    }
    for(auto word=std::uint64_t(threadIdx.x);word<count*4;word+=blockDim.x)destination[4+word]=source[4+word];
}
template<typename Input>
__global__ void quantize_q8(const Input* input, std::int64_t count, block_q8_1* output) {
    const auto i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count)store_q8(input[i],i,output); // complete 32-lane blocks
}
template<bool Half>
__global__ void swiglu_q8(const float* gate,const float* up,std::int64_t count,float limit,block_q8_1* output) {
    const auto i=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=count)return;
    const auto value=swiglu_q8_value<Half>(gate[i],up[i],limit);
    store_q8(value,i,output);
}

template<int Format,int Rows>
__global__ void project_dense_q8_rows(const std::uint8_t* packed,std::uint64_t stride,
        std::int64_t rows,std::int64_t columns,int tokens,const block_q8_1* input,void* output,bool half_output) {
    const auto row=std::int64_t(blockIdx.x)*4+threadIdx.y;
    if(row>=rows)return;
    float sums[Rows]{};
    q8_dot_row_partials<Format,Rows>(packed+row*stride,input,columns,threadIdx.x,tokens,sums);
    constexpr int Capacity=warp_sums::capacity(Rows);
    float values[Capacity]{};
    #pragma unroll
    for(int t=0;t<Rows;++t)values[t]=sums[t];
    const int lane=threadIdx.x,token=warp_sums::index<Capacity>(lane);
    const float sum=warp_sums::reduce<Capacity>(values,lane);
    if(token<tokens && lane==warp_sums::owner<Capacity>(token))
        store_projection(output,std::int64_t(token)*rows+row,sum,half_output);
}

template<bool PositionMajor>
__global__ void swiglu_paired_half_q8(const half* input,int64_t tokens,int64_t width,block_q8_1* output) {
    const auto i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=tokens*width)return;
    const auto at=(i/width)*(2*width)+i%width;
    const float value=swiglu_q8_value<true>(__half2float(input[at]),__half2float(input[at+width]),0.f);
    if constexpr(PositionMajor)store_q8_position_major(value,i,width,int(tokens),output);
    else store_q8(value,i,output);
}

#include "gguf_dense_group.cuh"

template<int Format, bool Routed>
__global__ void project_q8(const std::uint8_t* packed, const std::uint64_t* addresses,
    std::uint64_t stride, std::int64_t experts, std::int64_t rows, std::int64_t columns,
    const std::int64_t* ids, std::int64_t tokens, std::int64_t routes,
    const block_q8_1* input, bool routed_input, void* output,bool preserve_missing=false,bool half_output=false) {
    const auto index = std::int64_t(blockIdx.x) * 4 + threadIdx.y;
    const auto count = Routed ? tokens * routes * rows : rows;
    if (index >= count) return;
    const auto row = Routed ? index % rows : index;
    const auto slot = Routed ? index / rows : 0;
    const auto expert = Routed ? ids[slot] : 0;
    const auto address = Routed ? (expert >= 0 && expert < experts ? addresses[expert] : 0) :
        reinterpret_cast<std::uint64_t>(packed);
    const auto* weight = reinterpret_cast<const std::uint8_t*>(address) + row * stride;
    const int lane = threadIdx.x;
    for (std::int64_t token = 0; token < (Routed ? 1 : tokens); ++token) {
        const auto input_row = Routed ? (routed_input ? slot : slot / routes) : token;
        const float value = address ? q8_dot<Format>(weight, input + input_row * (columns / 32), columns, lane) : 0.f;
        if (!lane && (address || !preserve_missing)) {
            if constexpr(Routed)store_projection(static_cast<float*>(output),index,value);
            else store_projection(output,token*rows+row,value,half_output);
        }
    }
}
// A one-warp row leaves only ceil(N/4) CTAs. Cooperating on K gives
// each output its own CTA when that grid cannot fill the actual resident
// capacity. Large matrices keep the lower-overhead four-output layout.
template<int Format>
bool single_token_dense_cooperative(std::int64_t rows,std::int64_t parts,std::uint64_t group_blocks=0) {
    constexpr bool supported=Format==GGML_TYPE_Q8_0 || Format==GGML_TYPE_Q5_K ||
        Format==GGML_TYPE_Q6_K || Format==GGML_TYPE_IQ4_XS;
    if constexpr(!supported)return false;
    else {
        if(parts<64 || rows<1 || rows>INT_MAX)return false;
        if(const auto* option=std::getenv("TYLOINFER_GGUF_DENSE_M1_COOPERATIVE");option && option[0]=='0')return false;
        int device=0,sms=0,active=0;
        const auto checked=[](cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));};
        checked(cudaGetDevice(&device));checked(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device));
        const void* function=group_blocks?reinterpret_cast<const void*>(project_dense_group<1>):
            reinterpret_cast<const void*>(project_q8<Format,false>);
        checked(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,function,128,0));
        const auto blocks=group_blocks?group_blocks:std::uint64_t((rows+3)/4);
        return active>0 && blocks<std::uint64_t(sms)*std::uint64_t(active);
    }
}
__global__ void snapshot_addresses(const std::uint64_t* mapped,std::uint64_t* device,
        const std::int64_t* ids,std::int64_t slots,std::int64_t experts,const std::uint32_t* admission_ready) {
    if(admission_ready) {
        if(!threadIdx.x) {
            const auto* observed=reinterpret_cast<const volatile std::uint32_t*>(admission_ready);
            while(!*observed) {
#if __CUDA_ARCH__ >= 700
                __nanosleep(64);
#endif
            }
            __threadfence_system();
        }
        __syncthreads();
    }
    const auto slot=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(slot>=slots)return;
    const auto expert=ids[slot];
    if(expert<0 || expert>=experts)return;
    for(std::int64_t earlier=0;earlier<slot;++earlier)if(ids[earlier]==expert)return;
    for(int role=0;role<3;++role)device[role*experts+expert]=mapped[role*experts+expert];
}
__global__ void group_routes(const std::int64_t* ids,std::int64_t count,int capacity,RouteGroup* groups) {
    const auto slot=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(slot>=count)return;
    const auto expert=ids[slot];
    int previous=0;
    for(std::int64_t i=0;i<slot;++i)previous+=ids[i]==expert;
    // Repeated routes are allowed. Split an expert's occurrences into bounded
    // groups, preserving each routed activation and every output slot.
    if(previous%capacity)return;
    auto& group=groups[1+atomicAdd(&groups[0].count,1)];group.count=0;
    group.expert=expert;
    for(auto i=slot;i<count && group.count<capacity;++i)
        if(ids[i]==expert)group.slots[group.count++]=int(i);
}
template<int Format>
__global__ void project_grouped_q8(const std::uint64_t* addresses,std::uint64_t stride,
        std::int64_t experts,std::int64_t rows,std::int64_t columns,const RouteGroup* groups,
        std::int64_t routes,const block_q8_1* input,bool routed_input,float* output,bool preserve_missing) {
    const auto tiles=(rows+3)/4;
    // Separate the two dimensions: no 64-bit division is needed to recover
    // an expert group and an output row inside the persistent work loop.
    for(int index=blockIdx.y;index<groups[0].count;index+=gridDim.y) {
    const auto* group=groups+1+index;
    for(auto tile=std::int64_t(blockIdx.x);tile<tiles;tile+=gridDim.x) {
        const auto row=tile*4+threadIdx.y;
        if(row>=rows)continue;
        const auto address=group->expert>=0 && group->expert<experts?addresses[group->expert]:0;
        if(group->count==1) {
            const auto slot=group->slots[0],input_row=routed_input?slot:slot/int(routes);
            const float value=address?q8_dot<Format>(reinterpret_cast<const std::uint8_t*>(address)+row*stride,
                input+std::int64_t(input_row)*(columns/32),columns,threadIdx.x):0.f;
            if(!threadIdx.x && (address || !preserve_missing))output[slot*rows+row]=value;
            continue;
        }
        const auto apply=[&]<int Lanes>() {
            const int token=threadIdx.x/Lanes;
            const bool active=token<group->count;
            const auto slot=active?group->slots[token]:0;
            const auto input_row=routed_input?slot:slot/int(routes);
            const float value=address?q8_dot_subwarp<Format,Lanes>(reinterpret_cast<const std::uint8_t*>(address)+row*stride,
                input+std::int64_t(input_row)*(columns/32),columns,threadIdx.x,active):0.f;
            if(threadIdx.x%Lanes==0 && active && (address || !preserve_missing))output[slot*rows+row]=value;
        };
        if(group->count<=2)apply.template operator()<16>();
        else if(group->count<=4)apply.template operator()<8>();
        else apply.template operator()<4>();
    }
    }
}
__device__ float staged_value(int type, const std::uint8_t* row, std::uint64_t column) {
    switch (type) {
#define TYLOINFER_GGUF_VALUE(name) case GGML_TYPE_##name: return gguf_value<GGML_TYPE_##name>(row, column);
        TYLOINFER_GGUF_FORMATS(TYLOINFER_GGUF_VALUE)
#undef TYLOINFER_GGUF_VALUE
        default: return 0.f;
    }
}

template<typename Output>
__global__ void decode_packet(const std::uint8_t* packet, std::int64_t rows,
    std::int64_t columns, Output* output) {
    const auto* descriptors = reinterpret_cast<const StagedRow*>(packet);
    for (auto i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
            i < rows * columns; i += std::int64_t(gridDim.x) * blockDim.x) {
        const auto descriptor = descriptors[i / columns];
        output[i] = Output(staged_value(descriptor.type, packet + descriptor.offset, i % columns));
    }
}

template<int Format, typename Output, bool Embedding>
__global__ void decode_rows(const std::uint8_t* packed, std::uint64_t stride,
    std::int64_t rows, std::int64_t columns, const std::int64_t* ids,
    std::int64_t count, Output* output) {
    const auto index = (std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x)*8;
    const auto step = std::int64_t(gridDim.x) * blockDim.x*8;
    for (auto i = index; i < count * columns; i += step) {
        const auto row = Embedding ? ids[i / columns] : i / columns;
        float values[8]{};
        if(row>=0 && row<rows)gguf_values8<Format>(packed+std::uint64_t(row)*stride,i%columns,values);
        gguf_store8(output+i,values);
    }
}

// One warp per output row; all tokens reuse the packed row in the same launch.
// This exact FP32 path also provides a reference for subsequent integer dots.
template<int Format>
__global__ void project_rows(const std::uint8_t* packed, std::uint64_t stride,
    std::int64_t rows, std::int64_t columns, const float* input,
    std::int64_t tokens, float* output) {
    const auto row = std::int64_t(blockIdx.x) * 4 + threadIdx.y;
    if (row >= rows) return;
    const int lane = threadIdx.x;
    for (std::int64_t token = 0; token < tokens; ++token) {
        float sum = 0.f;
        for (std::int64_t k = lane; k < columns; k += 32)
            sum = fmaf(gguf_value<Format>(packed + std::uint64_t(row) * stride, k),
                input[token * columns + k], sum);
        for (int offset = 16; offset; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (!lane) output[token * rows + row] = sum;
    }
}

void checked_launch() {
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

template<int Format>
__global__ void project_experts(const std::uint64_t* addresses, std::uint64_t stride,
    std::int64_t experts, std::int64_t rows, std::int64_t columns, const std::int64_t* ids,
    std::int64_t slots, std::int64_t routes, const float* input, bool routed_input, float* output) {
    const auto index = std::int64_t(blockIdx.x) * 4 + threadIdx.y;
    if (index >= slots * rows) return;
    const auto slot = index / rows, row = index % rows, expert = ids[slot];
    const auto address = expert >= 0 && expert < experts ? addresses[expert] : 0;
    const int lane = threadIdx.x;
    float sum = 0.f;
    if (address) {
        const auto* packed = reinterpret_cast<const std::uint8_t*>(address) + row * stride;
        const auto* x = input + (routed_input ? slot : slot / routes) * columns;
        for (std::int64_t k = lane; k < columns; k += 32)
            sum = fmaf(gguf_value<Format>(packed, k), x[k], sum);
    }
    for (int offset = 16; offset; offset >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (!lane) output[index] = sum;
}

template<typename Output, bool Embedding>
void decode(ggml_type type, const void* packed, std::uint64_t stride,
    std::int64_t rows, std::int64_t columns, const std::int64_t* ids,
    std::int64_t count, Output* output, cudaStream_t stream) {
    if (count == 0) return;
    const auto blocks = static_cast<unsigned>(std::min<std::int64_t>(65535, (count * columns + 2047) / 2048));
    switch (type) {
#define TYLOINFER_GGUF_DECODE(name) case GGML_TYPE_##name: \
        decode_rows<GGML_TYPE_##name, Output, Embedding><<<blocks, 256, 0, stream>>>( \
            static_cast<const std::uint8_t*>(packed), stride, rows, columns, ids, count, output); break;
        TYLOINFER_GGUF_FORMATS(TYLOINFER_GGUF_DECODE)
#undef TYLOINFER_GGUF_DECODE
        default: throw std::invalid_argument("unsupported CUDA GGUF decoder");
    }
    checked_launch();
}
}

void snapshot_direct_plan(const void* mapped,const std::uint32_t* cancelled,void* snapshot,
        std::uint64_t capacity,const std::int32_t* cpu_slots,std::int32_t* device_slots,
        std::int64_t slots,const RouteGroup* host_groups,RouteGroup* device_groups,
        std::int64_t group_capacity,cudaStream_t stream) {
    snapshot_direct_cpu_plan<<<1,256,0,stream>>>(static_cast<const std::uint64_t*>(mapped),cancelled,
        static_cast<std::uint64_t*>(snapshot),capacity,cpu_slots,device_slots,slots,host_groups,device_groups,group_capacity);checked_launch();
}
void publish_cpu_input(const float* input,float* mapped,std::int64_t count,cudaStream_t stream) {
    if(count<=0)return;
    copy_cpu_input<<<unsigned((count+255)/256),256,0,stream>>>(input,mapped,count);checked_launch();
}
void reduce_moe_output(const float* pairs,const float* weights,std::int64_t tokens,
        std::int64_t routes,std::int64_t columns,bool half_output,void* output,
        const float* cpu_values,const std::int32_t* cpu_slots,const std::uint32_t* cpu_ready,cudaStream_t stream,
        std::uint64_t* cpu_waited_until) {
    if(!tokens || !columns)return;
    if(tokens<0 || routes<=0 || columns<0 || tokens>INT64_MAX/columns ||
        ((cpu_values!=nullptr || cpu_slots!=nullptr || cpu_ready!=nullptr) &&
        (!cpu_values || !cpu_slots || !cpu_ready)))throw std::invalid_argument("GGUF routed output geometry");
    const auto blocks=unsigned(std::min<std::int64_t>(65535,(tokens*columns+255)/256));
    // One system-memory flag read for the whole grid. Repeating it in every
    // reduction CTA makes PCIe control latency grow with matrix width.
    if(cpu_ready)mfq::cuda::wait_mapped_flag(cpu_ready,stream,cpu_waited_until);
    const auto launch=[&](auto half) {
        reduce_expert_pairs<decltype(half)::value><<<blocks,256,0,stream>>>(
            pairs,weights,tokens,routes,columns,output,cpu_values,cpu_slots);
    };
    if(half_output)launch(std::true_type{});else launch(std::false_type{});
    checked_launch();
}
void staged_rows(const void* packet, std::int64_t rows, std::int64_t columns,
    void* output, bool half_output, cudaStream_t stream) {
    if (!rows) return;
    const auto blocks = static_cast<unsigned>(std::min<std::int64_t>(65535, (rows * columns + 255) / 256));
    const auto* source = static_cast<const std::uint8_t*>(packet);
    if (half_output) decode_packet<<<blocks, 256, 0, stream>>>(source, rows, columns, static_cast<half*>(output));
    else decode_packet<<<blocks, 256, 0, stream>>>(source, rows, columns, static_cast<float*>(output));
    checked_launch();
}

void dequantize_rows(ggml_type type, const void* packed, std::uint64_t stride,
    std::int64_t rows, std::int64_t columns, void* output, bool half_output, cudaStream_t stream) {
    if (type >= GGML_TYPE_PTQ1_0 && type <= GGML_TYPE_TQK4) {
        compact_dequantize(CompactFormat(type), packed, stride, rows, columns, output, half_output, stream);
        return;
    }
    if (half_output) decode<half, false>(type, packed, stride, rows, columns, nullptr, rows,
        static_cast<half*>(output), stream);
    else decode<float, false>(type, packed, stride, rows, columns, nullptr, rows,
        static_cast<float*>(output), stream);
}

void embedding_rows(ggml_type type, const void* packed, std::uint64_t stride,
    std::int64_t rows, std::int64_t columns, const std::int64_t* ids,
    std::int64_t count, float* output, cudaStream_t stream) {
    decode<float, true>(type, packed, stride, rows, columns, ids, count, output, stream);
}

void matvec(ggml_type type, const void* packed, std::uint64_t stride,
    std::int64_t rows, std::int64_t columns, const float* input,
    std::int64_t tokens, float* output, cudaStream_t stream) {
    if (!tokens || !rows) return;
    if (type >= GGML_TYPE_PTQ1_0 && type <= GGML_TYPE_TQK4) {
        compact_project(CompactFormat(type), packed, stride, rows, columns, input,
            CompactInput::Float32, tokens, output, false, stream);
        return;
    }
    const dim3 threads(32, 4);
    const auto blocks = static_cast<unsigned>((rows + 3) / 4);
    switch (type) {
#define TYLOINFER_GGUF_MATVEC(name) case GGML_TYPE_##name: \
        project_rows<GGML_TYPE_##name><<<blocks, threads, 0, stream>>>( \
            static_cast<const std::uint8_t*>(packed), stride, rows, columns, input, tokens, output); break;
        TYLOINFER_GGUF_FORMATS(TYLOINFER_GGUF_MATVEC)
#undef TYLOINFER_GGUF_MATVEC
        default: throw std::invalid_argument("unsupported CUDA GGUF projection");
    }
    checked_launch();
}

std::uint64_t q8_row_bytes(std::int64_t columns) {
    if (columns <= 0 || columns % 32) throw std::invalid_argument("GGUF Q8 activation requires complete 32-value blocks");
    return std::uint64_t(columns / 32) * sizeof(block_q8_1);
}
void quantize_input(const float* input, std::int64_t tokens, std::int64_t columns,
    void* packed, cudaStream_t stream) {
    q8_row_bytes(columns);
    if (!tokens) return;
    quantize_q8<<<static_cast<unsigned>((tokens * columns + 255) / 256), 256, 0, stream>>>(
        input, tokens * columns, static_cast<block_q8_1*>(packed));
    checked_launch();
}
void quantize_input_f16(const void* input,std::int64_t tokens,std::int64_t columns,void* packed,cudaStream_t stream) {
    q8_row_bytes(columns);if(!tokens)return;
    quantize_q8<<<static_cast<unsigned>((tokens*columns+255)/256),256,0,stream>>>(
        static_cast<const half*>(input),tokens*columns,static_cast<block_q8_1*>(packed));
    checked_launch();
}
bool supports_interleaved_activation(ggml_type type,std::int64_t tokens) noexcept {
    return tokens>=2 && tokens<=4 &&
        (type==GGML_TYPE_IQ3_S || type==GGML_TYPE_IQ3_XXS || type==GGML_TYPE_IQ4_XS);
}
std::uint64_t interleaved_activation_bytes(std::int64_t tokens,std::int64_t columns) {
    q8_row_bytes(columns);
    if(tokens<2 || tokens>4 || columns>INT_MAX)
        throw std::invalid_argument("GGUF interleaved activation geometry differs");
    return std::uint64_t(columns)*(tokens==2?2:4)*9/8;
}
void quantize_input_interleaved(const void* input,bool half_input,std::int64_t tokens,std::int64_t columns,
        void* output,void* native,cudaStream_t stream) {
    interleaved_activation_bytes(tokens,columns);
    if(!input || !output)throw std::invalid_argument("GGUF interleaved activation storage missing");
    const auto blocks=unsigned((tokens*columns+255)/256);
    if(half_input)quantize_position_major<<<blocks,256,0,stream>>>(static_cast<const half*>(input),columns,int(tokens),output,static_cast<block_q8_1*>(native));
    else quantize_position_major<<<blocks,256,0,stream>>>(static_cast<const float*>(input),columns,int(tokens),output,static_cast<block_q8_1*>(native));
    checked_launch();
}
void matvec_interleaved(ggml_type type,const void* packed,std::uint64_t stride,std::int64_t rows,std::int64_t columns,
        const void* input,std::int64_t tokens,void* output,bool half_output,cudaStream_t stream) {
    interleaved_activation_bytes(tokens,columns);
    if(!supports_interleaved_activation(type,tokens) || rows<0 || rows>INT_MAX ||
            columns%ggml_blck_size(type) || stride<std::uint64_t(columns/ggml_blck_size(type))*ggml_type_size(type))
        throw std::invalid_argument("GGUF interleaved projection geometry differs");
    if(!rows)return;
    const auto capacity=tokens==2?2:4;
    const auto* scales=reinterpret_cast<const float*>(static_cast<const std::uint8_t*>(input)+columns*capacity);
    const auto launch=[&]<int Format,int Count>() {
        const auto run=[&]<bool Cooperative,int Warps>() {
            constexpr int Outputs=Cooperative?2:4;
            const auto function=project_dense_position_major<Format,Count,Cooperative,Warps>;
            function<<<unsigned((rows+Outputs-1)/Outputs),dim3(32,Warps),0,stream>>>(
                static_cast<const std::uint8_t*>(packed),stride,rows,int(columns),input,scales,output,half_output);
            if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(stream,
                "gguf_dense_position_major",reinterpret_cast<const void*>(function),
                "format="+std::to_string(Format)+" M="+std::to_string(Count)+" cooperative="+std::to_string(Cooperative)+
                " warps="+std::to_string(Warps));
        };
        if constexpr(Format==GGML_TYPE_IQ4_XS) {
            const int warps=projection_reduction_warps(columns/Dot<Format>::qk*Dot<Format>::parts);
            if(warps==2)run.template operator()<true,2>();
            else if(warps==3)run.template operator()<true,3>();
            else if(warps==4)run.template operator()<true,4>();
            else run.template operator()<false,4>();
        }else run.template operator()<false,4>();
    };
    const auto select=[&]<int Format>() {
        if(tokens==2)launch.template operator()<Format,2>();
        else if(tokens==3)launch.template operator()<Format,3>();
        else launch.template operator()<Format,4>();
    };
    if(type==GGML_TYPE_IQ3_S)select.template operator()<GGML_TYPE_IQ3_S>();
    else if(type==GGML_TYPE_IQ3_XXS)select.template operator()<GGML_TYPE_IQ3_XXS>();
    else select.template operator()<GGML_TYPE_IQ4_XS>();
    checked_launch();
}

void matvec_group(const DenseProjection* projections,std::size_t count,std::int64_t columns,
        const void* native_input,const void* interleaved_input,std::int64_t tokens,
        bool half_output,cudaStream_t stream) {
    if(tokens<1 || tokens>4 || columns<1 || columns>INT_MAX || (!projections && count))
        throw std::invalid_argument("GGUF dense group geometry differs");
    // Judge M1 against all submitted matrices, so a tiny projection beside
    // a large saturated one does not multiply its launch work unnecessarily.
    std::uint64_t group_blocks=0;
    if(tokens==1)for(std::size_t i=0;i<count;++i) {
        if(projections[i].rows>0 && projections[i].rows<=INT_MAX)
            group_blocks+=std::uint64_t(projections[i].rows+3)/4;
    }
    DenseGroupPacket packet{};unsigned blocks=0;
    const auto submit=[&] {
        if(!packet.count)return;
        const auto launch=[&]<int M>() {
            const auto function=project_dense_group<M>;
            function<<<blocks,dim3(32,4),0,stream>>>(packet,int(columns),
                static_cast<const block_q8_1*>(native_input),interleaved_input,half_output);
            if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(stream,
                "gguf_dense_group",reinterpret_cast<const void*>(function),
                "M="+std::to_string(M)+" matrices="+std::to_string(packet.count));
        };
        if(tokens==1)launch.template operator()<1>();
        else if(tokens==2)launch.template operator()<2>();
        else if(tokens==3)launch.template operator()<3>();
        else launch.template operator()<4>();
        checked_launch();packet={};blocks=0;
    };
    for(std::size_t i=0;i<count;++i) {
        const auto& p=projections[i];
        if(p.rows<0 || p.rows>INT_MAX)
            throw std::invalid_argument("GGUF dense group projection differs");
        // Empty matrices require no storage or activation packet. In
        // particular, a zero-size tensor may legitimately have a null pointer.
        if(!p.rows)continue;
        if(!p.packed || !p.output)
            throw std::invalid_argument("GGUF dense group projection storage is absent");
        DenseGroupEntry entry{};entry.projection=p;
        int outputs=4;
        const auto geometry=[&]<int Format>() {
            if(columns%Dot<Format>::qk || p.stride<std::uint64_t(columns/Dot<Format>::qk)*ggml_type_size(p.type))
                throw std::invalid_argument("GGUF dense group row encoding differs");
            entry.position_major=supports_interleaved_activation(p.type,tokens);
            entry.warps=projection_reduction_warps(columns/Dot<Format>::qk*Dot<Format>::parts);
            constexpr bool supports_cooperative=Format==GGML_TYPE_Q8_0 || Format==GGML_TYPE_Q5_K ||
                Format==GGML_TYPE_Q6_K || Format==GGML_TYPE_IQ4_XS;
            entry.cooperative=supports_cooperative && p.rows<=INT_MAX/tokens && p.stride<=INT_MAX && entry.warps>1 &&
                (tokens>1 || single_token_dense_cooperative<Format>(p.rows,columns/Dot<Format>::qk*Dot<Format>::parts,group_blocks));
            if(entry.position_major) {
                if constexpr(Format==GGML_TYPE_IQ4_XS)if(entry.warps>1)outputs=2;
            }else if(entry.cooperative)outputs=tokens==1 || (Format==GGML_TYPE_Q6_K && tokens==2)?1:2;
            if(entry.position_major?!interleaved_input:!native_input)
                throw std::invalid_argument("GGUF dense group activation layout is absent");
        };
        switch(p.type) {
#define TYLOINFER_GGUF_BATCH_GEOMETRY(name) case GGML_TYPE_##name: geometry.template operator()<GGML_TYPE_##name>();break;
            TYLOINFER_GGUF_FORMATS(TYLOINFER_GGUF_BATCH_GEOMETRY)
#undef TYLOINFER_GGUF_BATCH_GEOMETRY
        default:throw std::invalid_argument("GGUF dense group format unsupported");
        }
        const auto tiles=std::uint64_t(p.rows+outputs-1)/outputs;
        // CUDA's x grid dimension is bounded by INT_MAX. Start another packet
        // before a legal matrix would overflow the combined grid.
        if(packet.count==DenseGroupCapacity || std::uint64_t(blocks)+tiles>INT_MAX)submit();
        entry.begin=blocks;blocks+=unsigned(tiles);entry.end=blocks;
        packet.entries[packet.count++]=entry;
    }
    submit();
}

template<int Format,typename Output>
bool try_dense_cooperative(const void* packed,std::uint64_t stride,std::int64_t rows,std::int64_t columns,
        const void* input,std::int64_t tokens,Output* output,cudaStream_t stream) {
    // These formats spend their multi-token time on repeated activation
    // loads/dots. Two to four warps cooperate on the reduction and share a
    // decoded fragment across tokens. Short axes retain the one-warp path;
    // cooperative block size follows the number of independent weight parts.
    constexpr bool supported=Format==GGML_TYPE_Q8_0 || Format==GGML_TYPE_Q5_K ||
        Format==GGML_TYPE_Q6_K || Format==GGML_TYPE_IQ4_XS;
    if constexpr(!supported)return false;
    else {
        if(tokens<1 || tokens>8 || rows>INT_MAX/tokens || columns>INT_MAX || stride>INT_MAX ||
                columns/Dot<Format>::qk*Dot<Format>::parts<64)return false;
        if(tokens==1 && !single_token_dense_cooperative<Format>(rows,columns/Dot<Format>::qk*Dot<Format>::parts))return false;
        const auto launch=[&](auto capacity) {
            constexpr int Count=decltype(capacity)::value;
            // Bound the accumulator and shared-partial footprint to eight
            // output/token pairs. Longer windows process one output row.
            // Q6_K's two-row tile now shares decoded activation words and
            // scales. Three or four tokens amortize that tile; a two-token
            // Q6_K window retains the lighter one-row projection.
            constexpr int Outputs=Count>1 && Count<=4 && (Format!=GGML_TYPE_Q6_K || Count>2)?2:1;
            const auto run=[&]<int Warps>() {
                const auto function=project_dense_cooperative_q8<Format,Count,Outputs,Warps>;
                function<<<unsigned((rows+Outputs-1)/Outputs),dim3(32,Warps),0,stream>>>(
                    static_cast<const std::uint8_t*>(packed),int(stride),int(rows),int(columns),int(tokens),
                    static_cast<const block_q8_1*>(input),output,std::is_same_v<Output,half>);
                if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(stream,
                    "gguf_dense_cooperative",reinterpret_cast<const void*>(function),
                    "format="+std::to_string(Format)+" M="+std::to_string(tokens)+" outputs="+std::to_string(Outputs)+
                    " warps="+std::to_string(Warps));
            };
            const int warps=projection_reduction_warps(columns/Dot<Format>::qk*Dot<Format>::parts);
            if(warps==2)run.template operator()<2>();
            else if(warps==3)run.template operator()<3>();
            else run.template operator()<4>();
        };
        if(tokens==1)launch(std::integral_constant<int,1>{});
        else if(tokens==2)launch(std::integral_constant<int,2>{});
        else if(tokens==3)launch(std::integral_constant<int,3>{});
        else if(tokens==4)launch(std::integral_constant<int,4>{});
        else launch(std::integral_constant<int,8>{});
        return true;
    }
}

template<typename Output>
void matvec_q8_typed(ggml_type type, const void* packed, std::uint64_t stride,
    std::int64_t rows, std::int64_t columns, const void* input,
    std::int64_t tokens, Output* output, cudaStream_t stream) {
    if (!tokens || !rows) return;
    if (type >= GGML_TYPE_PTQ1_0 && type <= GGML_TYPE_TQK4) {
        compact_project(CompactFormat(type), packed, stride, rows, columns, input,
            CompactInput::Q8_1, tokens, output, std::is_same_v<Output, half>, stream);
        return;
    }
    if(tokens>8) {
        // Keep the existing bounded accumulator tiles instead of falling
        // back to one complete weight read per token. Native MTP history
        // expands each token into multiple HC streams before this projection.
        const auto* activation=static_cast<const block_q8_1*>(input);
        for(std::int64_t first=0;first<tokens;first+=8)
            matvec_q8_typed(type,packed,stride,rows,columns,activation+first*(columns/32),
                std::min<std::int64_t>(8,tokens-first),output+first*rows,stream);
        if(mfq::cuda::dispatch_trace::enabled() && tokens<=32 && type>=0 && type<GGML_TYPE_COUNT) {
            static thread_local std::uint64_t seen[GGML_TYPE_COUNT]{};
            const auto bit=std::uint64_t(1)<<tokens;
            if(!(seen[type]&bit)) {
                seen[type]|=bit;
                std::fprintf(stderr,"native_gguf_dense_history_tiles format=%d rows=%lld tile_rows=8 calls=%lld\n",
                    int(type),static_cast<long long>(tokens),static_cast<long long>((tokens+7)/8));
            }
        }
        return;
    }
    const dim3 threads(32, 4);
    const auto blocks = static_cast<unsigned>((rows + 3) / 4);
    const bool shared_rows=tokens>=2 && tokens<=8;
    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]!='0' &&
            type>=0 && type<GGML_TYPE_COUNT && tokens<=32) {
        static thread_local std::uint64_t seen[GGML_TYPE_COUNT]{};
        const auto bit=std::uint64_t(1)<<tokens;
        if(!(seen[type]&bit)) {
            seen[type]|=bit;
            std::fprintf(stderr,"native_gguf_dense_dispatch format=%d rows=%lld shared_weight_blocks=%d output_half=%d\n",
                int(type),static_cast<long long>(tokens),int(shared_rows),int(std::is_same_v<Output,half>));
        }
    }
    switch (type) {
#define TYLOINFER_GGUF_Q8(name) case GGML_TYPE_##name: \
        if(try_dense_cooperative<GGML_TYPE_##name,Output>(packed,stride,rows,columns,input,tokens,output,stream)) {} \
        else if(shared_rows) { \
            const auto launch=[&](auto capacity) { \
                project_dense_q8_rows<GGML_TYPE_##name,decltype(capacity)::value><<<blocks,threads,0,stream>>>( \
                    static_cast<const std::uint8_t*>(packed),stride,rows,columns,int(tokens),static_cast<const block_q8_1*>(input),output,std::is_same_v<Output,half>); \
            }; \
            if(tokens==2)launch(std::integral_constant<int,2>{}); \
            else if(tokens==3)launch(std::integral_constant<int,3>{}); \
            else if(tokens==4)launch(std::integral_constant<int,4>{}); \
            else launch(std::integral_constant<int,8>{}); \
        } else \
        project_q8<GGML_TYPE_##name, false><<<blocks, threads, 0, stream>>>( \
            static_cast<const std::uint8_t*>(packed), nullptr, stride, 0, rows, columns, nullptr, \
            tokens, 0, static_cast<const block_q8_1*>(input), false, output,false,std::is_same_v<Output,half>); break;
        TYLOINFER_GGUF_FORMATS(TYLOINFER_GGUF_Q8)
#undef TYLOINFER_GGUF_Q8
        default: throw std::invalid_argument("unsupported CUDA GGUF integer projection");
    }
    checked_launch();
}

void matvec_q8(ggml_type type,const void* packed,std::uint64_t stride,std::int64_t rows,std::int64_t columns,
        const void* input,std::int64_t tokens,float* output,cudaStream_t stream) {
    matvec_q8_typed(type,packed,stride,rows,columns,input,tokens,output,stream);
}
void matvec_q8_f16(ggml_type type,const void* packed,std::uint64_t stride,std::int64_t rows,std::int64_t columns,
        const void* input,std::int64_t tokens,void* output,cudaStream_t stream) {
    matvec_q8_typed(type,packed,stride,rows,columns,input,tokens,static_cast<half*>(output),stream);
}

void quantize_moe_hidden(const float* gate,const float* up,std::int64_t rows,std::int64_t columns,
        bool half_output,float limit,void* output,cudaStream_t stream) {
    q8_row_bytes(columns);if(!rows)return;
    const auto launch=[&](auto half) {
        swiglu_q8<decltype(half)::value><<<unsigned((rows*columns+255)/256),256,0,stream>>>(
            gate,up,rows*columns,limit,static_cast<block_q8_1*>(output));
        if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(stream,"gguf_moe_swiglu_q8",
            reinterpret_cast<const void*>(swiglu_q8<decltype(half)::value>),"half="+std::to_string(half_output));
    };
    if(half_output)launch(std::true_type{});else launch(std::false_type{});
    checked_launch();
}

void gate_up_swiglu_q8_f16(const void* gate_up,int64_t tokens,int64_t width,
        void* quantized,bool position_major,cudaStream_t stream) {
    q8_row_bytes(width);
    if(tokens<0 || (position_major && (tokens<2 || tokens>4)))
        throw std::invalid_argument("GGUF shared SwiGLU Q8 row geometry invalid");
    if(!tokens)return;
    const auto launch=[&](auto interleaved) {
        swiglu_paired_half_q8<decltype(interleaved)::value><<<unsigned((tokens*width+255)/256),256,0,stream>>>(
            static_cast<const half*>(gate_up),tokens,width,static_cast<block_q8_1*>(quantized));
        if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(stream,"gguf_shared_swiglu_q8",
            reinterpret_cast<const void*>(swiglu_paired_half_q8<decltype(interleaved)::value>),
            "rows="+std::to_string(tokens)+" position_major="+std::to_string(position_major));
    };
    if(position_major)launch(std::true_type{});else launch(std::false_type{});
    checked_launch();
}

void routed_matvec_q8(ggml_type type, const std::uint64_t* addresses,
    std::uint64_t stride, std::int64_t experts, std::int64_t rows, std::int64_t columns,
    const std::int64_t* ids, std::int64_t tokens, std::int64_t routes,
    const void* input, bool routed_input, float* output, cudaStream_t stream,bool preserve_missing) {
    if (!tokens || !rows || !routes) return;
    if(try_compact_routed_project(CompactFormat(type),addresses,stride,experts,rows,columns,
            ids,nullptr,tokens,routes,input,routed_input,output,stream,preserve_missing))return;
    // The compact indexing path needs 32-bit element and activation-block
    // offsets. Large tensors retain the original 64-bit implementation.
    const bool compact=experts>0 && experts<=INT_MAX && rows>0 && rows<=INT_MAX && columns>0 && columns<=INT_MAX &&
        routes>0 && routes<=INT_MAX && tokens>0 && tokens<=INT_MAX/routes &&
        tokens*routes<=INT_MAX/rows && columns/32>0 && tokens*routes<=INT_MAX/(columns/32);
    if(compact) {
        const auto launch=[&]<int Format>() {
            // Small reduction axes leave most of a full warp idle. Evaluate
            // more rows per block while retaining each virtual lane's sum.
            const int items=int(columns/Dot<Format>::qk*Dot<Format>::parts);
            const auto invoke=[&](auto width) {
                constexpr int Lanes=decltype(width)::value;
                const auto count=tokens*routes*rows;
                project_routed_subrow_q8<Format,Lanes><<<unsigned((count+128/Lanes-1)/(128/Lanes)),128,0,stream>>>(
                    addresses,stride,int(experts),int(rows),int(columns),ids,int(tokens*routes),int(routes),
                    static_cast<const block_q8_1*>(input),routed_input,output,preserve_missing);
                if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(stream,
                    "gguf_routed_subrow",reinterpret_cast<const void*>(project_routed_subrow_q8<Format,Lanes>),
                    "format="+std::to_string(Format)+" M="+std::to_string(tokens)+" lanes="+std::to_string(Lanes));
            };
            if(items<64 && Dot<Format>::qk<=64)invoke(std::integral_constant<int,8>{});
            else if(items<128)invoke(std::integral_constant<int,16>{});
            else {
                // Wide reductions already fill a warp. Preserve the original
                // dot rather than serializing its fragments into sub-rows.
                project_q8<Format,true><<<unsigned((tokens*routes*rows+3)/4),dim3(32,4),0,stream>>>(nullptr,addresses,
                    stride,experts,rows,columns,ids,tokens,routes,static_cast<const block_q8_1*>(input),routed_input,output,preserve_missing);
                if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(stream,"gguf_routed_warp",
                    reinterpret_cast<const void*>(project_q8<Format,true>),"format="+std::to_string(Format)+" M="+std::to_string(tokens));
            }
        };
        switch(type) {
#define TYLOINFER_GGUF_SUBROW(name) case GGML_TYPE_##name:launch.template operator()<GGML_TYPE_##name>();break;
            TYLOINFER_GGUF_FORMATS(TYLOINFER_GGUF_SUBROW)
#undef TYLOINFER_GGUF_SUBROW
            default:throw std::invalid_argument("unsupported CUDA GGUF integer routed projection");
        }
        checked_launch();return;
    }
    const dim3 threads(32, 4);
    const auto blocks = static_cast<unsigned>((tokens * routes * rows + 3) / 4);
    switch (type) {
#define TYLOINFER_GGUF_ROUTED_Q8(name) case GGML_TYPE_##name: \
        project_q8<GGML_TYPE_##name, true><<<blocks, threads, 0, stream>>>(nullptr, addresses, stride, \
            experts, rows, columns, ids, tokens, routes, static_cast<const block_q8_1*>(input), routed_input, output,preserve_missing); break;
        TYLOINFER_GGUF_FORMATS(TYLOINFER_GGUF_ROUTED_Q8)
#undef TYLOINFER_GGUF_ROUTED_Q8
        default: throw std::invalid_argument("unsupported CUDA GGUF integer routed projection");
    }
    checked_launch();
}

void prepare_route_groups(const std::int64_t* ids,std::int64_t slots,int tokens,RouteGroup* groups,cudaStream_t stream) {
    if(!slots)return;
    const int capacity=route_group_capacity(tokens);
    const auto status=cudaMemsetAsync(&groups[0].count,0,sizeof(int),stream);
    if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
    group_routes<<<unsigned((slots+127)/128),128,0,stream>>>(ids,slots,capacity,groups);
    checked_launch();
}
void snapshot_routed_addresses(const std::uint64_t* mapped,std::uint64_t* device,
        const std::int64_t* ids,std::int64_t slots,std::int64_t experts,cudaStream_t stream,
        const std::uint32_t* admission_ready) {
    if(!slots)return;
    snapshot_addresses<<<unsigned((slots+127)/128),128,0,stream>>>(mapped,device,ids,slots,experts,admission_ready);
    checked_launch();
    if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(stream,
        "gguf_resident_snapshot",reinterpret_cast<const void*>(snapshot_addresses),
        "slots="+std::to_string(slots)+" layer_admission_guard="+std::to_string(admission_ready!=nullptr));
}
bool supports_shared_routed_decode(ggml_type type) noexcept {
    switch(type) {
#define TYLOINFER_GGUF_SHARED_SUPPORT(name) case GGML_TYPE_##name: \
        return has_decoded_weight<GGML_TYPE_##name>;
        TYLOINFER_GGUF_FORMATS(TYLOINFER_GGUF_SHARED_SUPPORT)
#undef TYLOINFER_GGUF_SHARED_SUPPORT
        default:return false;
    }
}
namespace {
struct RoutedProjectionLaunch {
    const uint64_t* gate_addresses;const uint64_t* up_addresses;
    uint64_t gate_stride,up_stride;
    int experts,rows,columns,tokens,routes;
    const int64_t* ids;const RouteGroup* groups;const void* input;
    float* gate;float* up;cudaStream_t stream;
    bool preserve_missing,routed_input;void* hidden;bool half_output;float limit;
};
void projection_check(cudaError_t status) {
    if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}
template<int Format,int Lanes>
bool launch_direct_projection_pair(const RoutedProjectionLaunch& a) {
    const void* function;
    if constexpr(StagedPreparedDot<Format>::words) {
        constexpr int Threads=256;
        project_gate_up_table_q8<Format,Lanes><<<dim3(unsigned((a.rows+Threads/Lanes-1)/(Threads/Lanes)),unsigned(a.tokens*a.routes)),Threads,0,a.stream>>>(
            a.gate_addresses,a.up_addresses,a.gate_stride,a.up_stride,a.experts,a.rows,a.columns,a.tokens*a.routes,
            a.ids,a.routes,static_cast<const block_q8_1*>(a.input),a.gate,a.up,a.preserve_missing);
        function=reinterpret_cast<const void*>(project_gate_up_table_q8<Format,Lanes>);
    } else {
        project_gate_up_direct_q8<Format,Lanes><<<unsigned((int64_t(a.tokens)*a.routes*a.rows+128/Lanes-1)/(128/Lanes)),128,0,a.stream>>>(
            a.gate_addresses,a.up_addresses,a.gate_stride,a.up_stride,a.experts,a.rows,a.columns,a.tokens*a.routes,
            a.ids,a.routes,static_cast<const block_q8_1*>(a.input),a.gate,a.up,a.preserve_missing);
        function=reinterpret_cast<const void*>(project_gate_up_direct_q8<Format,Lanes>);
    }
    if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(a.stream,"gguf_routed_gate_up_direct",function,
        "format="+std::to_string(Format)+" M=1 lanes="+std::to_string(Lanes)+
        " expert_cta="+std::to_string(bool(StagedPreparedDot<Format>::words))+
        " shared_lookup="+std::to_string(bool(StagedPreparedDot<Format>::words)));
    return true;
}
template<int Format,int Capacity,int Lanes,bool Paired,bool Quantized>
bool launch_planar_projection(const RoutedProjectionLaunch& a) {
    constexpr int Threads=Quantized?32*Lanes:256,TileRows=Threads/Lanes;
    const auto function=reinterpret_cast<const void*>(project_gate_up_planar_q8<Format,Capacity,Lanes,Paired,Quantized>);
    const auto bytes=std::size_t(Capacity)*((a.columns/32)*sizeof(block_q8_1)+(Quantized?64*sizeof(float):0));
    int device=0,sms=0,max_shared=0;
    projection_check(cudaGetDevice(&device));
    projection_check(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device));
    projection_check(cudaDeviceGetAttribute(&max_shared,cudaDevAttrMaxSharedMemoryPerBlockOptin,device));
    cudaFuncAttributes attributes;projection_check(cudaFuncGetAttributes(&attributes,function));
    if(bytes+attributes.sharedSizeBytes>std::size_t(max_shared))return false;
    if(bytes>std::size_t(attributes.maxDynamicSharedSizeBytes))
        projection_check(cudaFuncSetAttribute(function,cudaFuncAttributeMaxDynamicSharedMemorySize,int(bytes)));
    int active=0;projection_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,function,Threads,bytes));
    if(active<=0)return false;
    const auto x=unsigned((a.rows+int64_t(TileRows)-1)/TileRows);
    const auto y=unsigned(std::min<int64_t>(a.tokens*a.routes,std::max<int64_t>(1,int64_t(active)*sms*2/x)));
    project_gate_up_planar_q8<Format,Capacity,Lanes,Paired,Quantized><<<dim3(x,y),dim3(32,Threads/32),bytes,a.stream>>>(
        a.gate_addresses,a.up_addresses,a.gate_stride,a.up_stride,a.experts,a.rows,a.columns,a.tokens*a.routes,a.ids,a.groups,a.routes,
        static_cast<const block_q8_1*>(a.input),a.gate,a.up,a.preserve_missing,a.routed_input,
        static_cast<block_q8_1*>(a.hidden),a.half_output,a.limit);
    if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(a.stream,
        Quantized?"gguf_routed_gate_up_hidden_q8":Paired?"gguf_routed_gate_up_planar":"gguf_routed_planar",function,
        "paired="+std::to_string(Paired)+" format="+std::to_string(Format)+" M="+std::to_string(a.tokens)+
        " lanes="+std::to_string(Lanes)+" shared_bytes="+std::to_string(bytes));
    return true;
}
template<int Format,int Lanes,bool Paired,bool Quantized>
bool launch_projection_rows(const RoutedProjectionLaunch& a) {
    switch(a.tokens) {
        case 1:return launch_planar_projection<Format,1,Lanes,Paired,Quantized>(a);
        case 2:return launch_planar_projection<Format,2,Lanes,Paired,Quantized>(a);
        case 3:return launch_planar_projection<Format,3,Lanes,Paired,Quantized>(a);
        default:return launch_planar_projection<Format,4,Lanes,Paired,Quantized>(a);
    }
}
template<int Format,bool Paired,bool Quantized>
bool launch_projection_format(const RoutedProjectionLaunch& a) {
    if(a.columns%Dot<Format>::qk)return false;
    const int64_t items=int64_t(a.columns)/Dot<Format>::qk*Dot<Format>::parts;
    if constexpr(Paired && !Quantized)if(a.tokens==1) {
        if(items<64 && Dot<Format>::qk<=64)return launch_direct_projection_pair<Format,8>(a);
        if(items<128)return launch_direct_projection_pair<Format,16>(a);
        return launch_direct_projection_pair<Format,32>(a);
    }
    if constexpr(Dot<Format>::qk<=64) {
        if(items<64)return launch_projection_rows<Format,8,Paired,Quantized>(a);
        if(items<128)return launch_projection_rows<Format,16,Paired,Quantized>(a);
        return launch_projection_rows<Format,32,Paired,Quantized>(a);
    }
    return launch_projection_rows<Format,16,Paired,Quantized>(a);
}
template<bool Paired,bool Quantized=false>
bool try_planar_routed_q8(ggml_type type,const uint64_t* gate_addresses,const uint64_t* up_addresses,
        uint64_t gate_stride,uint64_t up_stride,int64_t experts,int64_t rows,int64_t columns,
        const int64_t* ids,const RouteGroup* groups,int64_t tokens,int64_t routes,const void* input,
        float* gate,float* up,cudaStream_t stream,bool preserve_missing,bool routed_input,
        void* hidden=nullptr,bool half_output=false,float limit=0.f) {
    if(tokens<1 || tokens>8 || (tokens>1 && !groups) || experts<1 || experts>INT_MAX || rows<1 || rows>INT_MAX ||
            columns<1 || columns>INT_MAX || routes<1 || routes>INT_MAX/tokens ||
            tokens*routes>INT_MAX/rows || columns%32)return false;
    if constexpr(Quantized)if(rows%32 || !hidden)return false;
    if constexpr(!Paired)if(tokens==1)return false;
    const RoutedProjectionLaunch a{gate_addresses,up_addresses,gate_stride,up_stride,int(experts),int(rows),int(columns),
        int(tokens),int(routes),ids,groups,input,gate,up,stream,preserve_missing,routed_input,hidden,half_output,limit};
    bool launched=false;
    // Explicit function templates keep all three independent modes in their
    // dispatch types: a projection, a pair, and a pair with a Q8 epilogue.
    switch(type) {
        case GGML_TYPE_IQ3_XXS:launched=launch_projection_format<GGML_TYPE_IQ3_XXS,Paired,Quantized>(a);break;
        case GGML_TYPE_IQ3_S:launched=launch_projection_format<GGML_TYPE_IQ3_S,Paired,Quantized>(a);break;
        case GGML_TYPE_IQ2_XXS:launched=launch_projection_format<GGML_TYPE_IQ2_XXS,Paired,Quantized>(a);break;
        case GGML_TYPE_IQ2_XS:launched=launch_projection_format<GGML_TYPE_IQ2_XS,Paired,Quantized>(a);break;
        case GGML_TYPE_IQ2_S:launched=launch_projection_format<GGML_TYPE_IQ2_S,Paired,Quantized>(a);break;
        case GGML_TYPE_Q2_0:launched=launch_projection_format<GGML_TYPE_Q2_0,Paired,Quantized>(a);break;
        default:return false;
    }
    if(launched)checked_launch();
    return launched;
}
}
bool try_gate_up_matvec_q8(ggml_type type,const uint64_t* gate_addresses,const uint64_t* up_addresses,
        uint64_t gate_stride,uint64_t up_stride,int64_t experts,int64_t rows,int64_t columns,
        const int64_t* ids,const RouteGroup* groups,int64_t tokens,int64_t routes,const void* input,
        float* gate,float* up,cudaStream_t stream,bool preserve_missing) {
    if(try_compact_routed_project(CompactFormat(type),gate_addresses,gate_stride,experts,rows,columns,
            ids,groups,tokens,routes,input,false,gate,stream,preserve_missing,up_addresses,up_stride,up))return true;
    return try_planar_routed_q8<true>(type,gate_addresses,up_addresses,gate_stride,up_stride,experts,rows,columns,
        ids,groups,tokens,routes,input,gate,up,stream,preserve_missing,false);
}
bool try_gate_up_hidden_q8(ggml_type type,const uint64_t* gate_addresses,const uint64_t* up_addresses,
        uint64_t gate_stride,uint64_t up_stride,int64_t experts,int64_t rows,int64_t columns,
        const int64_t* ids,const RouteGroup* groups,int64_t tokens,int64_t routes,const void* input,
        void* hidden,bool half_output,float limit,cudaStream_t stream) {
    return try_planar_routed_q8<true,true>(type,gate_addresses,up_addresses,gate_stride,up_stride,experts,rows,columns,
        ids,groups,tokens,routes,input,nullptr,nullptr,stream,false,false,hidden,half_output,limit);
}
void grouped_routed_matvec_q8(ggml_type type,const std::uint64_t* addresses,
        std::uint64_t stride,std::int64_t experts,std::int64_t rows,std::int64_t columns,
        const std::int64_t* ids,const RouteGroup* groups,std::int64_t tokens,std::int64_t routes,
        const void* input,bool routed_input,float* output,cudaStream_t stream,bool preserve_missing) {
    if(try_compact_routed_project(CompactFormat(type),addresses,stride,experts,rows,columns,
            ids,groups,tokens,routes,input,routed_input,output,stream,preserve_missing))return;
    if(!groups || tokens<=1 || tokens*routes>INT_MAX) {
        routed_matvec_q8(type,addresses,stride,experts,rows,columns,ids,tokens,routes,input,routed_input,output,stream,preserve_missing);return;
    }
    if(!rows || !routes)return;
    if(try_planar_routed_q8<false>(type,addresses,nullptr,stride,0,experts,rows,columns,ids,groups,tokens,routes,
            input,output,nullptr,stream,preserve_missing,routed_input))return;
    if(tokens<=8) {
        const bool compact=experts>0 && experts<=INT_MAX && rows>0 && rows<=INT_MAX && columns>0 && columns<=INT_MAX &&
            routes>0 && routes<=INT_MAX && tokens<=INT_MAX/routes && tokens*routes<=INT_MAX/rows &&
            columns/32>0 && tokens*routes<=INT_MAX/(columns/32);
        bool launched=false;
        if(compact && supports_shared_routed_decode(type)) {
            const auto check=[](cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));};
            int device=0,sms=0,max_shared=0;
            check(cudaGetDevice(&device));check(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device));
            check(cudaDeviceGetAttribute(&max_shared,cudaDevAttrMaxSharedMemoryPerBlockOptin,device));
            const auto bytes=std::size_t(route_group_capacity(int(tokens)))*columns/32*sizeof(block_q8_1);
            const auto invoke=[&]<int Format>() {
                if constexpr(requires {typename PreparedDot<Format>::DecodedWeight;}) {
                    const auto launch=[&]<int Capacity>() {
                        const auto function=reinterpret_cast<const void*>(project_routed_shared_q8<Format,Capacity>);
                        cudaFuncAttributes attributes;check(cudaFuncGetAttributes(&attributes,function));
                        // Full activation staging is optional. The existing flat
                        // kernel remains available if an input exceeds real SMEM.
                        if(bytes+attributes.sharedSizeBytes>std::size_t(max_shared))return;
                        if(bytes>std::size_t(attributes.maxDynamicSharedSizeBytes))
                            check(cudaFuncSetAttribute(function,cudaFuncAttributeMaxDynamicSharedMemorySize,int(bytes)));
                        int active=0;check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,function,256,bytes));
                        if(active<=0)return;
                        constexpr int Lanes=Dot<Format>::qk<=64?8:16;
                        const auto x=unsigned((rows+256/Lanes-1)/(256/Lanes));
                        const auto y=unsigned(std::max<std::int64_t>(1,std::min<std::int64_t>(tokens*routes,
                            std::max<std::int64_t>(1,std::int64_t(active)*sms*2/x))));
                        project_routed_shared_q8<Format,Capacity><<<dim3(x,y),dim3(32,8),bytes,stream>>>(
                            addresses,stride,int(experts),int(rows),int(columns),int(tokens*routes),ids,groups,int(routes),
                            static_cast<const block_q8_1*>(input),routed_input,output,preserve_missing);
                        if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(stream,"gguf_routed_shared",
                            function,"format="+std::to_string(Format)+" M="+std::to_string(tokens)+" lanes="+
                            std::to_string(Lanes)+" shared_bytes="+std::to_string(bytes));
                        launched=true;
                    };
                    if(tokens==2)launch.template operator()<2>();
                    else if(tokens==3)launch.template operator()<3>();else launch.template operator()<4>();
                }
            };
            switch(type) {
#define TYLOINFER_GGUF_SHARED_LAUNCH(name) case GGML_TYPE_##name:invoke.template operator()<GGML_TYPE_##name>();break;
                TYLOINFER_GGUF_FORMATS(TYLOINFER_GGUF_SHARED_LAUNCH)
#undef TYLOINFER_GGUF_SHARED_LAUNCH
                default:break;
            }
        }
        if(launched)checked_launch();
        else routed_matvec_q8(type,addresses,stride,experts,rows,columns,ids,tokens,routes,input,routed_input,output,stream,preserve_missing);
        return;
    }
    const dim3 threads(32,4);
    int device=0,multiprocessors=0;
    const auto device_status=cudaGetDevice(&device);
    if(device_status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(device_status));
    const auto sm_status=cudaDeviceGetAttribute(&multiprocessors,cudaDevAttrMultiProcessorCount,device);
    if(sm_status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(sm_status));
    const int capacity=route_group_capacity(int(tokens));
    switch(type) {
#define TYLOINFER_GGUF_ROUTED_GROUP(name) case GGML_TYPE_##name: { \
            int active=0;const auto status=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, \
                project_grouped_q8<GGML_TYPE_##name>,128,0); \
            if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status)); \
            const auto target=std::max(1,active*multiprocessors*2); \
            const auto x=unsigned(std::min<std::int64_t>((rows+3)/4,target)); \
            const auto y=unsigned(std::min<std::int64_t>(tokens*routes,std::max(1,target/int(x)))); \
            project_grouped_q8<GGML_TYPE_##name><<<dim3(x,y),threads,0,stream>>>( \
                addresses,stride,experts,rows,columns,groups,routes,static_cast<const block_q8_1*>(input),routed_input,output,preserve_missing); \
        break; }
        TYLOINFER_GGUF_FORMATS(TYLOINFER_GGUF_ROUTED_GROUP)
#undef TYLOINFER_GGUF_ROUTED_GROUP
        default:throw std::invalid_argument("unsupported CUDA GGUF grouped routed projection");
    }
    checked_launch();
    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        static std::atomic<std::uint64_t> reported[9]{};
        const auto mask=std::uint64_t(1)<<unsigned(type);
        if(!(reported[capacity].fetch_or(mask,std::memory_order_relaxed)&mask))
            std::fprintf(stderr,"native_gguf_routed_dispatch format=%d rows=%lld group_capacity=%d shared_weight_decode=1\n",
                int(type),static_cast<long long>(tokens),capacity);
    }
}

void routed_matvec(ggml_type type, const std::uint64_t* addresses,
    std::uint64_t stride, std::int64_t experts, std::int64_t rows, std::int64_t columns,
    const std::int64_t* ids, std::int64_t tokens, std::int64_t routes,
    const float* input, bool routed_input, float* output, cudaStream_t stream) {
    if (!tokens || !rows || !routes) return;
    const auto slots = tokens * routes;
    const dim3 threads(32, 4);
    const auto blocks = static_cast<unsigned>((slots * rows + 3) / 4);
    switch (type) {
#define TYLOINFER_GGUF_ROUTED(name) case GGML_TYPE_##name: \
        project_experts<GGML_TYPE_##name><<<blocks, threads, 0, stream>>>(addresses, stride, \
            experts, rows, columns, ids, slots, routes, input, routed_input, output); break;
        TYLOINFER_GGUF_FORMATS(TYLOINFER_GGUF_ROUTED)
#undef TYLOINFER_GGUF_ROUTED
        default: throw std::invalid_argument("unsupported CUDA GGUF routed projection");
    }
    checked_launch();
}
}
