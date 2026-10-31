#include "gguf_compact.h"
#include "gguf_compact_adjacent.cuh"
#include "gguf.h"
#include "kernel_dispatch_trace.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <cstdlib>

namespace mfq::cuda::gguf {
namespace {
using compact_detail::decode_four;
using compact_detail::scale_at;
using compact_detail::ternary_four;
struct alignas(4) Q8 { __half d, s; std::int8_t qs[32]; };
static_assert(sizeof(Q8) == 36);

template<int Bits,int Steps,int Tile,bool Paired,bool Shared,bool Narrow>
__global__ void project_compact_routed(const std::uint64_t* addresses,const std::uint64_t* second_addresses,
        std::uint64_t stride,std::uint64_t second_stride,std::int64_t experts,std::int64_t rows,std::int64_t columns,
        const std::int64_t* ids,const RouteGroup* groups,std::int64_t slots,std::int64_t routes,
        const Q8* input,bool routed,float* output,float* second_output,bool preserve_missing) {
    extern __shared__ std::uint32_t book[];
    bool book_ready=false;
    constexpr int Bytes=2+Bits*Steps/8,WarpWidth=Narrow?16:32,RowTile=(Shared?8:4)*(Narrow?2:1);
    const int lane=threadIdx.x%WarpWidth,local=lane%(WarpWidth/2);
    if constexpr(Paired)if(blockIdx.z) {addresses=second_addresses;stride=second_stride;output=second_output;}
    // Tile 1 is the ungrouped entry selected below. Its slot is already
    // bounded by this loop, so no route-group table or per-fragment slot
    // validation is needed. Wider fallback tiles still accept route groups.
    const auto group_count=Tile==1?slots:(groups?std::int64_t(groups[0].count):slots);
    for(auto group=std::int64_t(blockIdx.y);group<group_count;group+=gridDim.y) {
        const auto count=Tile==1?1:(groups?groups[group+1].count:1);
        const auto expert=Tile==1?ids[group]:(groups?groups[group+1].expert:ids[group]);
        const auto pointer=expert>=0 && expert<experts?addresses[expert]:0;
        if(!pointer && preserve_missing)continue;
        // A uniform expert pointer guards the cooperative load. Reuse this
        // exact FP16 table across every row/group handled by the CTA.
        if constexpr(Shared)if(pointer && !book_ready) {
            for(int at=threadIdx.x;at<2048;at+=blockDim.x)
                const_cast<std::uint32_t*>(book)[at]=__ldg(reinterpret_cast<const std::uint32_t*>(compact_detail::gyro_book)+at);
            __syncthreads();book_ready=true;
        }
        // External callers may supply wider route groups. Reuse the same
        // bounded activation tile rather than increasing its live registers.
        for(int first=0;first<count;first+=Tile) {
            const auto entries=min(Tile,count-first);
            std::int64_t selected[Tile];const Q8* activation[Tile];
            #pragma unroll
            for(int m=0;m<Tile;++m)if(m<entries) {
                const auto slot=Tile==1?group:(groups?std::int64_t(groups[group+1].slots[first+m]):group);
                selected[m]=slot;
                activation[m]=(Tile==1 || (slot>=0 && slot<slots))?input+(routed?slot:slot/routes)*(columns/32):input;
            }
            for(auto row_tile=std::int64_t(blockIdx.x);row_tile<(rows+RowTile-1)/RowTile;row_tile+=gridDim.x) {
                const auto row=row_tile*RowTile+threadIdx.x/WarpWidth;
                if constexpr(!Narrow)if(row>=rows)continue;
                float sum[Tile]{};
                if(pointer && row<rows)for(auto block=std::int64_t(lane/(WarpWidth/2));block<columns/128;block+=2) {
                    const auto* q=reinterpret_cast<const std::uint8_t*>(pointer)+row*stride+block*Bytes;
                    const float scale=scale_at(q);
                    // Every fragment owned by this lane uses the same Q8
                    // scale. Accumulate its original FP16 book values first;
                    // apply the two block scales once, without requantizing.
                    float dot[Tile]{};
                    const auto window=[&] {if constexpr(Narrow)return compact_detail::sixteen_window<Bits,Steps>(q+2,local);
                        else return compact_detail::eight_window<Bits,Steps>(q+2,local);}();
                    const auto apply=[&]<int Part>() {
                        auto w=[&] {if constexpr(Narrow)return compact_detail::sixteen_part<Bits,Steps,Part,Shared>(window,book);
                            else return compact_detail::eight_part<Bits,Steps,Part,Shared>(window,book);}();
                        if constexpr(Tile!=1) {
                            w.x=__fmul_rn(w.x,scale);w.y=__fmul_rn(w.y,scale);w.z=__fmul_rn(w.z,scale);w.w=__fmul_rn(w.w,scale);
                        }
                        #pragma unroll
                        for(int m=0;m<Tile;++m)if(m<entries && (Tile==1 || (selected[m]>=0 && selected[m]<slots))) {
                            const auto& x=activation[m][block*4+local/((WarpWidth/2)/4)];
                            const auto codes=*reinterpret_cast<const std::uint32_t*>(x.qs+((128/(WarpWidth/2)/4)*(local%((WarpWidth/2)/4))+Part)*4);
                            const float d=__half2float(x.d);
                            if constexpr(Tile==1) {
                            dot[m]=__fmaf_rn(w.x,float(std::int8_t(codes)),dot[m]);
                            dot[m]=__fmaf_rn(w.y,float(std::int8_t(codes>>8)),dot[m]);
                            dot[m]=__fmaf_rn(w.z,float(std::int8_t(codes>>16)),dot[m]);
                            dot[m]=__fmaf_rn(w.w,float(std::int8_t(codes>>24)),dot[m]);
                            } else {
                            sum[m]=__fmaf_rn(w.x,float(std::int8_t(codes))*d,sum[m]);
                            sum[m]=__fmaf_rn(w.y,float(std::int8_t(codes>>8))*d,sum[m]);
                            sum[m]=__fmaf_rn(w.z,float(std::int8_t(codes>>16))*d,sum[m]);
                            sum[m]=__fmaf_rn(w.w,float(std::int8_t(codes>>24))*d,sum[m]);
                            }
                        }
                    };
                    apply.template operator()<0>();apply.template operator()<1>();
                    if constexpr(Narrow){apply.template operator()<2>();apply.template operator()<3>();}
                    if constexpr(Tile==1)for(int m=0;m<Tile;++m)if(m<entries && (Tile==1 || (selected[m]>=0 && selected[m]<slots))) {
                        const auto& x=activation[m][block*4+local/((WarpWidth/2)/4)];
                        sum[m]=__fmaf_rn(dot[m],scale*__half2float(x.d),sum[m]);
                    }
                }
                #pragma unroll
                for(int m=0;m<Tile;++m)if(m<entries) {
                    for(int offset=WarpWidth/2;offset;offset>>=1)
                        sum[m]+=__shfl_down_sync(0xffffffffu,sum[m],offset,WarpWidth);
                    if(!lane && row<rows && (Tile==1 || (selected[m]>=0 && selected[m]<slots)))output[selected[m]*rows+row]=sum[m];
                }
            }
        }
    }
}

template<int Bits,int Steps,int Tile,bool Paired,bool Shared,bool Decoded,bool Narrow=false>
__device__ __forceinline__ void project_compact_routed_staged_body(const std::uint64_t* addresses,const std::uint64_t* second_addresses,
        std::uint64_t stride,std::uint64_t second_stride,std::int64_t experts,std::int64_t rows,std::int64_t columns,
        const RouteGroup* groups,std::int64_t slots,std::int64_t routes,const Q8* input,bool routed,
        float* output,float* second_output,bool preserve_missing) {
    extern __shared__ std::uint32_t storage[];
    const auto* book=storage;
    auto* words=storage+(Shared?2048:0);
    bool book_ready=false;
    constexpr int Bytes=2+Bits*Steps/8,Threads=256,Lanes=Narrow?8:16,RowTile=Threads/Lanes,ActivationChunk=1024;
    const int tid=threadIdx.x,lane=tid%Lanes;
    const auto blocks=columns/32,words_per_row=blocks*9;
    if constexpr(Paired)if(blockIdx.z){addresses=second_addresses;stride=second_stride;output=second_output;}
    const auto group_count=groups[0].count;
    for(auto index=std::int64_t(blockIdx.y);index<group_count;index+=gridDim.y) {
        const auto& group=groups[index+1];
        const auto pointer=group.expert>=0 && group.expert<experts?addresses[group.expert]:0;
        if(!pointer && preserve_missing)continue;
        // A uniform expert pointer guards the cooperative load. Reuse this
        // exact FP16 table across every row/group handled by the CTA.
        if constexpr(Shared)if(pointer && !book_ready) {
            for(int at=threadIdx.x;at<2048;at+=blockDim.x)
                const_cast<std::uint32_t*>(book)[at]=__ldg(reinterpret_cast<const std::uint32_t*>(compact_detail::gyro_book)+at);
            __syncthreads();book_ready=true;
        }
        for(int first=0;first<group.count;first+=Tile) {
            const int entries=min(Tile,group.count-first);
            if(pointer && entries>1 && (!Decoded || entries<3)) {
                for(int m=0;m<entries;++m) {
                    const auto slot=std::int64_t(group.slots[first+m]);
                    const bool valid=slot>=0 && slot<slots;
                    const auto source=(valid?(routed?slot:slot/routes):0)*words_per_row;
                    const auto* raw=reinterpret_cast<const std::uint32_t*>(input);
                    for(auto word=std::int64_t(tid);word<words_per_row;word+=Threads)
                        words[m*words_per_row+word]=valid?raw[source+word]:0;
                }
                __syncthreads();
            }
            const int route_first=first;
            const auto compute=[&]<int Count>() {
                const Q8* activation=reinterpret_cast<const Q8*>(words);
                if constexpr(Count==1) {
                    const auto slot=std::int64_t(group.slots[first]);
                    activation=input+(slot>=0 && slot<slots?(routed?slot:slot/routes)*blocks:0);
                }
                for(auto tile=std::int64_t(blockIdx.x);tile<(rows+RowTile-1)/RowTile;tile+=gridDim.x) {
                    const auto row=tile*RowTile+tid/Lanes;
                    float sum[Count]{};
                    const auto compute_block=[&](std::int64_t block,std::int64_t first,std::int64_t chunk) {
                        const auto* q=reinterpret_cast<const std::uint8_t*>(pointer)+row*stride+block*Bytes;
                        const float scale=scale_at(q);
                        const auto window=[&] {if constexpr(Narrow)return compact_detail::sixteen_window<Bits,Steps>(q+2,lane);
                            else return compact_detail::eight_window<Bits,Steps>(q+2,lane);}();
                        const auto apply=[&]<int Part>() {
                            auto w=[&] {if constexpr(Narrow)return compact_detail::sixteen_part<Bits,Steps,Part,Shared>(window,book);
                                else return compact_detail::eight_part<Bits,Steps,Part,Shared>(window,book);}();
                            w.x=__fmul_rn(w.x,scale);w.y=__fmul_rn(w.y,scale);w.z=__fmul_rn(w.z,scale);w.w=__fmul_rn(w.w,scale);
                            #pragma unroll
                            for(int m=0;m<Count;++m) {
                                float4 value;
                                if constexpr(Decoded && Count>=3) {
                                    // Each part is contiguous across the half warp;
                                    // lane*8 would alias shared-memory banks.
                                    const auto* decoded=reinterpret_cast<const float*>(words);
                                    value=*reinterpret_cast<const float4*>(decoded+m*chunk+block*128-first+Part*Lanes*4+lane*4);
                                } else {
                                    const auto& x=activation[m*blocks+block*4+lane/(Lanes/4)];
                                    const auto codes=*reinterpret_cast<const std::uint32_t*>(x.qs+((128/Lanes/4)*(lane%(Lanes/4))+Part)*4);
                                    const float d=__half2float(x.d);
                                    value=make_float4(float(std::int8_t(codes))*d,float(std::int8_t(codes>>8))*d,
                                        float(std::int8_t(codes>>16))*d,float(std::int8_t(codes>>24))*d);
                                }
                                sum[m]=__fmaf_rn(w.x,value.x,sum[m]);
                                sum[m]=__fmaf_rn(w.y,value.y,sum[m]);
                                sum[m]=__fmaf_rn(w.z,value.z,sum[m]);
                                sum[m]=__fmaf_rn(w.w,value.w,sum[m]);
                            }
                        };
                        apply.template operator()<0>();apply.template operator()<1>();
                        if constexpr(Narrow){apply.template operator()<2>();apply.template operator()<3>();}
                    };
                    if constexpr(Decoded && Count>=3) {
                        // Decode Q8 once for all output rows. Bounded K slices
                        // avoid a full float activation image per CTA. Fewer
                        // than three reused rows keep the cheaper Q8 staging.
                        const int chunk=int(min(columns,std::int64_t(ActivationChunk)));
                        if(pointer)for(std::int64_t first=0;first<columns;first+=chunk) {
                            const int count=int(min(std::int64_t(chunk),columns-first));
                            auto* decoded=reinterpret_cast<float*>(words);
                            // One float4 per thread per input row. Keeping the
                            // rows outside this loop avoids 64-bit division by
                            // the runtime K tail and its register overhead.
                            const int destination=tid*4;
                            if(destination<count)
                            #pragma unroll 1
                            for(int m=0;m<Count;++m) {
                                const auto k=(destination/128)*128+((destination%(Lanes*4))/4)*(128/Lanes)+(destination%128)/(Lanes*4)*4;
                                const auto selected=std::int64_t(group.slots[route_first+m]);
                                const bool valid=selected>=0 && selected<slots;
                                const auto source=(valid?(routed?selected:selected/routes):0)*blocks+(first+k)/32;
                                const auto& x=input[source];
                                const auto codes=*reinterpret_cast<const std::uint32_t*>(x.qs+k%32);
                                const float d=valid?__half2float(x.d):0.f;
                                *reinterpret_cast<float4*>(decoded+m*chunk+destination)=make_float4(
                                    float(std::int8_t(codes))*d,float(std::int8_t(codes>>8))*d,
                                    float(std::int8_t(codes>>16))*d,float(std::int8_t(codes>>24))*d);
                            }
                            __syncthreads();
                            if(row<rows)for(auto block=first/128;block<(first+count)/128;++block)
                                compute_block(block,first,chunk);
                            __syncthreads();
                        }
                    } else if(pointer && row<rows)
                        for(std::int64_t block=0;block<columns/128;++block)compute_block(block,0,0);
                    #pragma unroll
                    for(int m=0;m<Count;++m) {
                        for(int offset=Lanes/2;offset;offset>>=1)sum[m]+=__shfl_down_sync(0xffffffffu,sum[m],offset,Lanes);
                        const auto slot=std::int64_t(group.slots[first+m]);
                        if(!lane && row<rows && slot>=0 && slot<slots)output[slot*rows+row]=sum[m];
                    }
                }
            };
            if(entries==1)compute.template operator()<1>();
            else if constexpr(Tile==2)compute.template operator()<2>();
            else if constexpr(Tile>=4) {
                if(entries==2)compute.template operator()<2>();else if(entries==3)compute.template operator()<3>();else compute.template operator()<4>();
            }
            if(pointer && entries>1 && (first+Tile<group.count || index+gridDim.y<group_count))__syncthreads();
        }
    }
}

// Keep the Q8 reference entry unconstrained. Decoded inputs add a staging
// phase; bound its register scheduling so low-reuse groups retain parallelism.
// This is a compiler target, not a claim that four CTAs fit every device:
// launch sizing below still queries the actual registers/shared-memory limit.
template<int Bits,int Steps,int Tile,bool Paired,bool Shared>
__global__ void project_compact_routed_staged(const std::uint64_t* addresses,const std::uint64_t* second_addresses,
        std::uint64_t stride,std::uint64_t second_stride,std::int64_t experts,std::int64_t rows,std::int64_t columns,
        const RouteGroup* groups,std::int64_t slots,std::int64_t routes,const Q8* input,bool routed,
        float* output,float* second_output,bool preserve_missing) {
    project_compact_routed_staged_body<Bits,Steps,Tile,Paired,Shared,false>(addresses,second_addresses,
        stride,second_stride,experts,rows,columns,groups,slots,routes,input,routed,output,second_output,preserve_missing);
}
template<int Bits,int Steps,int Tile,bool Paired,bool Shared>
__global__ __launch_bounds__(256,4) void project_compact_routed_decoded(const std::uint64_t* addresses,const std::uint64_t* second_addresses,
        std::uint64_t stride,std::uint64_t second_stride,std::int64_t experts,std::int64_t rows,std::int64_t columns,
        const RouteGroup* groups,std::int64_t slots,std::int64_t routes,const Q8* input,bool routed,
        float* output,float* second_output,bool preserve_missing) {
    project_compact_routed_staged_body<Bits,Steps,Tile,Paired,Shared,true>(addresses,second_addresses,
        stride,second_stride,experts,rows,columns,groups,slots,routes,input,routed,output,second_output,preserve_missing);
}

// Eight lanes decode sixteen weights each. This increases independent output
// rows per CTA, while retaining the original FP16 book and FP32 products.
// The narrower reduction changes only the FP32 addition tree.
template<int Bits,int Steps,int Tile,bool Paired,bool Shared,bool Decoded>
__global__ __launch_bounds__(256,4) void project_compact_routed_narrow(const std::uint64_t* addresses,const std::uint64_t* second_addresses,
        std::uint64_t stride,std::uint64_t second_stride,std::int64_t experts,std::int64_t rows,std::int64_t columns,
        const RouteGroup* groups,std::int64_t slots,std::int64_t routes,const Q8* input,bool routed,
        float* output,float* second_output,bool preserve_missing) {
    project_compact_routed_staged_body<Bits,Steps,Tile,Paired,Shared,Decoded,true>(addresses,second_addresses,
        stride,second_stride,experts,rows,columns,groups,slots,routes,input,routed,output,second_output,preserve_missing);
}

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template<class T>
__device__ float4 input_four(const T* input, std::int64_t at, bool aligned) {
    if constexpr (std::is_same_v<T, Q8>) {
        const auto& x = input[at / 32];
        const auto codes = *reinterpret_cast<const std::uint32_t*>(x.qs + at % 32);
        const float d = __half2float(x.d);
        return make_float4(d * std::int8_t(codes), d * std::int8_t(codes >> 8),
            d * std::int8_t(codes >> 16), d * std::int8_t(codes >> 24));
    } else if constexpr (std::is_same_v<T, __half>) {
        if (aligned) {
            const auto a = __half22float2(*reinterpret_cast<const __half2*>(input + at));
            const auto b = __half22float2(*reinterpret_cast<const __half2*>(input + at + 2));
            return make_float4(a.x, a.y, b.x, b.y);
        }
        return make_float4(float(input[at]), float(input[at + 1]), float(input[at + 2]), float(input[at + 3]));
    } else {
        if (aligned) return *reinterpret_cast<const float4*>(input + at);
        return make_float4(input[at], input[at + 1], input[at + 2], input[at + 3]);
    }
}

template<int Bits, int Steps, int Tile, bool ExactTile, class T, class Output>
__global__ void project_compact(const std::uint8_t* weights, std::uint64_t stride,
        std::int64_t rows, std::int64_t columns, const T* input, std::int64_t tokens,
        Output* output, bool aligned) {
    constexpr int Bytes = Bits ? 2 + Bits * Steps / 8 : 28;
    const auto row_tiles = (rows + 3) / 4;
    const auto token_tiles = ExactTile ? 1 : (tokens + Tile - 1) / Tile;
    const int lane = threadIdx.x % 32;
    for (auto token_tile = std::int64_t(blockIdx.y); token_tile < token_tiles; token_tile += gridDim.y)
    for (auto row_tile = std::int64_t(blockIdx.x); row_tile < row_tiles; row_tile += gridDim.x) {
        const auto row = row_tile * 4 + threadIdx.x / 32;
        const auto first = ExactTile ? 0 : token_tile * Tile;
        if (row >= rows) continue; // Entire warp leaves together.
        float sums[Tile]{};
        for (std::int64_t block = 0; block < columns / 128; ++block) {
            const auto* q = weights + row * stride + block * Bytes;
            const float scale = scale_at(q + (Bits ? 0 : 26));
            if constexpr (!Bits && std::is_same_v<T, Q8>) {
                const auto codes = ternary_four(q, lane);
                #pragma unroll
                for (int m = 0; m < Tile; ++m) if (ExactTile || first + m < tokens) {
                    const auto& x = input[(first + m) * (columns / 32) + block * 4 + lane / 8];
                    const auto values = *reinterpret_cast<const int*>(x.qs + (lane % 8) * 4);
                    int dot = 0;
#if __CUDA_ARCH__ >= 610
                    dot = __dp4a(int(codes), values, 0); // dp4a is available from SM 6.1.
#else
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        dot += std::int8_t(codes >> (j * 8)) * std::int8_t(std::uint32_t(values) >> (j * 8));
#endif
                    sums[m] += scale * __half2float(x.d) * float(dot);
                }
            } else {
                auto w = decode_four<Bits, Steps>(q + (Bits ? 2 : 0), lane);
                w.x = __fmul_rn(w.x, scale); w.y = __fmul_rn(w.y, scale);
                w.z = __fmul_rn(w.z, scale); w.w = __fmul_rn(w.w, scale);
                #pragma unroll
                for (int m = 0; m < Tile; ++m) if (ExactTile || first + m < tokens) {
                    const auto x = input_four(input, (first + m) * columns + block * 128 + lane * 4, aligned);
                    sums[m] = __fmaf_rn(w.x, x.x, sums[m]); sums[m] = __fmaf_rn(w.y, x.y, sums[m]);
                    sums[m] = __fmaf_rn(w.z, x.z, sums[m]); sums[m] = __fmaf_rn(w.w, x.w, sums[m]);
                }
            }
        }
        #pragma unroll
        for (int m = 0; m < Tile; ++m) if (ExactTile || first + m < tokens) {
            for (int offset = 16; offset; offset >>= 1) sums[m] += __shfl_down_sync(0xffffffffu, sums[m], offset);
            if (!lane) output[(first + m) * rows + row] = Output(sums[m]);
        }
    }
}

template<int Bits, int Steps, class Output>
__global__ void dequantize_compact(const std::uint8_t* weights, std::uint64_t stride,
        std::int64_t rows, std::int64_t columns, Output* output) {
    constexpr int Bytes = Bits ? 2 + Bits * Steps / 8 : 28;
    const auto blocks = columns / 128;
    const int lane = threadIdx.x % 32;
    for (auto group = std::int64_t(blockIdx.x) * 4 + threadIdx.x / 32;
            group < rows * blocks; group += std::int64_t(gridDim.x) * 4) {
        const auto* q = weights + (group / blocks) * stride + (group % blocks) * Bytes;
        const float scale = scale_at(q + (Bits ? 0 : 26));
        const auto w = decode_four<Bits, Steps>(q + (Bits ? 2 : 0), lane);
        const auto at = group * 128 + lane * 4;
        output[at] = Output(__fmul_rn(w.x, scale)); output[at + 1] = Output(__fmul_rn(w.y, scale));
        output[at + 2] = Output(__fmul_rn(w.z, scale)); output[at + 3] = Output(__fmul_rn(w.w, scale));
    }
}

template<class F> void dispatch(CompactFormat format, F&& fn) {
#define COMPACT_CASE(Name, Bits, Steps) case CompactFormat::Name: fn.template operator()<Bits, Steps>(); break;
    switch (format) {
        COMPACT_CASE(PTQ1_0, 0, 0) COMPACT_CASE(TQ2_T, 8, 32) COMPACT_CASE(TQK6, 6, 32)
        COMPACT_CASE(TQK7, 7, 32) COMPACT_CASE(TQP5, 5, 64) COMPACT_CASE(TQP6, 6, 64)
        COMPACT_CASE(TQP7, 7, 64) COMPACT_CASE(TQK5, 5, 32) COMPACT_CASE(TQK4, 4, 32)
        default: throw std::invalid_argument("unknown compact GGUF format");
    }
#undef COMPACT_CASE
}
void validate(CompactFormat format, std::uint64_t stride, std::int64_t rows,
        std::int64_t columns, std::int64_t tokens = 1) {
    const auto row_bytes = compact_row_bytes(format, columns);
    const auto limit = std::uint64_t(std::numeric_limits<std::int64_t>::max() / sizeof(float));
    if (rows < 0 || tokens < 0 || stride < row_bytes ||
            (rows && (std::uint64_t(rows) > limit / std::uint64_t(columns) ||
                std::uint64_t(tokens) > limit / std::uint64_t(rows) ||
                stride > (std::numeric_limits<std::uint64_t>::max() - row_bytes) / std::uint64_t(rows))))
        throw std::invalid_argument("compact GGUF projection geometry overflow");
    if (std::uint64_t(tokens) > limit / std::uint64_t(columns))
        throw std::invalid_argument("compact GGUF input geometry overflow");
}
}

std::uint64_t compact_row_bytes(CompactFormat format, std::int64_t columns) {
    if (columns <= 0 || columns % 128) throw std::invalid_argument("compact GGUF requires full 128-value blocks");
    std::uint64_t bytes = 0;
    dispatch(format, [&]<int Bits, int Steps>() { bytes = Bits ? 2 + Bits * Steps / 8 : 28; });
    return std::uint64_t(columns / 128) * bytes;
}

void compact_project(CompactFormat format, const void* packed, std::uint64_t stride,
        std::int64_t rows, std::int64_t columns, const void* input, CompactInput input_type,
        std::int64_t tokens, void* output, bool half_output, cudaStream_t stream) {
    validate(format, stride, rows, columns, tokens);
    if (input_type != CompactInput::Float16 && input_type != CompactInput::Float32 && input_type != CompactInput::Q8_1)
        throw std::invalid_argument("compact GGUF activation type");
    if (!rows || !tokens) return;
    if (!packed || !input || !output) throw std::invalid_argument("compact GGUF null projection buffer");
    const auto launch_input = [&](auto* x) {
        using T = std::remove_cv_t<std::remove_pointer_t<decltype(x)>>;
        if (reinterpret_cast<std::uintptr_t>(x) % alignof(T)) throw std::invalid_argument("compact GGUF activation alignment");
        const auto launch_output = [&](auto* dest) {
            using Output = std::remove_pointer_t<decltype(dest)>;
            if (reinterpret_cast<std::uintptr_t>(dest) % alignof(Output)) throw std::invalid_argument("compact GGUF output alignment");
            const bool aligned = reinterpret_cast<std::uintptr_t>(x) % (std::is_same_v<T, __half> ? 4 : 16) == 0;
            dispatch(format, [&]<int Bits, int Steps>() {
                const auto launch = [&]<int Tile, bool ExactTile>() {
                    // Orthogonal grid axes avoid a wide integer divide per
                    // CTA. Exact tiles omit token-tail checks in the K loop.
                    const dim3 grid(unsigned(std::min<std::int64_t>(65535, (rows + 3) / 4)),
                        unsigned(std::min<std::int64_t>(65535, (tokens + Tile - 1) / Tile)));
                    project_compact<Bits, Steps, Tile, ExactTile><<<grid, 128, 0, stream>>>(
                        static_cast<const std::uint8_t*>(packed), stride, rows, columns, x, tokens, dest, aligned);
                    check(cudaGetLastError());
                    if (dispatch_trace::enabled()) dispatch_trace::record(stream, "gguf_compact_project",
                        reinterpret_cast<const void*>(project_compact<Bits, Steps, Tile, ExactTile, T, Output>),
                        "format=" + std::to_string(int(format)) + " token_tile=" + std::to_string(Tile) + " decode_reuse=1");
                };
                switch (tokens) {
                    case 1: launch.template operator()<1, true>(); break;
                    case 2: launch.template operator()<2, true>(); break;
                    case 3: launch.template operator()<3, true>(); break;
                    case 4: launch.template operator()<4, true>(); break;
                    default: launch.template operator()<4, false>();
                }
            });
        };
        if (half_output) launch_output(static_cast<__half*>(output)); else launch_output(static_cast<float*>(output));
    };
    switch (input_type) {
        case CompactInput::Float16: launch_input(static_cast<const __half*>(input)); break;
        case CompactInput::Float32: launch_input(static_cast<const float*>(input)); break;
        case CompactInput::Q8_1: launch_input(static_cast<const Q8*>(input)); break;
    }
}

void compact_dequantize(CompactFormat format, const void* packed, std::uint64_t stride,
        std::int64_t rows, std::int64_t columns, void* output, bool half_output, cudaStream_t stream) {
    validate(format, stride, rows, columns);
    if (!rows) return;
    if (!packed || !output) throw std::invalid_argument("compact GGUF null dequantization buffer");
    const auto launch = [&](auto* dest) {
        using T = std::remove_pointer_t<decltype(dest)>;
        if (reinterpret_cast<std::uintptr_t>(dest) % alignof(T)) throw std::invalid_argument("compact GGUF output alignment");
        dispatch(format, [&]<int Bits, int Steps>() {
            const auto groups = rows * (columns / 128);
            dequantize_compact<Bits, Steps><<<unsigned(std::min<std::int64_t>(65535, (groups + 3) / 4)), 128, 0, stream>>>(
                static_cast<const std::uint8_t*>(packed), stride, rows, columns, dest);
        });
    };
    if (half_output) launch(static_cast<__half*>(output)); else launch(static_cast<float*>(output));
    check(cudaGetLastError());
}

bool try_compact_routed_project(CompactFormat format,const std::uint64_t* addresses,
        std::uint64_t stride,std::int64_t experts,std::int64_t rows,std::int64_t columns,
        const std::int64_t* ids,const RouteGroup* groups,std::int64_t tokens,std::int64_t routes,
        const void* input,bool routed_input,float* output,cudaStream_t stream,bool preserve_missing,
        const std::uint64_t* second_addresses,std::uint64_t second_stride,float* second_output,bool allow_shared_lookup,bool allow_decoded_input,bool allow_narrow_rows) {
    if(int(format)<int(CompactFormat::TQ2_T) || int(format)>int(CompactFormat::TQK4))return false;
    if(const auto* enabled=std::getenv("TYLOINFER_GGUF_COMPACT_ROUTED");enabled && enabled[0]=='0')return false;
    validate(format,stride,rows,columns,tokens);
    const bool paired=second_addresses!=nullptr;
    if(paired)validate(format,second_stride,rows,columns,tokens);
    if(experts<1 || routes<1 || tokens>INT64_MAX/routes || (paired && !second_output) || (!paired && second_output))
        throw std::invalid_argument("compact GGUF routed geometry");
    const auto slots=tokens*routes;
    const auto input_rows=routed_input?slots:tokens;
    if(std::uint64_t(input_rows)>std::uint64_t(INT64_MAX)/sizeof(Q8)/std::uint64_t(columns/32))
        throw std::invalid_argument("compact GGUF routed input overflow");
    if(rows && std::uint64_t(slots)>std::uint64_t(INT64_MAX/sizeof(float))/std::uint64_t(rows))
        throw std::invalid_argument("compact GGUF routed output overflow");
    if(!rows || !tokens)return true;
    if(!addresses || !ids || !input || !output)throw std::invalid_argument("compact GGUF null routed buffer");
    dispatch(format,[&]<int Bits,int Steps>() {
        if constexpr(Bits) {
            const auto launch=[&]<int Tile,bool Paired,bool Narrow>() ->bool {
                int device=0,sms=0,shared_limit=0,regular_limit=0;
                check(cudaGetDevice(&device));check(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device));
                check(cudaDeviceGetAttribute(&shared_limit,cudaDevAttrMaxSharedMemoryPerBlockOptin,device));
                check(cudaDeviceGetAttribute(&regular_limit,cudaDevAttrMaxSharedMemoryPerBlock,device));
                shared_limit=std::max(shared_limit,regular_limit);
                // Read-only caching already reuses nearby lookups. Amortize
                // the cooperative copy over at least four table reads.
                // Each 128-weight block uses 64 packed points;
                // this is work amortization, independent of GPU/model names.
                const auto amortized=[&](int row_tile) {
                    return allow_shared_lookup && std::min<std::int64_t>(rows,row_tile)*(columns/128)>=(Narrow?256:128);
                };
                if(groups) {
                    const auto staged=[&]<bool Shared,bool Decoded>() {
                        constexpr int Threads=256,RowTile=Threads/(Narrow?8:16);
                        const auto function=[] {
                            if constexpr(Narrow)return project_compact_routed_narrow<Bits,Steps,Tile,Paired,Shared,Decoded>;
                            else if constexpr(Decoded)return project_compact_routed_decoded<Bits,Steps,Tile,Paired,Shared>;
                            else return project_compact_routed_staged<Bits,Steps,Tile,Paired,Shared>;
                        }();
                        cudaFuncAttributes attributes{};check(cudaFuncGetAttributes(&attributes,function));
                        const auto book_bytes=std::size_t(Shared?8192:0);
                        const auto fixed=attributes.sharedSizeBytes+book_bytes;
                        const auto capacity=std::size_t(shared_limit)>fixed?std::size_t(shared_limit)-fixed:0;
                        if(attributes.maxThreadsPerBlock<Threads || std::uint64_t(columns/32)>capacity/(Tile*sizeof(Q8)))return false;
                        const auto q8_bytes=std::size_t(Tile)*(columns/32)*sizeof(Q8);
                        const auto decoded_bytes=Decoded?std::size_t(Tile)*std::min<std::int64_t>(columns,1024)*sizeof(float):0;
                        if constexpr(Decoded)if(decoded_bytes>capacity)return false;
                        const auto shared=book_bytes+std::max(q8_bytes,decoded_bytes);
                        if(shared>std::size_t(attributes.maxDynamicSharedSizeBytes))
                            check(cudaFuncSetAttribute(function,cudaFuncAttributeMaxDynamicSharedMemorySize,int(shared)));
                        int active=0;check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,function,Threads,shared));
                        if(active<1)return false;
                        const auto x=std::min<std::int64_t>(65535,(rows+RowTile-1)/RowTile);
                        const auto y=std::min<std::int64_t>(65535,std::min(slots,std::max<std::int64_t>(1,std::int64_t(active)*sms*2/(x*(Paired?2:1)))));
                        function<<<dim3(unsigned(x),unsigned(y),Paired?2:1),Threads,shared,stream>>>(
                            addresses,second_addresses,stride,second_stride,experts,rows,columns,groups,slots,routes,
                            static_cast<const Q8*>(input),routed_input,output,second_output,preserve_missing);
                        check(cudaGetLastError());
                        if(dispatch_trace::enabled())dispatch_trace::record(stream,"gguf_compact_routed_staged",
                            reinterpret_cast<const void*>(function),"format="+std::to_string(int(format))+" token_tile="+std::to_string(Tile)+
                            " codebook=fp16 row_lanes="+std::to_string(Narrow?8:16)+" paired="+std::to_string(Paired)+" shared_lookup="+std::to_string(Shared)+" decoded_input="+std::to_string(Decoded)+" shared_bytes="+std::to_string(shared));
                        return true;
                    };
                    if constexpr(Tile>=3)if(allow_decoded_input) {
                        if(amortized(Narrow?32:16) && staged.template operator()<true,true>())return true;
                        if(staged.template operator()<false,true>())return true;
                    }
                    if(amortized(Narrow?32:16) && staged.template operator()<true,false>())return true;
                    if(staged.template operator()<false,false>())return true;
                }
                const auto direct=[&]<bool Shared>() {
                    const auto function=project_compact_routed<Bits,Steps,Tile,Paired,Shared,Narrow>;
                    constexpr int shared=Shared?8192:0,Threads=Shared?256:128,RowTile=Threads/(Narrow?16:32);
                    {
                        cudaFuncAttributes attributes{};check(cudaFuncGetAttributes(&attributes,function));
                        if(attributes.maxThreadsPerBlock<Threads || attributes.sharedSizeBytes+shared>std::size_t(shared_limit))return false;
                        int active=0;check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,function,Threads,shared));
                        if(active<1)return false;
                    }
                    const dim3 grid(unsigned(std::min<std::int64_t>(65535,(rows+RowTile-1)/RowTile)),
                        unsigned(std::min<std::int64_t>(65535,slots)),Paired?2:1);
                    project_compact_routed<Bits,Steps,Tile,Paired,Shared,Narrow><<<grid,Threads,shared,stream>>>(addresses,second_addresses,stride,second_stride,
                        experts,rows,columns,ids,groups,slots,routes,static_cast<const Q8*>(input),routed_input,output,second_output,preserve_missing);
                    check(cudaGetLastError());
                    if(dispatch_trace::enabled())dispatch_trace::record(stream,"gguf_compact_routed",
                        reinterpret_cast<const void*>(function),"format="+std::to_string(int(format))+" token_tile="+std::to_string(Tile)+
                        " k_groups=2 codebook=fp16 row_lanes="+std::to_string(Narrow?8:16)+" paired="+std::to_string(Paired)+" shared_lookup="+std::to_string(Shared)+" factored_scale="+std::to_string(Tile==1)+" group_metadata="+std::to_string(Tile!=1));
                    return true;
                };
                if(amortized(Narrow?16:8) && direct.template operator()<true>())return true;
                return direct.template operator()<false>();
            };
            const auto tile=[&]<bool Paired>() {
                const auto run=[&]<int Tile>() {
                    if(allow_narrow_rows && launch.template operator()<Tile,Paired,true>())return;
                    if(!launch.template operator()<Tile,Paired,false>())
                        throw std::runtime_error("compact GGUF routed kernel exceeds device resources");
                };
                if(!groups)run.template operator()<1>();
                else if(tokens==2)run.template operator()<2>();
                else run.template operator()<4>();
            };
            if(paired)tile.template operator()<true>();else tile.template operator()<false>();
        }
    });
    return true;
}
}
