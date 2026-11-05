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
#include <climits>

namespace mfq::cuda::gguf {
namespace {
using compact_detail::decode_four;
using compact_detail::scale_at;
using compact_detail::ternary_four;
struct alignas(4) Q8 { __half d, s; std::int8_t qs[32]; };
static_assert(sizeof(Q8) == 36);

__device__ __forceinline__ compact_detail::SixteenWindow shared_window6(const uint8_t* q,int lane) {
    const unsigned start=unsigned(__cvta_generic_to_shared(q+2+(21-3*lane)));
    const unsigned aligned=start&~3u;unsigned lo,hi;
    asm("ld.shared.u32 %0,[%1];":"=r"(lo):"r"(aligned));
    asm("ld.shared.u32 %0,[%1];":"=r"(hi):"r"(aligned+4));
    uint64_t bits=(uint64_t(lo)|(uint64_t(hi)<<32))>>((start&3u)*8);
    if(!lane) {const unsigned base=unsigned(__cvta_generic_to_shared(q+2));unsigned prefix;asm("ld.shared.u16 %0,[%1];":"=r"(prefix):"r"(base));bits=(bits&0xffffffu)|(uint64_t(prefix)<<24);}
    return {bits,0};
}
// Read aligned shared words from a complete original compressed slice.
// Circular tails retain the normative bit stream, including wrapped states.
template<int Bits,int Steps>
__device__ __forceinline__ compact_detail::SixteenWindow shared_window(const uint8_t* q,int lane) {
    if constexpr(Steps==32 && Bits==6)return shared_window6(q,lane);
    else if constexpr(Steps==32 && Bits!=8) {
        constexpr int Bytes=Bits*Steps/8;
        const int bit=(Steps-4-4*lane)*Bits,byte=bit/8;
        if(byte+5<=Bytes) {
            const unsigned start=unsigned(__cvta_generic_to_shared(q+2+byte)),aligned=start&~3u;unsigned lo,hi;
            asm("ld.shared.u32 %0,[%1];":"=r"(lo):"r"(aligned));
            asm("ld.shared.u32 %0,[%1];":"=r"(hi):"r"(aligned+4));
            return {(uint64_t(lo)|(uint64_t(hi)<<32))>>((start&3u)*8+bit%8),0};
        }
    }
    return compact_detail::sixteen_window<Bits,Steps>(q+2,lane);
}
template<int Bytes,int RowTile,int StageStride>
__device__ __noinline__ void unaligned_weight_slice(uint8_t* destination,const uint8_t* source,uint64_t stride,int64_t first_row,int64_t rows,int64_t first_block,int64_t columns) {
    constexpr int RowBytes=4*Bytes;
    for(int index=int(threadIdx.x);index<RowTile*RowBytes;index+=blockDim.x) {
        const auto row=first_row+index/RowBytes;
        const int at=index%RowBytes;
        if(row<rows)destination[(index/RowBytes)*StageStride+at]=at<(columns/128-first_block)*Bytes?source[row*stride+first_block*Bytes+at]:0;
    }
}
template<int Bits,int Steps,int Tile,bool Paired,bool Shared,bool Decoded,bool Narrow=false,bool Full=false>
__device__ __forceinline__ void project_compact_routed_staged_body(const std::uint64_t* addresses,const std::uint64_t* second_addresses,
        std::uint64_t stride,std::uint64_t second_stride,std::int64_t experts,std::int64_t rows,std::int64_t columns,
        const RouteGroup* groups,std::int64_t slots,std::int64_t routes,const Q8* input,bool routed,
        float* output,float* second_output,bool preserve_missing) {
    extern __shared__ std::uint32_t storage[];
    const auto* book=storage;
    auto* words=storage+(Shared?2048:0);
    auto* weight_stage=reinterpret_cast<uint8_t*>(storage)+(Shared?8192:0)+max(uint64_t(Tile)*uint64_t(columns/32)*sizeof(Q8),Tile>=3?uint64_t(Tile)*uint64_t(min(columns,int64_t(1024)))*sizeof(float):uint64_t(0));
    bool book_ready=false;
    constexpr int Bytes=2+Bits*Steps/8,Threads=256,Lanes=Narrow?8:16,RowTile=(Narrow?2:1)*(Threads/Lanes),ActivationChunk=1024,WeightBlocks=4,StageStride=(WeightBlocks*Bytes+15)&~15;
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
                    const auto other_row=min(row+Threads/Lanes,rows-1);
                    float sum[Count]{},sum2[Count]{};
                    int weight_first=0;
                    const auto stage_weights=[&](int64_t first_block) {
                        constexpr int WordsPerRow=(WeightBlocks*Bytes)/8;
                        if(pointer) {
                            #if __CUDA_ARCH__ >= 800
                            if(!(pointer&7)) {
                                for(int word=tid;word<RowTile*WordsPerRow;word+=Threads) {
                                    const int relative=word/WordsPerRow,offset=(word%WordsPerRow)*8;
                                    const auto source_row=tile*RowTile+relative;
                                    if(source_row<rows) {
                                        const auto* source=reinterpret_cast<const uint8_t*>(pointer)+source_row*stride+first_block*Bytes+offset;
                                        const auto target=unsigned(__cvta_generic_to_shared(weight_stage+relative*StageStride+offset));
                                        if constexpr(Full)asm volatile("cp.async.ca.shared.global [%0], [%1], 8;"::"r"(target),"l"(source):"memory");
                                        else {
                                            const int valid=int(max(int64_t(0),min(int64_t(8),(columns/128-first_block)*Bytes-offset)));
                                            asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;"::"r"(target),"l"(valid?source:reinterpret_cast<const uint8_t*>(pointer)),"r"(valid):"memory");
                                        }
                                    }
                                }
                            }else
                            #endif
                            unaligned_weight_slice<Bytes,RowTile,StageStride>(weight_stage,reinterpret_cast<const uint8_t*>(pointer),stride,tile*RowTile,rows,first_block,columns);
                        }
                        #if __CUDA_ARCH__ >= 800
                        asm volatile("cp.async.commit_group;");asm volatile("cp.async.wait_group 0;");
                        #endif
                        __syncthreads();
                    };

                    const auto compute_block=[&](std::int64_t block,std::int64_t first,std::int64_t chunk) {
                        // Native Q8 fragments otherwise rescale every coordinate.
                        // The predecoded regime already amortizes that work over
                        // the row tile, so it retains the existing FMA order.
                        constexpr bool RawNativeQ8=!(Decoded && Count>=3);
                        const auto* q=weight_stage+(tid/Lanes)*StageStride+(int(block)-weight_first)*Bytes;
                        const auto* q2=weight_stage+int(other_row-tile*RowTile)*StageStride+(int(block)-weight_first)*Bytes;
                        const float scale=scale_at(q);
                        const float scale2=Narrow?scale_at(q2):0.f;
                        const auto window=[&] {if constexpr(Narrow)return shared_window<Bits,Steps>(q,lane);
                            else return compact_detail::eight_window<Bits,Steps>(q+2,lane);}();
                        const auto window2=[&] {if constexpr(Narrow)return shared_window<Bits,Steps>(q2,lane);
                            else return window;}();
                        float dot[RawNativeQ8?Count:1]{},dot2[RawNativeQ8?Count:1]{};
                        const auto apply=[&]<int Part>() {
                            auto w=[&] {if constexpr(Narrow)return compact_detail::sixteen_part<Bits,Steps,Part,Shared>(window,book);
                                else return compact_detail::eight_part<Bits,Steps,Part,Shared>(window,book);}();
                            auto w2=[&] {if constexpr(Narrow)return compact_detail::sixteen_part<Bits,Steps,Part,Shared>(window2,book);
                                else return w;}();
                            if constexpr(!RawNativeQ8) {
                                w.x=__fmul_rn(w.x,scale);w.y=__fmul_rn(w.y,scale);w.z=__fmul_rn(w.z,scale);w.w=__fmul_rn(w.w,scale);
                                if constexpr(Narrow){w2.x=__fmul_rn(w2.x,scale2);w2.y=__fmul_rn(w2.y,scale2);w2.z=__fmul_rn(w2.z,scale2);w2.w=__fmul_rn(w2.w,scale2);}
                            }
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
                                    const float d=1.f;
                                    value=make_float4(float(std::int8_t(codes))*d,float(std::int8_t(codes>>8))*d,
                                        float(std::int8_t(codes>>16))*d,float(std::int8_t(codes>>24))*d);
                                }
                                if constexpr(RawNativeQ8) {
                                dot[m]=__fmaf_rn(w.x,value.x,dot[m]);
                                if constexpr(Narrow)dot2[m]=__fmaf_rn(w2.x,value.x,dot2[m]);
                                dot[m]=__fmaf_rn(w.y,value.y,dot[m]);
                                if constexpr(Narrow)dot2[m]=__fmaf_rn(w2.y,value.y,dot2[m]);
                                dot[m]=__fmaf_rn(w.z,value.z,dot[m]);
                                if constexpr(Narrow)dot2[m]=__fmaf_rn(w2.z,value.z,dot2[m]);
                                dot[m]=__fmaf_rn(w.w,value.w,dot[m]);
                                if constexpr(Narrow)dot2[m]=__fmaf_rn(w2.w,value.w,dot2[m]);
                                } else {
                                sum[m]=__fmaf_rn(w.x,value.x,sum[m]);
                                if constexpr(Narrow)sum2[m]=__fmaf_rn(w2.x,value.x,sum2[m]);
                                sum[m]=__fmaf_rn(w.y,value.y,sum[m]);
                                if constexpr(Narrow)sum2[m]=__fmaf_rn(w2.y,value.y,sum2[m]);
                                sum[m]=__fmaf_rn(w.z,value.z,sum[m]);
                                if constexpr(Narrow)sum2[m]=__fmaf_rn(w2.z,value.z,sum2[m]);
                                sum[m]=__fmaf_rn(w.w,value.w,sum[m]);
                                if constexpr(Narrow)sum2[m]=__fmaf_rn(w2.w,value.w,sum2[m]);
                                }
                            }
                        };
                        apply.template operator()<0>();apply.template operator()<1>();
                        if constexpr(Narrow){apply.template operator()<2>();apply.template operator()<3>();}
                        if constexpr(RawNativeQ8) {
#pragma unroll
                        for(int m=0;m<Count;++m) {
                            const float d=__half2float(activation[m*blocks+block*4+lane/(Lanes/4)].d);
                            sum[m]=__fmaf_rn(dot[m],scale*d,sum[m]);
                            if constexpr(Narrow)sum2[m]=__fmaf_rn(dot2[m],scale2*d,sum2[m]);
                        }
                        }

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
                            for(auto start=first/128;start<(first+count)/128;start+=4) {
                                weight_first=start;stage_weights(start);
                                if(row<rows)for(auto block=start;block<(Full?start+4:min(start+4,columns/128));++block)compute_block(block,first,chunk);
                                __syncthreads();
                            }
                            __syncthreads();
                        }
                    } else {
                        for(int64_t start=0;start<columns/128;start+=4) {
                            weight_first=start;stage_weights(start);
                            if(pointer && row<rows)for(int64_t block=start;block<(Full?start+4:min(start+4,columns/128));++block)compute_block(block,0,0);
                            __syncthreads();
                        }
                    }
                    #pragma unroll
                    for(int m=0;m<Count;++m) {
                        for(int offset=Lanes/2;offset;offset>>=1) {
                            sum[m]+=__shfl_down_sync(0xffffffffu,sum[m],offset,Lanes);
                            if constexpr(Narrow)sum2[m]+=__shfl_down_sync(0xffffffffu,sum2[m],offset,Lanes);
                        }
                        const auto slot=std::int64_t(group.slots[first+m]);
                        if(!lane && row<rows && slot>=0 && slot<slots)output[slot*rows+row]=sum[m];
                        if constexpr(Narrow)if(!lane && row+Threads/Lanes<rows && slot>=0 && slot<slots)output[slot*rows+row+Threads/Lanes]=sum2[m];
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

// Eight lanes compute two independent output rows and reuse each decoded
// activation fragment. Independent weight streams hide lookup/decode latency;
// Native-Q8 groups hoist both block scales; predecoded groups retain their
// FP32 normalization, FMA and reduction order.
template<int Bits,int Steps,int Tile,bool Paired,bool Shared,bool Decoded,bool Full>
__global__ __launch_bounds__(256,3) void project_compact_routed_async(const std::uint64_t* addresses,const std::uint64_t* second_addresses,
        std::uint64_t stride,std::uint64_t second_stride,std::int64_t experts,std::int64_t rows,std::int64_t columns,
        const RouteGroup* groups,std::int64_t slots,std::int64_t routes,const Q8* input,bool routed,
        float* output,float* second_output,bool preserve_missing) {
    project_compact_routed_staged_body<Bits,Steps,Tile,Paired,Shared,Decoded,true,Full>(addresses,second_addresses,
        stride,second_stride,experts,rows,columns,groups,slots,routes,input,routed,output,second_output,preserve_missing);
}

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
}
bool try_compact_routed_project_async(CompactFormat format,const uint64_t* addresses,
        uint64_t stride,int64_t experts,int64_t rows,int64_t columns,
        const int64_t* ids,const RouteGroup* groups,int64_t tokens,int64_t routes,
        const void* input,bool routed_input,float* output,cudaStream_t stream,bool preserve_missing,
        const uint64_t* second_addresses,uint64_t second_stride,float* second_output,
        bool allow_shared_lookup,bool allow_decoded_input,bool allow_narrow_rows) {
    const auto fallback=[&] {
        return try_compact_routed_project(format,addresses,stride,experts,rows,columns,ids,groups,tokens,routes,
            input,routed_input,output,stream,preserve_missing,second_addresses,second_stride,second_output,
            allow_shared_lookup,allow_decoded_input,allow_narrow_rows);
    };
    // An eight-byte transfer requires aligned row starts. Original layouts
    // that do not meet it retain the ordinary compact decoder. Each bank's
    // base is checked on device, including changing pointer-table contents.
    if(int(format)<int(CompactFormat::TQ2_T) || int(format)>int(CompactFormat::TQK4) ||
            !groups || tokens<2 || !allow_shared_lookup || !allow_decoded_input || !allow_narrow_rows ||
            experts<1 || rows<1 || columns<128 || columns%128 || columns/128>INT_MAX ||
            routes<1 || tokens>INT64_MAX/routes || stride%8 ||
            (second_addresses && (!second_output || second_stride%8)) || (!second_addresses && second_output) ||
            !addresses || !ids || !input || !output)return fallback();
    if(const auto* enabled=std::getenv("TYLOINFER_GGUF_COMPACT_ROUTED");enabled && enabled[0]=='0')return false;
    // Match the existing shared-book amortization requirement. Short work
    // uses the existing global-book path instead of loading an 8 KiB book.
    if(std::min<int64_t>(rows,32)*(columns/128)<256)return fallback();
    const auto minimum_stride=compact_row_bytes(format,columns);
    if(stride<minimum_stride || (second_addresses && second_stride<minimum_stride) ||
            uint64_t(tokens*routes)>uint64_t(INT64_MAX/sizeof(float))/uint64_t(rows))return fallback();
    int device=0,major=0,regular_limit=0,optin_limit=0;
    check(cudaGetDevice(&device));
    check(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device));
    // cp.async is available from compute capability 8.0 onward. The device
    // body also has a scalar implementation so other build targets compile.
    if(major<8)return fallback();
    check(cudaDeviceGetAttribute(&regular_limit,cudaDevAttrMaxSharedMemoryPerBlock,device));
    check(cudaDeviceGetAttribute(&optin_limit,cudaDevAttrMaxSharedMemoryPerBlockOptin,device));
    bool launched=false;
    const auto dispatch=[&]<int Bits,int Steps>() {
        const auto token=[&]<int Tile>() {
            const auto pair=[&]<bool Paired>() {
                const auto lookup=[&]<bool Shared,bool Full>() ->bool {
                    const auto function=project_compact_routed_async<Bits,Steps,Tile,Paired,Shared,true,Full>;
                    cudaFuncAttributes attributes{};check(cudaFuncGetAttributes(&attributes,function));
                    if(attributes.maxThreadsPerBlock<256 || uint64_t(columns/32)>uint64_t(std::max(regular_limit,optin_limit))/(Tile*sizeof(Q8)))return false;
                    const auto shared=(Shared?8192:0)+std::max(size_t(Tile)*(columns/32)*sizeof(Q8),
                        Tile>=3?size_t(Tile)*std::min<int64_t>(columns,1024)*sizeof(float):0)+64*((4*(2+Bits*Steps/8)+15)&~15);
                    if(shared+attributes.sharedSizeBytes>size_t(std::max(regular_limit,optin_limit)))
                        return false;
                    if(shared>size_t(attributes.maxDynamicSharedSizeBytes))
                        check(cudaFuncSetAttribute(function,cudaFuncAttributeMaxDynamicSharedMemorySize,int(shared)));
                    int active=0;check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,function,256,shared));
                    if(active<1)return false;
                    const auto x=std::min<int64_t>(65535,(rows+63)/64);
                    const auto y=std::min<int64_t>(65535,tokens*routes);
                    function<<<dim3(unsigned(x),unsigned(y),Paired?2:1),256,shared,stream>>>(addresses,second_addresses,stride,second_stride,
                        experts,rows,columns,groups,tokens*routes,routes,static_cast<const Q8*>(input),routed_input,output,second_output,preserve_missing);
                    check(cudaGetLastError());
                    if(dispatch_trace::enabled())dispatch_trace::record(stream,"gguf_compact_routed_async",
                        reinterpret_cast<const void*>(function),"format="+std::to_string(int(format))+" token_tile="+std::to_string(Tile)+
                        " rows_per_lane=2 row_lanes=8 weight_tile_blocks=4 full_k_tiles="+std::to_string(Full)+
                        " paired="+std::to_string(Paired)+" shared_bytes="+std::to_string(shared));
                    return true;
                };
                if(columns%512==0)launched=lookup.template operator()<true,true>();else launched=lookup.template operator()<true,false>();
            };
            if(second_addresses)pair.template operator()<true>();else pair.template operator()<false>();
        };
        if(tokens==2)token.template operator()<2>();else token.template operator()<4>();
    };
    switch(format) {
#define CASE(Name,Bits,Steps) case CompactFormat::Name:dispatch.template operator()<Bits,Steps>();break;
        CASE(TQ2_T,8,32) CASE(TQK4,4,32) CASE(TQK5,5,32) CASE(TQK6,6,32)
        CASE(TQK7,7,32) CASE(TQP5,5,64) CASE(TQP6,6,64) CASE(TQP7,7,64)
#undef CASE
        default:return false;
    }
    const auto status=cudaGetLastError();if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
    return launched || fallback();
}
} // namespace mfq::cuda::gguf
