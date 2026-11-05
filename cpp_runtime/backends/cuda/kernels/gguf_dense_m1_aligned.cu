#include "gguf.h"
#include "tyloinfer/gguf_quant.h"
#include "ggml-cuda/common.cuh"
#include "gguf_decode.cuh"
#include "gguf_dot.cuh"
#include "gguf_quantize.cuh"
#include "gguf_dense_rows.cuh"
#include "kernel_dispatch_trace.h"
#include <algorithm>
#include <climits>
#include <stdexcept>
#define ALIGNED_FORMATS(X) X(IQ1_S) X(IQ1_M) X(IQ2_XXS) X(IQ2_XS) X(IQ2_S) X(IQ3_XXS) X(IQ3_S) X(Q2_K) X(Q4_K) X(Q5_K) X(Q6_K) X(IQ4_XS)
namespace mfq::cuda::gguf {namespace {
#include "gguf_dense_m1_aligned.cuh"
void checked_m1(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));}
template<class T>__global__ void quantize_m1_aligned_kernel(const T* input,int64_t columns,void* output){
 const int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<columns)store_q8_m1_aligned(float(input[i]),i,columns,output);
}
template<int F,int R,int W,int BlockWarps=4,bool Cached=false>__global__ void __launch_bounds__(32*BlockWarps) project_aligned_m1(const uint8_t* packed,uint64_t stride,int rows,int columns,const uint4* input,const unsigned* scales,void* output,bool half_output){
 __shared__ float scratch[4*BlockWarps];
 if constexpr(Cached) {
  using Dot=SharedM1PreparedDot<F>;Dot::stage(threadIdx.y*32+threadIdx.x,32*BlockWarps);
  project_aligned_m1_body<F,R,W,BlockWarps,M1ProjectionStore,Dot>(packed,stride,rows,columns,input,scales,output,half_output,blockIdx.x,scratch);
 }else project_aligned_m1_body<F,R,W,BlockWarps>(packed,stride,rows,columns,input,scales,output,half_output,blockIdx.x,scratch);
}
template<int F>int aligned_k_warps(int64_t rows,int64_t columns,uint64_t group_blocks=0){
 if constexpr(F==GGML_TYPE_IQ4_XS)return canonical_iq4_k_warps(columns);
 else if constexpr(F==GGML_TYPE_IQ1_S||F==GGML_TYPE_IQ1_M)return 1;
 else {const auto parts=columns/M1PreparedDot<F>::qk*M1PreparedDot<F>::parts;if(parts<4*32)return 1;
  const int cooperative_warps=parts%(3*32)==0&&parts%(4*32)!=0?3:4;
  int device=0;checked_m1(cudaGetDevice(&device));static thread_local int cached_device=-1,sms=0,active=0;
  if(cached_device!=device){checked_m1(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device));checked_m1(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,project_aligned_m1<F,2,1>,128,0));cached_device=device;}
  const auto blocks=group_blocks?group_blocks:uint64_t((rows+7)/8);return active>0&&blocks<uint64_t(sms)*unsigned(active)?cooperative_warps:1;
 }
}
struct AlignedEntry{DenseProjection p;unsigned begin=0,end=0;int warps=1;};
struct AlignedPacket{AlignedEntry entries[8];int count=0;};
template<int F,bool AllowKCooperation=true>__device__ __forceinline__ void aligned_entry(const AlignedEntry& entry,int columns,const uint4* input,const unsigned* scales,bool half_output,unsigned tile,float* scratch){
 const auto& p=entry.p;
 if constexpr(F==GGML_TYPE_IQ1_S||F==GGML_TYPE_IQ1_M){if(columns/M1PreparedDot<F>::qk*M1PreparedDot<F>::parts>=64){project_aligned_m1_body<F,4,2>(static_cast<const uint8_t*>(p.packed),p.stride,int(p.rows),columns,input,scales,p.output,half_output,tile,scratch);return;}}
 if constexpr(AllowKCooperation){if(entry.warps==3){project_aligned_m1_body<F,2,3>(static_cast<const uint8_t*>(p.packed),p.stride,int(p.rows),columns,input,scales,p.output,half_output,tile,scratch);return;}if(entry.warps==4){project_aligned_m1_body<F,2,4>(static_cast<const uint8_t*>(p.packed),p.stride,int(p.rows),columns,input,scales,p.output,half_output,tile,scratch);return;}}
 project_aligned_m1_body<F,2,1>(static_cast<const uint8_t*>(p.packed),p.stride,int(p.rows),columns,input,scales,p.output,half_output,tile,scratch);
}
template<int F,bool SplitK>__device__ __forceinline__ void aligned_pair_entry(const AlignedEntry& entry,int columns,const uint4* input,const unsigned* scales,bool half_output,unsigned tile,float* scratch){
 if constexpr(SplitK&&F!=GGML_TYPE_IQ1_S&&F!=GGML_TYPE_IQ1_M&&F!=GGML_TYPE_IQ4_XS){const auto& p=entry.p;project_aligned_m1_body<F,2,2>(static_cast<const uint8_t*>(p.packed),p.stride,int(p.rows),columns,input,scales,p.output,half_output,tile,scratch);}
 else aligned_entry<F,false>(entry,columns,input,scales,half_output,tile,scratch);
}
template<int A,int B,bool Half,bool SplitK>__global__ void __launch_bounds__(128) project_aligned_pair(AlignedEntry a,AlignedEntry b,int columns,const uint4* input,const unsigned* scales){
 // The host selects typed pairs only when neither entry uses four K warps.
 // Keep those unreachable bodies out of this kernel's register allocation.
 __shared__ float scratch[16];if(blockIdx.x<a.end)aligned_pair_entry<A,SplitK>(a,columns,input,scales,Half,blockIdx.x,scratch);else aligned_pair_entry<B,SplitK>(b,columns,input,scales,Half,blockIdx.x-b.begin,scratch);
}
template<int A>void launch_aligned_second(const AlignedPacket& p,int columns,const uint4* input,const unsigned* scales,bool half_output,cudaStream_t stream){
 const auto launch=[&]<int B,bool Half,bool SplitK>(){auto a=p.entries[0],b=p.entries[1];constexpr int ARows=SplitK&&A!=GGML_TYPE_IQ1_S&&A!=GGML_TYPE_IQ1_M&&A!=GGML_TYPE_IQ4_XS?4:8,BRows=SplitK&&B!=GGML_TYPE_IQ1_S&&B!=GGML_TYPE_IQ1_M&&B!=GGML_TYPE_IQ4_XS?4:8;
  a.end=unsigned((a.p.rows+ARows-1)/ARows);b.begin=a.end;b.end=b.begin+unsigned((b.p.rows+BRows-1)/BRows);
  const auto fn=project_aligned_pair<A,B,Half,SplitK>;fn<<<b.end,dim3(32,4),0,stream>>>(a,b,columns,input,scales);
  if(dispatch_trace::enabled())dispatch_trace::record(stream,"gguf_dense_m1_aligned_pair",reinterpret_cast<const void*>(fn),"M=1 formats="+std::to_string(A)+"/"+std::to_string(B)+" split_k="+std::to_string(SplitK));};
 const auto select=[&]<int B,bool Half>(){if(columns/M1PreparedDot<A>::qk*M1PreparedDot<A>::parts>=128&&columns/M1PreparedDot<B>::qk*M1PreparedDot<B>::parts>=128)launch.template operator()<B,Half,true>();else launch.template operator()<B,Half,false>();};
 switch(p.entries[1].p.type){
 #define CASE(F) case GGML_TYPE_##F:if(half_output)select.template operator()<GGML_TYPE_##F,true>();else select.template operator()<GGML_TYPE_##F,false>();break;
 ALIGNED_FORMATS(CASE)
 #undef CASE
 default:throw std::invalid_argument("Unsupported aligned M=1 pair");}
}
__global__ void __launch_bounds__(128,8) project_aligned_group(AlignedPacket packet,int columns,const uint4* input,const unsigned* scales,bool half_output){
 __shared__ float scratch[16];int index=0;while(index+1<packet.count&&blockIdx.x>=packet.entries[index].end)++index;const auto& entry=packet.entries[index];
 switch(entry.p.type){
 #define CASE(F) case GGML_TYPE_##F:aligned_entry<GGML_TYPE_##F>(entry,columns,input,scales,half_output,blockIdx.x-entry.begin,scratch);break;
 ALIGNED_FORMATS(CASE)
 #undef CASE
 default:break;}
}
void validate_aligned_projection(const DenseProjection& p,int64_t columns){
 if(!supports_m1_aligned(p.type)||p.rows<0||p.rows>INT_MAX||columns<1||columns>INT_MAX||columns%ggml_blck_size(p.type))throw std::invalid_argument("Aligned M=1 projection geometry/format differs");
 if(p.rows&&(!p.packed||!p.output||p.stride<ggml_row_size(p.type,columns)||p.stride>UINT64_MAX/uint64_t(p.rows)))throw std::invalid_argument("Aligned M=1 projection storage differs");
}
}
bool supports_m1_aligned(ggml_type type)noexcept{switch(type){
 #define CASE(F) case GGML_TYPE_##F:return true;
 ALIGNED_FORMATS(CASE)
 #undef CASE
 default:return false;}}
bool prefers_m1_aligned(ggml_type type)noexcept{
 // IQ1's four-row reuse amortizes aligned vector reads over more outputs.
 return type==GGML_TYPE_IQ1_S||type==GGML_TYPE_IQ1_M;
}
void quantize_input_m1_aligned(const void* input,bool half_input,int64_t columns,void* output,cudaStream_t stream){
 q8_row_bytes(columns);if(columns>INT_MAX||!input||!output||(reinterpret_cast<uintptr_t>(output)&15))throw std::invalid_argument("Aligned M=1 activation geometry/storage differs");
 const auto blocks=unsigned((columns+255)/256);
 if(half_input)quantize_m1_aligned_kernel<<<blocks,256,0,stream>>>(static_cast<const half*>(input),columns,output);
 else quantize_m1_aligned_kernel<<<blocks,256,0,stream>>>(static_cast<const float*>(input),columns,output);checked_m1(cudaGetLastError());
}
void matvec_m1_aligned(ggml_type type,const void* w,uint64_t stride,int64_t rows,int64_t columns,const void* input,void* output,bool half_output,cudaStream_t stream){
 validate_aligned_projection({w,stride,rows,output,type},columns);if(!rows)return;if(!input||(reinterpret_cast<uintptr_t>(input)&15))throw std::invalid_argument("Aligned M=1 activation storage differs");
 const auto launch=[&]<int F,int R,int W,bool Cached=false>(){constexpr int BlockWarps=W==3?3:4;const auto fn=project_aligned_m1<F,R,W,BlockWarps,Cached>;constexpr int Outputs=BlockWarps/W*R;constexpr unsigned Shared=[] {if constexpr(Cached)return SharedM1PreparedDot<F>::shared_bytes;else return 0u;}();fn<<<unsigned((rows+Outputs-1)/Outputs),dim3(32,BlockWarps),Shared,stream>>>(static_cast<const uint8_t*>(w),stride,int(rows),int(columns),static_cast<const uint4*>(input),reinterpret_cast<const unsigned*>(static_cast<const uint8_t*>(input)+columns),output,half_output);
  if(dispatch_trace::enabled())dispatch_trace::record(stream,"gguf_dense_m1_aligned",reinterpret_cast<const void*>(fn),"M=1 format="+std::to_string(F)+" k_warps="+std::to_string(W)+" shared_codebook="+std::to_string(Cached));};
 const auto format=[&]<int F>(){if constexpr(F==GGML_TYPE_IQ1_S||F==GGML_TYPE_IQ1_M){if(columns/M1PreparedDot<F>::qk*M1PreparedDot<F>::parts>=64){launch.template operator()<F,4,2,true>();return;}}
  const auto warps=aligned_k_warps<F>(rows,columns);if(warps==3)launch.template operator()<F,2,3>();else if(warps==4)launch.template operator()<F,2,4>();else launch.template operator()<F,2,1>();};
 switch(type){
 #define CASE(F) case GGML_TYPE_##F:format.template operator()<GGML_TYPE_##F>();break;
 ALIGNED_FORMATS(CASE)
 #undef CASE
 default:throw std::invalid_argument("Unsupported aligned M=1 projection");}checked_m1(cudaGetLastError());
}
void matvec_group_m1_aligned(const DenseProjection* projections,size_t count,int64_t columns,const void* input,bool half_output,cudaStream_t stream){
 if((!projections&&count)||columns<1||columns>INT_MAX)throw std::invalid_argument("Aligned M=1 group geometry differs");
 uint64_t group_blocks=0;for(size_t i=0;i<count;++i){validate_aligned_projection(projections[i],columns);group_blocks+=uint64_t(projections[i].rows+7)/8;}
 if(!group_blocks)return;if(!input||(reinterpret_cast<uintptr_t>(input)&15))throw std::invalid_argument("Aligned M=1 activation storage differs");
 AlignedPacket packet{};unsigned blocks=0;const auto* codes=static_cast<const uint4*>(input);const auto* scales=reinterpret_cast<const unsigned*>(static_cast<const uint8_t*>(input)+columns);
 const auto submit=[&]{if(!packet.count)return;const bool cooperative=std::any_of(packet.entries,packet.entries+packet.count,[](const auto& e){return e.warps>1;});
  if(!cooperative){std::stable_sort(packet.entries,packet.entries+packet.count,[](const auto& a,const auto& b){return a.p.rows<b.p.rows;});unsigned next=0;for(int i=0;i<packet.count;++i){auto& e=packet.entries[i];const auto length=e.end-e.begin;e.begin=next;next+=length;e.end=next;}}
  if(packet.count==2&&!cooperative){switch(packet.entries[0].p.type){
   #define CASE(F) case GGML_TYPE_##F:launch_aligned_second<GGML_TYPE_##F>(packet,int(columns),codes,scales,half_output,stream);break;
   ALIGNED_FORMATS(CASE)
   #undef CASE
   default:throw std::invalid_argument("Unsupported aligned M=1 pair");}
  }else{project_aligned_group<<<blocks,dim3(32,4),0,stream>>>(packet,int(columns),codes,scales,half_output);
   if(dispatch_trace::enabled())dispatch_trace::record(stream,"gguf_dense_m1_aligned_group",reinterpret_cast<const void*>(project_aligned_group),"M=1 matrices="+std::to_string(packet.count));}
  checked_m1(cudaGetLastError());packet={};blocks=0;};
 for(size_t i=0;i<count;++i){const auto& p=projections[i];if(!p.rows)continue;AlignedEntry e{};e.p=p;
  switch(p.type){
   #define CASE(F) case GGML_TYPE_##F:e.warps=aligned_k_warps<GGML_TYPE_##F>(p.rows,columns,group_blocks);break;
   ALIGNED_FORMATS(CASE)
   #undef CASE
   default:throw std::invalid_argument("Unsupported aligned M=1 group");}
  const auto tiles=uint64_t(p.rows+(e.warps>1?1:7))/(e.warps>1?2:8);
  if(packet.count==8||uint64_t(blocks)+tiles>INT_MAX)submit();e.begin=blocks;blocks+=unsigned(tiles);e.end=blocks;packet.entries[packet.count++]=e;
 }submit();
}
}
