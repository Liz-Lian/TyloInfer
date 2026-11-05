#include "gguf_q6_group.h"
#include "tyloinfer/gguf_quant.h"
#include "ggml-cuda/common.cuh"
#include "gguf_decode.cuh"
#include "gguf_dot.cuh"
#include "gguf_quantize.cuh"
#include "gguf_dense_rows.cuh"
#include "kernel_dispatch_trace.h"
#include <climits>
#include <stdexcept>
#include <string>
namespace mfq::cuda::gguf {
namespace {
constexpr unsigned Capacity=8;
// Homogeneous groups do not carry a format tag or wide geometry through
// every lane. Host validation proves the existing dot's int bounds.
struct Entry {const uint8_t* packed;void* output;int rows,stride;unsigned end;};
struct Group {Entry entries[Capacity];unsigned count;};
template<int M,int Warps=4>
__global__ void project_q6_group(Group group,int columns,const block_q8_1* input,bool half_output) {
    __shared__ float partial[M==1?2*Warps:(Warps-1)*M*32];
    unsigned i=0;
    while(i+1<group.count && blockIdx.x>=group.entries[i].end)++i;
    const auto& p=group.entries[i];
    const unsigned row=blockIdx.x-(i?group.entries[i-1].end:0);
    // Additional K warps fill small row grids without a separate split-K
    // allocation or reduction launch. The four-warp M1 path retains its
    // original three/four-warp arithmetic and two-row activation reuse.
    if constexpr(M==1) {
        if constexpr(Warps>4)
            project_dense_reused_q8_body<GGML_TYPE_Q6_K,2,Warps,Warps>(p.packed,p.stride,p.rows,columns,input,p.output,half_output,row,partial);
        else {
            const int parts=columns/M1PreparedDot<GGML_TYPE_Q6_K>::qk*M1PreparedDot<GGML_TYPE_Q6_K>::parts;
            if(parts%96==0 && parts%128!=0)
                project_dense_reused_q8_body<GGML_TYPE_Q6_K,2,3>(p.packed,p.stride,p.rows,columns,input,p.output,half_output,row,partial);
            else project_dense_reused_q8_body<GGML_TYPE_Q6_K,2,4>(p.packed,p.stride,p.rows,columns,input,p.output,half_output,row,partial);
        }
    } else project_dense_cooperative_q8_body<GGML_TYPE_Q6_K,M,1,Warps>(p.packed,p.stride,p.rows,
        columns,M,input,p.output,half_output,row,partial);
}
struct Resources {int device=-1,sm_count=0,active[4][3]{};};
template<int M,int Warps>bool resource_entry(Resources& r) {
    int active=0;
    if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,project_q6_group<M,Warps>,32*Warps,0)!=cudaSuccess)return false;
    r.active[M-1][Warps==4?0:Warps==8?1:2]=active;return active>0;
}
template<int M>bool resource_entries(Resources& r) {
    if(!resource_entry<M,4>(r))return false;
    resource_entry<M,8>(r);resource_entry<M,16>(r);return true;
}
bool resources(Resources& r) {
    int device=-1;if(cudaGetDevice(&device)!=cudaSuccess)return false;
    if(device==r.device)return true;
    Resources next;next.device=device;
    if(cudaDeviceGetAttribute(&next.sm_count,cudaDevAttrMultiProcessorCount,device)!=cudaSuccess || next.sm_count<1)return false;
    if(!resource_entries<1>(next)||!resource_entries<2>(next)||!resource_entries<3>(next)||!resource_entries<4>(next))return false;
    r=next;return true;
}
template<int M,int Warps>void submit_width(Group group,unsigned rows,int columns,const void* input,bool half_output,cudaStream_t stream) {
    const auto function=project_q6_group<M,Warps>;
    function<<<rows,dim3(32,Warps),0,stream>>>(group,columns,static_cast<const block_q8_1*>(input),half_output);
    const auto error=cudaGetLastError();if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
    if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(stream,
        "gguf_underfilled_q6_group",reinterpret_cast<const void*>(function),
        "M="+std::to_string(M)+" matrices="+std::to_string(group.count)+
        " tiles="+std::to_string(rows)+" block_warps="+std::to_string(Warps)+
        " ordered_partition=1");
}
template<int M>void submit(Group group,unsigned rows,int columns,const void* input,bool half_output,
        cudaStream_t stream,const Resources& r) {
    using F=std::conditional_t<M==1,M1PreparedDot<GGML_TYPE_Q6_K>,PreparedDot<GGML_TYPE_Q6_K>>;
    const int fragments=columns/F::qk*F::parts;
    // A wide CTA must have useful work in at least half of its warps, and
    // the entire row grid must fit its measured resident CTA capacity.
    // Otherwise keep four warps rather than adding another scheduling wave.
    if(fragments>=16*16 && rows<=std::uint64_t(r.sm_count)*r.active[M-1][2])
        submit_width<M,16>(group,rows,columns,input,half_output,stream);
    else if(fragments>=8*16 && rows<=std::uint64_t(r.sm_count)*r.active[M-1][1])
        submit_width<M,8>(group,rows,columns,input,half_output,stream);
    else submit_width<M,4>(group,rows,columns,input,half_output,stream);
}
}
bool try_underfilled_q6_group(const DenseProjection* p,std::size_t count,std::int64_t columns,
        const void* input,std::int64_t tokens,bool half_output,cudaStream_t stream) {
    if(!p || !count || count>Capacity || !input || tokens<1 || tokens>4 || columns<1 || columns>INT_MAX || columns%256)return false;
    using F=PreparedDot<GGML_TYPE_Q6_K>;
    // Four K warps require a complete fragment for every lane.
    if(columns/F::qk*F::parts<128)return false;
    Group group{};std::uint64_t rows=0;
    for(std::size_t i=0;i<count;++i) {
        if(p[i].type!=GGML_TYPE_Q6_K || p[i].rows<0 || p[i].rows>INT_MAX/tokens ||
           p[i].stride>INT_MAX || p[i].stride<std::uint64_t(columns/256)*sizeof(block_q6_K))return false;
        if(!p[i].rows)continue;
        if(!p[i].packed || !p[i].output)return false;
        rows+=(p[i].rows+(tokens==1?1:0))/(tokens==1?2:1);if(rows>INT_MAX)return false;
        group.entries[group.count++]={static_cast<const uint8_t*>(p[i].packed),p[i].output,int(p[i].rows),int(p[i].stride),unsigned(rows)};
    }
    if(!group.count)return true;
    thread_local Resources cached;
    if(!resources(cached))return false;
    // An already saturated row grid retains the old wider row reuse.
    if(rows>std::uint64_t(cached.sm_count)*cached.active[tokens-1][0])return false;
    switch(tokens){case 1:submit<1>(group,unsigned(rows),int(columns),input,half_output,stream,cached);break;
    case 2:submit<2>(group,unsigned(rows),int(columns),input,half_output,stream,cached);break;
    case 3:submit<3>(group,unsigned(rows),int(columns),input,half_output,stream,cached);break;
    case 4:submit<4>(group,unsigned(rows),int(columns),input,half_output,stream,cached);break;}
    return true;
}
}
