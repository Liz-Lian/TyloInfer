#pragma once
// Included inside gguf.cu's private namespace, after the row-dot helpers.
// Small by-value packets avoid allocating/copying device pointer tables.
// Arbitrarily large groups are split into packets at submission time.
constexpr int DenseGroupCapacity=8;
struct DenseGroupEntry {
    DenseProjection projection;
    unsigned begin=0,end=0;
    int warps=1;
    bool position_major=false,cooperative=false;
};
struct DenseGroupPacket {DenseGroupEntry entries[DenseGroupCapacity];int count=0;};

template<int Format>
__device__ __forceinline__ void project_dense_group_single_entry(const DenseGroupEntry& entry,int columns,
        const block_q8_1* input,void* output,bool half_output,unsigned tile,float* scratch) {
    const auto& p=entry.projection;
    if constexpr(dense_m1_row_reuse<Format> && Format!=GGML_TYPE_IQ1_S && Format!=GGML_TYPE_IQ1_M)if(entry.warps==3) {
        project_dense_reused_q8_body<Format,2,3>(static_cast<const uint8_t*>(p.packed),p.stride,int(p.rows),columns,input,output,half_output,tile,scratch);
        return;
    }
    if constexpr(dense_m1_row_reuse<Format>)if(entry.warps==4) {
        project_dense_reused_q8_body<Format,2,4>(static_cast<const uint8_t*>(p.packed),p.stride,int(p.rows),columns,input,output,half_output,tile,scratch);
        return;
    }
    project_dense_single_q8_body<Format>(static_cast<const uint8_t*>(p.packed),p.stride,int(p.rows),columns,input,output,half_output,tile,scratch);
}

// M=1 keeps its row-reuse tile separate from multi-token projection state.
// IQ1 uses only sixteen shared floats to join its two K partitions.
// Bound register growth from inactive format cases in this mixed kernel.
// This is a compiler resource target, independent of device and matrix shape.
__global__ void __launch_bounds__(128,8) project_dense_group_single(DenseGroupPacket packet,int columns,
        const block_q8_1* input,bool half_output) {
    __shared__ float scratch[16];
    int index=0;
    while(index+1<packet.count && blockIdx.x>=packet.entries[index].end)++index;
    const auto& entry=packet.entries[index];const auto& p=entry.projection;
    const auto tile=blockIdx.x-entry.begin;
    switch(p.type) {
#define TYLOINFER_GGUF_SINGLE(name) case GGML_TYPE_##name: \
        project_dense_single_q8_body<GGML_TYPE_##name>(static_cast<const uint8_t*>(p.packed),p.stride,int(p.rows),columns,input,p.output,half_output,tile,scratch);break;
        TYLOINFER_GGUF_FORMATS(TYLOINFER_GGUF_SINGLE)
#undef TYLOINFER_GGUF_SINGLE
    default:break;
    }
}

// Keep long-K cooperation outside the ordinary mixed-format instruction body.
__global__ void __launch_bounds__(128) project_dense_group_single_long(DenseGroupPacket packet,int columns,
        const block_q8_1* input,bool half_output) {
    __shared__ float scratch[16];
    int index=0;
    while(index+1<packet.count && blockIdx.x>=packet.entries[index].end)++index;
    const auto& entry=packet.entries[index];const auto& p=entry.projection;
    const auto tile=blockIdx.x-entry.begin;
    switch(p.type) {
#define TYLOINFER_GGUF_SINGLE(name) case GGML_TYPE_##name: \
        project_dense_group_single_entry<GGML_TYPE_##name>(entry,columns,input,p.output,half_output,tile,scratch);break;
        TYLOINFER_GGUF_FORMATS(TYLOINFER_GGUF_SINGLE)
#undef TYLOINFER_GGUF_SINGLE
    default:break;
    }
}

template<int Format,int M>
__device__ __forceinline__ void project_dense_group_entry(const DenseGroupEntry& entry,unsigned tile,
        int columns,const block_q8_1* native,const void* position_major,void* output,
        bool half_output,float* scratch) {
    const auto& p=entry.projection;
    const auto* packed=static_cast<const uint8_t*>(p.packed);
    if constexpr(M==1)if(!entry.cooperative) {
        project_dense_group_single_entry<Format>(entry,columns,native,output,half_output,tile,scratch);
        return;
    }
    if constexpr(Format==GGML_TYPE_IQ3_S || Format==GGML_TYPE_IQ3_XXS || Format==GGML_TYPE_IQ4_XS) {
        if(entry.position_major) {
            const auto* scales=reinterpret_cast<const float*>(static_cast<const uint8_t*>(position_major)+int64_t(columns)*(M==2?2:4));
            if constexpr(Format==GGML_TYPE_IQ4_XS) {
                if(entry.warps==2)project_dense_position_major_body<Format,M,true,2>(packed,p.stride,p.rows,columns,position_major,scales,output,half_output,tile,scratch);
                else if(entry.warps==3)project_dense_position_major_body<Format,M,true,3>(packed,p.stride,p.rows,columns,position_major,scales,output,half_output,tile,scratch);
                else if(entry.warps==4)project_dense_position_major_body<Format,M,true,4>(packed,p.stride,p.rows,columns,position_major,scales,output,half_output,tile,scratch);
                else project_dense_position_major_body<Format,M>(packed,p.stride,p.rows,columns,position_major,scales,output,half_output,tile,nullptr);
            }else project_dense_position_major_body<Format,M>(packed,p.stride,p.rows,columns,position_major,scales,output,half_output,tile,nullptr);
            return;
        }
    }
    if constexpr(Format==GGML_TYPE_Q8_0 || Format==GGML_TYPE_Q5_K || Format==GGML_TYPE_Q6_K || Format==GGML_TYPE_IQ4_XS || dense_long_reduction<Format>) {
        if(entry.cooperative) {
            constexpr int Outputs=M==1 || (Format==GGML_TYPE_Q6_K && M==2)?1:2;
            if(entry.warps==2)project_dense_cooperative_q8_body<Format,M,Outputs,2,PreparedDot<Format>,Format==GGML_TYPE_IQ4_XS>(packed,int(p.stride),int(p.rows),columns,M,native,output,half_output,tile,scratch);
            else if(entry.warps==3)project_dense_cooperative_q8_body<Format,M,Outputs,3,PreparedDot<Format>,Format==GGML_TYPE_IQ4_XS>(packed,int(p.stride),int(p.rows),columns,M,native,output,half_output,tile,scratch);
            else project_dense_cooperative_q8_body<Format,M,Outputs,4,PreparedDot<Format>,Format==GGML_TYPE_IQ4_XS>(packed,int(p.stride),int(p.rows),columns,M,native,output,half_output,tile,scratch);
            return;
        }
    }
    const auto row=int64_t(tile)*4+threadIdx.y;
    if(row>=p.rows)return;
    float sums[M]{};
    q8_dot_row_partials<Format,M>(packed+row*p.stride,native,columns,threadIdx.x,M,sums);
    constexpr int Capacity=warp_sums::capacity(M);
    float values[Capacity]{};
    #pragma unroll
    for(int t=0;t<M;++t)values[t]=sums[t];
    const int lane=threadIdx.x,token=warp_sums::index<Capacity>(lane);
    const float sum=warp_sums::reduce<Capacity>(values,lane);
    if(token<M && lane==warp_sums::owner<Capacity>(token))store_projection(output,int64_t(token)*p.rows+row,sum,half_output);
}
template<int M>
__global__ void project_dense_group(DenseGroupPacket packet,int columns,const block_q8_1* native,
        const void* position_major,bool half_output) {
    // Share scratch across encodings instead of allocating one static array
    // per format specialization in the mixed kernel.
    __shared__ float scratch[3*2*M*32];
    int index=0;
    while(index+1<packet.count && blockIdx.x>=packet.entries[index].end)++index;
    const auto& entry=packet.entries[index];
    const unsigned tile=blockIdx.x-entry.begin;
    switch(entry.projection.type) {
#define TYLOINFER_GGUF_BATCH(name) case GGML_TYPE_##name: \
        project_dense_group_entry<GGML_TYPE_##name,M>(entry,tile,columns,native,position_major,entry.projection.output,half_output,scratch);break;
        TYLOINFER_GGUF_FORMATS(TYLOINFER_GGUF_BATCH)
#undef TYLOINFER_GGUF_BATCH
    default:break;
    }
}
