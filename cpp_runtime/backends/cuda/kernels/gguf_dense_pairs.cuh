#pragma once
// Included in gguf.cu's private namespace. Specializing an ordinary row-reuse
// pair prevents inactive formats from setting the whole kernel's register
// allocation. Both matrices retain their own compressed storage and outputs.
struct DensePairEntry {
    const void* packed;
    std::uint64_t stride;
    int rows;
    void* output;
};

template<int Format,bool SplitK>
__device__ __forceinline__ void project_dense_pair_entry(const DensePairEntry& entry,
        int columns,const block_q8_1* input,bool half_output,unsigned tile,float* scratch) {
    // IQ1 already uses four-row reuse and two K warps. Other long rows
    // split their decode chain between two warps without changing storage.
    if constexpr(SplitK && Format!=GGML_TYPE_IQ1_S && Format!=GGML_TYPE_IQ1_M)
        project_dense_reused_q8_body<Format,2,2>(static_cast<const std::uint8_t*>(entry.packed),
            entry.stride,entry.rows,columns,input,entry.output,half_output,tile,scratch);
    else project_dense_single_q8_body<Format>(static_cast<const std::uint8_t*>(entry.packed),
        entry.stride,entry.rows,columns,input,entry.output,half_output,tile,scratch);
}

template<int First,int Second,bool Half,bool SplitK>
__global__ void __launch_bounds__(128) project_dense_pair_single(
        DensePairEntry first,DensePairEntry second,unsigned split,int columns,const block_q8_1* input) {
    __shared__ float scratch[16];
    if(blockIdx.x<split)
        project_dense_pair_entry<First,SplitK>(first,columns,input,Half,blockIdx.x,scratch);
    else
        project_dense_pair_entry<Second,SplitK>(second,columns,input,Half,blockIdx.x-split,scratch);
}

template<int First>
bool try_dense_pair_second(const DenseGroupPacket& packet,int columns,const block_q8_1* input,
        bool half_output,cudaStream_t stream) {
    const auto launch=[&]<int Second,bool Half,bool SplitK>() {
        const auto& a=packet.entries[0];const auto& b=packet.entries[1];
        const DensePairEntry first{a.projection.packed,a.projection.stride,int(a.projection.rows),a.projection.output};
        const DensePairEntry second{b.projection.packed,b.projection.stride,int(b.projection.rows),b.projection.output};
        constexpr int FirstRows=SplitK && First!=GGML_TYPE_IQ1_S && First!=GGML_TYPE_IQ1_M?4:8;
        constexpr int SecondRows=SplitK && Second!=GGML_TYPE_IQ1_S && Second!=GGML_TYPE_IQ1_M?4:8;
        const unsigned split=unsigned((a.projection.rows+FirstRows-1)/FirstRows);
        const unsigned end=split+unsigned((b.projection.rows+SecondRows-1)/SecondRows);
        const auto function=project_dense_pair_single<First,Second,Half,SplitK>;
        function<<<end,dim3(32,4),0,stream>>>(first,second,split,columns,input);
        if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(stream,
            "gguf_dense_pair_single",reinterpret_cast<const void*>(function),
            "M=1 matrices=2 formats="+std::to_string(First)+"/"+std::to_string(Second)+
            " split_k="+std::to_string(SplitK));
    };
    const auto select=[&]<int Second,bool Half>() {
        // At least two complete fragments per cooperating lane amortize
        // the shared reduction. Underfilled grids retain the separate W4
        // path selected before typed-pair dispatch.
        if(columns/M1PreparedDot<First>::qk*M1PreparedDot<First>::parts>=128 &&
                columns/M1PreparedDot<Second>::qk*M1PreparedDot<Second>::parts>=128)
            launch.template operator()<Second,Half,true>();
        else launch.template operator()<Second,Half,false>();
    };
    switch(packet.entries[1].projection.type) {
#define TYLOINFER_PAIR_SECOND(Name) case GGML_TYPE_##Name: \
        if constexpr(dense_m1_row_reuse<GGML_TYPE_##Name>) { \
            if(half_output)select.template operator()<GGML_TYPE_##Name,true>(); \
            else select.template operator()<GGML_TYPE_##Name,false>(); \
            return true; \
        }else return false;
        TYLOINFER_GGUF_FORMATS(TYLOINFER_PAIR_SECOND)
#undef TYLOINFER_PAIR_SECOND
    default:return false;
    }
}

bool try_dense_pair_single(const DenseGroupPacket& packet,int columns,const block_q8_1* input,
        bool half_output,cudaStream_t stream) {
    if(packet.count!=2)return false;
    switch(packet.entries[0].projection.type) {
#define TYLOINFER_PAIR_FIRST(Name) case GGML_TYPE_##Name: \
        if constexpr(dense_m1_row_reuse<GGML_TYPE_##Name>) \
            return try_dense_pair_second<GGML_TYPE_##Name>(packet,columns,input,half_output,stream); \
        else return false;
        TYLOINFER_GGUF_FORMATS(TYLOINFER_PAIR_FIRST)
#undef TYLOINFER_PAIR_FIRST
    default:return false;
    }
}
