#pragma once
// Radix threshold selection with stable ties. Block IDs are emitted in input
// order rather than score order; attention consumes the same top-k set.
__device__ unsigned qsa_score_key(float score) {
    unsigned bits=__float_as_uint(score);
    if(!(bits&0x7fffffffu))bits=0; // CUB treats signed zero as equal.
    return bits&0x80000000u?~bits:bits^0x80000000u;
}
struct QsaThresholdCountAdd {
    __device__ __forceinline__ int2 operator()(const int2 a,const int2 b)const {
        return make_int2(a.x+b.x,a.y+b.y);
    }
};
template<int Items=16> __device__ void live_topk_threshold(const float* source,const int64_t* ids,float* output,int64_t* out_ids,
        int row,int columns,int begin,int available,int pools,int count,int64_t out,
        unsigned char* temporary,bool emit_tokens) {
    static_assert(Items>0 && Items<=16 && (Items&(Items-1))==0);
    using Scan=cub::BlockScan<int2,256>;
    static_assert(sizeof(Scan::TempStorage)<=8*256*sizeof(int));
    auto* histogram=reinterpret_cast<int(*)[256]>(temporary);
    const int thread=threadIdx.x,lane=thread&31,warp=thread/32;
    const int per=(available+255)/256,first=thread*per;
    uint32_t keys[Items]{};
    #pragma unroll
    for(int i=0;i<Items;++i)if(i<per && first+i<available)
        keys[i]=qsa_score_key(source[int64_t(row)*columns+begin+first+i]);
    __shared__ int warp_totals[8],digit,above_digit,digit_population;
    uint32_t threshold=0;int above=0;
    for(int shift=24;shift>=0;shift-=8) {
        for(int bin=lane;bin<256;bin+=32)histogram[warp][bin]=0;
        __syncwarp();
        const uint32_t mask=shift==24?0u:0xffffffffu<<(shift+8);
        #pragma unroll
        for(int i=0;i<Items;++i)if(i<per && first+i<available && (keys[i]&mask)==(threshold&mask))
            atomicAdd(&histogram[warp][(keys[i]>>shift)&255],1);
        __syncthreads();
        int population=0;for(int w=0;w<8;++w)population+=histogram[w][255-thread];
        int inclusive=population;
        #pragma unroll
        for(int step=1;step<32;step*=2) {
            const int other=__shfl_up_sync(0xffffffffu,inclusive,step);
            if(lane>=step)inclusive+=other;
        }
        if(lane==31)warp_totals[warp]=inclusive;
        __syncthreads();
        int before=above+inclusive-population;
        for(int w=0;w<warp;++w)before+=warp_totals[w];
        if(before<count && before+population>=count) {digit=255-thread;above_digit=before;digit_population=population;}
        __syncthreads();
        threshold|=uint32_t(digit)<<shift;above=above_digit;
        const bool complete=above+digit_population==count;
        __syncthreads();
        // Selecting this entire prefix already gives the exact top-k set.
        // Lower radix digits cannot change membership or stable tie order.
        if(complete)break;
    }
    int greater=0,equal=0;
    #pragma unroll
    for(int i=0;i<Items;++i)if(i<per && first+i<available) {
        greater+=keys[i]>threshold;equal+=keys[i]==threshold;
    }
    // One scan gives both prefixes. Earlier selected equal keys are the
    // equal prefix clipped to the tie quota, independently of greater keys.
    int2 before;
    Scan(*reinterpret_cast<Scan::TempStorage*>(temporary)).ExclusiveScan(
        make_int2(greater,equal),before,make_int2(0,0),QsaThresholdCountAdd{});
    __syncthreads();
    const int quota=max(0,count-above);
    int remaining=min(equal,max(0,quota-before.y));
    int selected_before=before.x+min(before.y,quota);
    auto* published=emit_tokens?reinterpret_cast<int64_t*>(temporary):out_ids+out;
    #pragma unroll
    for(int i=0;i<Items;++i)if(i<per && first+i<available) {
        bool take=keys[i]>threshold;
        if(keys[i]==threshold && remaining>0){take=true;--remaining;}
        if(take) {
            const int column=begin+first+i;const auto at=int64_t(row)*columns+column;
            if(!emit_tokens)output[out+selected_before]=source[at];
            published[selected_before]=ids?ids[at]:column;++selected_before;
        }
    }
}
