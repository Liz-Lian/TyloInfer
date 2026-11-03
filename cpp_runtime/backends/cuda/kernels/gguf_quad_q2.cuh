#pragma once
// Q2_K keeps its two-bit payload and affine metadata exactly. Four adjacent
// rows share a storage plane; each 32-coordinate fragment needs two words.
// Native rows may have only two-byte alignment, including padded row views.
__device__ __forceinline__ unsigned quad_q2_native_word(const void* pointer,unsigned index){
    const auto* p=static_cast<const unsigned short*>(pointer)+2*index;
    return unsigned(p[0])|(unsigned(p[1])<<16);
}
__device__ __forceinline__ void quad_q2_native_store(void* pointer,unsigned index,unsigned value){
    auto* p=static_cast<unsigned short*>(pointer)+2*index;p[0]=value;p[1]=value>>16;
}
__device__ __forceinline__ void pack_quad_q2_fragment(const block_q2_K* source,
        unsigned char* packed,std::uint64_t total,std::uint64_t qb,unsigned row,unsigned part){
    unsigned low=0,high=0;const unsigned shift=2*(part%4),begin=8*(part/4);
    #pragma unroll
    for(unsigned j=0;j<4;++j){
        low|=((quad_q2_native_word(source->qs,begin+j)>>shift)&0x03030303u)<<(2*j);
        high|=((quad_q2_native_word(source->qs,begin+4+j)>>shift)&0x03030303u)<<(2*j);
    }
    const auto entry=qb*8+part;
    reinterpret_cast<uint2*>(packed+entry*32)[row]=make_uint2(low,high);
    reinterpret_cast<unsigned short*>(packed+total*256)[entry*4+row]=
        reinterpret_cast<const unsigned short*>(source->scales)[part];
    if(!part)reinterpret_cast<unsigned*>(packed+total*320)[qb*4+row]=quad_q2_native_word(&source->dm,0);
}
__device__ __forceinline__ void restore_quad_q2_fragment(const unsigned char* packed,
        std::uint64_t total,std::uint64_t qb,unsigned row,unsigned part,block_q2_K* output){
    // A fragment restores one native word in each 128-coordinate half.
    // All eight fragment owners therefore write disjoint words.
    #pragma unroll
    for(unsigned group=0;group<2;++group){
        unsigned value=0;
        #pragma unroll
        for(unsigned p=0;p<4;++p){
            const auto entry=qb*8+group*4+p;
            const unsigned code=*reinterpret_cast<const unsigned*>(packed+entry*32+row*8+(part/4)*4);
            value|=((code>>(2*(part%4)))&0x03030303u)<<(2*p);
        }
        quad_q2_native_store(output->qs,group*8+part,value);
    }
    reinterpret_cast<unsigned short*>(output->scales)[part]=
        reinterpret_cast<const unsigned short*>(packed+total*256)[(qb*8+part)*4+row];
    if(!part)quad_q2_native_store(&output->dm,0,reinterpret_cast<const unsigned*>(packed+total*320)[qb*4+row]);
}
