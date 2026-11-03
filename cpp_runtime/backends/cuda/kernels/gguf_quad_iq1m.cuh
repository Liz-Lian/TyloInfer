#pragma once
// The native IQ1_M scales contain 48 local-scale bits and 16 global-scale
// bits per row. Group four rows' six-bit local scales into 24-bit fields,
// followed by four FP16 bit patterns. The metadata remains exactly 32 bytes.
__device__ __forceinline__ unsigned quad_iq1m_native_word(const void* pointer,unsigned index){
    const auto* p=static_cast<const unsigned short*>(pointer)+2*index;
    return unsigned(p[0])|(unsigned(p[1])<<16);
}
__device__ __forceinline__ void pack_quad_iq1m_metadata(const unsigned char* source,
        std::uint64_t stride,unsigned char* output){
    unsigned groups[8]{};unsigned short scales[4];
    #pragma unroll
    for(unsigned row=0;row<4;++row){
        const auto* b=reinterpret_cast<const block_iq1_m*>(source+row*stride);
        const unsigned a=quad_iq1m_native_word(b->scales,0),c=quad_iq1m_native_word(b->scales,1);
        scales[row]=((a>>12)&15)|((a>>24)&240)|((c>>4)&3840)|((c>>16)&61440);
        #pragma unroll
        for(unsigned part=0;part<8;++part){
            const unsigned value=((part&4)?c:a)>>((part&2)*8+6*(part&1));
            groups[part]|=(value&63)<<(6*row);
        }
    }
    #pragma unroll
    for(unsigned part=0;part<8;++part){
        const auto value=groups[part];output[3*part]=value;output[3*part+1]=value>>8;output[3*part+2]=value>>16;
    }
    #pragma unroll
    for(unsigned row=0;row<4;++row)reinterpret_cast<unsigned short*>(output+24)[row]=scales[row];
}
__device__ __forceinline__ void restore_quad_iq1m_metadata(const unsigned char* source,
        unsigned row,block_iq1_m* output){
    unsigned scales[4]{};
    #pragma unroll
    for(unsigned part=0;part<8;++part){
        const unsigned value=unsigned(source[3*part])|(unsigned(source[3*part+1])<<8)|(unsigned(source[3*part+2])<<16);
        scales[part/2]|=((value>>(6*row))&63)<<(6*(part%2));
    }
    const unsigned global=reinterpret_cast<const unsigned short*>(source+24)[row];
    #pragma unroll
    for(unsigned part=0;part<4;++part)
        reinterpret_cast<unsigned short*>(output->scales)[part]=scales[part]|(((global>>(4*part))&15)<<12);
}
