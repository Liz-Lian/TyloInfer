#pragma once
// All dot products and affine corrections retain the native Q2_K integer
// arithmetic. Only the payload traversal changes; no decoded weights persist.
template<int Rows,int Warps,bool Aligned,bool Half,class Index,class Store>
__device__ __forceinline__ void quad_q2_m1_body(QuadProjection p,int columns,
        const void* input,unsigned tile,Store store){
    static_assert(Rows==2 || Rows==4);
    using F=M1PreparedDot<GGML_TYPE_Q2_K>;
    const auto* packed=static_cast<const uint8_t*>(p.projection.packed);
    const unsigned lane=threadIdx.x,warp=threadIdx.y;
    const bool active=warp<(4/Warps)*Warps;
    const unsigned row=(tile*(4/Warps)+warp/Warps)*Rows;
    const Index source_row=Index(row)+Index(p.row_begin),quad=source_row/4,pair=(source_row%4)/2;
    const Index blocks=Index(columns/256),groups=blocks*8,total=Index(p.storage_rows/4)*blocks;
    float sums[Rows]{};
    for(unsigned i=(warp%Warps)*32+lane;active && row<unsigned(p.projection.rows) && i<groups;i+=32*Warps){
        const Index entry=quad*groups+i,qb=quad*blocks+i/8;
        uint4 c0{},c1{},dm{};uint2 scales{};
        if constexpr(Rows==2){
            c0=load_four(packed,entry*32+pair*16);
            scales.x=load_one(packed,total*256+entry*8+pair*4);
            const auto bits=load_two(packed,total*320+qb*16+pair*8);dm=make_uint4(bits.x,bits.y,0,0);
        }else{
            c0=load_four(packed,entry*32);c1=load_four(packed,entry*32+16);
            scales=load_two(packed,total*256+entry*8);dm=load_four(packed,total*320+qb*16);
        }
        typename F::Activation a;
        if constexpr(Aligned){
            const auto* codes=static_cast<const uint4*>(input);const auto low=__ldg(codes+2*i),high=__ldg(codes+2*i+1);
            a.codes[0]=low.x;a.codes[1]=low.y;a.codes[2]=low.z;a.codes[3]=low.w;
            a.codes[4]=high.x;a.codes[5]=high.y;a.codes[6]=high.z;a.codes[7]=high.w;
            const auto bits=__ldg(reinterpret_cast<const unsigned*>(static_cast<const uint8_t*>(input)+columns)+i);
            a.scale=__half2float(__ushort_as_half(bits));a.lo_sum=0;a.hi_sum=0;
            #pragma unroll
            for(int j=0;j<4;++j){a.lo_sum=ggml_cuda_dp4a(0x01010101,a.codes[j],a.lo_sum);a.hi_sum=ggml_cuda_dp4a(0x01010101,a.codes[j+4],a.hi_sum);}
        }else a=F::load_activation(static_cast<const block_q8_1*>(input)+i,0);
        const auto apply=[&]<int R>(){
            const auto codes=R<2?c0:c1;
            const unsigned low=R%2?codes.z:codes.x,high=R%2?codes.w:codes.y;
            const unsigned sc=((R<2?scales.x:scales.y)>>(16*(R%2)))&65535,bits=field<R>(dm);
            int lo=0,hi=0;
            #pragma unroll
            for(int j=0;j<4;++j){lo=ggml_cuda_dp4a((low>>(2*j))&0x03030303,a.codes[j],lo);hi=ggml_cuda_dp4a((high>>(2*j))&0x03030303,a.codes[j+4],hi);}
            const int dot=lo*int(sc&15)+hi*int((sc>>8)&15);
            const int minimum=a.lo_sum*int((sc>>4)&15)+a.hi_sum*int(sc>>12);
            const float d=__half2float(__ushort_as_half(bits)),m=__half2float(__ushort_as_half(bits>>16));
            sums[R]+=d*(a.scale*dot)-m*(a.scale*minimum);
        };
        apply.template operator()<0>();apply.template operator()<1>();
        if constexpr(Rows==4){apply.template operator()<2>();apply.template operator()<3>();}
    }
    const float value=warp_sums::reduce<Rows>(sums,lane);const int r=warp_sums::index<Rows>(lane);
    const bool owner=lane==warp_sums::owner<Rows>(r);__shared__ float scratch[16];
    if constexpr(Warps==1){if(owner && row+r<unsigned(p.projection.rows))store(p.projection.output,row+r,value,Half);}
    else{
        if(owner && active)scratch[warp*Rows+r]=value;__syncthreads();
        if(owner && active && warp%Warps==0 && row+r<unsigned(p.projection.rows)){
            float sum=scratch[warp*Rows+r];
            #pragma unroll
            for(int k=1;k<Warps;++k)sum+=scratch[(warp+k)*Rows+r];
            store(p.projection.output,row+r,sum,Half);
        }
    }
}
