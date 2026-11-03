#pragma once

template<int Tokens,class Index,int Rows>
__device__ __forceinline__ void finish_small_m(float (&sums)[Tokens][Rows],
        const QuadProjection& p,unsigned row,unsigned lane,bool half_output) {
    #pragma unroll
    for(int token=0;token<Tokens;++token) {
        const float value=warp_sums::reduce<Rows>(sums[token],lane);
        const int offset=warp_sums::index<Rows>(lane);
        if(lane==warp_sums::owner<Rows>(offset) && row+offset<unsigned(p.projection.rows))
            store_projection(p.projection.output,Index(token)*p.projection.rows+row+offset,value,half_output);
    }
}

template<bool Medium,int Tokens,class Index>
__global__ void __launch_bounds__(128) quad_small_m_iq1_kernel(
        QuadProjection p,int columns,const void* input,bool half_output) {
    constexpr int Format=Medium?GGML_TYPE_IQ1_M:GGML_TYPE_IQ1_S;
    constexpr int Rows=Tokens<=4?4:2;
    using Decoder=QuadIq1Dot<Medium,0>;
    const unsigned lane=threadIdx.x,warp=threadIdx.y,row=(blockIdx.x*4+warp)*Rows;
    stage_global_grid<Format,0>(warp*32+lane);__syncthreads();
    const auto* packed=static_cast<const uint8_t*>(p.projection.packed);
    const auto* activations=static_cast<const block_q8_1*>(input);
    const Index blocks=Index(columns/256),groups=blocks*8;
    const Index source_row=Index(row)+p.row_begin,quad=source_row/4,pair=(source_row%4)/2,total=Index(p.storage_rows/4)*blocks;
    float sums[Tokens][Rows]{};
    for(unsigned i=lane;row<unsigned(p.projection.rows) && i<groups;i+=32) {
        const Index entry=quad*groups+i,qb=quad*blocks+i/8,part=i%8;
        uint4 codes;uint2 high;
        if constexpr(Rows==4){codes=load_four(packed,entry*16);high=load_two(packed,total*128+entry*8);}
        else {const auto value=load_two(packed,entry*16+pair*8);codes=make_uint4(value.x,value.y,0,0);high=make_uint2(load_one(packed,total*128+entry*8+pair*4),0);}
        uint2 scales;unsigned local=0;
        if constexpr(Medium) {
            const Index base=total*192+qb*32,byte=3*part,aligned=byte&~Index(3);
            local=__funnelshift_r(load_one(packed,base+aligned),load_one(packed,base+aligned+4),unsigned(8*(byte&3)))&0xffffffu;
            scales=load_two(packed,base+24);
            if constexpr(Rows==2){local=(local>>(12*pair))&4095u;scales=make_uint2(pair?scales.y:scales.x,0);}
        }else {
            if constexpr(Rows==4)scales=load_two(packed,total*192+qb*8);
            else scales=make_uint2(load_one(packed,total*192+qb*8+pair*4),0);
        }
        float2 input_scale[Tokens];
        #pragma unroll
        for(int token=0;token<Tokens;++token)
            input_scale[token]=__half22float2(activations[Index(token)*groups+i].ds);
        const auto apply=[&]<int R>() {
            const unsigned code=field<R>(codes),hi=(R<2?high.x:high.y)>>(16*(R%2));
            float scale=__half2float(__ushort_as_half((R<2?scales.x:scales.y)>>(16*(R%2))));
            if constexpr(!Medium)scale*=((hi>>11)&14)+1;
            int low_sum[Tokens]{},high_sum[Tokens]{};
            const auto dot=[&]<int J>() {
                const unsigned selector=hi>>((Medium?4:3)*J);
                const unsigned index=((code>>(8*J))&255)|((selector&7)<<8);
                const int2 weight=[&] {
                    if constexpr(Medium)return Decoder::decode_grid(index,(selector&8)!=0);
                    else return Decoder::decode_rank_planes(index);
                }();
                #pragma unroll
                for(int token=0;token<Tokens;++token) {
                    const auto* x=activations+Index(token)*groups+i;
                    int first=get_int_b4(x->qs,2*J),second=get_int_b4(x->qs,2*J+1);
                    if constexpr(!Medium) {
                        const int a=first,b=second;
                        first=__byte_perm(a,b,0x6420);second=__byte_perm(a,b,0x7531);
                    }
                    if constexpr(Medium && J>=2)
                        high_sum[token]=ggml_cuda_dp4a(weight.y,second,ggml_cuda_dp4a(weight.x,first,high_sum[token]));
                    else low_sum[token]=ggml_cuda_dp4a(weight.y,second,ggml_cuda_dp4a(weight.x,first,low_sum[token]));
                }
            };
            dot.template operator()<0>();dot.template operator()<1>();dot.template operator()<2>();dot.template operator()<3>();
            #pragma unroll
            for(int token=0;token<Tokens;++token) {
                if constexpr(Medium) {
                    const unsigned sc=(local>>(6*R))&63;
                    const int sum=low_sum[token]*(2*(sc&7)+1)+high_sum[token]*(2*((sc>>3)&7)+1);
                    sums[token][R]+=(scale*input_scale[token].x)*(float(sum)*.125f);
                }else {
                    // The Q8_1 sum is rounded separately from its scale;
                    // reconstructing it from quantized coordinates changes IQ1_S.
                    const float delta=-1.f+IQ1S_DELTA-(hi&0x8000)*(2.f*IQ1S_DELTA/0x8000);
                    sums[token][R]+=scale*(input_scale[token].x*low_sum[token]+input_scale[token].y*delta);
                }
            }
        };
        apply.template operator()<0>();apply.template operator()<1>();
        if constexpr(Rows==4){apply.template operator()<2>();apply.template operator()<3>();}
    }
    finish_small_m<Tokens,Index>(sums,p,row,lane,half_output);
}

template<int Tokens,class Index>
__global__ void __launch_bounds__(128) quad_small_m_q2_kernel(
        QuadProjection p,int columns,const void* input,bool half_output) {
    constexpr int Rows=Tokens<=4?4:2;
    const unsigned lane=threadIdx.x,warp=threadIdx.y,row=(blockIdx.x*4+warp)*Rows;
    const auto* packed=static_cast<const uint8_t*>(p.projection.packed);
    const auto* activations=static_cast<const block_q8_1*>(input);
    const Index blocks=Index(columns/256),groups=blocks*8;
    const Index source_row=Index(row)+p.row_begin,quad=source_row/4,pair=(source_row%4)/2,total=Index(p.storage_rows/4)*blocks;
    float sums[Tokens][Rows]{};
    for(unsigned i=lane;row<unsigned(p.projection.rows) && i<groups;i+=32) {
        const Index entry=quad*groups+i,qb=quad*blocks+i/8;
        uint4 first,second{},dm;uint2 scales;
        if constexpr(Rows==4) {
            first=load_four(packed,entry*32);second=load_four(packed,entry*32+16);
            scales=load_two(packed,total*256+entry*8);dm=load_four(packed,total*320+qb*16);
        }else {
            first=load_four(packed,entry*32+pair*16);scales=make_uint2(load_one(packed,total*256+entry*8+pair*4),0);
            const auto bits=load_two(packed,total*320+qb*16+pair*8);dm=make_uint4(bits.x,bits.y,0,0);
        }
        float input_scale[Tokens];int input_low[Tokens]{},input_high[Tokens]{};
        #pragma unroll
        for(int token=0;token<Tokens;++token) {
            const auto* x=activations+Index(token)*groups+i;
            input_scale[token]=__low2float(x->ds);
            #pragma unroll
            for(int j=0;j<4;++j) {
                input_low[token]=ggml_cuda_dp4a(0x01010101,get_int_b4(x->qs,j),input_low[token]);
                input_high[token]=ggml_cuda_dp4a(0x01010101,get_int_b4(x->qs,j+4),input_high[token]);
            }
        }
        const auto apply=[&]<int R>() {
            const auto codes=R<2?first:second;
            const unsigned low=R%2?codes.z:codes.x,high=R%2?codes.w:codes.y;
            const unsigned sc=((R<2?scales.x:scales.y)>>(16*(R%2)))&65535,bits=field<R>(dm);
            int low_sum[Tokens]{},high_sum[Tokens]{};
            #pragma unroll
            for(int j=0;j<4;++j) {
                const int lo=(low>>(2*j))&0x03030303,hi=(high>>(2*j))&0x03030303;
                #pragma unroll
                for(int token=0;token<Tokens;++token) {
                    const auto* x=activations+Index(token)*groups+i;
                    low_sum[token]=ggml_cuda_dp4a(lo,get_int_b4(x->qs,j),low_sum[token]);
                    high_sum[token]=ggml_cuda_dp4a(hi,get_int_b4(x->qs,j+4),high_sum[token]);
                }
            }
            const float d=__half2float(__ushort_as_half(bits)),minimum_scale=__half2float(__ushort_as_half(bits>>16));
            #pragma unroll
            for(int token=0;token<Tokens;++token) {
                const int dot=low_sum[token]*int(sc&15)+high_sum[token]*int((sc>>8)&15);
                const int minimum=input_low[token]*int((sc>>4)&15)+input_high[token]*int(sc>>12);
                sums[token][R]+=d*(input_scale[token]*dot)-minimum_scale*(input_scale[token]*minimum);
            }
        };
        apply.template operator()<0>();apply.template operator()<1>();
        if constexpr(Rows==4){apply.template operator()<2>();apply.template operator()<3>();}
    }
    finish_small_m<Tokens,Index>(sums,p,row,lane,half_output);
}
