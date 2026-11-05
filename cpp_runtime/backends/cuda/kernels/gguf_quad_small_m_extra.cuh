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

// Sequential nibbles avoid permuting every token's IQ1_S activation.
// The 2^11-entry codebook occupies 8 KiB of CTA scratch; model weights
// and the independently rounded Q8 scale/sum remain unchanged.
constexpr unsigned Iq1SSequentialGridCount=2048;
__device__ __forceinline__ void stage_iq1s_sequential_grid(unsigned tid) {
    extern __shared__ __align__(16) unsigned iq1s_sequential_grid[];
    for(unsigned i=tid*4;i<Iq1SSequentialGridCount;i+=128*4) {
        // Source entries only promise scalar alignment. Widen the shared
        // stores without imposing extra alignment on the bundled codebook.
        const uint4 words=make_uint4(__ldg(iq1s_grid_gpu+i),__ldg(iq1s_grid_gpu+i+1),
            __ldg(iq1s_grid_gpu+i+2),__ldg(iq1s_grid_gpu+i+3));
        reinterpret_cast<uint4*>(iq1s_sequential_grid)[i/4]=words;
    }
}
__device__ __forceinline__ int2 decode_iq1s_sequential_grid(unsigned index) {
    extern __shared__ __align__(16) unsigned iq1s_sequential_grid[];
    const unsigned value=iq1s_sequential_grid[index];
    return make_int2(value&0x0f0f0f0fu,(value>>4)&0x0f0f0f0fu);
}

template<bool Medium,int Tokens,class Index,bool Split=false>
__device__ __forceinline__ void quad_small_m_iq1_body(
        QuadProjection p,int columns,const void* input,bool half_output,unsigned tile) {
    constexpr int Format=Medium?GGML_TYPE_IQ1_M:GGML_TYPE_IQ1_S;
    constexpr int Rows=4;
    using Decoder=QuadIq1Dot<Medium,0>;
    const unsigned lane=threadIdx.x,warp=threadIdx.y,row=(tile*4+warp)*Rows;
    if constexpr(!Medium)stage_iq1s_sequential_grid(warp*32+lane);
    else stage_global_grid<Format,0>(warp*32+lane);
    __syncthreads();
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
        if constexpr(Split) {
            static_assert(Tokens>=5 && Tokens<=8);
        int2 decoded[Rows][4];float row_scale[Rows],row_delta[Rows];
        #pragma unroll
        for(int r=0;r<Rows;++r) {
            unsigned code;
            if(r==0)code=codes.x;else if(r==1)code=codes.y;else if(r==2)code=codes.z;else code=codes.w;
            const unsigned hi=(r<2?high.x:high.y)>>(16*(r%2));
            row_scale[r]=__half2float(__ushort_as_half((r<2?scales.x:scales.y)>>(16*(r%2))));
            if constexpr(!Medium)row_scale[r]*=((hi>>11)&14)+1;
            row_delta[r]=-1.f+IQ1S_DELTA-(hi&0x8000)*(2.f*IQ1S_DELTA/0x8000);
            #pragma unroll
            for(int j=0;j<4;++j) {
                const unsigned selector=hi>>((Medium?4:3)*j);
                const unsigned index=((code>>(8*j))&255)|((selector&7)<<8);
                if constexpr(Medium)decoded[r][j]=Decoder::decode_grid(index,(selector&8)!=0);
                else decoded[r][j]=decode_iq1s_sequential_grid(index);
            }
        }
        #pragma unroll
        for(int token=0;token<Tokens;++token) {
            const auto codes=load_row_vector<Tokens,Index>(input,groups,token,i);
            const auto input_scale=__half22float2(row_packet_scales<Tokens,Index>(input,groups,token,i));
            int activation[8];
            #pragma unroll
            for(int j=0;j<4;++j) {
                const int first=codes.value[2*j],second=codes.value[2*j+1];
                if constexpr(Medium){activation[2*j]=first;activation[2*j+1]=second;}
                else {activation[2*j]=first;activation[2*j+1]=second;}
            }
            #pragma unroll
            for(int r=0;r<Rows;++r) {
                int low_sum=ggml_cuda_dp4a(decoded[r][0].y,activation[1],ggml_cuda_dp4a(decoded[r][0].x,activation[0],0));
                low_sum=ggml_cuda_dp4a(decoded[r][1].y,activation[3],ggml_cuda_dp4a(decoded[r][1].x,activation[2],low_sum));
                if constexpr(Medium) {
                    int high_sum=ggml_cuda_dp4a(decoded[r][2].y,activation[5],ggml_cuda_dp4a(decoded[r][2].x,activation[4],0));
                    high_sum=ggml_cuda_dp4a(decoded[r][3].y,activation[7],ggml_cuda_dp4a(decoded[r][3].x,activation[6],high_sum));
                    const unsigned sc=(local>>(6*r))&63;
                    const int sum=low_sum*(2*(sc&7)+1)+high_sum*(2*((sc>>3)&7)+1);
                    sums[token][r]+=(row_scale[r]*input_scale.x)*(float(sum)*.125f);
                }else {
                    low_sum=ggml_cuda_dp4a(decoded[r][2].y,activation[5],ggml_cuda_dp4a(decoded[r][2].x,activation[4],low_sum));
                    low_sum=ggml_cuda_dp4a(decoded[r][3].y,activation[7],ggml_cuda_dp4a(decoded[r][3].x,activation[6],low_sum));
                    sums[token][r]+=row_scale[r]*(input_scale.x*low_sum+input_scale.y*row_delta[r]);
                }
            }
            asm volatile("" ::: "memory");
        }        } else {
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
                    else return decode_iq1s_sequential_grid(index);
                }();
                #pragma unroll
                for(int token=0;token<Tokens;++token) {
                    const auto* x=activations+Index(token)*groups+i;
                    int first=get_int_b4(x->qs,2*J),second=get_int_b4(x->qs,2*J+1);
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
        if constexpr(Rows==4){apply.template operator()<2>();apply.template operator()<3>();}        }

    }
    finish_small_m<Tokens,Index>(sums,p,row,lane,half_output);
}

// The split Q8 packet already stores exact signed INT16 sums of both
// 16-coordinate halves. Its positive Q2 minima fit in two INT8 bytes.
// DP2A consumes both pairs directly without sign extraction or repacking.
__device__ __forceinline__ int packed_q2_affine_sum(int sums,unsigned minima) {
#if __CUDA_ARCH__>=610
    // DP2A was introduced in compute capability 6.1.
    return __dp2a_lo(sums,int(minima),0);
#else
    return int(int16_t(sums))*int(minima&255)+
        int(int16_t(unsigned(sums)>>16))*int((minima>>8)&255);
#endif
}

template<bool Medium,int Tokens,class Index,bool Split=false>
__global__ void __launch_bounds__(128) quad_small_m_iq1_kernel(
        QuadProjection p,int columns,const void* input,bool half_output) {
    quad_small_m_iq1_body<Medium,Tokens,Index,Split>(p,columns,input,half_output,blockIdx.x);
}

template<int Tokens,class Index,bool Split=false>
__device__ __forceinline__ void quad_small_m_q2_body(
        QuadProjection p,int columns,const void* input,bool half_output,unsigned tile) {
    constexpr int Rows=4;
    const unsigned lane=threadIdx.x,warp=threadIdx.y,row=(tile*4+warp)*Rows;
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
        // Wide token tiles amortize the local Q2 scales
        // while unpacking weights: q is 0..3 and its scale is 0..15,
        // so four byte products fit without carries (maximum 45).
        // The integer dot is unchanged, including the separate affine
        // minimum. Share one token activation across all output rows.
        if constexpr(Split || Tokens>=6) {
        int decoded[Rows][8];unsigned row_scales[Rows];float2 factors[Rows];
        #pragma unroll
        for(int r=0;r<Rows;++r) {
            const auto codes=r<2?first:second;
            const unsigned low=r%2?codes.z:codes.x,high=r%2?codes.w:codes.y;
            row_scales[r]=((r<2?scales.x:scales.y)>>(16*(r%2)))&65535;
            unsigned bits;
            if(r==0)bits=dm.x;else if(r==1)bits=dm.y;else if(r==2)bits=dm.z;else bits=dm.w;
            factors[r]=make_float2(__half2float(__ushort_as_half(bits)),__half2float(__ushort_as_half(bits>>16)));
            #pragma unroll
            for(int j=0;j<4;++j) {
                decoded[r][j]=int(((low>>(2*j))&0x03030303)*(row_scales[r]&15));
                decoded[r][j+4]=int(((high>>(2*j))&0x03030303)*((row_scales[r]>>8)&15));
            }
        }
        #pragma unroll
        for(int token=0;token<Tokens;++token) {
            float input_scale;int activation[8],input_low=0,input_high=0;int input_halfsums=0;
            if constexpr(Split) {
                const auto packet=Index(token)*groups+i;
                const auto meta=__ldg(reinterpret_cast<const uint2*>(static_cast<const uint8_t*>(input)+uint64_t(groups)*Tokens*36)+packet);
                input_scale=__half2float(__ushort_as_half(meta.x&65535u));
                input_halfsums=int(meta.y);
                const auto codes=load_row_vector<Tokens,Index>(input,groups,token,i);
                #pragma unroll
                for(int j=0;j<8;++j)activation[j]=codes.value[j];
            } else {
            const auto* x=activations+Index(token)*groups+i;
            input_scale=__low2float(x->ds);

            #pragma unroll
            for(int j=0;j<4;++j) {
                activation[j]=get_int_b4(x->qs,j);activation[j+4]=get_int_b4(x->qs,j+4);
                input_low=ggml_cuda_dp4a(0x01010101,activation[j],input_low);
                input_high=ggml_cuda_dp4a(0x01010101,activation[j+4],input_high);
            }
            }
            #pragma unroll
            for(int r=0;r<Rows;++r) {
                int dot=0;
                #pragma unroll
                for(int j=0;j<4;++j) {
                    dot=ggml_cuda_dp4a(decoded[r][j],activation[j],dot);
                    dot=ggml_cuda_dp4a(decoded[r][j+4],activation[j+4],dot);
                }
                const unsigned sc=row_scales[r];
                const int minimum=[&] {
                    if constexpr(Split)return packed_q2_affine_sum(input_halfsums,(sc>>4)&0x0f0fu);
                    else return input_low*int((sc>>4)&15)+input_high*int(sc>>12);
                }();
                sums[token][r]+=factors[r].x*(input_scale*dot)-factors[r].y*(input_scale*minimum);
            }
            asm volatile("" ::: "memory");
        }
        } else {
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
    }
    finish_small_m<Tokens,Index>(sums,p,row,lane,half_output);
}

template<int Tokens,class Index>
__global__ void __launch_bounds__(128) quad_small_m_q2_kernel(
        QuadProjection p,int columns,const void* input,bool half_output) {
    quad_small_m_q2_body<Tokens,Index>(p,columns,input,half_output,blockIdx.x);
}

// The four-row accumulator tile grows with the verification window. Bound
// its register allocation together with the preweighted-code schedule;
// shorter windows retain their original instruction and launch contract.
template<int Tokens,class Index>
__global__ void __launch_bounds__(128,4) quad_wide_q2_kernel(
        QuadProjection p,int columns,const void* input,bool half_output) {
    static_assert(Tokens>=6 && Tokens<=8);
    quad_small_m_q2_body<Tokens,Index>(p,columns,input,half_output,blockIdx.x);
}

template<int Tokens,class Index>
__global__ void __launch_bounds__(128,4) quad_split_q2_kernel(QuadProjection p,int columns,const void* input,bool half_output){
    static_assert(Tokens>=5 && Tokens<=8);quad_small_m_q2_body<Tokens,Index,true>(p,columns,input,half_output,blockIdx.x);
}
