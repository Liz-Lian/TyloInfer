#pragma once
// Eight-query-head Tensor Core tile with joint four-token KV decoding.
// m16n8k16 FP16 operands require SM80; FP32 accumulation is retained.
__device__ __forceinline__ void qsa_mma16x8(float* d,const unsigned* a,const unsigned* b) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
#endif
}
__device__ __forceinline__ unsigned qsa_pack_half(half a,half b) {
    return unsigned(__half_as_ushort(a))|(unsigned(__half_as_ushort(b))<<16);
}
template<int W,bool Snapshots,bool Staged=false> __global__ void __launch_bounds__(256,(W==256?4:2)) sparse_attention_mma8(const float* query,const half* relative,
    const TqBlockView* blocks,const SelectedBlock* selected,const int64_t* positions,
    float* partial,float* stats,float* block_log_mass,int heads,int kv_heads,int width,int count,int parts,const half* staged_keys=nullptr,const half* staged_values=nullptr) {
#if __CUDA_ARCH__ >= 800
    constexpr int P=W+8,Rings=W/256;
    const int tid=threadIdx.x,lane=tid&31,warp=tid/32,part=blockIdx.x,token=blockIdx.z;
    const int gqa=heads/kv_heads,groups=(gqa+7)/8,kh=blockIdx.y/groups,base=(blockIdx.y%groups)*8;
    extern __shared__ __align__(32) unsigned char memory[];
    auto* key=reinterpret_cast<half*>(memory);auto* value=key;
    auto* probability=value+16*P;
    auto* scores=reinterpret_cast<float*>(probability+8*16);
    __shared__ TqRowView kr[16*Rings],vr[16*Rings];
    __shared__ SelectedBlock chosen[4];
    __shared__ float factors[8];
    // Keep one K eighth of the eight query heads per warp. Query
    // operands and output fragments use half as many registers per thread.
    unsigned query_words[W/64];
    const int qhead=kh*gqa+base+lane/4,qk=(lane%4)*2;
    #pragma unroll
    for(int i=0;i<W/128;++i) {
        const int d=warp*16+i*128+qk;
        const auto* q=base+lane/4<gqa?query+(int64_t(token)*heads+qhead)*W+d:query;
        query_words[2*i]=base+lane/4<gqa?qsa_pack_half(__float2half_rn(q[0]),__float2half_rn(q[1])):0;
        query_words[2*i+1]=base+lane/4<gqa?qsa_pack_half(__float2half_rn(q[8]),__float2half_rn(q[9])):0;
    }
    float accum[W/128][4]{};
    float maximum=-CUDART_INF_F,mass=0;
    const int tiles=(count+3)/4,begin=4*(part*tiles/parts),end=min(count,4*((part+1)*tiles/parts));
    const auto boundary=positions[token];
    for(int first=begin;first<end;first+=4) {
        if(tid<4)chosen[tid]=first+tid<end?selected[int64_t(token)*count+first+tid]:SelectedBlock{};
        __syncthreads();
        if constexpr(Staged) {
            for(int at=4*tid;at<16*W;at+=256*4) {
                const int token_row=at/W,b=token_row/4,d=at%W;
                const auto item=chosen[b];
                half2 a=__float2half2_rn(0.f),c=a;
                if(item.id>=0 && item.token_mask) {
                    const auto* source=staged_keys+((int64_t(item.id)*kv_heads+kh)*4+token_row%4)*W+d;
                    a=*reinterpret_cast<const half2*>(source);c=*reinterpret_cast<const half2*>(source+2);
                }
                auto* destination=key+token_row*P+d;
                *reinterpret_cast<half2*>(destination)=a;*reinterpret_cast<half2*>(destination+2)=c;
            }
        } else {
        if(tid<16*Rings) {
            const int b=tid/(4*Rings),r=tid%(4*Rings);
            const auto item=chosen[b];const bool valid=item.id>=0 && item.token_mask;
            const auto view=valid?blocks[Snapshots?int64_t(token)*count+first+b:item.id]:TqBlockView{};
            kr[tid]=valid?tq_row(view.key,view.key_bits,kh*4*Rings+r,kv_heads*4*Rings,relative):TqRowView{};
            vr[tid]=valid?tq_row(view.value,view.value_bits,kh*4*Rings+r,kv_heads*4*Rings,relative):TqRowView{};
        }
        __syncthreads();
        for(int at=4*tid;at<4*W;at+=256*4) {
            const int b=at/W,d=at%W;const bool valid=chosen[b].id>=0 && chosen[b].token_mask;
            float4 k[4];
            #pragma unroll
            for(int m=0;m<4;++m) {
                const int r=(b*4+m)*Rings+d/256;
                k[m]=valid?tq_values4(kr[r],d&255):make_float4(0,0,0,0);
            }
            #pragma unroll
            for(int t=0;t<4;++t) {
                float4 a=make_float4(0,0,0,0);
                #pragma unroll
                for(int m=0;m<4;++m) {
                    const float sign=(__popc(unsigned(t&m))&1)?-.5f:.5f;
                    a.x+=sign*k[m].x;a.y+=sign*k[m].y;a.z+=sign*k[m].z;a.w+=sign*k[m].w;
                }
                auto* kp=key+(b*4+t)*P+d;
                reinterpret_cast<half2*>(kp)[0]=__floats2half2_rn(a.x,a.y);reinterpret_cast<half2*>(kp)[1]=__floats2half2_rn(a.z,a.w);
            }
        }
        }
        __syncthreads();
        {
            float c[4]{};const int r=lane/4,k=(lane%4)*2;
            #pragma unroll
            for(int i=0;i<W/128;++i) {
                const int d=warp*16+i*128;
                const unsigned a[4]={*reinterpret_cast<unsigned*>(key+r*P+d+k),*reinterpret_cast<unsigned*>(key+(r+8)*P+d+k),
                    *reinterpret_cast<unsigned*>(key+r*P+d+k+8),*reinterpret_cast<unsigned*>(key+(r+8)*P+d+k+8)};
                const unsigned b[2]={query_words[2*i],query_words[2*i+1]};
                qsa_mma16x8(c,a,b);
            }
            const int h=(lane%4)*2;
            scores[warp*128+h*16+r]=c[0];scores[warp*128+(h+1)*16+r]=c[1];scores[warp*128+h*16+r+8]=c[2];scores[warp*128+(h+1)*16+r+8]=c[3];
        }
        __syncthreads();
        // A half warp owns one complete16-token softmax row. Both halves
        // are live, so the mask remains uniform even for padded query heads.
        if(warp<4) {
            const int local=2*warp+(lane>>4),column=lane&15;
            const int head=kh*gqa+base+local;
            const bool active=base+local<gqa;
            float score=-CUDART_INF_F;
            const auto item=chosen[column/4];
            if(active && item.id>=0 && (item.token_mask&(1u<<(column&3))) && int64_t(item.id)*4+(column&3)<=boundary)
                score=(((scores[local*16+column]+scores[128+local*16+column])+
                        (scores[256+local*16+column]+scores[384+local*16+column]))+
                       ((scores[512+local*16+column]+scores[640+local*16+column])+
                        (scores[768+local*16+column]+scores[896+local*16+column])))*rsqrtf(float(width));
            float top=score;
            #pragma unroll
            for(int delta=8;delta;delta/=2)top=fmaxf(top,__shfl_xor_sync(0xffffffffu,top,delta,16));
            top=fmaxf(maximum,top);
            const float factor=isfinite(maximum)?expf(maximum-top):0;
            const float weight=isfinite(score)?expf(score-top):0;
            if(!column)factors[local]=factor;
            float total=weight;
            #pragma unroll
            for(int delta=8;delta;delta/=2)total+=__shfl_xor_sync(0xffffffffu,total,delta,16);
            mass=mass*factor+total;maximum=top;
            probability[local*16+column]=__float2half_rn(weight);
            const float sum2=weight+__shfl_xor_sync(0xffffffffu,weight,1,16);
            const float sum4=sum2+__shfl_xor_sync(0xffffffffu,sum2,2,16);
            if(active && !(column&3) && first+column/4<end)
                block_log_mass[(int64_t(token)*heads+head)*count+first+column/4]=sum4>0?top+logf(sum4):-CUDART_INF_F;
        }
        if constexpr(Staged) {
            for(int at=4*tid;at<16*W;at+=256*4) {
                const int token_row=at/W,b=token_row/4,d=at%W;
                const auto item=chosen[b];
                half2 a=__float2half2_rn(0.f),c=a;
                if(item.id>=0 && item.token_mask) {
                    const auto* source=staged_values+((int64_t(item.id)*kv_heads+kh)*4+token_row%4)*W+d;
                    a=*reinterpret_cast<const half2*>(source);c=*reinterpret_cast<const half2*>(source+2);
                }
                auto* destination=value+token_row*P+d;
                *reinterpret_cast<half2*>(destination)=a;*reinterpret_cast<half2*>(destination+2)=c;
            }
        } else {
        for(int at=4*tid;at<4*W;at+=256*4) {
            const int b=at/W,d=at%W;const bool valid=chosen[b].id>=0 && chosen[b].token_mask;
            float4 k[4];
            #pragma unroll
            for(int m=0;m<4;++m) {
                const int r=(b*4+m)*Rings+d/256;
                k[m]=valid?tq_values4(vr[r],d&255):make_float4(0,0,0,0);
            }
            #pragma unroll
            for(int t=0;t<4;++t) {
                float4 a=make_float4(0,0,0,0);
                #pragma unroll
                for(int m=0;m<4;++m) {
                    const float sign=(__popc(unsigned(t&m))&1)?-.5f:.5f;
                    a.x+=sign*k[m].x;a.y+=sign*k[m].y;a.z+=sign*k[m].z;a.w+=sign*k[m].w;
                }
                auto* kp=value+(b*4+t)*P+d;
                reinterpret_cast<half2*>(kp)[0]=__floats2half2_rn(a.x,a.y);reinterpret_cast<half2*>(kp)[1]=__floats2half2_rn(a.z,a.w);
            }
        }
        }
        __syncthreads();
        const int row=lane/4,k=(lane%4)*2,head_pair=(lane%4)*2;
        for(int tile=0;tile<W/128;++tile) {
            const int d=(tile*8+warp)*16+row;
            const unsigned a[4]={qsa_pack_half(value[k*P+d],value[(k+1)*P+d]),qsa_pack_half(value[k*P+d+8],value[(k+1)*P+d+8]),
                qsa_pack_half(value[(k+8)*P+d],value[(k+9)*P+d]),qsa_pack_half(value[(k+8)*P+d+8],value[(k+9)*P+d+8])};
            const unsigned b[2]={*reinterpret_cast<unsigned*>(probability+row*16+k),*reinterpret_cast<unsigned*>(probability+row*16+k+8)};
            float next[4]{};qsa_mma16x8(next,a,b);
            accum[tile][0]=fmaf(accum[tile][0],factors[head_pair],next[0]);
            accum[tile][1]=fmaf(accum[tile][1],factors[head_pair+1],next[1]);
            accum[tile][2]=fmaf(accum[tile][2],factors[head_pair],next[2]);
            accum[tile][3]=fmaf(accum[tile][3],factors[head_pair+1],next[3]);
        }
        __syncthreads();
    }
    for(int tile=0;tile<W/128;++tile)for(int h=0;h<2;++h) {
        const int local=base+(lane%4)*2+h,head=kh*gqa+local,d=(tile*8+warp)*16+lane/4;
        if(local<gqa) {
            partial[((int64_t(token)*heads+head)*parts+part)*W+d]=accum[tile][h];
            partial[((int64_t(token)*heads+head)*parts+part)*W+d+8]=accum[tile][h+2];
        }
    }
    const int stat_local=2*warp+(lane>>4);
    if(warp<4 && !(lane&15) && base+stat_local<gqa) {
        const auto at=(int64_t(token)*heads+kh*gqa+base+stat_local)*parts+part;
        stats[at*2]=maximum;stats[at*2+1]=mass;
    }

#else
    // A lower-architecture PTX image may be JIT-compiled on a newer GPU.
    // Preserve actual computation even when that image lacks m16n8k16.
    // Eight warps cover the full eight-head group in the portable SIMT path.
    sparse_attention_impl<W,Snapshots,8>(query,relative,blocks,selected,positions,partial,stats,
        block_log_mass,heads,kv_heads,width,count,parts,0);

#endif
}
