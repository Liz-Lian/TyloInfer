#pragma once
// FP16 m16n8k16 uses SM80 instructions. The host also checks the compiled
// virtual architecture so lower-architecture PTX keeps the vector path.
__device__ __forceinline__ float paged_warp_max(float value) {
    for(int d=16;d;d>>=1)value=fmaxf(value,__shfl_xor_sync(0xffffffffu,value,d));
    return value;
}
__device__ __forceinline__ void paged_mma16x8(float* d,const unsigned* a,const unsigned* b) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
#endif
}
__device__ __forceinline__ unsigned paged_pack_half(half a,half b) {
    return unsigned(__half_as_ushort(a))|(unsigned(__half_as_ushort(b))<<16);
}
// The MMA path already requires SM80. Transfer raw FP16 KV in 16-byte
// groups without keeping a copied vector live in general registers.
__device__ __forceinline__ void paged_copy_half8(half* destination,const half* source,int d,int width) {
#if __CUDA_ARCH__ >= 800
    if(source && !(width&7) && d+7<width) {
        const auto address=static_cast<unsigned>(__cvta_generic_to_shared(destination));
        asm volatile("cp.async.ca.shared.global [%0],[%1],16;"::"r"(address),"l"(source+d):"memory");
        return;
    }
#endif
    #pragma unroll
    for(int i=0;i<8;++i)destination[i]=source && d+i<width?source[d+i]:__float2half_rn(0);
}
__device__ __forceinline__ float paged_halfwarp_max(float value) {
    #pragma unroll
    for(int d=8;d;d>>=1)value=fmaxf(value,__shfl_xor_sync(0xffffffffu,value,d,16));
    return value;
}
__device__ __forceinline__ float paged_halfwarp_sum(float value) {
    #pragma unroll
    for(int d=8;d;d>>=1)value+=__shfl_xor_sync(0xffffffffu,value,d,16);
    return value;
}
template<int W> __global__ void __launch_bounds__(128) paged_attention_decode_mma8(
    const half* query,const int64_t* key_chunks,const int64_t* value_chunks,
    const int32_t* page_table,const int64_t* lengths,float* partial,float* maxima,float* sums,
    int heads,int kv_heads,int width,int logical_pages,int page_size,int pages_per_chunk,int chunks,
    int parts,int workspace_parts,float scale,int dynamic_parts) {
#if __CUDA_ARCH__ >= 800
    constexpr int P=W+8;
    const int tid=threadIdx.x,lane=tid&31,warp=tid/32,part=blockIdx.x,token=blockIdx.z;
    const int gqa=heads/kv_heads,groups=(gqa+7)/8,kh=blockIdx.y/groups,base=(blockIdx.y%groups)*8;
    extern __shared__ __align__(32) unsigned char memory[];
    auto* qh=reinterpret_cast<half*>(memory);
    auto* key=qh+8*P;auto* value=key;
    auto* probability=value+16*P;
    auto* scores=reinterpret_cast<float*>(probability+8*16);
    __shared__ const half *key_rows[16],*value_rows[16];
    __shared__ float factors[8];
    for(int at=tid;at<8*W;at+=128) {
        const int h=at/W,d=at%W,head=kh*gqa+base+h;
        qh[h*P+d]=base+h<gqa && d<width?query[(int64_t(token)*heads+head)*width+d]:__float2half_rn(0);
    }
    float accum[W/64][4]{};
    float maximum[2]={-CUDART_INF_F,-CUDART_INF_F},mass[2]{};
    const int tokens=int(max(int64_t(0),min(lengths[token],int64_t(logical_pages)*page_size)));
    const int active_parts=paged_active_parts(tokens,parts,dynamic_parts);
    if(part>=active_parts)return;
    const int tiles=(tokens+15)/16,begin=16*(part*tiles/active_parts),end=min(tokens,16*((part+1)*tiles/active_parts));
    const size_t page_elements=size_t(kv_heads)*page_size*width;
    for(int first=begin;first<end;first+=16) {
        if(tid<16) {
            const int position=first+tid,logical=position/page_size;
            const int physical=position<end?page_table[size_t(token)*logical_pages+logical]:-1;
            const int chunk=physical>=0?physical/pages_per_chunk:-1;
            const size_t offset=(size_t(kh)*page_size+position%page_size)*width;
            key_rows[tid]=chunk>=0 && chunk<chunks && key_chunks[chunk]
                ? page_address<half>(key_chunks,physical,pages_per_chunk,page_elements)+offset:nullptr;
            value_rows[tid]=chunk>=0 && chunk<chunks && value_chunks[chunk]
                ? page_address<half>(value_chunks,physical,pages_per_chunk,page_elements)+offset:nullptr;
        }
        __syncthreads();
        for(int at=8*tid;at<16*W;at+=128*8) {
            const int row=at/W,d=at%W;
            paged_copy_half8(key+row*P+d,key_rows[row],d,width);
        }
        asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;":::"memory");
        __syncthreads();
        if(!warp) {
            float c[4]{};const int r=lane/4,k=(lane%4)*2;
            for(int d=0;d<W;d+=16) {
                const unsigned a[4]={*reinterpret_cast<unsigned*>(key+r*P+d+k),*reinterpret_cast<unsigned*>(key+(r+8)*P+d+k),
                    *reinterpret_cast<unsigned*>(key+r*P+d+k+8),*reinterpret_cast<unsigned*>(key+(r+8)*P+d+k+8)};
                const unsigned b[2]={*reinterpret_cast<unsigned*>(qh+r*P+d+k),*reinterpret_cast<unsigned*>(qh+r*P+d+k+8)};
                paged_mma16x8(c,a,b);
            }
            const int h=(lane%4)*2;
            scores[h*16+r]=c[0];scores[(h+1)*16+r]=c[1];scores[h*16+r+8]=c[2];scores[(h+1)*16+r+8]=c[3];
        }
        __syncthreads();
        for(int r=0;r<2;++r) {
            const int local=warp+4*r;
            const bool active=base+local<gqa;
            float score=-CUDART_INF_F;
            if(lane<16 && active) {
                if(first+lane<end)score=scores[local*16+lane]*scale;
            }
            const float top=fmaxf(maximum[r],paged_halfwarp_max(score));
            const float factor=isfinite(maximum[r])?expf(maximum[r]-top):0;
            const float weight=isfinite(score)?expf(score-top):0;
            if(!lane)factors[local]=factor;
            mass[r]=mass[r]*factor+paged_halfwarp_sum(weight);maximum[r]=top;
            if(lane<16)probability[local*16+lane]=__float2half_rn(weight);
        }
        // QK completed at the barrier before softmax. Scores and
        // probabilities do not alias the shared K/V tile.
        for(int at=8*tid;at<16*W;at+=128*8) {
            const int row=at/W,d=at%W;
            paged_copy_half8(value+row*P+d,value_rows[row],d,width);
        }
        asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;":::"memory");
        __syncthreads();
        const int row=lane/4,k=(lane%4)*2,head_pair=(lane%4)*2;
        for(int tile=0;tile<W/64;++tile) {
            const int d=(tile*4+warp)*16+row;
            const unsigned a[4]={paged_pack_half(value[k*P+d],value[(k+1)*P+d]),paged_pack_half(value[k*P+d+8],value[(k+1)*P+d+8]),
                paged_pack_half(value[(k+8)*P+d],value[(k+9)*P+d]),paged_pack_half(value[(k+8)*P+d+8],value[(k+9)*P+d+8])};
            const unsigned b[2]={*reinterpret_cast<unsigned*>(probability+row*16+k),*reinterpret_cast<unsigned*>(probability+row*16+k+8)};
            float next[4]{};paged_mma16x8(next,a,b);
            accum[tile][0]=fmaf(accum[tile][0],factors[head_pair],next[0]);
            accum[tile][1]=fmaf(accum[tile][1],factors[head_pair+1],next[1]);
            accum[tile][2]=fmaf(accum[tile][2],factors[head_pair],next[2]);
            accum[tile][3]=fmaf(accum[tile][3],factors[head_pair+1],next[3]);
        }
        __syncthreads();
    }
    for(int tile=0;tile<W/64;++tile)for(int h=0;h<2;++h) {
        const int local=base+(lane%4)*2+h,head=kh*gqa+local,d=(tile*4+warp)*16+lane/4;
        if(local<gqa) {
            const size_t at=(size_t(token)*heads+head)*workspace_parts+part;
            if(d<width)partial[at*width+d]=accum[tile][h];
            if(d+8<width)partial[at*width+d+8]=accum[tile][h+2];
        }
    }
    for(int r=0;r<2;++r)if(!lane && base+warp+4*r<gqa) {
        const size_t at=(size_t(token)*heads+kh*gqa+base+warp+4*r)*workspace_parts+part;
        maxima[at]=isfinite(maximum[r])?maximum[r]:-1e30f;sums[at]=mass[r];
    }
#endif
}
