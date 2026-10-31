#pragma once
// Adapted from Strata src/kernels/cuda/dflash2.cu, Copyright (c) 2026
// Niko1221 and contributors. MIT: cpp_runtime/third_party/strata.LICENSE.
// Retile arbitrary query/GQA counts and token-major KV rings without fixing
// the number of heads, rows, context length or device architecture.
template<int Dimension,class History=float>
__global__ void __launch_bounds__(128) attend_tiled(const float* __restrict__ query,
    const History* __restrict__ ring_key,const History* __restrict__ ring_value,
    const float* __restrict__ block_key,const float* __restrict__ block_value,float* partial,
    int rows,int heads,int kv_heads,int window,int64_t position,const int64_t* device_position) {
    static_assert(Dimension==64 || Dimension==128,"eight-dimensional PV tiles must cover 32 queries in 128 threads");
    constexpr int AS=Dimension+4,PS=36,Loads=Dimension/16;
    constexpr int DimThreads=Dimension/8,QueriesPerThread=32*DimThreads/128;
    __shared__ __align__(16) float qs[32*AS],ks[32*AS],probabilities[32*PS],scores[32*33];
    __shared__ float peaks[32],masses[32],corrections[32];
    const int tid=threadIdx.x,gqa=heads/kv_heads,queries=rows*gqa,query_tiles=(queries+31)/32;
    const int kv=blockIdx.x/query_tiles,qbase=(blockIdx.x%query_tiles)*32,chunk=blockIdx.y;
    if(device_position)position=device_position[0];
    const int64_t lo=max(int64_t(0),position-window+1);
    const int past=int(position-lo),total=past+rows,start=chunk*128,end=min(start+128,total);
    const auto destination=[&](int qi) {
        const int actual=qbase+qi,row=actual/gqa,head=kv*gqa+actual%gqa;
        return partial+(int64_t(chunk)*rows*heads+int64_t(row)*heads+head)*(Dimension+2);
    };
    if(start>=total) {
        for(int i=tid;i<32*(Dimension+2);i+=128) {
            const int qi=i/(Dimension+2),d=i%(Dimension+2);
            if(qbase+qi<queries)destination(qi)[d]=d==0?-INFINITY:0.f;
        }
        return;
    }
    const float scale=rsqrtf(float(Dimension));
    for(int i=tid;i<32*Dimension/4;i+=128) {
        const int qi=i/(Dimension/4),d4=i%(Dimension/4),actual=qbase+qi;
        float4 q=make_float4(0,0,0,0);
        if(actual<queries) {
            const int row=actual/gqa,head=kv*gqa+actual%gqa;
            q=*reinterpret_cast<const float4*>(query+(int64_t(row)*heads+head)*Dimension+d4*4);
            q.x*=scale;q.y*=scale;q.z*=scale;q.w*=scale;
        }
        *reinterpret_cast<float4*>(qs+qi*AS+d4*4)=q;
    }
    if(tid<32){peaks[tid]=-INFINITY;masses[tid]=0.f;}
    float output[QueriesPerThread][8]{};
    const int output_q=tid/DimThreads*QueriesPerThread,output_d=tid%DimThreads*8;
    const auto fetch=[&](int first,float4 (&reg)[Loads],const History* ring,const float* block) {
#pragma unroll
        for(int u=0;u<Loads;++u) {
            const int i=tid+u*128,k=first+i/(Dimension/4),d4=i%(Dimension/4);
            float4 v=make_float4(0,0,0,0);
            if(k<past)v=history_four(ring+(((lo+k)%window)*kv_heads+kv)*Dimension+d4*4);
            else if(k<total)v=*reinterpret_cast<const float4*>(block+(int64_t(k-past)*kv_heads+kv)*Dimension+d4*4);
            reg[u]=v;
        }
    };
    const auto put=[&](const float4 (&reg)[Loads]) {
#pragma unroll
        for(int u=0;u<Loads;++u) {
            const int i=tid+u*128,k=i/(Dimension/4),d4=i%(Dimension/4);
            *reinterpret_cast<float4*>(ks+k*AS+d4*4)=reg[u];
        }
    };
    float4 kreg[Loads],vreg[Loads];fetch(start,kreg,ring_key,block_key);
    for(int first=start;first<end;first+=32) {
        __syncthreads();put(kreg);fetch(first,vreg,ring_value,block_value);__syncthreads();
        if(tid<64) {
            const int qgroup=tid>>3,kgroup=tid&7;float accum[4][4]{};
#pragma unroll 4
            for(int d=0;d<Dimension;d+=4) {
                float4 q[4],k[4];
#pragma unroll
                for(int i=0;i<4;++i)q[i]=*reinterpret_cast<const float4*>(qs+(qgroup*4+i)*AS+d);
#pragma unroll
                for(int j=0;j<4;++j)k[j]=*reinterpret_cast<const float4*>(ks+(kgroup*4+j)*AS+d);
#pragma unroll
                for(int i=0;i<4;++i)
#pragma unroll
                    for(int j=0;j<4;++j)accum[i][j]+=q[i].x*k[j].x+q[i].y*k[j].y+q[i].z*k[j].z+q[i].w*k[j].w;
            }
#pragma unroll
            for(int i=0;i<4;++i)
#pragma unroll
                for(int j=0;j<4;++j) {
                    const int qi=qgroup*4+i,t=kgroup*4+j,k=first+t,row=(qbase+qi)/gqa;
                    const bool valid=qbase+qi<queries && k<total && (k>=past || lo+k>position+row-window);
                    scores[qi*33+t]=valid?accum[i][j]:-INFINITY;
                }
        }
        __syncthreads();
        {
            const int qi=tid>>2,sub=tid&3;float values[8],maximum=-INFINITY;
#pragma unroll
            for(int u=0;u<8;++u){values[u]=scores[qi*33+sub*8+u];maximum=fmaxf(maximum,values[u]);}
            maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,1));
            maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,2));
            const float peak=fmaxf(peaks[qi],maximum),correction=peaks[qi]==-INFINITY?0.f:__expf(peaks[qi]-peak);
            float mass=0;
#pragma unroll
            for(int u=0;u<8;++u) {
                const float p=values[u]==-INFINITY?0.f:__expf(values[u]-peak);
                probabilities[(sub*8+u)*PS+qi]=p;mass+=p;
            }
            mass+=__shfl_xor_sync(0xffffffff,mass,1);mass+=__shfl_xor_sync(0xffffffff,mass,2);
            if(!sub){masses[qi]=masses[qi]*correction+mass;peaks[qi]=peak;corrections[qi]=correction;}
        }
        __syncthreads();put(vreg);
        if(first+32<end)fetch(first+32,kreg,ring_key,block_key);
        __syncthreads();
#pragma unroll
        for(int i=0;i<QueriesPerThread;++i)
#pragma unroll
            for(int j=0;j<8;++j)output[i][j]*=corrections[output_q+i];
#pragma unroll 4
        for(int k=0;k<32;++k) {
            const float4 a=*reinterpret_cast<const float4*>(ks+k*AS+output_d);
            const float4 b=*reinterpret_cast<const float4*>(ks+k*AS+output_d+4);
            const float v[8]={a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w};
#pragma unroll
            for(int i=0;i<QueriesPerThread;++i) {
                const float p=probabilities[k*PS+output_q+i];
#pragma unroll
                for(int j=0;j<8;++j)output[i][j]=fmaf(p,v[j],output[i][j]);
            }
        }
    }
    __syncthreads();
    if(tid<32 && qbase+tid<queries){auto* out=destination(tid);out[0]=peaks[tid];out[1]=masses[tid];}
#pragma unroll
    for(int i=0;i<QueriesPerThread;++i)if(qbase+output_q+i<queries) {
        auto* out=destination(output_q+i)+2+output_d;
#pragma unroll
        for(int j=0;j<8;j+=2)*reinterpret_cast<float2*>(out+j)=make_float2(output[i][j],output[i][j+1]);
    }
}
