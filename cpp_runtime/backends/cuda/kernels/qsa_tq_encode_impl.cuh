// Copyright 2026 MFQ contributors. SPDX-License-Identifier: Apache-2.0
// Included once per trellis profile namespace; no include guard by design.
// STATE_BITS is supplied by the including profile namespace.
constexpr int WARMUP=32;
template<int K> constexpr int E=1<<(STATE_BITS-K);
template<int K> constexpr int THREADS=std::min(512,E<K>/2);
// Internal traceback compression; serialized row streams stay unchanged.
template<int K> constexpr bool PAIRED=WARMUP<128&&(K==2||K==3);
template<int K> constexpr int TRACE_DIV=K==1?8:((PAIRED<K>)?2:1);
__device__ __forceinline__ unsigned qsa_byte_sum(unsigned value) {
#if __CUDA_ARCH__ >= 610
 return __dp4a(value,0x01010101u,0u);
#else
 return (value&255)+((value>>8)&255)+((value>>16)&255)+(value>>24);
#endif
}
__device__ __forceinline__ half2 book(unsigned s){
 unsigned a=qsa_byte_sum(s*0x83dcd12du),b=qsa_byte_sum((s+1)*0x83dcd12du);
 return __hfma2(__halves2half2(__ushort_as_half(0x6400+a),__ushort_as_half(0x6400+b)),
  __half2half2(__ushort_as_half(0x1eee)),__half2half2(__ushort_as_half(0xc931)));
}
__device__ __forceinline__ unsigned warp_min_unsigned(unsigned value) {
#if __CUDA_ARCH__ >= 800
 return __reduce_min_sync(0xffffffff,value);
#else
 for(int delta=16;delta;delta/=2)value=min(value,__shfl_xor_sync(0xffffffff,value,delta));
 return value;
#endif
}

template<int K> __device__ __forceinline__ int choice(const uint8_t*trace,int t,int e){
 if constexpr(K==1){int off=t*(E<K>/32)+(e/64)*2+(e&1);return (reinterpret_cast<const uint32_t*>(trace)[off]>>((e/2)&31))&1;}
 else return trace[int64_t(t)*E<K>+e];
}
template<int K> __global__ __launch_bounds__(THREADS<K>,1) void solve(
 float* values,uint8_t* symbols_out,uint8_t* trace,const int* rowids,const int* lengths,int capacity){
 constexpr int norm=16;
 constexpr int N=E<K>/(2*THREADS<K>),C=1<<K;
 int tid=threadIdx.x;
 const int total=lengths[K-1];
 if(int(blockIdx.x)>=total)return;
 float* xbase=values;
 trace+=int64_t(blockIdx.x)*256*E<K>/TRACE_DIV<K>;
 extern __shared__ half costs[];
 uint8_t*previous_choice=reinterpret_cast<uint8_t*>(costs)+4*E<K>;
 __shared__ half target_row[256];
 __shared__ uint32_t minima[THREADS<K>/32];__shared__ float offset;__shared__ int edge;__shared__ uint16_t symbols[256];
 // High bit rates have up to 256 choices per edge. Recompute their cheap
 // fixed-book values instead of spilling a 256-register point cache.
 half2 points[N][K<=4?C:1];
 {
  #pragma unroll
  for(int i=0;i<N;++i){int e=2*(tid+i*THREADS<K>);
   #pragma unroll
   for(int j=0;j<(K<=4?C:1);++j)points[i][j]=book(e+j*E<K>);
  }
 }
 int count=1;
 if constexpr(WARMUP<128)count=(total-int(blockIdx.x)+int(gridDim.x)-1)/int(gridDim.x);
 for(int item=0;item<count;++item){
 const int64_t r=rowids[(K-1)*capacity+int64_t(blockIdx.x)+int64_t(item)*gridDim.x];
 float* x=xbase+r*256;
 for(int d=tid;d<256;d+=THREADS<K>)target_row[d]=__float2half_rn(x[d]);
 __syncthreads();
 for(int pass=0;pass<2;++pass){
  half*old=costs;half*next=costs+(K==1?0:E<K>);
  for(int e=tid;e<E<K>;e+=THREADS<K>)old[e]=__float2half(pass&&e!=edge?CUDART_INF_F:0.f);
  __syncthreads();
  int steps=pass?256:2*WARMUP;
  for(int t=0;t<steps;++t){int at=pass?t:(256-WARMUP+t)&255;
   half2 target=__half2half2(target_row[at]);
   [[maybe_unused]] half2 staged[K==1?N:1];
   #pragma unroll
   for(int i=0;i<N;++i){int e=2*(tid+i*THREADS<K>);half2 best=__half2half2(__ushort_as_half(0x7c00));uint32_t picks=0;
    auto consider=[&](int j){int s=e+j*E<K>;half2 q;
     if constexpr(K<=4)q=points[i][j];else q=book(s);
     half2 d=__hsub2(q,target);
     half2 v=__hfma2(d,d,__half2half2(old[s>>K]));
     uint32_t take=__hlt2_mask(v,best);picks=(picks&~take)|(uint32_t(j|(j<<16))&take);best=__hmin2(v,best);
        };
    if constexpr(K<=4){
     #pragma unroll
     for(int j=0;j<C;++j)consider(j);
    }else{
     // Bound live procedural candidates at high rates to avoid register spills.
     #pragma unroll 4
     for(int j=0;j<C;++j)consider(j);
    }
    uint32_t lo=picks&65535,hi=picks>>16;
    if constexpr(K==1)staged[i]=best;else reinterpret_cast<half2*>(next)[e/2]=best;
    if (PAIRED<K>){
     if(!(t&1))reinterpret_cast<uint16_t*>(previous_choice)[e/2]=uint16_t(lo|(hi<<8));
    }
    if(pass||t>=WARMUP){
     if constexpr(K==1){unsigned a=__ballot_sync(0xffffffff,lo&1),b=__ballot_sync(0xffffffff,hi&1);
      if(!(tid&31)){int o=t*(E<K>/32)+(e/64)*2;reinterpret_cast<uint32_t*>(trace)[o]=a;reinterpret_cast<uint32_t*>(trace)[o+1]=b;}
     }else if (PAIRED<K>){
      if(t&1){
       uint32_t a=lo|(uint32_t(previous_choice[(e+lo*E<K>)>>K])<<K);
       uint32_t b=hi|(uint32_t(previous_choice[(e+1+hi*E<K>)>>K])<<K);
       reinterpret_cast<uint16_t*>(trace)[(int64_t(t/2)*E<K>+e)/2]=uint16_t(a|(b<<8));
      }
     }else reinterpret_cast<uint16_t*>(trace)[(int64_t(t)*E<K>+e)/2]=uint16_t((lo&255)|((hi&255)<<8));
    }
   }
   if constexpr(K==1){__syncthreads();
    #pragma unroll
    for(int i=0;i<N;++i)reinterpret_cast<half2*>(next)[tid+i*THREADS<K>]=staged[i];
   }
   __syncthreads();half*tmp=old;old=next;next=tmp;
   if(norm&&((t+1)%norm==0)&&t!=steps-1){
    float v=CUDART_INF_F;for(int e=tid;e<E<K>;e+=THREADS<K>)v=fminf(v,__half2float(old[e]));
    if constexpr(WARMUP<128)v=__uint_as_float(warp_min_unsigned(__float_as_uint(v)));
    else for(int d=16;d;d/=2)v=fminf(v,__shfl_down_sync(0xffffffff,v,d));
    if(!(tid&31))minima[tid/32]=__float_as_uint(v);__syncthreads();
    if(!tid){uint32_t b=minima[0];for(int i=1;i<THREADS<K>/32;++i)b=min(b,minima[i]);offset=__uint_as_float(b);}
    __syncthreads();for(int e=tid;e<E<K>;e+=THREADS<K>)old[e]=__float2half_rn(__half2float(old[e])-offset);__syncthreads();
   }
  }
  if(!pass){uint32_t b=0x7c00ffffu;
   for(int e=tid;e<E<K>;e+=THREADS<K>)b=min(b,(uint32_t(__half_as_ushort(old[e]))<<16)|e);
   if constexpr(WARMUP<128)b=warp_min_unsigned(b);
   else for(int d=16;d;d/=2)b=min(b,__shfl_down_sync(0xffffffff,b,d));
   if(!(tid&31))minima[tid/32]=b;__syncthreads();
   if(!tid){for(int i=1;i<THREADS<K>/32;++i)minima[0]=min(minima[0],minima[i]);int e=minima[0]&65535;
    if (PAIRED<K>){
     for(int t=2*WARMUP-1;t>=WARMUP;t-=2){
      int p=trace[int64_t(t/2)*E<K>+e];
      e=(e+(p&((1<<K)-1))*E<K>)>>K;e=(e+(p>>K)*E<K>)>>K;
     }
    }else for(int t=2*WARMUP-1;t>=WARMUP;--t)e=(e+choice<K>(trace,t,e)*E<K>)>>K;
    edge=e;
   }__syncthreads();
  }else{
   if(!tid){int e=edge;
    if (PAIRED<K>){
     for(int t=255;t>=0;t-=2){
      int p=trace[int64_t(t/2)*E<K>+e],s=e+(p&((1<<K)-1))*E<K>;
      symbols[t]=s&((1<<K)-1);e=s>>K;
      s=e+(p>>K)*E<K>;symbols[t-1]=s&((1<<K)-1);e=s>>K;
     }
    }else for(int t=255;t>=0;--t){int s=e+choice<K>(trace,t,e)*E<K>;symbols[t]=s&((1<<K)-1);e=s>>K;}
   }
   __syncthreads();for(int out=tid;out<256;out+=THREADS<K>){unsigned s=0;
    #pragma unroll
    for(int j=(16+K-1)/K-1;j>=0;--j)s=(s<<K)|symbols[(out-j+256)&255];
    symbols_out[r*256+out]=uint8_t(s&((1<<K)-1));x[out]=__half2float(__low2half(book(s&((1<<STATE_BITS)-1))));
   }
  }
 }
 if constexpr(WARMUP<128)__syncthreads();
 }
}
template<int K> constexpr int SH=E<K>*(K==1?2:((PAIRED<K>)?5:4));
