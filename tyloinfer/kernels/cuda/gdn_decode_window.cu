#include "mfq_cuda_linear_attention_ops.h"
#include "mfq_cuda_gdn_replay_ops.h"
#define GGML_COMMON_DECL_CUDA
#include "ggml-common.h"
#include "../../../cpp_runtime/backends/cuda/kernels/gguf_quantize.cuh"
#include "../../../cpp_runtime/backends/cuda/kernels/warp_sums.cuh"
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <array>
#include <algorithm>
#include "reduce.cuh"
#include "gdn_input.cuh"
#include <atomic>
#include <limits>
#include <type_traits>
namespace tb=mfq_tensor_backend;
// A warp retains four state rows for each of its sixteen value columns.
// Time steps reuse those registers; only rollback prefixes and the final
// state are written. Convolution history remains read-only until completion.
struct GdnWindowSnapshots {float* conv[8]{};float* state[8]{};int confirmed=1;float* updates=nullptr;block_q8_1* quantized=nullptr;bool position_major=false;};

enum class GdnWindowMode {Snapshots,ReadOnly,Commit,RecordUpdates};

template<class Gate,class Output,GdnWindowMode Mode,int Columns=16>
__device__ __forceinline__ void gdn_decode_window_body(
 const void* __restrict__ qkv,const Gate* __restrict__ output_gate,
 const void* __restrict__ alpha,const void* __restrict__ beta,
 const float* __restrict__ conv,float* __restrict__ state,const float* __restrict__ weights,
 const float* __restrict__ bias,const float* __restrict__ a_log,const float* __restrict__ norm,
 Output* __restrict__ output,float* __restrict__ next_conv,
 int nk,int nv,int tokens,int layout,bool transposed,bool silu,float conv_eps,double norm_eps,
 int64_t qs,int64_t gs,int64_t as,int64_t ah,int64_t bs,int64_t bh,GdnWindowSnapshots snapshots,bool qkv_float,bool tiled_heads,bool gates_float=std::is_same_v<Gate,float>) {
 constexpr int D=128,Warps=8;
 static_assert(Columns>0 && 16%Columns==0);
 static_assert(Mode!=GdnWindowMode::ReadOnly || Columns==16);
 constexpr bool Readout=Mode!=GdnWindowMode::Commit,
  WriteState=Mode==GdnWindowMode::Snapshots || Mode==GdnWindowMode::Commit;
 const int head=blockIdx.x,tid=threadIdx.x,lane=tid&31,warp=tid>>5;
 const int key_head=tiled_heads?head%nk:head/(nv/nk),channels=(2*nk+nv)*D,qk=2*nk*D;
 const int qc=key_head*D+tid,kc=nk*D+qc,vc=qk+head*D+tid;
 __shared__ float query[D],key[D],value[D],attended[D],qsum[256],ksum[256],norm_sum[D];
 __shared__ float decay_value,beta_value,inverse;
 float states[Columns][4];
 #pragma unroll
 for(int c=0;c<Columns;++c) {
  const int column=warp+(int(blockIdx.z)*Columns+c)*Warps;
  #pragma unroll
  for(int r=0;r<4;++r) {
   const int i=r*32+lane;
   states[c][r]=state[size_t(head)*D*D+(transposed?size_t(column)*D+i:size_t(i)*D+column)];
  }
 }
 for(int t=0;t<tokens;++t) {
  // Every token reads the original history or an already projected QKV row.
  // No head may overwrite shared Q/K history while another still needs it.
  float q=0,k=0,v=0;
  if(tid<D) {
   #pragma unroll
   for(int tap=0;tap<4;++tap) {
    const int from=t+tap-3;
    const float qv=Readout?(from<0?conv[size_t(from+3)*channels+qc]:gdn_qkv_value(qkv,int64_t(from)*qs+qc,qkv_float)):0.f;
    const float kv=from<0?conv[size_t(from+3)*channels+kc]:gdn_qkv_value(qkv,int64_t(from)*qs+kc,qkv_float);
    const float vv=from<0?conv[size_t(from+3)*channels+vc]:gdn_qkv_value(qkv,int64_t(from)*qs+vc,qkv_float);
    if constexpr(Readout)q+=qv*weights[layout==0?size_t(qc)*4+tap:size_t(tap)*channels+qc];
    k+=kv*weights[layout==0?size_t(kc)*4+tap:size_t(tap)*channels+kc];
    v+=vv*weights[layout==0?size_t(vc)*4+tap:size_t(tap)*channels+vc];
   }
   if constexpr(Readout)q=q/(1.f+expf(-q));
   k=k/(1.f+expf(-k));v=v/(1.f+expf(-v));
   if constexpr(Readout)query[tid]=q;
   key[tid]=k;value[tid]=v;
  }
  if constexpr(Readout)qsum[tid]=tid<D?q*q:0;
  ksum[tid]=tid<D?k*k:0;
  if(tid==0) {
   const float a=gdn_qkv_value(alpha,int64_t(t)*as+int64_t(head)*ah,gates_float)+bias[head];
   const float sp=fmaxf(a,0.f)+log1pf(expf(-fabsf(a)));
   decay_value=expf(sp*-expf(a_log[head]));
   beta_value=1.f/(1.f+expf(-gdn_qkv_value(beta,int64_t(t)*bs+int64_t(head)*bh,gates_float)));
  }
  __syncthreads();
  if(tid<32) {
   float qr=0;
   if constexpr(Readout)qr=(qsum[tid]+qsum[tid+128])+(qsum[tid+64]+qsum[tid+192]);
   float kr=(ksum[tid]+ksum[tid+128])+(ksum[tid+64]+ksum[tid+192]);
   if constexpr(Readout)qr+=(qsum[tid+32]+qsum[tid+160])+(qsum[tid+96]+qsum[tid+224]);
   kr+=(ksum[tid+32]+ksum[tid+160])+(ksum[tid+96]+ksum[tid+224]);
   #pragma unroll
   for(int stride=16;stride>0;stride>>=1) {
    const float qother=__shfl_down_sync(0xffffffffu,qr,stride),kother=__shfl_down_sync(0xffffffffu,kr,stride);
    if(tid<stride){qr+=qother;kr+=kother;}
   }
   if(tid==0) {
    if constexpr(Readout)qsum[0]=1.f/fmaxf(sqrtf(qr),conv_eps);
    ksum[0]=1.f/fmaxf(sqrtf(kr),conv_eps);
   }
  }
  __syncthreads();
  if(tid<D) {
   if constexpr(Readout)query[tid]*=qsum[0];
   key[tid]*=ksum[0];
  }
  __syncthreads();
  const int checkpoint=t+1-snapshots.confirmed;
  if constexpr(Mode==GdnWindowMode::RecordUpdates)if(blockIdx.z==0) {
   const auto base=(size_t(t)*nv+head)*257;
   if(tid<D)snapshots.updates[base+tid]=key[tid];
   if(!tid)snapshots.updates[base+256]=decay_value;
  }
  float* snapshot_state=nullptr;
  if constexpr(Mode==GdnWindowMode::Snapshots)
   snapshot_state=checkpoint>=0 && t+1<tokens?snapshots.state[checkpoint]:nullptr;
  const float decay=decay_value,bt=beta_value;
  float keys[4],queries[4];
  #pragma unroll
  for(int r=0;r<4;++r){keys[r]=key[r*32+lane];if constexpr(Readout)queries[r]=query[r*32+lane];}
  // Reduce all columns together, retaining the same XOR addition tree.
  // A scalar reduction per column repeats five warp shuffles each time;
  // the transposed butterfly shares those levels across the live columns.
  float column_sums[Columns]{};
  #pragma unroll
  for(int c=0;c<Columns;++c) {
   #pragma unroll
   for(int r=0;r<4;++r)column_sums[c]+=states[c][r]*keys[r];
  }
  const float projected_columns=mfq::cuda::warp_sums::reduce<Columns>(column_sums,lane);
  #pragma unroll
  for(int c=0;c<Columns;++c) {
   const int column=warp+(int(blockIdx.z)*Columns+c)*Warps;
   const float projected=__shfl_sync(0xffffffffu,projected_columns,mfq::cuda::warp_sums::owner<Columns>(c));
   const float delta=(value[column]-decay*projected)*bt;
   if constexpr(Mode==GdnWindowMode::RecordUpdates)
    if(!lane)snapshots.updates[(size_t(t)*nv+head)*257+128+column]=delta;
   float result=0;
   #pragma unroll
   for(int r=0;r<4;++r) {
    states[c][r]=__fmaf_rn(decay,states[c][r],__fmul_rn(keys[r],delta));
    if constexpr(Readout)result+=states[c][r]*queries[r];
    if(snapshot_state) {
     const int i=r*32+lane;
     snapshot_state[size_t(head)*D*D+(transposed?size_t(column)*D+i:size_t(i)*D+column)]=states[c][r];
    }
   }
   if constexpr(Readout)column_sums[c]=result;
  }
  if constexpr(Readout) {
   const float readout=mfq::cuda::warp_sums::reduce<Columns>(column_sums,lane);
   const int c=mfq::cuda::warp_sums::index<Columns>(lane);
   if(lane==mfq::cuda::warp_sums::owner<Columns>(c)) {
     const int column=warp+(int(blockIdx.z)*Columns+c)*Warps;
     const float value=readout*(1.f/sqrtf(float(D)));
     if constexpr(Columns<16)output[(size_t(t)*nv+head)*D+column]=value;
     else attended[column]=value;
   }
  }
  if constexpr(Readout && Columns==16) {
  __syncthreads();
  if(tid<D)norm_sum[tid]=__fadd_rn(0.f,__fmul_rn(attended[tid],attended[tid]));
  __syncthreads();
  if(tid<32) {
   float sum=__fadd_rn(__fadd_rn(norm_sum[tid],norm_sum[tid+64]),__fadd_rn(norm_sum[tid+32],norm_sum[tid+96]));
   #pragma unroll
   for(int stride=16;stride>0;stride>>=1) {
    const float other=__shfl_down_sync(0xffffffffu,sum,stride);
    if(tid<stride)sum=__fadd_rn(sum,other);
   }
   if(tid==0)inverse=__fdiv_rn(1.f,__fsqrt_rn(float(double(__fdiv_rn(sum,float(D)))+norm_eps)));
  }
  __syncthreads();
  if(tid<D) {
   const float z=float(output_gate[int64_t(t)*gs+head*D+tid]);
   const float sigmoid=__fdiv_rn(1.f,__fadd_rn(1.f,expf(-z)));
   const float gate=silu?__fmul_rn(z,sigmoid):sigmoid;
   const auto at=(size_t(t)*nv+head)*D+tid;
   const auto rounded=static_cast<Output>(__fmul_rn(__fmul_rn(__fmul_rn(attended[tid],inverse),norm[tid]),gate));
   if(snapshots.quantized) {
    if(snapshots.position_major)mfq::cuda::gguf::store_q8_position_major(float(rounded),int64_t(at),int64_t(nv)*D,tokens,snapshots.quantized);
    else mfq::cuda::gguf::store_q8(float(rounded),int64_t(at),snapshots.quantized);
   }
   else output[at]=rounded;
  }
  }
  if constexpr(WriteState)if(tid<D && (Columns==16 || blockIdx.z==0)) {
   float* destination=t+1==tokens?next_conv:nullptr;
   if constexpr(Mode==GdnWindowMode::Snapshots)
    if(t+1<tokens && checkpoint>=0)destination=snapshots.conv[checkpoint];
   if(destination) {
    #pragma unroll
    for(int tap=0;tap<3;++tap) {
     const int from=t+tap-2;
     destination[size_t(tap)*channels+vc]=from<0?conv[size_t(from+3)*channels+vc]:gdn_qkv_value(qkv,int64_t(from)*qs+vc,qkv_float);
     if(tiled_heads?head<nk:head%(nv/nk)==0) {
      destination[size_t(tap)*channels+qc]=from<0?conv[size_t(from+3)*channels+qc]:gdn_qkv_value(qkv,int64_t(from)*qs+qc,qkv_float);
      destination[size_t(tap)*channels+kc]=from<0?conv[size_t(from+3)*channels+kc]:gdn_qkv_value(qkv,int64_t(from)*qs+kc,qkv_float);
     }
    }
   }
  }
  __syncthreads();
 }
 if constexpr(WriteState) {
 #pragma unroll
 for(int c=0;c<Columns;++c) {
  const int column=warp+(int(blockIdx.z)*Columns+c)*Warps;
  #pragma unroll
  for(int r=0;r<4;++r) {
   const int i=r*32+lane;
   state[size_t(head)*D*D+(transposed?size_t(column)*D+i:size_t(i)*D+column)]=states[c][r];
  }
 }
}

}
template<class Gate,class Output,GdnWindowMode Mode>
__global__ __launch_bounds__(256,(Mode==GdnWindowMode::RecordUpdates || Mode==GdnWindowMode::Snapshots)?2:1) void gdn_decode_window_kernel(
 const void* __restrict__ qkv,const Gate* __restrict__ output_gate,
 const Gate* __restrict__ alpha,const Gate* __restrict__ beta,
 const float* __restrict__ conv,float* __restrict__ state,const float* __restrict__ weights,
 const float* __restrict__ bias,const float* __restrict__ a_log,const float* __restrict__ norm,
 Output* __restrict__ output,float* __restrict__ next_conv,
 int nk,int nv,int tokens,int layout,bool transposed,bool silu,float conv_eps,double norm_eps,
 int64_t qs,int64_t gs,int64_t as,int64_t ah,int64_t bs,int64_t bh,GdnWindowSnapshots snapshots,bool qkv_float,bool tiled_heads) {
 // Split state columns into four smaller tiles. Besides reducing the live
 // state per thread, this exposes more independent blocks for short windows.
 // Output normalization waits for all tiles; each column retains its exact
 // recurrence and checkpoint arithmetic.
 constexpr int Columns=(Mode==GdnWindowMode::RecordUpdates || Mode==GdnWindowMode::Snapshots)?4:16;
 gdn_decode_window_body<Gate,Output,Mode,Columns>(qkv,output_gate,alpha,beta,conv,state,weights,bias,a_log,norm,
  output,next_conv,nk,nv,tokens,layout,transposed,silu,conv_eps,norm_eps,qs,gs,as,ah,bs,bh,snapshots,qkv_float,tiled_heads);
}

__global__ __launch_bounds__(256,2) void gdn_replay_commit_layers_kernel(const GdnReplayLayer* layers,int keep) {
 const auto& d=layers[blockIdx.y];if(blockIdx.x>=unsigned(d.value_heads))return;
 GdnWindowSnapshots snapshots;snapshots.confirmed=keep;
 // All layers use one body. Runtime gate loads avoid separate F16/F32
 // instantiations with duplicated shared storage and control-flow state.
 gdn_decode_window_body<float,float,GdnWindowMode::Commit,8>(d.qkv,nullptr,d.alpha,d.beta,
   d.convolution,d.state,d.weight,d.bias,d.a_log,nullptr,nullptr,d.next_convolution,
   d.key_heads,d.value_heads,keep,d.weight_layout,d.transposed,false,d.convolution_eps,1e-6,
   d.qkv_stride,0,d.alpha_stride,d.alpha_head_stride,d.beta_stride,d.beta_head_stride,snapshots,d.qkv_float,d.tiled_heads,d.gate_float);
}
__global__ void gdn_replay_copy_convolutions(const GdnReplayLayer* layers) {
 const auto& d=layers[blockIdx.y];const auto i=int64_t(blockIdx.x)*256+threadIdx.x;
 if(i<int64_t(3)*d.channels)d.convolution[i]=d.next_convolution[i];
}

template<class Gate,bool Half,bool Quantized>
__global__ void gdn_finish_recorded_readout(const float* attended,const Gate* gate,const float* norm,
        void* output,int nv,int64_t gate_stride,bool silu,double norm_eps,bool position_major) {
 constexpr int D=128;const int head=blockIdx.x,t=blockIdx.y,tid=threadIdx.x;
 const auto at=(size_t(t)*nv+head)*D+tid;
 __shared__ float square[D],inverse;
 square[tid]=__fadd_rn(0.f,__fmul_rn(attended[at],attended[at]));
 __syncthreads();
 if(tid<32) {
  float sum=__fadd_rn(__fadd_rn(square[tid],square[tid+64]),__fadd_rn(square[tid+32],square[tid+96]));
  #pragma unroll
  for(int offset=16;offset;offset>>=1) {
   const float other=__shfl_down_sync(0xffffffffu,sum,offset);
   if(tid<offset)sum=__fadd_rn(sum,other);
  }
  if(!tid)inverse=__fdiv_rn(1.f,__fsqrt_rn(float(double(__fdiv_rn(sum,float(D)))+norm_eps)));
 }
 __syncthreads();
 const float z=float(gate[int64_t(t)*gate_stride+head*D+tid]);
 const float sigmoid=__fdiv_rn(1.f,__fadd_rn(1.f,expf(-z)));
 const float activated=silu?__fmul_rn(z,sigmoid):sigmoid;
 float value=__fmul_rn(__fmul_rn(__fmul_rn(attended[at],inverse),norm[tid]),activated);
 if constexpr(Half)value=__half2float(__float2half_rn(value));
 if constexpr(Quantized) {
  if(position_major)mfq::cuda::gguf::store_q8_position_major(value,int64_t(at),int64_t(nv)*D,int(gridDim.y),output);
  else mfq::cuda::gguf::store_q8(value,int64_t(at),static_cast<block_q8_1*>(output));
 }
 else if constexpr(Half)static_cast<__half*>(output)[at]=__float2half_rn(value);
 else static_cast<float*>(output)[at]=value;
}
// A recorded update is S'=fma(decay,S,mul(key,delta)), exactly the main
// recurrence's explicit operations. Committing it needs no dot-product,
// normalization, convolution or sigmoid recomputation.
__global__ void gdn_apply_recorded_updates(const GdnReplayLayer* layers,int keep) {
 const auto& d=layers[blockIdx.y];const int head=blockIdx.x;
 if(head>=d.value_heads)return;
 extern __shared__ float updates[];
 for(int i=threadIdx.x;i<keep*257;i+=blockDim.x)
  updates[i]=d.updates[(size_t(i/257)*d.value_heads+head)*257+i%257];
 __syncthreads();
 auto* state=d.state+size_t(head)*128*128;
 const bool aligned=(reinterpret_cast<uintptr_t>(state)&15u)==0;
 for(int at=threadIdx.x*4;at<128*128;at+=blockDim.x*4) {
  float4 value;
  if(aligned)value=*reinterpret_cast<const float4*>(state+at);
  else value=make_float4(state[at],state[at+1],state[at+2],state[at+3]);
  float* parts=reinterpret_cast<float*>(&value);
  #pragma unroll
  for(int part=0;part<4;++part) {
   const int row=d.transposed?(at+part)%128:(at+part)/128;
   const int column=d.transposed?(at+part)/128:(at+part)%128;
   for(int t=0;t<keep;++t)parts[part]=__fmaf_rn(updates[t*257+256],parts[part],
    __fmul_rn(updates[t*257+row],updates[t*257+128+column]));
  }
  if(aligned)*reinterpret_cast<float4*>(state+at)=value;
  else {state[at]=value.x;state[at+1]=value.y;state[at+2]=value.z;state[at+3]=value.w;}
 }
 // Prepare histories before the following copy kernel changes any layer's
 // old convolution. Key heads may be shared by several value heads.
 if(threadIdx.x<128) {
  const int i=threadIdx.x,nk=d.key_heads,nv=d.value_heads;
  const int kh=d.tiled_heads?head%nk:head/(nv/nk);
  const int qc=kh*128+i,kc=nk*128+qc,vc=2*nk*128+head*128+i;
  for(int tap=0;tap<3;++tap) {
   const int from=keep+tap-3;
   const auto load=[&](int channel) {
    return from<0?d.convolution[size_t(from+3)*d.channels+channel]:
     gdn_qkv_value(d.qkv,int64_t(from)*d.qkv_stride+channel,d.qkv_float);
   };
   d.next_convolution[size_t(tap)*d.channels+vc]=load(vc);
   if(d.tiled_heads?head<nk:head%(nv/nk)==0) {
    d.next_convolution[size_t(tap)*d.channels+qc]=load(qc);
    d.next_convolution[size_t(tap)*d.channels+kc]=load(kc);
   }
  }
 }
}
void gdn_replay_commit_layers_cuda(const tb::Tensor& descriptors,int layers,int maximum_heads,int maximum_channels,int keep,bool recorded_updates) {
 TYLOINFER_RUNTIME_CHECK(descriptors.is_cuda() && descriptors.is_contiguous() && descriptors.scalar_type()==tb::kUInt8 &&
  layers>0 && maximum_heads>0 && maximum_channels>0 && keep>=1 && keep<=8 &&
  descriptors.numel()>=int64_t(layers)*int64_t(sizeof(GdnReplayLayer)),"GDN replay layer descriptors are invalid");
 const MfqCudaGuard guard(descriptors.device());const auto stream=mfq_current_cuda_stream();
 auto resident=descriptors;resident.record_stream(reinterpret_cast<std::uintptr_t>(stream));
 const auto* data=reinterpret_cast<const GdnReplayLayer*>(resident.data_ptr());
 // CUDA's y grid dimension is bounded even when a model has many layers.
 for(int first=0;first<layers;first+=65535) {
  const int count=std::min(65535,layers-first);
  // Commit has no cross-column readout. Two tiles halve the live state;
  // launch bounds discourage retaining all state addresses across the time
  // loop. Each output column keeps its original arithmetic order.
  if(recorded_updates)
   gdn_apply_recorded_updates<<<dim3(maximum_heads,count),256,size_t(keep)*257*sizeof(float),stream>>>(data+first,keep);
  else gdn_replay_commit_layers_kernel<<<dim3(maximum_heads,count,2),256,0,stream>>>(data+first,keep);
  gdn_replay_copy_convolutions<<<dim3(unsigned((int64_t(3)*maximum_channels+255)/256),count),256,0,stream>>>(data+first);
 }
 TYLOINFER_CUDA_CHECK(cudaGetLastError());
 static std::atomic<unsigned> reported{0};
 const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");
 const auto bit=recorded_updates?2u:1u;
 if(trace && trace[0]=='1' && !(reported.fetch_or(bit)&bit))
  std::fprintf(stderr,"runtime_gdn_replay_batch enabled=1 layers=%d kernels=2 maximum_heads=%d recorded_updates=%d\n",layers,maximum_heads,int(recorded_updates));
}

static std::vector<tb::Tensor> gdn_window_impl(const tb::Tensor& qkv,const tb::Tensor& gate,
 const tb::Tensor& alpha,const tb::Tensor& beta,const tb::Tensor& conv,const tb::Tensor& state,
 const tb::Tensor& weights,const tb::Tensor& bias,const tb::Tensor& a_log,const tb::Tensor& norm,
 int nk,int nv,int confirmed,double conv_eps,double norm_eps,bool silu,bool half,bool transposed,
 const std::vector<tb::Tensor>& checkpoint_conv,const std::vector<tb::Tensor>& checkpoint_state,bool tiled_heads,GdnWindowMode mode,bool quantize_output=false,bool position_major=false) {
 TYLOINFER_RUNTIME_CHECK(!quantize_output || mode!=GdnWindowMode::Commit,"GDN output preparation requires a readout");
 TYLOINFER_RUNTIME_CHECK(qkv.dim()==3 && qkv.size(0)==1 && qkv.size(1)>=1 && qkv.size(1)<=8 &&
  nk>0 && nv>0 && nv<=65535 && nv%nk==0 && int64_t(2)*nk+nv<=std::numeric_limits<int>::max()/128,
  "GDN window heads/sequence invalid");
 const int tokens=int(qkv.size(1)),channels=(2*nk+nv)*128;
 TYLOINFER_RUNTIME_CHECK(!position_major || (quantize_output && tokens>=2 && tokens<=4),"GDN position-major output requires 2..4 Q8 rows");
 TYLOINFER_RUNTIME_CHECK(confirmed>=1 && confirmed<=tokens &&
  checkpoint_conv.size()==size_t(tokens-confirmed) && checkpoint_state.size()==checkpoint_conv.size(),"GDN window geometry invalid");
 const auto device=qkv.device();
 const auto real=[&](const tb::Tensor& x){return x.is_cuda() && x.device()==device && x.is_contiguous() && x.scalar_type()==tb::kFloat32;};
 TYLOINFER_RUNTIME_CHECK(qkv.is_cuda() && (qkv.scalar_type()==tb::kFloat16 || qkv.scalar_type()==tb::kFloat32) && qkv.size(2)==channels &&
  qkv.stride(2)==1 && qkv.stride(1)>=channels && std::isfinite(conv_eps) && conv_eps>0 && std::isfinite(norm_eps) && norm_eps>0,
  "GDN window QKV/epsilon invalid");
 TYLOINFER_RUNTIME_CHECK(real(conv) && real(state) && real(weights) && real(bias) && real(a_log) && real(norm) &&
  conv.sizes().vec()==std::vector<int64_t>({1,3,channels}) && state.sizes().vec()==std::vector<int64_t>({1,nv,128,128}) &&
  (weights.sizes().vec()==std::vector<int64_t>({channels,1,4}) || weights.sizes().vec()==std::vector<int64_t>({4,channels})) &&
  bias.numel()==nv && a_log.numel()==nv && norm.numel()==128,"GDN window state/weights invalid");
 const auto gate_type=gate.scalar_type();
 const auto projection=[&](const tb::Tensor& x,const std::vector<int64_t>& shape) {
  return x.is_cuda() && x.device()==device && x.scalar_type()==gate_type && x.sizes().vec()==shape &&
   x.stride(1)>0 && x.stride(2)>0;
 };
 TYLOINFER_RUNTIME_CHECK((gate_type==tb::kFloat16 || gate_type==tb::kFloat32) && gate.stride(2)==1 &&
  projection(gate,{1,tokens,nv*128}) && projection(alpha,{1,tokens,nv}) && projection(beta,{1,tokens,nv}),
  "GDN window gate projections invalid");
 for(int i=0;i<tokens-confirmed;++i) {
  TYLOINFER_RUNTIME_CHECK(real(checkpoint_conv[i]) && real(checkpoint_state[i]) &&
   checkpoint_conv[i].sizes()==conv.sizes() && checkpoint_state[i].sizes()==state.sizes() &&
   checkpoint_conv[i].data_ptr()!=conv.data_ptr() && checkpoint_state[i].data_ptr()!=state.data_ptr(),
   "GDN window checkpoint storage invalid");
 }
 const MfqCudaGuard guard(device);
 auto output=mode==GdnWindowMode::Commit?tb::Tensor{}:quantize_output?
  tb::empty({int64_t(position_major?(tokens==2?2:4):tokens)*nv*4*int64_t(sizeof(block_q8_1))},qkv.options().dtype(tb::kUInt8)):
  tb::empty({1,tokens,nv*128},qkv.options().dtype(half?tb::kFloat16:tb::kFloat32));
 const bool deferred=mode==GdnWindowMode::RecordUpdates || mode==GdnWindowMode::Snapshots;
 auto attended=deferred?tb::empty({1,tokens,nv*128},state.options()):tb::Tensor{};
 const bool read_only=mode==GdnWindowMode::ReadOnly || mode==GdnWindowMode::RecordUpdates;
 auto next_conv=read_only?tb::Tensor{}:tb::empty_like(conv);GdnWindowSnapshots snapshots;snapshots.confirmed=confirmed;
 auto updates=mode==GdnWindowMode::RecordUpdates?tb::empty({tokens,nv,257},state.options()):tb::Tensor{};
 if(updates.defined())snapshots.updates=updates.data_ptr<float>();
 if(quantize_output && !deferred)snapshots.quantized=static_cast<block_q8_1*>(output.data_ptr());
 snapshots.position_major=position_major;
 for(int i=0;i<tokens-confirmed;++i) {
  snapshots.conv[i]=checkpoint_conv[i].data_ptr<float>();snapshots.state[i]=checkpoint_state[i].data_ptr<float>();
 }
 const auto launch=[&]<GdnWindowMode Mode>(auto g,auto o) {
  using Gate=decltype(g);using Output=decltype(o);
  const auto kernel=gdn_decode_window_kernel<Gate,Output,Mode>;
  static bool reported=false;
  if(!reported && std::getenv("TYLOINFER_GDN_PROBE_RESOURCES")) {
   cudaFuncAttributes a;TYLOINFER_CUDA_CHECK(cudaFuncGetAttributes(&a,kernel));reported=true;
   std::fprintf(stderr,"gdn_register_resources mode=%d registers=%d local=%zu shared=%zu\n",int(Mode),a.numRegs,a.localSizeBytes,a.sharedSizeBytes);
  }
  const dim3 grid(nv,1,(Mode==GdnWindowMode::RecordUpdates || Mode==GdnWindowMode::Snapshots)?4:1);
  kernel<<<grid,256,0,mfq_current_cuda_stream()>>>(qkv.data_ptr(),gate.data_ptr<Gate>(),alpha.data_ptr<Gate>(),beta.data_ptr<Gate>(),
   conv.data_ptr<float>(),state.data_ptr<float>(),weights.data_ptr<float>(),bias.data_ptr<float>(),a_log.data_ptr<float>(),norm.data_ptr<float>(),
   (Mode==GdnWindowMode::RecordUpdates || Mode==GdnWindowMode::Snapshots)?attended.data_ptr<Output>():output.defined() && !quantize_output?output.data_ptr<Output>():nullptr,
   next_conv.defined()?next_conv.data_ptr<float>():nullptr,nk,nv,tokens,weights.dim()==3?0:1,transposed,silu,float(conv_eps),norm_eps,
   qkv.stride(1),gate.stride(1),alpha.stride(1),alpha.stride(2),beta.stride(1),beta.stride(2),snapshots,qkv.scalar_type()==tb::kFloat32,tiled_heads);
 };
 const auto precision=[&]<GdnWindowMode Mode>() {
  if(gate.scalar_type()==tb::kFloat16){if(half)launch.template operator()<Mode>(__half{},__half{});else launch.template operator()<Mode>(__half{},float{});}
  else {if(half)launch.template operator()<Mode>(float{},__half{});else launch.template operator()<Mode>(float{},float{});}
 };
 if(deferred) {
  const auto recorded=[&]<GdnWindowMode Mode>(auto g) {
   using Gate=decltype(g);launch.template operator()<Mode>(g,float{});
   const auto finish=[&](auto half_tag,auto quantized_tag) {
    gdn_finish_recorded_readout<Gate,decltype(half_tag)::value,decltype(quantized_tag)::value><<<dim3(nv,tokens),128,0,mfq_current_cuda_stream()>>>(
     attended.data_ptr<float>(),gate.data_ptr<Gate>(),norm.data_ptr<float>(),output.data_ptr(),nv,gate.stride(1),silu,norm_eps,position_major);
   };
   if(half){if(quantize_output)finish(std::true_type{},std::true_type{});else finish(std::true_type{},std::false_type{});}
   else {if(quantize_output)finish(std::false_type{},std::true_type{});else finish(std::false_type{},std::false_type{});}
  };
  const auto launch_mode=[&]<GdnWindowMode Mode>() {
   if(gate.scalar_type()==tb::kFloat16)recorded.template operator()<Mode>(__half{});
   else recorded.template operator()<Mode>(float{});
  };
  if(mode==GdnWindowMode::Snapshots)launch_mode.template operator()<GdnWindowMode::Snapshots>();
  else launch_mode.template operator()<GdnWindowMode::RecordUpdates>();
 }
 else if(mode==GdnWindowMode::ReadOnly)precision.template operator()<GdnWindowMode::ReadOnly>();
 else if(mode==GdnWindowMode::Commit)precision.template operator()<GdnWindowMode::Commit>();
 else precision.template operator()<GdnWindowMode::Snapshots>();
 TYLOINFER_CUDA_CHECK(cudaGetLastError());
 static std::atomic<unsigned> seen{0};const unsigned bit=1u<<(tokens+(quantize_output?8:0));
 if(!(seen.fetch_or(bit,std::memory_order_relaxed)&bit))std::fprintf(stderr,
  "runtime_mtp_gdn_window enabled=1 tokens=%d key_heads=%d value_heads=%d checkpoints=%d output_q8=%d state_tiles=%d reduction=columns\n",
  tokens,nk,nv,tokens-confirmed,int(quantize_output),(mode==GdnWindowMode::RecordUpdates || mode==GdnWindowMode::Snapshots)?4:1);
 if(updates.defined())return {output,next_conv,state,updates};
 return {output,next_conv,state};
}

std::vector<tb::Tensor> gdn_decode_window_cuda(const tb::Tensor& qkv,const tb::Tensor& gate,
 const tb::Tensor& alpha,const tb::Tensor& beta,const tb::Tensor& conv,const tb::Tensor& state,
 const tb::Tensor& weights,const tb::Tensor& bias,const tb::Tensor& a_log,const tb::Tensor& norm,
 int nk,int nv,int confirmed,double conv_eps,double norm_eps,bool silu,bool half,bool transposed,
 const std::vector<tb::Tensor>& checkpoint_conv,const std::vector<tb::Tensor>& checkpoint_state,bool tiled_heads,bool quantize_output,bool position_major) {
 return gdn_window_impl(qkv,gate,alpha,beta,conv,state,weights,bias,a_log,norm,nk,nv,confirmed,
  conv_eps,norm_eps,silu,half,transposed,checkpoint_conv,checkpoint_state,tiled_heads,GdnWindowMode::Snapshots,quantize_output,position_major);
}
std::vector<tb::Tensor> gdn_replay_window_cuda(const tb::Tensor& qkv,const tb::Tensor& gate,
 const tb::Tensor& alpha,const tb::Tensor& beta,const tb::Tensor& conv,const tb::Tensor& state,
 const tb::Tensor& weights,const tb::Tensor& bias,const tb::Tensor& a_log,const tb::Tensor& norm,
 int nk,int nv,double conv_eps,double norm_eps,bool silu,bool half,bool transposed,bool commit,bool tiled_heads,bool record_updates,bool quantize_output,bool position_major) {
 TYLOINFER_RUNTIME_CHECK(!commit || !record_updates,"GDN update records are produced by read-only verification");
 return gdn_window_impl(qkv,gate,alpha,beta,conv,state,weights,bias,a_log,norm,nk,nv,int(qkv.size(1)),
  conv_eps,norm_eps,silu,half,transposed,{}, {},tiled_heads,
  commit?GdnWindowMode::Commit:record_updates?GdnWindowMode::RecordUpdates:GdnWindowMode::ReadOnly,quantize_output,position_major);
}
