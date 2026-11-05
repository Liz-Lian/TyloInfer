#include "mfq_tensor_backend.h"
#include "mfq_cuda_linear_attention_ops.h"
#include "mfq_cuda_gdn_replay_ops.h"
#include "mfq_cuda_context.h"
#include "../kernels/gguf.h"
#include <vector>
#include <cmath>
#include <cstring>
#include <iostream>
#include <algorithm>
namespace tb=mfq_tensor_backend;
using tb::Tensor;
Tensor data(std::vector<int64_t> shape,int salt,float scale,tb::ScalarType dtype=tb::kFloat32) {
 int64_t n=1;for(auto d:shape)n*=d;std::vector<float> values(n);
 for(int64_t i=0;i<n;++i)values[i]=std::sin(float(i)*.173f+salt)*scale;
 return tb::tensor(values).reshape(shape).to(tb::kCUDA).to(dtype);
}
void exact(const Tensor& a,const Tensor& b,const char* label) {
 auto x=a.contiguous().cpu(),y=b.contiguous().cpu();
 if(x.scalar_type()!=y.scalar_type() || x.sizes()!=y.sizes())throw std::runtime_error(std::string(label)+" shape/dtype differs");
 if(std::memcmp(x.data_ptr(),y.data_ptr(),x.nbytes())) {
  auto xf=x.to(tb::kFloat32),yf=y.to(tb::kFloat32);int64_t first=0;double peak=0;
  for(int64_t i=0;i<xf.numel();++i)if(xf.data_ptr<float>()[i]!=yf.data_ptr<float>()[i]) {
   if(peak==0)first=i;peak=std::max(peak,std::abs(double(xf.data_ptr<float>()[i])-yf.data_ptr<float>()[i]));
  }
  throw std::runtime_error(std::string(label)+" bytes differ first="+std::to_string(first)+" max="+std::to_string(peak));
 }
}
int main(int argc,char** argv)try {
 const bool benchmark=argc==2 && std::string(argv[1])=="--bench";
 const bool replay_benchmark=argc==2 && std::string(argv[1])=="--replay-bench";
 const bool snapshot_benchmark=argc==2 && std::string(argv[1])=="--snapshot-bench";
 const bool recorded_benchmark=argc==2 && std::string(argv[1])=="--recorded-bench";
 const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
 const auto stream=mfq_current_cuda_stream();
 int cases=0,replays=0;
 for(int tokens:{2,3,4,5,6,7,8})for(int combination=0;combination<64;++combination) {
  const bool tiled=combination&32;
  const int nk=combination&1?16:2,nv=nk*(combination&2?3:1),channels=(2*nk+nv)*128;
  const auto gate_type=combination&4?tb::kFloat32:tb::kFloat16;
  const bool output_half=!(combination&8),transposed=!(combination&8),silu=combination&2;
  const int confirmed=combination&8?2:1;
  const double eps=combination&8?.013456789123:1e-6;
  auto qkv=data({1,tokens,channels+17},151,.4f,combination&16?tb::kFloat32:tb::kFloat16).narrow(2,0,channels);
  auto gate=data({1,tokens,nv*128+7},152,88,gate_type).narrow(2,0,nv*128);
  const float gate_amplitude=gate_type==tb::kFloat32?88.f:4.f;
  auto alpha=data({1,tokens,nv*2},153,gate_amplitude,gate_type).slice(2,0,nv*2,2);
  auto beta=data({1,tokens,nv*2},154,gate_amplitude,gate_type).slice(2,0,nv*2,2);
  auto initial_conv=data({1,3,channels},155,.3f),initial_state=data({1,nv,128,128},156,.17f);
  auto weight=data(combination&8?std::vector<int64_t>{4,channels}:std::vector<int64_t>{channels,1,4},157,.19f);
  auto bias=data({nv},158,1.3f),a=data({nv},159,.3f),norm=data({128},160,1.3f);
  Tensor conv[2]={initial_conv.clone(),initial_conv.clone()},state[2]={initial_state.clone(),initial_state.clone()};
  auto workspace=tb::zeros({1,nv,129},conv[0].options());
  std::vector<Tensor> cc[2],cs[2];
  for(int kind=0;kind<2;++kind)for(int i=0;i<tokens-confirmed;++i){cc[kind].push_back(tb::empty_like(conv[kind]));cs[kind].push_back(tb::empty_like(state[kind]));}
  const auto old=[&] {
   std::vector<Tensor> outputs;
   for(int t=0;t<tokens;++t) {
    const int cp=t+1-confirmed;
    auto result=cp>=0 && t+1<tokens?
     gdn_decode_core_checkpoint_cuda(qkv.narrow(1,t,1),gate.narrow(1,t,1),alpha.narrow(1,t,1),beta.narrow(1,t,1),
      conv[0],state[0],weight,bias,a,norm,nk,nv,128,1e-6,eps,silu,output_half,transposed,workspace,true,cc[0][cp],cs[0][cp],tiled):
     gdn_decode_core_cuda(qkv.narrow(1,t,1),gate.narrow(1,t,1),alpha.narrow(1,t,1),beta.narrow(1,t,1),
      conv[0],state[0],weight,bias,a,norm,nk,nv,128,1e-6,eps,silu,output_half,transposed,workspace,true,{},tiled);
    if(result[1].data_ptr()!=conv[0].data_ptr())conv[0].copy_(result[1]);outputs.push_back(result[0]);
   }
   return tb::cat(outputs,1);
  };
  const auto candidate=[&](bool quantize=false,bool position_major=false) {
   auto result=gdn_decode_window_cuda(qkv,gate,alpha,beta,conv[1],state[1],weight,bias,a,norm,
    nk,nv,confirmed,1e-6,eps,silu,output_half,transposed,cc[1],cs[1],tiled,quantize,position_major);
   conv[1].copy_(result[1]);return result[0];
  };
  const auto reset=[&](int k){conv[k].copy_(initial_conv);state[k].copy_(initial_state);};
  const auto check=[&](const Tensor& actual,const Tensor& expected,bool position_major=false) {
   if(actual.scalar_type()==tb::kUInt8) {
    auto reference=tb::empty({int64_t(position_major?mfq::cuda::gguf::interleaved_activation_bytes(tokens,nv*128)
     :tokens*mfq::cuda::gguf::q8_row_bytes(nv*128))},expected.options().dtype(tb::kUInt8));
    if(position_major)mfq::cuda::gguf::quantize_input_interleaved(expected.data_ptr(),output_half,tokens,nv*128,reference.data_ptr(),nullptr,stream);
    else if(output_half)mfq::cuda::gguf::quantize_input_f16(expected.data_ptr(),tokens,nv*128,reference.data_ptr(),stream);
    else mfq::cuda::gguf::quantize_input(expected.data_ptr<float>(),tokens,nv*128,reference.data_ptr(),stream);
    exact(actual,reference,"prepared snapshot output");
   } else exact(actual,expected,"output");exact(state[1],state[0],"state");exact(conv[1],conv[0],"conv");
   for(int i=0;i<tokens-confirmed;++i){exact(cc[1][i],cc[0][i],"checkpoint conv");exact(cs[1][i],cs[0][i],"checkpoint state");}
  };
  auto expected=old(),actual=candidate();check(actual,expected);++cases;
  const auto saved_conv=initial_conv.clone(),saved_state=initial_state.clone();
  Tensor verification_updates;
  for(bool quantized:{false,true})for(bool position_major:{false,true}) {
   if(position_major && !quantized)continue;
   auto separate=gdn_verify_window_cuda(qkv,gate,alpha,beta,initial_conv,initial_state,weight,bias,a,norm,
    nk,nv,1e-6,eps,silu,output_half,transposed,tiled,quantized,position_major);
   check(separate[0],expected,position_major);
   exact(separate[1],conv[0],"separate final convolution");
   exact(separate[2],state[0],"separate final recurrence");
   exact(initial_conv,saved_conv,"verification changed initial convolution");
   exact(initial_state,saved_state,"verification changed initial recurrence");
   if(!verification_updates.defined())verification_updates=separate[3];
   else exact(separate[3],verification_updates,"output packet changed verification updates");
   ++cases;
  }
  auto replay_conv=initial_conv.clone(),replay_state=initial_state.clone();
  if(combination&8) {
   auto storage=tb::empty({initial_state.numel()+1},initial_state.options());
   replay_state=storage.narrow(0,1,initial_state.numel()).reshape(initial_state.sizes());replay_state.copy_(initial_state);
  }
  const auto read_window=[&] {
   return gdn_replay_window_cuda(qkv,gate,alpha,beta,replay_conv,replay_state,weight,bias,a,norm,
    nk,nv,1e-6,eps,silu,output_half,transposed,false,tiled);
  };
  auto read=read_window();exact(read[0],expected,"read-only output");
  exact(replay_conv,initial_conv,"read-only convolution");exact(replay_state,initial_state,"read-only state");
  auto recorded=gdn_replay_window_cuda(qkv,gate,alpha,beta,replay_conv,replay_state,weight,bias,a,norm,
   nk,nv,1e-6,eps,silu,output_half,transposed,false,tiled,true);
  exact(verification_updates,recorded[3],"fused verification update records differ");
  exact(recorded[0],read[0],"recorded-update output");
  auto prepared=gdn_replay_window_cuda(qkv,gate,alpha,beta,replay_conv,replay_state,weight,bias,a,norm,
   nk,nv,1e-6,eps,silu,output_half,transposed,false,tiled,true,true);
  auto quantized=tb::empty({int64_t(tokens*mfq::cuda::gguf::q8_row_bytes(nv*128))},initial_state.options().dtype(tb::kUInt8));
  if(output_half)mfq::cuda::gguf::quantize_input_f16(read[0].data_ptr(),tokens,nv*128,quantized.data_ptr(),stream);
  else mfq::cuda::gguf::quantize_input(read[0].data_ptr<float>(),tokens,nv*128,quantized.data_ptr(),stream);
  reset(1);check(candidate(true),expected);
  if(tokens<=8) {
   reset(1);check(candidate(true,true),expected,true);
   auto interleaved=gdn_replay_window_cuda(qkv,gate,alpha,beta,replay_conv,replay_state,weight,bias,a,norm,
    nk,nv,1e-6,eps,silu,output_half,transposed,false,tiled,true,true,true);
   auto reference=tb::empty({int64_t(mfq::cuda::gguf::interleaved_activation_bytes(tokens,nv*128))},quantized.options());
   mfq::cuda::gguf::quantize_input_interleaved(read[0].data_ptr(),output_half,tokens,nv*128,reference.data_ptr(),nullptr,stream);
   exact(interleaved[0],reference,"fused GDN position-major output packet");
   exact(interleaved[3],recorded[3],"position-major output changed update records");
  }
  exact(prepared[0],quantized,"fused GDN output Q8 packet");exact(prepared[3],recorded[3],"fused GDN output update records");
  exact(replay_conv,initial_conv,"recorded-update convolution");exact(replay_state,initial_state,"recorded-update state");
  auto next_conv=tb::empty_like(replay_conv);
  GdnReplayLayer update_layer{qkv.data_ptr(),alpha.data_ptr(),beta.data_ptr(),replay_conv.data_ptr<float>(),replay_state.data_ptr<float>(),
   next_conv.data_ptr<float>(),weight.data_ptr<float>(),bias.data_ptr<float>(),a.data_ptr<float>(),
   qkv.stride(1),alpha.stride(1),alpha.stride(2),beta.stride(1),beta.stride(2),nk,nv,channels,tokens,weight.dim()==3?0:1,
   int(transposed),int(tiled),int(qkv.scalar_type()==tb::kFloat32),int(gate_type==tb::kFloat32),1e-6f,recorded[3].data_ptr<float>()};
  auto host_update=tb::empty({int64_t(sizeof(update_layer))},initial_state.options().device(tb::kCPU).dtype(tb::kUInt8));
  std::memcpy(host_update.data_ptr(),&update_layer,sizeof(update_layer));auto update_descriptor=host_update.to(tb::kCUDA);
  const auto commit_prefix=[&](int keep) {
   auto result=gdn_replay_window_cuda(qkv.narrow(1,0,keep),gate.narrow(1,0,keep),alpha.narrow(1,0,keep),beta.narrow(1,0,keep),
    replay_conv,replay_state,weight,bias,a,norm,nk,nv,1e-6,eps,silu,output_half,transposed,true,tiled);
   replay_conv.copy_(result[1]);
  };
  for(int keep=1;keep<=tokens;++keep) {
   replay_conv.copy_(initial_conv);replay_state.copy_(initial_state);commit_prefix(keep);
   const auto expected_conv=keep==tokens?conv[0]:keep>=confirmed?cc[0][keep-confirmed]:Tensor{};
   const auto expected_state=keep==tokens?state[0]:keep>=confirmed?cs[0][keep-confirmed]:Tensor{};
   if(expected_conv.defined()){exact(replay_conv,expected_conv,"replayed prefix convolution");exact(replay_state,expected_state,"replayed prefix state");}
   auto legacy_conv=replay_conv.clone(),legacy_state=replay_state.clone();
   replay_conv.copy_(initial_conv);replay_state.copy_(initial_state);
   gdn_replay_commit_layers_cuda(update_descriptor,1,nv,channels,keep,true);
   exact(replay_conv,legacy_conv,"recorded prefix convolution");exact(replay_state,legacy_state,"recorded prefix state");
   if(!try_gdn_replay_updates_cuda(qkv,verification_updates,initial_conv,initial_state,
       next_conv,replay_state,keep,transposed))throw std::runtime_error("recorded prefix disjoint dispatch failed");
   exact(next_conv,legacy_conv,"one-kernel disjoint prefix convolution");
   exact(replay_state,legacy_state,"one-kernel disjoint prefix state");
   replay_conv.copy_(initial_conv);replay_state.copy_(initial_state);
   if(!try_gdn_replay_updates_cuda(qkv,verification_updates,replay_conv,replay_state,
       replay_conv,replay_state,keep,transposed))throw std::runtime_error("recorded prefix inplace dispatch failed");
   exact(replay_conv,legacy_conv,"one-kernel inplace prefix convolution");
   exact(replay_state,legacy_state,"one-kernel inplace prefix state");
   exact(initial_conv,saved_conv,"prefix replay changed immutable convolution");
   exact(initial_state,saved_state,"prefix replay changed immutable state");
   cases+=2;
  }
  if(recorded_benchmark && tokens<=4 && (combination==3 || combination==35)) {
   mfq::cuda::Graph graph;std::vector<Tensor> owned;
   const auto execute=[&]{owned=gdn_replay_window_cuda(qkv,gate,alpha,beta,replay_conv,replay_state,weight,bias,a,norm,
    nk,nv,1e-6,eps,silu,output_half,transposed,false,tiled,true,true,true);};
   graph.prepare_memory();execute();mfq_cuda_synchronize();owned.clear();
   graph.capture_begin();execute();graph.capture_end();
   std::vector<float> samples(5);
   for(auto& sample:samples) {
    cudaEvent_t begin,end;TYLOINFER_CUDA_CHECK(cudaEventCreate(&begin));TYLOINFER_CUDA_CHECK(cudaEventCreate(&end));
    TYLOINFER_CUDA_CHECK(cudaEventRecord(begin,stream));for(int i=0;i<100;++i)graph.replay();
    TYLOINFER_CUDA_CHECK(cudaEventRecord(end,stream));TYLOINFER_CUDA_CHECK(cudaEventSynchronize(end));
    TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&sample,begin,end));sample*=10;
    cudaEventDestroy(begin);cudaEventDestroy(end);
   }
   std::sort(samples.begin(),samples.end());
   std::cout<<"gdn_recorded_window tokens="<<tokens<<" tiled="<<tiled<<" heads="<<nv
    <<" verify_us="<<samples[2]<<std::endl;
  }
  if((replay_benchmark || snapshot_benchmark) && tokens<=4 && (combination==3 || combination==35))for(int keep=1;keep<=tokens;++keep) {
   mfq::cuda::Graph graph;std::vector<Tensor> owned;
   const auto execute=[&]{
    if(snapshot_benchmark) {
     owned={candidate()};
     if(keep<tokens){conv[1].copy_(cc[1][keep-confirmed]);state[1].copy_(cs[1][keep-confirmed]);}
    }else {owned=read_window();commit_prefix(keep);}
   };
   graph.prepare_memory();execute();mfq_cuda_synchronize();owned.clear();
   replay_conv.copy_(initial_conv);replay_state.copy_(initial_state);reset(1);
   graph.capture_begin();execute();graph.capture_end();
   std::vector<float> samples(5);
   for(auto& sample:samples) {
    cudaEvent_t begin,end;TYLOINFER_CUDA_CHECK(cudaEventCreate(&begin));TYLOINFER_CUDA_CHECK(cudaEventCreate(&end));
    TYLOINFER_CUDA_CHECK(cudaEventRecord(begin,stream));for(int i=0;i<100;++i)graph.replay();
    TYLOINFER_CUDA_CHECK(cudaEventRecord(end,stream));TYLOINFER_CUDA_CHECK(cudaEventSynchronize(end));
    TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&sample,begin,end));sample*=10;
    cudaEventDestroy(begin);cudaEventDestroy(end);
   }
   std::sort(samples.begin(),samples.end());
   std::cout<<(snapshot_benchmark?"gdn_snapshot_window":"gdn_replay_window")<<" tokens="<<tokens<<" keep="<<keep<<" tiled="<<tiled<<" heads="<<nv
    <<" read_and_commit_us="<<samples[2]<<" avoided_state_bytes="<<(tokens-1)*initial_state.nbytes()<<std::endl;
  }
  if(combination==3 || combination==35) {
   {
   mfq::cuda::Graph rank_graph;std::vector<Tensor> rank_owned;
   auto rank_conv=tb::empty_like(initial_conv),rank_state=tb::empty_like(initial_state);
   const auto rank_execute=[&] {
    rank_owned=gdn_verify_window_cuda(qkv,gate,alpha,beta,initial_conv,initial_state,weight,bias,a,norm,
      nk,nv,1e-6,eps,silu,output_half,transposed,tiled);
    if(!try_gdn_replay_updates_cuda(qkv,rank_owned[3],initial_conv,initial_state,
       rank_conv,rank_state,tokens-1,transposed))throw std::runtime_error("rank-update Graph dispatch failed");
   };
   rank_graph.prepare_memory();rank_execute();mfq_cuda_synchronize();rank_owned.clear();
   rank_graph.capture_begin();rank_execute();rank_graph.capture_end();
   auto rank_base=qkv.clone();
   for(float amplitude:{.3f,.8f,1.1f}) {
    qkv.copy_(rank_base*amplitude);rank_graph.replay();
    auto reference_conv=initial_conv.clone(),reference_state=initial_state.clone();
    const auto reference=gdn_replay_window_cuda(qkv.narrow(1,0,tokens-1),gate.narrow(1,0,tokens-1),
      alpha.narrow(1,0,tokens-1),beta.narrow(1,0,tokens-1),reference_conv,reference_state,
      weight,bias,a,norm,nk,nv,1e-6,eps,silu,output_half,transposed,true,tiled);
    exact(rank_conv,reference[1],"changed-input rank Graph convolution");
    exact(rank_state,reference_state,"changed-input rank Graph state");++replays;
   }
   qkv.copy_(rank_base);
   }
   mfq::cuda::Graph oldgraph,newgraph;Tensor oldout,newout;
   oldgraph.prepare_memory();oldout=old();mfq_cuda_synchronize();oldout={};reset(0);
   oldgraph.capture_begin();oldout=old();oldgraph.capture_end();
   {
    mfq::cuda::GraphWarmupScope warm(context,stream);
    newout=candidate(true,tokens<=4);mfq_cuda_synchronize();newout={};reset(1);
   }
   newgraph.capture_begin_shared();newout=candidate(true,tokens<=4);newgraph.capture_end();
   auto base=qkv.clone();reset(0);reset(1);
   for(float amplitude:{.3f,.8f,1.1f}) {
    qkv.copy_(base*amplitude);oldgraph.replay();newgraph.replay();check(newout,oldout,tokens<=4);++replays;
   }
   if(benchmark) {
   std::vector<float> times[2];
   for(int sample=0;sample<5;++sample)for(int order=0;order<2;++order) {
    const int kind=(sample+order)%2;reset(kind);mfq_cuda_synchronize();
    cudaEvent_t start,end;TYLOINFER_CUDA_CHECK(cudaEventCreate(&start));TYLOINFER_CUDA_CHECK(cudaEventCreate(&end));
    TYLOINFER_CUDA_CHECK(cudaEventRecord(start,stream));
    for(int i=0;i<100;++i)(kind?newgraph:oldgraph).replay();
    TYLOINFER_CUDA_CHECK(cudaEventRecord(end,stream));TYLOINFER_CUDA_CHECK(cudaEventSynchronize(end));
    float ms;TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&ms,start,end));times[kind].push_back(ms*10);
    TYLOINFER_CUDA_CHECK(cudaEventDestroy(start));TYLOINFER_CUDA_CHECK(cudaEventDestroy(end));
   }
   for(auto& v:times)std::sort(v.begin(),v.end());
   std::cout<<"gdn_register_window tokens="<<tokens<<" heads="<<nv<<" old_us="<<times[0][2]<<" new_us="<<times[1][2]<<std::endl;
   }
  }
 }
 {
  constexpr int tokens=3,nv=2,channels=768;
  auto qkv=data({1,tokens,channels},201,.2f),updates=data({tokens,nv,257},202,.1f);
  auto conv=data({1,3,channels},203,.2f),state=data({1,nv,128,128},204,.1f);
  auto output_conv=conv.clone(),output_state=state.clone();
  const auto rejected=[&](const Tensor& input,const Tensor& records,const Tensor& ic,const Tensor& is,
      Tensor& oc,Tensor& os,int keep) {
   auto before_c=oc.clone(),before_s=os.clone();
   if(try_gdn_replay_updates_cuda(input,records,ic,is,oc,os,keep,true))
    throw std::runtime_error("invalid rank-update input accepted");
   exact(oc,before_c,"rejected rank input changed convolution");
   exact(os,before_s,"rejected rank input changed state");++cases;
  };
  rejected(qkv,updates,conv,state,output_conv,output_state,0);
  rejected(qkv,updates,conv,state,output_conv,output_state,tokens+1);
  rejected(qkv,updates.narrow(2,0,256),conv,state,output_conv,output_state,1);
  rejected(qkv.cpu(),updates,conv,state,output_conv,output_state,1);
  auto ss=tb::empty({state.numel()+1},state.options());
  auto si=ss.narrow(0,0,state.numel()).reshape(state.sizes());
  auto so=ss.narrow(0,1,state.numel()).reshape(state.sizes());
  si.copy_(state);rejected(qkv,updates,conv,si,output_conv,so,1);
  auto cs=tb::empty({conv.numel()+1},conv.options());
  auto ci=cs.narrow(0,0,conv.numel()).reshape(conv.sizes());
  auto co=cs.narrow(0,1,conv.numel()).reshape(conv.sizes());
  ci.copy_(conv);rejected(qkv,updates,ci,state,co,output_state,1);
  auto shared=tb::empty({state.numel()+qkv.numel()+updates.numel()},state.options());
  auto shared_state=shared.narrow(0,0,state.numel()).reshape(state.sizes());
  auto shared_qkv=shared.narrow(0,0,qkv.numel()).reshape(qkv.sizes());
  rejected(shared_qkv,updates,conv,state,output_conv,shared_state,1);
  auto shared_updates=shared.narrow(0,0,updates.numel()).reshape(updates.sizes());
  rejected(qkv,shared_updates,conv,state,output_conv,shared_state,1);
 }
 mfq_cuda_synchronize();
 std::cout<<"gdn_register_window exact_cases="<<cases<<" changing_graph="<<replays<<" PASS\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
