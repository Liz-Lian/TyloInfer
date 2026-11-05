#include "runtime/gdn_recorded_replay_batch.h"
#include "runtime/decode_window.h"
#include "runtime/graph_pool_lease.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
namespace tb=mfq_tensor_backend;using Batch=mfq::cuda::RecordedGdnReplayBatch;
static tb::Tensor data(std::vector<int64_t> shape,float phase,float scale) {
 auto a=tb::empty(shape,tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
 for(int64_t i=0;i<a.numel();++i)a.data_ptr<float>()[i]=scale*std::sin(float(i)*.037f+phase);
 return a.to(tb::kCUDA);
}
static void exact(const tb::Tensor& a,const tb::Tensor& b,const char* label) {
 auto x=a.cpu().contiguous(),y=b.cpu().contiguous();
 if(x.sizes()!=y.sizes()||x.scalar_type()!=y.scalar_type()||std::memcmp(x.data_ptr(),y.data_ptr(),x.nbytes()))throw std::runtime_error(label);
}
static void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Fixture {
 std::vector<Batch::Layer> layers;
 std::vector<tb::Tensor> initial,reference_state,reference_conv;
 int tokens;
 Fixture(int n,int t,bool uniform=false):tokens(t) {
  for(int i=0;i<n;++i) {
   const int nk=uniform?16:(i%3+1),nv=uniform?48:2*nk,channels=(2*nk+nv)*128;
   auto qkv=data({1,t,channels+(uniform?0:7)},.3f+i,.4f);
   if(i%2)qkv=qkv.to(tb::kFloat16);
   qkv=qkv.narrow(2,0,channels);
   auto updates=data({t,nv,257},.17f+i,.02f);
   auto host=updates.cpu();for(int x=0;x<t*nv;++x)host.data_ptr<float>()[x*257+256]=.98f;updates.copy_(host);
   auto conv=data({1,3,channels},.1f+i,.2f),state=data({1,nv,128,128},.2f+i,.01f);
   initial.push_back(state.clone());reference_state.push_back(state.clone());reference_conv.push_back(tb::empty_like(conv));
   if(!uniform && i==1) {auto offset=tb::empty({state.numel()+1},state.options()).narrow(0,1,state.numel()).reshape_as(state);offset.copy_(state);state=std::move(offset);}
   layers.push_back({qkv,updates,conv,tb::empty_like(conv),state,nk,bool(i%2),bool((i/2)%2)});
  }
 }
 void reset(){for(size_t i=0;i<layers.size();++i){layers[i].recurrent.copy_(initial[i]);reference_state[i].copy_(initial[i]);layers[i].convolution.fill_(-.37f);reference_conv[i].fill_(.91f);}}
 void reference(int keep) {
  for(size_t i=0;i<layers.size();++i) {
   auto& x=layers[i];if(!try_gdn_replay_updates_cuda(x.qkv,x.updates,x.initial_convolution,reference_state[i],reference_conv[i],reference_state[i],keep,x.transposed))throw std::runtime_error("Single layer reference unavailable");
  }
 }
 void verify(){for(size_t i=0;i<layers.size();++i){exact(layers[i].convolution,reference_conv[i],"Batch convolution differs");exact(layers[i].recurrent,reference_state[i],"Batch recurrence differs");}}
};
int main(int argc,char** argv)try {
 int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||!devices)return 77;
 const auto stream=mfq_current_cuda_stream();auto pool=mfq::cuda::GraphPoolLease::acquire(mfq::cuda::default_context(mfq_current_cuda_device()),stream);tb::NoGradGuard guard;
 if(argc==2 && std::string(argv[1])=="--copy-batch-override-check") {
  const auto* option=std::getenv("TYLOINFER_RECURRENT_COPY_BATCH");
  if(!option || option[0]!='0')throw std::runtime_error("Copy batch override test requires disabled batching");
  Fixture f(3,4);std::vector<tb::Tensor> scratch;
  auto batch=Batch::create(f.layers,scratch);if(!batch)throw std::runtime_error("Valid override batch rejected");
  f.reset();f.reference(3);batch->run(3);f.verify();
  cudaGraph_t captured=nullptr;check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
  batch->run(2);check(cudaStreamEndCapture(stream,&captured));
  size_t count=0;check(cudaGraphGetNodes(captured,nullptr,&count));
  std::vector<cudaGraphNode_t> nodes(count);check(cudaGraphGetNodes(captured,nodes.data(),&count));
  int copies=0,kernels=0;
  for(auto node:nodes) {
   cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));
   copies+=type==cudaGraphNodeTypeMemcpy;kernels+=type==cudaGraphNodeTypeKernel;
  }
  check(cudaGraphDestroy(captured));
  if(copies!=int(f.layers.size()) || kernels!=2)throw std::runtime_error("Copy batch override did not select individual copies");
  mfq::cuda::DecodeWindow graph(stream);graph.capture([&]{batch->run(2);},[]{});
  for(auto& layer:f.layers)layer.qkv.copy_(layer.qkv*-.25f);
  f.reset();f.reference(2);graph.run();f.verify();
  std::cout<<"Recorded GDN copy batch override PASS exact_states=12 changed_graphs=1\n";return 0;
 }
 int cases=0,graph_cases=0,rejections=0;
 for(int tokens:{1,2,3,4,5,6,7,8}) {
  Fixture f(3,tokens);std::vector<tb::Tensor> scratch{f.layers.front().convolution}; // Must replace an aliased workspace.
  auto batch=Batch::create(f.layers,scratch);if(!batch)throw std::runtime_error("Valid mixed batch rejected");
  if(scratch.front().data_ptr()==f.layers.front().convolution.data_ptr())throw std::runtime_error("Aliased scratch reused");
  std::vector<tb::Tensor> immutable;for(const auto& x:f.layers)immutable.push_back(x.initial_convolution.clone());
  for(int keep=1;keep<=tokens;++keep){f.reset();f.reference(keep);batch->run(keep);f.verify();for(size_t i=0;i<f.layers.size();++i)exact(f.layers[i].initial_convolution,immutable[i],"Initial convolution changed");cases+=int(f.layers.size())*2;}
  mfq::cuda::DecodeWindow graph(stream);graph.capture([&]{batch->run(std::max(1,tokens-1));},[]{});
  for(float factor:{.75f,-.25f,0.f}){for(auto& x:f.layers){x.qkv.copy_(x.qkv*factor);x.updates.copy_(x.updates*factor);}f.reset();f.reference(std::max(1,tokens-1));graph.run();f.verify();++graph_cases;}
  // One shared output cannot belong to two layers in a parallel commit.
  auto invalid=f.layers;invalid.push_back(invalid.front());auto before=f.layers.front().recurrent.clone();
  if(Batch::create(invalid,scratch))throw std::runtime_error("Overlapping outputs accepted");exact(before,f.layers.front().recurrent,"Rejected batch changed state");++rejections;
  invalid=f.layers;invalid[0].initial_convolution=invalid[0].recurrent.narrow(3,0,1);if(Batch::create(invalid,scratch))throw std::runtime_error("Invalid initial geometry accepted");++rejections;
  for(int bad:{0,tokens+1}){bool rejected=false;try{batch->run(bad);}catch(const std::exception&){rejected=true;}if(!rejected)throw std::runtime_error("Invalid prefix accepted");++rejections;}
 }
 std::vector<tb::Tensor> shared;
 Fixture small(3,4),large(3,8,true);
 auto small_batch=Batch::create(small.layers,shared);auto large_batch=Batch::create(large.layers,shared);
 if(!small_batch||!large_batch)throw std::runtime_error("Shared workspace batch rejected");
 for(int keep:{1,3,4}){small.reset();small.reference(keep);small_batch->run(keep);small.verify();cases+=6;}
 for(int keep:{1,7,8}){large.reset();large.reference(keep);large_batch->run(keep);large.verify();cases+=6;}
 mfq::cuda::DecodeWindow lifetime_graph(stream);lifetime_graph.capture([&]{small_batch->run(3);},[]{});
 large_batch.reset();shared.clear();
 for(float scale:{.5f,-.5f}){for(auto& x:small.layers)x.qkv.copy_(x.qkv*scale);small.reset();small.reference(3);lifetime_graph.run();small.verify();++graph_cases;}
 Fixture alias(1,3);alias.reset();alias.layers[0].convolution.copy_(alias.layers[0].initial_convolution);
 alias.layers[0].initial_convolution=alias.layers[0].convolution;std::vector<tb::Tensor> alias_work;
 auto alias_batch=Batch::create(alias.layers,alias_work);if(!alias_batch)throw std::runtime_error("Exact conv alias rejected");alias.reference(2);alias_batch->run(2);alias.verify();cases+=2;
 auto partial=alias.layers;auto blob=tb::empty({partial[0].convolution.numel()+1},partial[0].convolution.options());
 partial[0].initial_convolution=blob.narrow(0,0,blob.numel()-1).reshape_as(partial[0].convolution);
 partial[0].convolution=blob.narrow(0,1,blob.numel()-1).reshape_as(partial[0].convolution);
 if(Batch::create(partial,alias_work))throw std::runtime_error("Partial conv alias accepted");++rejections;
 if(Batch::create({},alias_work))throw std::runtime_error("Empty batch accepted");++rejections;
 std::cout<<"Recorded multilayer GDN replay PASS exact_states="<<cases<<" changed_graphs="<<graph_cases<<" invalid_cases="<<rejections<<'\n';return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
