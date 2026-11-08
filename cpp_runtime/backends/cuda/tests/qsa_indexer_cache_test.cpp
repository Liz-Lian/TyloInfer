#include "runtime/qsa_indexer.h"
#include "core/rope.h"
#include "mfq_cuda_context.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <cstring>
namespace tb=mfq_tensor_backend;
using namespace mfq::cuda;
int main()try {
 int devices=0;const auto status=cudaGetDeviceCount(&devices);
 if(status==cudaErrorNoDevice || status==cudaErrorInsufficientDriver || !devices){std::cout<<"SKIP: CUDA device unavailable\n";return 77;}
 TYLOINFER_RUNTIME_CHECK(status==cudaSuccess,"CUDA device query failed");
 const auto f=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
 int cases=0;double affine_sse=0,half_sse=0;
 for(int w:{17,64,128,256})for(int m:{1,2,3,4,7,8,16,39}) {
  std::vector<float> x(2*65*w),q(2*m*3*w);
  for(size_t i=0;i<x.size();++i)x[i]=std::sin(float(i)*.071f)*.45f+std::cos(float(i)*.011f)*.13f;
  for(size_t i=0;i<q.size();++i)q[i]=std::sin(float(i)*.017f)*.09f;
  auto source=tb::tensor(x,f).reshape({2,65,w}),query=tb::tensor(q,f).reshape({2,m,3,w});
  auto codes=qsa_indexer::encode(source);if(codes.scalar_type()!=tb::kInt16 || codes.size(2)!=w+2)throw std::runtime_error("Indexer stored as non-affine values");
  auto decoded=qsa_indexer::decode(codes),cpu=decoded.cpu(),half=source.to(tb::kFloat16).to(tb::kFloat32).cpu();
  for(size_t i=0;i<x.size();++i){double d=cpu.data_ptr<float>()[i]-x[i];affine_sse+=d*d;d=half.data_ptr<float>()[i]-x[i];half_sse+=d*d;}
  auto actual=qsa_indexer::scores(query,codes).cpu();
  for(int b=0;b<2;++b)for(int row=0;row<m;++row)for(int k=0;k<65;++k) {
   double expected=0;
   for(int h=0;h<3;++h){double dot=0;for(int d=0;d<w;++d)dot+=double(q[((b*m+row)*3+h)*w+d])*cpu.data_ptr<float>()[(b*65+k)*w+d];expected+=std::max(0.0,dot);}
   expected/=std::sqrt(double(w));float got=actual.data_ptr<float>()[(b*m+row)*65+k];
   if(!std::isfinite(got) || std::abs(got-expected)>2e-6+std::abs(expected)*4e-5)throw std::runtime_error("INT16 scores differ from independent FP64 reference");
  }
  ++cases;
 }
 if(affine_sse>=half_sse*.05)throw std::runtime_error("Affine encoding accidentally inherited FP16 rounding");
 for(int width:{17,64,128,256}) {
  auto rope=std::make_shared<RotaryEmbedding>(width-width%2,256,10000.);
  std::vector<float> values(52*width),weights(width);for(size_t i=0;i<values.size();++i)values[i]=std::sin(float(i)*.137f)*.4f;for(int i=0;i<width;++i)weights[i]=.01f*std::cos(float(i));
  auto raw=tb::tensor(values,f).reshape({1,52,width});
  auto weight=tb::tensor(weights,f);
  auto encoded=qsa_indexer::encode(raw);
  auto decoded=qsa_indexer::decode(encoded);
  for(int pool:{1,4}) {
   auto pos=tb::arange(52/pool,f.dtype(tb::kInt64))*pool;
   auto expected=qsa_indexer::encode(rope->pool_normalized(decoded,weight,pos,pool,1e-6));
   auto actual=rope->pool_normalized(encoded,weight,pos,pool,1e-6);
   auto a=actual.contiguous().cpu(),e=expected.contiguous().cpu();
   if(std::memcmp(a.data_ptr(),e.data_ptr(),size_t(a.numel()*2)))throw std::runtime_error("Fused INT16 pool differs from FP32 source -> affine reference");
   auto cache=tb::zeros({1,52/pool,width+2},f.dtype(tb::kInt16));
   auto cp=tb::arange(48,52,f.dtype(tb::kInt64)),p=cp;
   if(!rope->pool_normalized_into(encoded,weight,p,cp,cache,pool,52,1e-6))throw std::runtime_error("INT16 live pooling not dispatched");
   auto first=48/pool,count=52/pool-first;
   a=cache.narrow(1,first,count).contiguous().cpu();e=expected.narrow(1,first,count).contiguous().cpu();
   if(std::memcmp(a.data_ptr(),e.data_ptr(),size_t(a.numel()*2)))throw std::runtime_error("Live INT16 pool differs from compact reference");
   ++cases;
  }
 }
 std::cout<<"INT16 default raw/pooled Indexer: "<<cases<<" score/pool cases PASS; affine/FP16 SSE="<<affine_sse/half_sse<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
