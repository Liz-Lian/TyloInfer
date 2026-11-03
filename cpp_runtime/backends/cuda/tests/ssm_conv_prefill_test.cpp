
#include "mfq_tensor_backend.h"
#include "mfq_cuda_context.h"
#include "decode_window.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
namespace tb=mfq_tensor_backend;using Json=nlohmann::json;using Tensor=tb::Tensor;
Tensor ssm_conv_silu_cuda(Tensor,Tensor,Tensor,int64_t);

Tensor data(std::vector<int64_t> shape,unsigned seed){auto x=tb::empty(shape,tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));for(int64_t i=0;i<x.numel();++i){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;x.data_ptr<float>()[i]=(int(seed&65535)-32768)*(1.f/131072);}return x;}
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void same(const Tensor&a,const Tensor&b){auto x=a.cpu().contiguous(),y=b.cpu().contiguous();if(x.nbytes()!=y.nbytes()||std::memcmp(x.data_ptr(),y.data_ptr(),x.nbytes()))throw std::runtime_error("Tiled convolution changed FP32 output bits");}
int main()try{
 int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices){std::cout<<"CUDA unavailable\n";return 77;}
 auto test_stream=mfq::cuda::stream_from_pool(false);mfq::cuda::StreamGuard stream_guard(test_stream);
 tb::NoGradGuard guard;Json result={{"oracle_cases",0}};int cases=0;
 for(int B:{1,2})for(int C:{37,513})for(int T:{1,2,4,5,17})for(int K:{1,4,7})for(int layout:{0,1,2})for(bool bias:{false,true}){
  auto hx=data({B,T+K-1,C},19),hw=data(layout==0?std::vector<int64_t>{C,1,K}:layout==1?std::vector<int64_t>{C,K}:std::vector<int64_t>{K,C},71),hb=data({bias?C:0},29);
  auto x=hx.to(tb::kCUDA),w=hw.to(tb::kCUDA),b=hb.to(tb::kCUDA);auto y=ssm_conv_silu_cuda(x,w,b,T);
  auto cpu=y.cpu();float worst=0;for(int batch=0;batch<B;++batch)for(int t=0;t<T;++t)for(int c=0;c<C;++c){float sum=0;for(int j=0;j<K;++j)sum=std::fma(hx.data_ptr<float>()[((batch*(T+K-1)+t+j)*C+c)],hw.data_ptr<float>()[layout==2?j*C+c:c*K+j],sum);if(bias)sum+=hb.data_ptr<float>()[c];float ref=sum/(1.f+std::exp(-sum));worst=std::max(worst,std::abs(cpu.data_ptr<float>()[((batch*T+t)*C+c)]-ref));}if(worst>3.e-7f)throw std::runtime_error("Independent convolution oracle mismatch");++cases;
 }
 result["oracle_cases"]=cases;
 {auto stream=mfq_current_cuda_stream();auto context=mfq::cuda::default_context(mfq_current_cuda_device());context->begin_graph_pool(stream);std::shared_ptr<void> pool(nullptr,[context,stream](void*){context->end_graph_pool(stream);});
 auto x=data({1,20,513},31).to(tb::kCUDA),w=data({513,1,4},37).to(tb::kCUDA),b=data({0},0).to(tb::kCUDA);Tensor y;mfq::cuda::DecodeWindow window(stream);window.capture([&]{y=ssm_conv_silu_cuda(x,w,b,17);},[]{},[&]{y={};});window.run();same(y,ssm_conv_silu_cuda(x,w,b,17));x.fill_(.125);window.run();same(y,ssm_conv_silu_cuda(x,w,b,17));result["changing_graph_exact"]=true;}
 {const int B=65537;auto hx=data({B,4,1},53),hw=data({1,1,4},59);auto x=hx.to(tb::kCUDA),w=hw.to(tb::kCUDA),b=data({0},0).to(tb::kCUDA);auto y=ssm_conv_silu_cuda(x,w,b,1).cpu();
 for(int batch=0;batch<B;++batch){float sum=0;for(int j=0;j<4;++j)sum=std::fma(hx.data_ptr<float>()[batch*4+j],hw.data_ptr<float>()[j],sum);float ref=sum/(1.f+std::exp(-sum));if(std::abs(y.data_ptr<float>()[batch]-ref)>3e-7f)throw std::runtime_error("Grid-stride batch/tail oracle mismatch");}result["grid_stride_oracle"]=true;}
 std::cout<<result.dump()<<std::endl;return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}
