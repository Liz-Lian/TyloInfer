
#include "decode_window.h"
#include "mfq_cuda_context.h"
#include "mfq_tensor_backend.h"
#include <iostream>
#include <stdexcept>
namespace tb=mfq_tensor_backend;
int main()try{
 int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices){std::cout<<"CUDA unavailable\n";return 77;}
 tb::NoGradGuard no_grad;auto stream=mfq_current_cuda_stream();
 auto context=mfq::cuda::default_context(mfq_current_cuda_device());context->begin_graph_pool(stream);
 std::shared_ptr<void> pool_owner(nullptr,[context,stream](void*){context->end_graph_pool(stream);});
 auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
 auto value=tb::zeros({1},options),two=tb::full({1},2,options),four=tb::full({1},4,options),eight=tb::full({1},8,options);
 {
  mfq::cuda::DecodeWindow window(stream);int checks=0;
  auto body=[&]{value.fill_(1);window.host_barrier([&]{value.add_(two);});
   window.host_barrier([&]{if(value.item<float>()!=3)throw std::runtime_error("Empty middle segment lost preceding host GPU work");++checks;});
   value.add_(four);window.host_barrier([&]{value.add_(eight);});};
  window.capture(body,[&]{value.fill_(0);});
  if(window.segments()!=4 || !window.nodes())throw std::runtime_error("Wrong segmented graph topology");
  for(int i=0;i<8;++i){window.run();if(value.item<float>()!=15)throw std::runtime_error("Trailing segment failed to fence host GPU work");}
  if(checks!=8)throw std::runtime_error("Host barrier order changed");
 }
 {
  mfq::cuda::DecodeWindow window(stream);
  auto body=[&]{window.host_barrier([&]{value.fill_(5);});value.add_(two);window.host_barrier([&]{if(value.item<float>()!=7)throw std::runtime_error("Leading empty segment lost host work");});};
  window.capture(body,[&]{value.fill_(0);});window.run();
 }
 {
  mfq::cuda::GraphRegistry graphs(stream);std::string error;
  if(graphs.record(1,1,[]{},error) || error.find("produced no work")==std::string::npos)throw std::runtime_error("Ordinary empty capture was not rejected");
 }
 {
  mfq::cuda::DecodeWindow window(stream);bool rejected=false;
  try{window.capture([&]{window.host_barrier([]{});},[]{});}catch(const std::exception&e){rejected=std::string(e.what()).find("produced no work")!=std::string::npos;}
  if(!rejected)throw std::runtime_error("Entirely empty decode window was not rejected");
 }
 std::cout<<"Leading, consecutive, trailing empty segments; async host GPU fences; repeated replay; ordinary/all-empty capture rejection PASS\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}
