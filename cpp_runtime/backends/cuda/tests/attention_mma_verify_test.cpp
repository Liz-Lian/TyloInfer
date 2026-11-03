#include "mfq_cuda_attention_ops.h"
#include "attention_mma_verify.h"
#include "mfq_cuda_context.h"
#include "runtime/decode_window.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace tb=mfq_tensor_backend;
static double close(const tb::Tensor& actual,const tb::Tensor& reference) {
    auto a=actual.to(tb::kCPU,tb::kFloat32).contiguous(),b=reference.to(tb::kCPU,tb::kFloat32).contiguous();
    if(a.sizes()!=b.sizes())throw std::runtime_error("verification attention shape changed");
    double worst=0;
    for(int64_t i=0;i<a.numel();++i) {
        const double error=std::abs(double(a.data_ptr<float>()[i])-b.data_ptr<float>()[i]);
        if(!std::isfinite(error))throw std::runtime_error("nonfinite verification attention");
        worst=std::max(worst,error);
    }
    if(worst>5e-4)throw std::runtime_error("verification attention differs from causal reference: "+std::to_string(worst));
    return worst;
}
int main()try {
    std::cout.setf(std::ios::unitbuf);
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices)return 77;
    auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    const auto stream=mfq_current_cuda_stream();context->begin_graph_pool(stream);
    const auto cpu=tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32);
    int cases=0,active=0;double worst=0;
    const auto run=[&](int width,int ratio,int tokens,int capacity,bool dynamic,bool poison_tail=false) {
        constexpr int batch=2,kv=2;const int heads=ratio*kv;
        auto hq=tb::empty({batch,heads,tokens,width},cpu);
        auto hk=tb::empty({batch,kv,capacity,width},cpu),hv=tb::empty_like(hk);
        for(int64_t i=0;i<hq.numel();++i)hq.data_ptr<float>()[i]=.1f*std::sin(float(i)*.317f);
        for(int64_t i=0;i<hk.numel();++i) {
            hk.data_ptr<float>()[i]=.1f*std::sin(float(i)*.119f);
            const int token=int(i/width%capacity);
            hv.data_ptr<float>()[i]=(.8f+.2f*std::cos(float(i)*.071f))*((token%64)==63?4.f:.1f);
            const int batch_row=int(i/(int64_t(kv)*capacity*width));
            if(poison_tail && token>=64+batch_row)
                hk.data_ptr<float>()[i]=hv.data_ptr<float>()[i]=std::numeric_limits<float>::quiet_NaN();
        }
        auto q=hq.to(tb::kCUDA,tb::kFloat16),k=hk.to(tb::kCUDA,tb::kFloat16),v=hv.to(tb::kCUDA,tb::kFloat16);
        auto lens=tb::tensor(std::vector<int64_t>{64,65},cpu.dtype(tb::kInt64)).to(tb::kCUDA);
        const double scale=1./std::sqrt(double(width));
        auto direct=try_attention_mma_verify_cuda(q,k,v,lens,scale,capacity);
        if(direct.defined())++active;
        tb::Tensor output;
        const auto body=[&]{output=attention_cache_swa_planned_cuda(q,k,v,lens,scale,capacity,capacity);};
        body();auto reference=attention_cache_swa_cuda(q,k,v,lens,scale,capacity);
        worst=std::max(worst,close(output,reference));++cases;
        if(dynamic) {
            mfq::cuda::DecodeWindow graph(stream);graph.capture(body,[]{},[&]{output={};});
            for(const auto& lengths:std::vector<std::vector<int64_t>>{{0,1},{tokens-1,tokens},{32,33},{64,65},{65,64},{127,128},{129,capacity}}) {
                lens.copy_(tb::tensor(lengths,cpu.dtype(tb::kInt64)).to(tb::kCUDA));
                graph.run();reference=attention_cache_swa_cuda(q,k,v,lens,scale,capacity);
                worst=std::max(worst,close(output,reference));++cases;
            }
        }
        std::cout<<"cached verification attention PASS width="<<width<<" ratio="<<ratio<<" M="<<tokens<<" dynamic="<<dynamic<<'\n';
    };
    for(int tokens:{2,3,4,5,6,7,8})run(256,6,tokens,192,true);
    for(int width:{64,128,256,512})for(int ratio:{1,3,8,12})for(int tokens:{3,8})run(width,ratio,tokens,192,false);
    run(256,6,4,192,false,true);
    context->end_graph_pool(stream);
    std::cout<<"cached verification attention checks PASS cases="<<cases<<" active_mma_geometries="<<active<<" max_error="<<worst<<'\n';
    return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
