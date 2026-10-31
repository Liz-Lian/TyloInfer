#include "mfq_tensor_backend.h"
#include "mfq_cuda_linear_attention_ops.h"
#include "mfq_cuda_context.h"
#include "runtime/decode_window.h"
#include "models/qwen4_exp/model.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
namespace tb=mfq_tensor_backend;
using tb::Tensor;
using mfq::cuda::DecodeWindow;
using mfq::cuda::GraphWarmupScope;
using mfq::cuda::Graph;
using mfq::cuda::Event;
using mfq::cuda::default_context;
void equal(const Tensor& actual,const Tensor& expected,const char* label,double absolute,double relative) {
    const auto error=(actual-expected.to(actual.device())).abs().max().item<float>();
    const auto bound=absolute+relative*expected.to(actual.device()).abs().max().item<float>();
    if(!std::isfinite(error) || error>bound)throw std::runtime_error(label);
}
Tensor data(std::vector<int64_t> shape,int salt,float scale) {
    int64_t count=1;for(const auto d:shape)count*=d;
    std::vector<float> values(count);
    for(int64_t i=0;i<count;++i)values[i]=std::sin(float(i)*.173f+salt)*scale;
    return tb::tensor(values).reshape(shape).to(tb::kCUDA);
}
#include "gdn_replay_model_test.h"
#include "gdn_replay_default_test.h"
#include "gdn_core_q8_test.h"
#include "gdn_prefill_columns_test.h"
#include "gdn_prefill_pipeline_test.h"
#include "gdn_output_warp_test.h"
void fp32_convolution() {
    for(int batch:{1,2})for(int tokens:{2,17,129})for(int width:{32,128,256})for(int taps:{2,4,8}) {
        const int nk=2,nv=6,channels=(2*nk+nv)*width;
        auto previous=data({batch,taps-1,channels},451,.1f),qkv=data({batch,tokens,channels},452,.2f);
        auto weight=data({channels,1,taps},453,.3f),bias=data({channels},454,.02f);
        auto prepared=linear_conv_qkv_prefill_cuda(previous,qkv.narrow(2,0,2*nk*width).contiguous(),
            qkv.narrow(2,2*nk*width,nv*width).contiguous(),weight,bias,nk,nv,width,width,1e-6);
        auto joined=tb::cat({previous,qkv},1).contiguous();
        auto reference=ssm_conv_silu_cuda(joined,weight,bias,tokens);
        const auto normalized=[&](int offset) {
            auto x=reference.narrow(2,offset,nk*width).reshape({batch,tokens,nk,width}).permute({0,2,1,3});
            return x/tb::clamp_min((x*x).sum(-1,true).sqrt(),1e-6);
        };
        equal(prepared[0],normalized(0),"FP32 fused convolution Q",2e-6,2e-5);
        equal(prepared[1],normalized(nk*width),"FP32 fused convolution K",2e-6,2e-5);
        equal(prepared[2],reference.narrow(2,2*nk*width,nv*width).reshape({batch,tokens,nv,width}).permute({0,2,1,3}),
            "FP32 fused convolution V",2e-6,2e-5);
        equal(prepared[3],joined.narrow(1,tokens,taps-1),"FP32 fused convolution state",0.,0.);
    }
    std::cout<<"GDN FP32 fused convolution Q/K/V/state PASS"<<std::endl;
}
void tiled_core_equivalence() {
    for(int width:{32,64,128})for(bool half:{false,true})for(bool transpose:{false,true}) {
        const int nk=2,nv=6,channels=(2*nk+nv)*width;
        const auto dtype=half?tb::kFloat16:tb::kFloat32;
        std::vector<int64_t> columns;
        for(int part=0;part<2;++part)for(int h=0;h<nv;++h)for(int d=0;d<width;++d)
            columns.push_back((part*nk+h%nk)*width+d);
        for(int d=0;d<nv*width;++d)columns.push_back(2*nk*width+d);
        auto index=tb::tensor(columns,tb::TensorOptions().device(tb::kCPU).dtype(tb::kInt64)).to(tb::kCUDA);
        auto conv=data({1,3,channels},461,.2f),expanded=conv.index_select(2,index).contiguous();
        auto state=data({1,nv,width,width},462,.03f),reference_state=state.clone();
        mfq::cuda::qwen4_exp::Gdn restored({},nk,nv,width,4,1e-6,true,true,transpose,true,true,true,true,true);
        restored.restore_session(expanded,state);
        equal(restored.conv_state(),conv,"legacy tiled GDN prefix convolution restore",0.,0.);
        restored.restore_session(expanded,state,true);
        equal(restored.conv_state(),conv,"legacy tiled GDN prefix restore into graph storage",0.,0.);
        auto weight=data({channels,1,4},463,.3f),expanded_weight=weight.index_select(0,index).contiguous();
        auto bias=data({nv},464,.1f),a=data({nv},465,.1f),norm=data({width},466,.8f);
        auto workspace=tb::zeros({1,nv,width+1},conv.options()),reference_workspace=workspace.clone();
        for(int step=0;step<5;++step) {
            auto qkv=data({1,1,channels},467+step,.3f).to(dtype);
            auto gate=data({1,1,nv*width},468+step,.2f).to(dtype);
            auto alpha=data({1,1,nv},469+step,.2f).to(dtype),beta=data({1,1,nv},470+step,.2f).to(dtype);
            const auto expected=gdn_decode_core_cuda(qkv.index_select(2,index).contiguous(),gate,alpha,beta,
                expanded,reference_state,expanded_weight,bias,a,norm,nv,nv,width,1e-6,1e-6,true,half,transpose,reference_workspace,true);
            const auto actual=gdn_decode_core_cuda(qkv,gate,alpha,beta,conv,state,weight,bias,a,norm,
                nk,nv,width,1e-6,1e-6,true,half,transpose,workspace,true,{},true);
            const auto bits=[](const Tensor& x,const Tensor& y) {
                const auto xc=x.cpu().contiguous(),yc=y.cpu().contiguous();
                if(xc.nbytes()!=yc.nbytes() || std::memcmp(xc.data_ptr(),yc.data_ptr(),xc.nbytes()))
                    throw std::runtime_error("compact tiled GDN differs from expanded reference");
            };
            bits(actual[0],expected[0]);bits(actual[2],expected[2]);
            bits(actual[1].index_select(2,index).contiguous(),expected[1]);
            if(actual[1].data_ptr()!=conv.data_ptr())conv.copy_(actual[1]);
            if(expected[1].data_ptr()!=expanded.data_ptr())expanded.copy_(expected[1]);
        }
    }
    std::cout<<"Compact tiled GDN matches expanded decode/state/conv over repeated steps PASS"<<std::endl;
}
namespace attention_ops=mfq::cuda::attention_ops;
Tensor composed_gdn_output(const Tensor& attended,const Tensor& projection,const Tensor& weight,
    double eps,bool silu_gate,bool output_half) {
    const auto batch=attended.size(0),heads=attended.size(1),tokens=attended.size(2),width=attended.size(3);
    auto z=projection.reshape({batch,tokens,heads,width}).permute({0,2,1,3}).to(tb::kFloat32);
    auto gate=tb::sigmoid(z);if(silu_gate)gate=z*gate;
    auto normalized=attention_ops::rms_norm(attended,weight,eps)*gate;
    return normalized.permute({0,2,1,3}).reshape({batch,tokens,heads*width})
        .to(output_half?tb::kFloat16:tb::kFloat32);
}
void gdn_output_case() {
    int cases=0;
    for(int width:{17,32,64,128,129,320})for(int tokens:{1,3,8,35})
        for(auto gate_dtype:{tb::kFloat16,tb::kFloat32})for(bool silu:{false,true})for(bool half:{false,true}) {
        const int batch=tokens==3?2:1,heads=48;
        auto attended=data({batch,heads,tokens,width},132,.17f).to(tb::kFloat32);
        auto projection=data({batch,tokens,heads*width},133,88.f).to(gate_dtype);
        auto weight=data({width},134,1.3f).to(tb::kFloat32);
        const double eps=half?1e-6:0.013456789123;
        auto output=gdn_rms_norm_gate_cuda(attended,projection,weight,eps,silu,half);
        equal(output,composed_gdn_output(attended,projection,weight,eps,silu,half),"GDN fused RMSNorm/gate/layout",half?1e-5f:2e-6f,half?1e-3f:2e-6f);
        if(output.scalar_type()!=(half?tb::kFloat16:tb::kFloat32) || !output.is_contiguous())
            throw std::runtime_error("GDN fused output dtype/layout changed");
        ++cases;
    }
    for(int tokens:{1,3,8})for(auto dtype:{tb::kFloat16,tb::kFloat32}) {
        auto attended=data({1,48,tokens,128},141,.00013f).to(tb::kFloat32);
        auto projection=data({1,tokens,48*128},142,14.f).to(dtype);
        auto weight=data({128},143,1.1f).to(tb::kFloat32);
        Tensor output;DecodeWindow window(mfq_current_cuda_stream());
        window.capture([&]{output=gdn_rms_norm_gate_cuda(attended,projection,weight,1e-6,true,true);},[]{},[&]{output={};});
        for(int replay=0;replay<3;++replay) {
            attended.copy_(replay==1?tb::zeros_like(attended):data(attended.sizes().vec(),144+replay,replay==2?.17f:.00013f).to(tb::kFloat32));
            projection.copy_(data(projection.sizes().vec(),147+replay,88.f).to(dtype));
            window.run();equal(output,composed_gdn_output(attended,projection,weight,1e-6,true,true),
                "GDN changing-input fused output graph",1e-5f,1e-3f);++cases;
        }
    }
    std::cout<<"GDN fused output FP32 reference cases="<<cases<<" PASS\n";
}
void packed_convolution() {
    for(int batch:{1,2})for(int tokens:{2,17,129})for(int width:{32,128,256})
    for(auto dtype:{tb::kFloat16,tb::kFloat32})for(auto gate_type:{tb::kFloat16,tb::kFloat32}) {
        const int nk=2,nv=6,taps=4,channels=(2*nk+nv)*width;
        auto previous=data({batch,taps-1,channels},471,.1f);
        auto projected=data({batch,tokens,channels+17},472,.2f).to(dtype).narrow(2,0,channels);
        auto weight=data({channels,1,taps},473,.3f),dt=data({nv},474,.1f),a=data({nv},475,.1f);
        auto alpha=data({batch,tokens,nv+7},476,.2f).to(gate_type).narrow(2,0,nv);
        auto beta=data({batch,tokens,nv+7},477,.2f).to(gate_type).narrow(2,0,nv);
        const auto expected=linear_conv_qkv_prefill_cuda(previous,projected.narrow(2,0,2*nk*width).contiguous(),
            projected.narrow(2,2*nk*width,nv*width).contiguous(),weight,tb::empty({0},previous.options()),nk,nv,width,width,1e-6);
        const auto gates=linear_gate_beta_cuda(alpha.to(tb::kFloat32).contiguous(),beta.to(tb::kFloat32).contiguous(),dt,a,true);
        const auto actual=linear_conv_qkv_gate_prefill_cuda(previous,projected,weight,alpha,beta,dt,a,nk,nv,width,1e-6);
        for(int i=0;i<4;++i)equal(actual[i],expected[i],"packed prefill convolution/QK/state",2e-6,2e-5);
        for(int i=0;i<2;++i)equal(actual[i+4],gates[i],"packed prefill decay/beta",2e-6,2e-5);
    }
    std::cout<<"Packed GDN prefill f16/f32 + strided QKV/gates PASS"<<std::endl;
}
int main(int argc,char** argv) {
    int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess || count==0)return 77;
    try {
        const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
        const auto stream=mfq_current_cuda_stream();context->begin_graph_pool(stream);
        struct ReleasePool {
            std::shared_ptr<mfq::cuda::Context> context;cudaStream_t stream;
            ~ReleasePool(){context->end_graph_pool(stream);}
        } pool{context,stream};
        if(argc==2 && std::string(argv[1])=="--gdn-replay-check"){gdn_replay_checkpoint_case();return 0;}
        if(argc==2 && std::string(argv[1])=="--gdn-output-warp-check"){gdn_output_warp_case(false);return 0;}
        if(argc==2 && std::string(argv[1])=="--gdn-keyhead-check"){gdn_prefill_keyhead_case(false);return 0;}
        if(argc==2 && std::string(argv[1])=="--gdn-keyhead-focused-check"){gdn_prefill_keyhead_case(false,true);return 0;}
        if(argc==2 && std::string(argv[1])=="--gdn-output-warp-bench"){gdn_output_warp_case(true);return 0;}
        if(argc==2 && std::string(argv[1])=="--gdn-prepared-replay-check"){gdn_prepared_replay_default_case();return 0;}
        if(argc==2 && std::string(argv[1])=="--gdn-core-q8-check"){gdn_core_q8_case();return 0;}
        if(argc==3 && std::string(argv[1])=="--gdn-core-q8-bench"){gdn_core_q8_benchmark(std::stoi(argv[2]));return 0;}
        if(argc==3 && std::string(argv[1])=="--gdn-replay-batch-bench"){gdn_replay_batch_benchmark(std::stoi(argv[2]));return 0;}
        if(argc==3 && std::string(argv[1])=="--gdn-recorded-update-bench"){gdn_recorded_update_benchmark(std::stoi(argv[2]));return 0;}
        gdn_output_case();packed_convolution();gdn_core_q8_case();
        tiled_core_equivalence();
        fp32_convolution();
        gdn_prefill_keyhead_case(argc==2 && std::string(argv[1])=="--bench");
        if(argc==1) {gdn_prefill_staged_case(false);gdn_prefill_columns_case();}
        return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
