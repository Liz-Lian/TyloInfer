#include "gguf_weight.h"
#include "quant_linear.h"
#include "../kernels/gguf.h"
#include "mfq_cuda_context.h"
#include "mfq_cuda_norm_ops.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace tb=mfq_tensor_backend;
namespace gg=mfq::cuda::gguf;
static void require(bool condition,const char* message) {if(!condition)throw std::runtime_error(message);}
static void same(const tb::Tensor& a,const tb::Tensor& b,const char* message) {
    auto x=a.cpu(),y=b.cpu();
    require(x.sizes()==y.sizes() && x.scalar_type()==y.scalar_type() && x.nbytes()==y.nbytes(),message);
    require(std::memcmp(x.data_ptr(),y.data_ptr(),x.nbytes())==0,message);
}
static tb::Tensor values(std::vector<int64_t> shape,tb::ScalarType dtype,float phase=0) {
    auto host=tb::empty(shape,tb::TensorOptions{}.device(tb::kCPU).dtype(tb::kFloat32));
    for(int64_t i=0;i<host.numel();++i)host.data_ptr<float>()[i]=std::sin(float(i)*.127f+phase)*1.7f;
    return host.to(dtype).to(tb::kCUDA);
}
static tb::Tensor gated(const tb::Tensor& up,const tb::Tensor& gate,bool silu) {
    return up*(silu?tb::silu(gate):tb::sigmoid(gate));
}
static double measure(const std::function<tb::Tensor()>& fn) {
    mfq::cuda::Graph graph;graph.prepare_memory();{auto warm=fn();}
    graph.capture_begin();for(int i=0;i<16;++i){auto output=fn();}graph.capture_end();
    auto stream=mfq_get_current_cuda_stream(0).stream();
    for(int i=0;i<32;++i)graph.replay();
    cudaEvent_t begin,end;TYLOINFER_CUDA_CHECK(cudaEventCreate(&begin));TYLOINFER_CUDA_CHECK(cudaEventCreate(&end));
    std::vector<double> samples;
    for(int i=0;i<3;++i) {
        TYLOINFER_CUDA_CHECK(cudaEventRecord(begin,stream));for(int n=0;n<32;++n)graph.replay();
        TYLOINFER_CUDA_CHECK(cudaEventRecord(end,stream));TYLOINFER_CUDA_CHECK(cudaEventSynchronize(end));
        float ms=0;TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end));samples.push_back(ms*1000./512);
    }
    TYLOINFER_CUDA_CHECK(cudaEventDestroy(begin));TYLOINFER_CUDA_CHECK(cudaEventDestroy(end));
    std::sort(samples.begin(),samples.end());return samples[1];
}
#include "gguf_small_window_rms_test.h"
int main(int argc,char** argv)try {
    int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess || !count)return 77;
    const MfqCudaGuard guard(0);const auto stream=mfq_get_current_cuda_stream(0);
    if(argc>1 && std::string_view(argv[1])=="--rms-only") {
        small_window_rms_checks();TYLOINFER_CUDA_CHECK(cudaDeviceSynchronize());ggml_quantize_free();return 0;
    }
    const bool benchmark=argc>1 && std::string_view(argv[1])=="--bench";
    int cases=0;
    for(int tokens:{2,3,4})for(int width:{32,256,512,17408})for(auto dtype:{tb::kFloat16,tb::kFloat32})for(bool silu:{false,true}) {
        auto up=values({tokens,width},dtype),gate=values({tokens,width},dtype,.7f);
        auto old_up=up.cpu(),old_gate=gate.cpu();
        auto materialized=gated(up,gate,silu);
        auto expected=tb::empty({int64_t(gg::interleaved_activation_bytes(tokens,width))},up.options().dtype(tb::kUInt8));
        auto storage=tb::empty({expected.numel()+32},expected.options());storage.fill_(165);
        auto actual=storage.narrow(0,16,expected.numel());
        gg::quantize_input_interleaved(materialized.data_ptr(),dtype==tb::kFloat16,tokens,width,expected.data_ptr(),nullptr,stream.stream());
        gg::quantize_gated_input_interleaved(up.data_ptr(),gate.data_ptr(),dtype==tb::kFloat16,tokens,width,silu,actual.data_ptr(),stream.stream());
        same(actual,expected,"interleaved gate differs from separate activation/multiply/quantization");
        same(up,old_up,"interleaved gate modified its up input");same(gate,old_gate,"interleaved gate modified its gate input");
        auto host=storage.cpu();for(int i=0;i<16;++i)require(host.data_ptr<uint8_t>()[i]==165 &&
            host.data_ptr<uint8_t>()[host.numel()-16+i]==165,"interleaved packing overwrote its bounds");
        ++cases;
    }
    for(auto geometry:{std::pair{1,2},std::pair{1,3},std::pair{1,4},std::pair{2,1},std::pair{2,2},std::pair{3,1},std::pair{4,1}})
    for(int width:{32,128,160})for(bool silu:{false,true}) {
        const auto [batch,tokens]=geometry;const int heads=8,columns=heads*width,rows=batch*tokens;
        for(auto dtype:{tb::kFloat16,tb::kFloat32}) {
            auto x=values({batch,tokens,columns},dtype);
            auto head_gate=values({batch,tokens,heads,width+7},dtype,.6f).narrow(3,0,width);
            auto flat_gate=head_gate.contiguous().reshape_as(x);auto materialized=gated(x,flat_gate,silu);
            auto expected=tb::empty({int64_t(gg::interleaved_activation_bytes(rows,columns))},x.options().dtype(tb::kUInt8));
            auto actual=tb::empty_like(expected);
            gg::quantize_input_interleaved(materialized.data_ptr(),dtype==tb::kFloat16,rows,columns,expected.data_ptr(),nullptr,stream.stream());
            gg::quantize_gated_head_input_interleaved(x.data_ptr(),head_gate.data_ptr(),dtype==tb::kFloat16,
                batch,tokens,heads,width,head_gate.stride(0),head_gate.stride(1),head_gate.stride(2),silu,actual.data_ptr(),stream.stream());
            same(actual,expected,"strided head gate changed its position-major packet");++cases;
        }
        auto attended=values({batch,heads,tokens,width},tb::kFloat32),norm=values({width},tb::kFloat32,.8f);
        auto gate=values({batch,tokens,columns+13},tb::kFloat32,.4f).narrow(2,0,columns);
        auto normalized=rms_norm_cuda(attended.reshape({-1,width}),norm,1e-5)
            .reshape_as(attended).transpose(1,2).contiguous().reshape({batch,tokens,columns});
        auto materialized=gated(normalized,gate.contiguous(),silu);
        auto expected=tb::empty({int64_t(gg::interleaved_activation_bytes(rows,columns))},attended.options().dtype(tb::kUInt8));
        auto actual=tb::empty_like(expected);
        gg::quantize_input_interleaved(materialized.data_ptr(),false,rows,columns,expected.data_ptr(),nullptr,stream.stream());
        gg::quantize_rms_gated_input_interleaved(attended.data_ptr<float>(),gate.data_ptr<float>(),norm.data_ptr<float>(),
            batch,heads,tokens,width,gate.stride(0),gate.stride(1),1e-5,silu,actual.data_ptr(),stream.stream());
        same(actual,expected,"grouped RMS/gate changed its position-major packet");++cases;
    }
    CudaExecutionContext execution;int projections=0;
    for(auto type:{GGML_TYPE_IQ3_XXS,GGML_TYPE_IQ3_S,GGML_TYPE_IQ4_XS}) {
        const int rows=36,columns=512,heads=4,width=columns/heads;
        auto host=values({rows,columns},tb::kFloat32).cpu();auto importance=tb::ones({columns},host.options());
        const auto row_bytes=ggml_row_size(type,columns);
        auto bytes=tb::empty({int64_t(rows*row_bytes)},host.options().dtype(tb::kUInt8));
        require(ggml_quantize_chunk(type,host.data_ptr<float>(),bytes.data_ptr(),0,rows,columns,
            importance.data_ptr<float>())==bytes.nbytes(),"synthetic GGUF quantization failed");
        QuantLinear linear;linear.kind=QuantLinearKind::Gguf;
        linear.gguf={bytes.to(tb::kCUDA),type,rows,columns,uint64_t(row_bytes)};linear.gguf.repack_inplace();
        for(int tokens:{1,2,3,4,5,8,32})for(auto dtype:{tb::kFloat16,tb::kFloat32})for(bool silu:{false,true}) {
            auto up=values({1,tokens,columns},dtype),gate=values({1,tokens,columns},dtype,.5f);
            const auto expected=linear.forward(execution,gated(up,gate,silu));
            same(linear.forward_input_mul(execution,up,gate,silu?2:1),expected,"GGUF gated projection bypassed or changed the fused path");
            auto head_gate=gate.reshape({1,tokens,heads,width});
            same(linear.forward_input_head_mul(execution,up,head_gate,silu?2:1),expected,"GGUF head-gated projection changed output");
            auto attended=up.to(tb::kFloat32).reshape({1,tokens,heads,width}).transpose(1,2).contiguous();
            auto norm=values({width},tb::kFloat32,.8f),fg=gate.to(tb::kFloat32);
            auto normalized=rms_norm_cuda(attended.reshape({-1,width}),norm,1e-5)
                .reshape_as(attended).transpose(1,2).contiguous().reshape({1,tokens,columns});
            const auto reference=linear.forward(execution,gated(normalized,fg,silu));
            same(linear.forward_input_rms_gate(execution,attended,fg,norm,1e-5,silu),reference,"GGUF grouped RMS/gate projection changed output");
            projections+=3;
            if(benchmark && tokens>=2 && tokens<=4 && dtype==tb::kFloat16 && silu) {
                double old_us=measure([&]{return linear.forward(execution,gated(up,gate,true));});
                double new_us=measure([&]{return linear.forward_input_mul(execution,up,gate,2);});
                std::cout<<nlohmann::json{{"format",ggml_type_name(type)},{"M",tokens},{"rows",rows},{"K",columns},
                    {"ordinary_us",old_us},{"fused_us",new_us},{"ratio",new_us/old_us},{"scope","gating, Q8 packing and projection"}}.dump()<<std::endl;
            }
            if(tokens==3 && dtype==tb::kFloat16 && silu) {
                {
                mfq::cuda::Graph graph;graph.prepare_memory();{auto warm=linear.forward_input_mul(execution,up,gate,2);}
                graph.capture_begin();auto output=linear.forward_input_mul(execution,up,gate,2);graph.capture_end();
                for(int iteration=0;iteration<3;++iteration) {
                    up.copy_(values({1,tokens,columns},dtype,float(iteration)*.3f));graph.replay();
                    same(output,linear.forward(execution,gated(up,gate,true)),"changing-input Graph replay changed fused projection");
                }
                }
                mfq::cuda::Graph rms_graph;rms_graph.prepare_memory();
                {auto warm=linear.forward_input_rms_gate(execution,attended,fg,norm,1e-5,true);}
                rms_graph.capture_begin();auto rms_output=linear.forward_input_rms_gate(execution,attended,fg,norm,1e-5,true);rms_graph.capture_end();
                for(int iteration=0;iteration<3;++iteration) {
                    attended.copy_(values({1,heads,tokens,width},tb::kFloat32,float(iteration)*.4f));rms_graph.replay();
                    auto normalized_reference=rms_norm_cuda(attended.reshape({-1,width}),norm,1e-5)
                        .reshape_as(attended).transpose(1,2).contiguous().reshape({1,tokens,columns});
                    same(rms_output,linear.forward(execution,gated(normalized_reference,fg,true)),"changing-input RMS/gate Graph replay changed projection");
                }
            }
        }
    }
    bool invalid=false;auto x=values({1,32},tb::kFloat32),packet=tb::empty({1024},x.options().dtype(tb::kUInt8));
    try{gg::quantize_gated_input_interleaved(x.data_ptr(),x.data_ptr(),false,1,32,true,packet.data_ptr(),stream.stream());}
    catch(const std::invalid_argument&){invalid=true;}require(invalid,"invalid interleaved token count accepted");
    small_window_rms_checks();TYLOINFER_CUDA_CHECK(cudaDeviceSynchronize());ggml_quantize_free();
    std::cout<<"Small-window gated packing PASS cases="<<cases<<" exact_projections="<<projections
        <<"; immutable inputs, guard bytes, padded strides, batched rows and changing-input Graph replay PASS"<<std::endl;
    return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<std::endl;return 1;}
