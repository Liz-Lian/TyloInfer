#pragma once
#include "../kernels/gguf_input.h"
#include "../ops/gguf/gguf_rotation.h"
#include "nepq.h"

namespace gguf_rotation_test {
inline void bench(int rows,int columns) {
    namespace tb=mfq_tensor_backend;namespace gg=mfq::cuda::gguf;
    if(rows<=0 || columns<=0 || columns%128)throw std::invalid_argument("Hadamard benchmark geometry");
    const auto stream=mfq_current_cuda_stream();
    GgufInputRotation rotation;rotation.block=128;rotation.signs_cpu.resize(columns);
    auto signs=tb::empty({columns},tb::TensorOptions().device(tb::kCPU).dtype(tb::kInt8));
    for(int k=0;k<columns;++k)signs.data_ptr<int8_t>()[k]=rotation.signs_cpu[k]=(k*17+k/7)%3?-1:1;
    rotation.signs_device=signs.to(tb::kCUDA);
    for(auto dtype:{tb::kFloat16,tb::kFloat32})for(bool hidden:{false,true}) {
        auto host=tb::empty({rows,columns},signs.options().dtype(tb::kFloat32));
        for(int64_t i=0;i<host.numel();++i)host.data_ptr<float>()[i]=std::sin(float(i)*.071f)*17.f;
        auto input=host.to(dtype).to(tb::kCUDA),gate=host.to(tb::kCUDA),up=gate*-.25f;
        auto packet=tb::empty({int64_t(rows*gg::q8_row_bytes(columns))},input.options().dtype(tb::kUInt8));
        auto temporary=tb::empty_like(input),rotated=tb::empty_like(input);
        const auto launch=[&](bool fused) {
            if(fused) {
                if(hidden)gg::try_quantize_rotated_moe_hidden(gate.data_ptr<float>(),up.data_ptr<float>(),rows,columns,
                    dtype==tb::kFloat16,.2f,rotation.signs_device.data_ptr<int8_t>(),128,packet.data_ptr(),stream);
                else gg::try_prepare_rotated_moe_input(input.data_ptr(),dtype==tb::kFloat16?gg::InputType::Float16:gg::InputType::Float32,
                    rows,columns,columns,1,rotation.signs_device.data_ptr<int8_t>(),128,packet.data_ptr(),nullptr,stream);
            } else {
                auto source=input;
                if(hidden) {
                    gg::materialize_moe_hidden(gate.data_ptr<float>(),up.data_ptr<float>(),rows,columns,dtype==tb::kFloat16,.2f,temporary.data_ptr(),stream);
                    source=temporary;
                }
                // Retain one preallocated output for an equally static graph.
                if(dtype==tb::kFloat16)mfq::cuda::hadamard::apply_f16(source.data_ptr(),rotation.signs_device.data_ptr<int8_t>(),rotated.data_ptr(),rows,columns,128,false,stream);
                else mfq::cuda::hadamard::apply_f32(source.data_ptr<float>(),rotation.signs_device.data_ptr<int8_t>(),rotated.data_ptr<float>(),rows,columns,128,false,stream);
                if(dtype==tb::kFloat16)gg::quantize_input_f16(rotated.data_ptr(),rows,columns,packet.data_ptr(),stream);
                else gg::quantize_input(rotated.data_ptr<float>(),rows,columns,packet.data_ptr(),stream);
            }
        };
        struct Graph {cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
            ~Graph(){if(exec)cudaGraphExecDestroy(exec);if(graph)cudaGraphDestroy(graph);}} graphs[2];
        for(int mode=0;mode<2;++mode) {
            TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));launch(mode==1);
            TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamEndCapture(stream,&graphs[mode].graph));
            size_t nodes=0;TYLOINFER_NATIVE_CUDA_CHECK(cudaGraphGetNodes(graphs[mode].graph,nullptr,&nodes));
            if(nodes!=size_t(mode==1?1:hidden?3:2))throw std::runtime_error("Hadamard benchmark missed its intended dispatch");
            TYLOINFER_NATIVE_CUDA_CHECK(cudaGraphInstantiate(&graphs[mode].exec,graphs[mode].graph,0));
        }
        for(int i=0;i<256;++i)for(auto& graph:graphs)TYLOINFER_NATIVE_CUDA_CHECK(cudaGraphLaunch(graph.exec,stream));
        cudaEvent_t begin,end;TYLOINFER_NATIVE_CUDA_CHECK(cudaEventCreate(&begin));TYLOINFER_NATIVE_CUDA_CHECK(cudaEventCreate(&end));
        std::vector<float> times[2];
        for(int trial=0;trial<7;++trial)for(int order=0;order<2;++order) {
            const int mode=(trial+order)%2;TYLOINFER_NATIVE_CUDA_CHECK(cudaEventRecord(begin,stream));
            for(int i=0;i<512;++i)TYLOINFER_NATIVE_CUDA_CHECK(cudaGraphLaunch(graphs[mode].exec,stream));
            TYLOINFER_NATIVE_CUDA_CHECK(cudaEventRecord(end,stream));TYLOINFER_NATIVE_CUDA_CHECK(cudaEventSynchronize(end));
            float ms=0;TYLOINFER_NATIVE_CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end));times[mode].push_back(ms*1000/512);
        }
        for(auto& values:times)std::sort(values.begin(),values.end());
        std::printf("hadamard_q8_pair rows=%d K=%d half=%d swiglu=%d staged_us=%.6f fused_us=%.6f ratio=%.6f actual_match=1\n",
            rows,columns,int(dtype==tb::kFloat16),int(hidden),times[0][3],times[1][3],times[1][3]/times[0][3]);
        TYLOINFER_NATIVE_CUDA_CHECK(cudaEventDestroy(begin));TYLOINFER_NATIVE_CUDA_CHECK(cudaEventDestroy(end));
    }
}
inline void check() {
    namespace tb=mfq_tensor_backend;
    namespace gg=mfq::cuda::gguf;
    const auto stream=mfq_current_cuda_stream();
    int can_map=0;TYLOINFER_NATIVE_CUDA_CHECK(cudaDeviceGetAttribute(&can_map,cudaDevAttrCanMapHostMemory,mfq_current_cuda_device()));
    int cases=0;
    for(int rows:{1,4,17})for(int columns:{32,96,128,640})for(int block:{1,2,4,8,16,32,64,128}) {
        if(columns%block)continue;
        for(auto dtype:{tb::kFloat16,tb::kFloat32,tb::kBFloat16})for(int view=0;view<2;++view)for(bool identity:{false,true}) {
            auto host=tb::empty({rows,2*columns+1},tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
            for(int64_t i=0;i<host.numel();++i)host.data_ptr<float>()[i]=i%43==0?0:std::sin(float(i)*.071f)*17.f;
            auto backing=host.to(dtype).to(tb::kCUDA);
            auto input=view?backing.slice(1,1,2*columns+1,2):backing.narrow(1,0,columns);
            GgufInputRotation rotation;rotation.block=block;
            if(!identity) {
                rotation.signs_cpu.resize(columns);
                auto signs=tb::empty({columns},host.options().dtype(tb::kInt8));
                for(int k=0;k<columns;++k)signs.data_ptr<int8_t>()[k]=rotation.signs_cpu[k]=(k*17+k/7)%3?-1:1;
                rotation.signs_device=signs.to(tb::kCUDA);
            }
            const auto values=rotation.apply(input).to(tb::kFloat32);
            const auto bytes=int64_t(rows*gg::q8_row_bytes(columns));
            auto reference=tb::empty({bytes},backing.options().dtype(tb::kUInt8));
            auto actual=tb::empty({bytes+32},reference.options());
            gg::quantize_input(values.data_ptr<float>(),rows,columns,reference.data_ptr(),stream);
            const auto expected=reference.cpu(),expected_values=values.cpu();
            for(int publish=0;publish<=(can_map?1:0);++publish) {
                std::unique_ptr<mfq::cuda::HostBuffer> host_output;
                void* mapped=nullptr;
                if(publish) {
                    host_output=std::make_unique<mfq::cuda::HostBuffer>(rows*columns*sizeof(float)+32,true);
                    std::memset(host_output->data(),0xa5,host_output->size());
                    TYLOINFER_NATIVE_CUDA_CHECK(cudaHostGetDevicePointer(&mapped,host_output->data(),0));
                }
                TYLOINFER_NATIVE_CUDA_CHECK(cudaMemsetAsync(actual.data_ptr(),0xa5,actual.nbytes(),stream));
                const auto type=dtype==tb::kFloat16?gg::InputType::Float16:dtype==tb::kBFloat16?gg::InputType::BFloat16:gg::InputType::Float32;
                if(!gg::try_prepare_rotated_moe_input(input.data_ptr(),type,rows,columns,input.stride(0),input.stride(1),
                        identity?nullptr:rotation.signs_device.data_ptr<int8_t>(),block,actual.data_ptr<uint8_t>()+16,static_cast<float*>(mapped),stream))
                    throw std::runtime_error("Register Hadamard/Q8 was not dispatched");
                const auto result=actual.cpu();
                if(std::memcmp(result.data_ptr<uint8_t>()+16,expected.data_ptr(),bytes))
                    throw std::runtime_error("Fused Hadamard changed native Q8 bytes");
                for(int j=0;j<16;++j)if(result.data_ptr<uint8_t>()[j]!=0xa5 || result.data_ptr<uint8_t>()[bytes+16+j]!=0xa5)
                    throw std::runtime_error("Fused Hadamard overwrote output guards");
                if(publish) {
                    if(std::memcmp(host_output->data(),expected_values.data_ptr(),rows*columns*sizeof(float)))
                        throw std::runtime_error("Fused Hadamard CPU input differs from staged precision");
                    const auto* guard=static_cast<const uint8_t*>(host_output->data())+rows*columns*sizeof(float);
                    for(int j=0;j<32;++j)if(guard[j]!=0xa5)throw std::runtime_error("Fused Hadamard overwrote mapped CPU guards");
                }
                ++cases;
            }
            if(dtype==tb::kBFloat16 || view)continue;
            const auto gate=host.narrow(1,0,columns).contiguous().to(tb::kCUDA),up=(gate*-.25f).contiguous();
            for(float limit:{0.f,1.3f}) {
                auto hidden=tb::empty({rows,columns},gate.options().dtype(dtype));
                gg::materialize_moe_hidden(gate.data_ptr<float>(),up.data_ptr<float>(),rows,columns,dtype==tb::kFloat16,limit,hidden.data_ptr(),stream);
                const auto transformed=rotation.apply(hidden).to(tb::kFloat32);
                gg::quantize_input(transformed.data_ptr<float>(),rows,columns,reference.data_ptr(),stream);
                if(!gg::try_quantize_rotated_moe_hidden(gate.data_ptr<float>(),up.data_ptr<float>(),rows,columns,
                        dtype==tb::kFloat16,limit,identity?nullptr:rotation.signs_device.data_ptr<int8_t>(),block,actual.data_ptr(),stream))
                    throw std::runtime_error("Fused SwiGLU/Hadamard/Q8 was not dispatched");
                const auto a=actual.cpu(),b=reference.cpu();
                if(std::memcmp(a.data_ptr(),b.data_ptr(),bytes))throw std::runtime_error("Fused SwiGLU/Hadamard changed hidden Q8 bytes");
                ++cases;
            }
        }
    }
    std::printf("Fused Hadamard/Q8 and SwiGLU/Hadamard/Q8 blocks1..128, strided inputs, CPU publication and guards cases=%d PASS\n",cases);
    std::fflush(stdout);
}
}
