#pragma once
#include "../kernels/gguf_input.h"
#include "../kernels/gguf.h"
#include <cstring>
#include <iostream>

namespace gguf_input_test {
inline void bench(int rows,int columns,int routes) {
    namespace tb=mfq_tensor_backend;
    using namespace mfq::cuda::gguf;
    if(rows<=0 || columns<=0 || columns%32 || routes<=0)
        throw std::invalid_argument("MoE input benchmark geometry");
    const auto device=mfq_current_cuda_device();
    const auto stream=mfq_get_current_cuda_stream(device).stream();
    const auto options=tb::TensorOptions().device(tb::Device(tb::kCUDA,device));
    auto source=tb::ones({rows,columns},options.dtype(tb::kFloat16));
    auto ids=tb::arange(rows*routes,options.dtype(tb::kInt64));
    auto weights=tb::ones({rows,routes},options.dtype(tb::kFloat32));
    auto old_input=tb::empty({rows,columns},options.dtype(tb::kFloat32));
    auto old_ids=tb::empty_like(ids),old_weights=tb::empty_like(weights);
    auto q8=tb::empty({std::int64_t(rows*q8_row_bytes(columns))},options.dtype(tb::kUInt8));
    const auto input_bytes=rows*columns*sizeof(float),ids_bytes=ids.nbytes();
    mfq::cuda::HostBuffer host(input_bytes+ids_bytes+64,true);
    void* mapped=nullptr;
    TYLOINFER_NATIVE_CUDA_CHECK(cudaHostGetDevicePointer(&mapped,host.data(),0));
    auto* cpu_input=static_cast<float*>(mapped);
    auto* cpu_ids=reinterpret_cast<std::int32_t*>(static_cast<std::uint8_t*>(mapped)+input_bytes);
    auto* ready=reinterpret_cast<std::uint32_t*>(static_cast<std::uint8_t*>(mapped)+input_bytes+ids_bytes);
    struct Graph {
        cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
        ~Graph(){if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);}
    } graph[2];
    TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream));
    for(int mode=0;mode<2;++mode) {
        TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        if(!mode) {
            old_input.copy_(source);old_ids.copy_(ids);old_weights.copy_(weights);
            publish_cpu_input(old_input.data_ptr<float>(),cpu_input,rows*columns,stream);
            mfq::cuda::moe_publish_routes(nullptr,reinterpret_cast<const std::int32_t*>(old_ids.data_ptr()),
                nullptr,cpu_ids,ready,0,rows*routes*2,stream);
        } else {
            prepare_moe_input(source.data_ptr(),InputType::Float16,rows,columns,columns,1,
                q8.data_ptr(),cpu_input,stream);
            mfq::cuda::moe_publish_routes(nullptr,reinterpret_cast<const std::int32_t*>(ids.data_ptr()),
                nullptr,cpu_ids,ready,0,rows*routes*2,stream);
        }
        if(!mode)quantize_input(old_input.data_ptr<float>(),rows,columns,q8.data_ptr(),stream);
        TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamEndCapture(stream,&graph[mode].graph));
        TYLOINFER_NATIVE_CUDA_CHECK(cudaGraphInstantiate(&graph[mode].executable,graph[mode].graph,0));
    }
    for(int i=0;i<512;++i)for(auto& g:graph)
        TYLOINFER_NATIVE_CUDA_CHECK(cudaGraphLaunch(g.executable,stream));
    cudaEvent_t begin,end;
    TYLOINFER_NATIVE_CUDA_CHECK(cudaEventCreate(&begin));
    TYLOINFER_NATIVE_CUDA_CHECK(cudaEventCreate(&end));
    std::vector<float> samples[2];
    for(int trial=0;trial<7;++trial)for(int order=0;order<2;++order) {
        const int mode=(trial+order)%2;
        TYLOINFER_NATIVE_CUDA_CHECK(cudaEventRecord(begin,stream));
        for(int i=0;i<512;++i)TYLOINFER_NATIVE_CUDA_CHECK(cudaGraphLaunch(graph[mode].executable,stream));
        TYLOINFER_NATIVE_CUDA_CHECK(cudaEventRecord(end,stream));
        TYLOINFER_NATIVE_CUDA_CHECK(cudaEventSynchronize(end));
        float ms=0;TYLOINFER_NATIVE_CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end));
        samples[mode].push_back(ms*1000.f/512);
    }
    for(auto& sample:samples)std::sort(sample.begin(),sample.end());
    std::printf("moe_input_pair rows=%d columns=%d routes=%d old_us=%.6f fused_us=%.6f ratio=%.6f\n",
        rows,columns,routes,samples[0][3],samples[1][3],samples[1][3]/samples[0][3]);
    TYLOINFER_NATIVE_CUDA_CHECK(cudaEventDestroy(begin));
    TYLOINFER_NATIVE_CUDA_CHECK(cudaEventDestroy(end));
}
inline void check() {
    namespace tb=mfq_tensor_backend;
    using namespace mfq::cuda::gguf;
    const auto device=mfq_current_cuda_device();
    const auto stream=mfq_get_current_cuda_stream(device).stream();
    int can_map=0;
    TYLOINFER_NATIVE_CUDA_CHECK(cudaDeviceGetAttribute(&can_map,cudaDevAttrCanMapHostMemory,device));
    int cases=0;
    for(int rows:{1,2,4,32})for(int columns:{32,96,288})
    for(auto dtype:{tb::kFloat32,tb::kFloat16,tb::kBFloat16})for(int layout=0;layout<4;++layout) {
        const auto options=tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32);
        const auto height=layout==2?columns:layout==3?1:rows;
        const auto width=layout==2?rows+3:layout==1?2*columns+3:columns;
        auto base=tb::empty({height,width},options);
        for(std::int64_t i=0;i<base.numel();++i)
            base.data_ptr<float>()[i]=i%67==0?0.f:17.f*std::sin(float(i)*.071f);
        base=base.to(dtype).to(tb::Device(tb::kCUDA,device));
        auto source=layout==3?base.expand({rows,columns}):layout==2?base.transpose(0,1).narrow(0,1,rows):
            layout==1?base.slice(1,1,2*columns+1,2):base;
        const auto canonical=source.to(tb::kFloat32).contiguous();
        const auto bytes=rows*q8_row_bytes(columns);
        auto reference=tb::empty({std::int64_t(bytes)},source.options().dtype(tb::kUInt8));
        auto actual=tb::empty({std::int64_t(bytes+32)},reference.options());
        quantize_input(canonical.data_ptr<float>(),rows,columns,reference.data_ptr(),stream);
        const auto type=dtype==tb::kFloat16?InputType::Float16:
            dtype==tb::kBFloat16?InputType::BFloat16:InputType::Float32;
        for(int publish=0;publish<=(can_map?3:0);++publish) {
            std::unique_ptr<mfq::cuda::HostBuffer> cpu;
            void* mapped=nullptr;
            if(publish%2) {
                cpu=std::make_unique<mfq::cuda::HostBuffer>(rows*columns*sizeof(float)+32,true);
                std::memset(cpu->data(),0xa5,cpu->size());
                TYLOINFER_NATIVE_CUDA_CHECK(cudaHostGetDevicePointer(&mapped,cpu->data(),0));
            }
            TYLOINFER_NATIVE_CUDA_CHECK(cudaMemsetAsync(actual.data_ptr(),0xa5,actual.nbytes(),stream));
            std::unique_ptr<mfq::cuda::HostBuffer> route_host;
            tb::Tensor ids_host,ids;
            if(publish>=2) {
                ids_host=tb::empty({rows,3},options.dtype(tb::kInt64));
                for(int i=0;i<rows*3;++i)ids_host.data_ptr<std::int64_t>()[i]=i%2?-i:(std::int64_t(1)<<39)+i;
                ids=ids_host.to(tb::Device(tb::kCUDA,device));
                route_host=std::make_unique<mfq::cuda::HostBuffer>(128+ids_host.nbytes()+32,true);
                std::memset(route_host->data(),0xa5,route_host->size());
                *static_cast<std::uint32_t*>(route_host->data())=0;
                void* route_mapped=nullptr;
                TYLOINFER_NATIVE_CUDA_CHECK(cudaHostGetDevicePointer(&route_mapped,route_host->data(),0));
                prepare_moe_input(source.data_ptr(),type,rows,columns,source.stride(0),source.stride(1),
                    actual.data_ptr<std::uint8_t>()+16,static_cast<float*>(mapped),stream);
                mfq::cuda::moe_publish_routes(nullptr,reinterpret_cast<const std::int32_t*>(ids.data_ptr()),nullptr,
                    reinterpret_cast<std::int32_t*>(static_cast<std::uint8_t*>(route_mapped)+128),
                    static_cast<std::uint32_t*>(route_mapped),0,rows*3*2,stream);
            } else prepare_moe_input(source.data_ptr(),type,rows,columns,source.stride(0),source.stride(1),
                actual.data_ptr<std::uint8_t>()+16,static_cast<float*>(mapped),stream);
            const auto host=actual.cpu(),expected=reference.cpu();
            if(std::memcmp(host.data_ptr<std::uint8_t>()+16,expected.data_ptr(),bytes))
                throw std::runtime_error("GGUF fused input changed Q8 activation bytes");
            for(int i=0;i<16;++i)
                if(host.data_ptr<std::uint8_t>()[i]!=0xa5 || host.data_ptr<std::uint8_t>()[bytes+16+i]!=0xa5)
                    throw std::runtime_error("GGUF fused input wrote outside activation packet");
            if(publish%2) {
                const auto expected_cpu=canonical.cpu();
                if(std::memcmp(cpu->data(),expected_cpu.data_ptr(),expected_cpu.nbytes()))
                    throw std::runtime_error("GGUF fused input changed CPU activations");
                const auto* tail=static_cast<const std::uint8_t*>(cpu->data())+expected_cpu.nbytes();
                for(int i=0;i<32;++i)if(tail[i]!=0xa5)
                    throw std::runtime_error("GGUF fused input wrote outside CPU packet");
            }
            if(publish>=2) {
                const auto* packet=static_cast<const std::uint8_t*>(route_host->data());
                if(*reinterpret_cast<const std::uint32_t*>(packet)!=1 ||
                    std::memcmp(packet+128,ids_host.data_ptr(),ids_host.nbytes()))
                    throw std::runtime_error("GGUF fused input route publication changed IDs/ready flag");
                for(std::size_t i=4;i<128;++i)if(packet[i]!=0xa5)
                    throw std::runtime_error("GGUF fused input route header guard changed");
                for(std::size_t i=128+ids_host.nbytes();i<route_host->size();++i)if(packet[i]!=0xa5)
                    throw std::runtime_error("GGUF fused input wrote outside route packet");
            }
            ++cases;
        }
    }
    std::cout<<"GGUF direct/strided FP32/FP16/BF16 input Q8/CPU bytes/tails PASS cases="<<cases<<'\n';
}
}
