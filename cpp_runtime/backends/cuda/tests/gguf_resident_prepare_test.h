#pragma once
#include "gguf_input_test.h"
#include "../kernels/gguf_route_groups.h"

namespace gguf_resident_prepare_test {
inline void run(int slots,int experts,int tokens,bool bench) {
    namespace tb=mfq_tensor_backend;
    using namespace mfq::cuda::gguf;
    if(slots<0 || experts<=0 || tokens<=0)throw std::invalid_argument("resident preparation test geometry");
    const auto device=mfq_current_cuda_device();
    const auto stream=mfq_get_current_cuda_stream(device).stream();
    const auto options=tb::TensorOptions().device(tb::Device(tb::kCUDA,device)).dtype(tb::kInt64);
    auto host_ids=tb::empty({slots},options.device(tb::kCPU));
    for(int i=0;i<slots;++i)host_ids.data_ptr<std::int64_t>()[i]=bench?i%experts:i%7==0?-1:(i*7+i/3)%(experts+2);
    auto ids=host_ids.to(tb::Device(tb::kCUDA,device));
    const auto table_bytes=3*experts*sizeof(std::uint64_t);
    mfq::cuda::HostBuffer host(table_bytes+64,true);
    void* mapped=nullptr;
    TYLOINFER_NATIVE_CUDA_CHECK(cudaHostGetDevicePointer(&mapped,host.data(),0));
    auto* pointers=static_cast<std::uint64_t*>(host.data());
    for(int i=0;i<3*experts;++i)pointers[i]=(i%5)?0x123400000000ull+i*256:0;
    auto* admission=reinterpret_cast<std::uint32_t*>(static_cast<std::uint8_t*>(mapped)+table_bytes);
    *reinterpret_cast<std::uint32_t*>(static_cast<std::uint8_t*>(host.data())+table_bytes)=1;
    auto* clock=reinterpret_cast<std::uint64_t*>(static_cast<std::uint8_t*>(mapped)+table_bytes+8);
    auto addresses=tb::zeros({3*experts},options);
    const int capacity=route_group_capacity(tokens);
    auto groups=tokens>1?tb::empty({std::int64_t((slots+1)*sizeof(RouteGroup))},options.dtype(tb::kUInt8)):tb::Tensor{};
    auto* grouped=groups.defined()?static_cast<RouteGroup*>(groups.data_ptr()):nullptr;
    const auto launch=[&](bool fused) {
        if(fused) {
            if(!try_prepare_resident_routes(static_cast<const std::uint64_t*>(mapped),
                reinterpret_cast<std::uint64_t*>(addresses.data_ptr()),ids.data_ptr<std::int64_t>(),slots,experts,
                capacity,grouped,admission,clock,stream))return false;
        } else {
            snapshot_routed_addresses(static_cast<const std::uint64_t*>(mapped),
                reinterpret_cast<std::uint64_t*>(addresses.data_ptr()),ids.data_ptr<std::int64_t>(),slots,experts,stream,admission);
            mfq::cuda::moe_stamp_clock(clock,stream);
            if(grouped)prepare_route_groups(ids.data_ptr<std::int64_t>(),slots,tokens,grouped,stream);
        }
        return true;
    };
    const auto read_groups=[&] {
        std::vector<std::vector<std::int64_t>> result;
        if(!grouped)return result;
        const auto cpu=groups.cpu();
        const auto* value=static_cast<const RouteGroup*>(cpu.data_ptr());
        for(int i=1;i<=value[0].count;++i) {
            std::vector<std::int64_t> group{value[i].expert};
            group.insert(group.end(),value[i].slots,value[i].slots+value[i].count);result.push_back(std::move(group));
        }
        std::sort(result.begin(),result.end());return result;
    };
    launch(false);const auto expected=addresses.cpu();const auto expected_groups=read_groups();
    TYLOINFER_NATIVE_CUDA_CHECK(cudaMemsetAsync(addresses.data_ptr(),0,addresses.nbytes(),stream));
    if(!launch(true)) {
        std::cout<<"GGUF fused resident preparation SKIP: route count "<<slots<<" exceeds kernel block-thread capacity\n";return;
    }
    const auto actual=addresses.cpu();
    if(std::memcmp(expected.data_ptr(),actual.data_ptr(),actual.nbytes()) || expected_groups!=read_groups())
        throw std::runtime_error("fused resident preparation changed pointers or route groups");
    if(!bench)return;
    cudaGraph_t graph[2]{};cudaGraphExec_t executable[2]{};
    for(int mode=0;mode<2;++mode) {
        TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        launch(mode==1);
        TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamEndCapture(stream,&graph[mode]));
        TYLOINFER_NATIVE_CUDA_CHECK(cudaGraphInstantiate(&executable[mode],graph[mode],0));
    }
    for(int i=0;i<512;++i)for(auto g:executable)TYLOINFER_NATIVE_CUDA_CHECK(cudaGraphLaunch(g,stream));
    cudaEvent_t begin,end;TYLOINFER_NATIVE_CUDA_CHECK(cudaEventCreate(&begin));TYLOINFER_NATIVE_CUDA_CHECK(cudaEventCreate(&end));
    std::vector<float> times[2];
    for(int trial=0;trial<7;++trial)for(int order=0;order<2;++order) {
        const int mode=(trial+order)%2;
        TYLOINFER_NATIVE_CUDA_CHECK(cudaEventRecord(begin,stream));
        for(int i=0;i<512;++i)TYLOINFER_NATIVE_CUDA_CHECK(cudaGraphLaunch(executable[mode],stream));
        TYLOINFER_NATIVE_CUDA_CHECK(cudaEventRecord(end,stream));TYLOINFER_NATIVE_CUDA_CHECK(cudaEventSynchronize(end));
        float ms=0;TYLOINFER_NATIVE_CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end));times[mode].push_back(ms*1000/512);
    }
    for(auto& time:times)std::sort(time.begin(),time.end());
    std::printf("resident_prepare_pair slots=%d experts=%d tokens=%d old_us=%.6f fused_us=%.6f ratio=%.6f\n",slots,experts,tokens,times[0][3],times[1][3],times[1][3]/times[0][3]);
    for(int mode=0;mode<2;++mode){cudaGraphExecDestroy(executable[mode]);cudaGraphDestroy(graph[mode]);}
    cudaEventDestroy(begin);cudaEventDestroy(end);
}
inline void check() {
    for(int tokens:{1,2,4,8,32})for(int slots:{1,7,33,130,320})run(slots,37,tokens,false);
    if(mfq::cuda::gguf::try_prepare_resident_routes(nullptr,nullptr,nullptr,INT64_MAX,0,1,nullptr,nullptr,nullptr,mfq_current_cuda_stream()))
        throw std::runtime_error("resident preparation ignored actual block-thread limit");
    std::cout<<"GGUF fused resident pointers/groups duplicate/invalid/tail slots PASS\n";
}
}
