#include "../runtime/moe_pipeline.h"
#include <cuda_fp16.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void check(cudaError_t status) {
    if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}
struct Stream {
    cudaStream_t value=nullptr;
    Stream(){check(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking));}
    ~Stream(){if(value)cudaStreamDestroy(value);}
};
struct Device {
    void* value=nullptr;
    explicit Device(std::size_t bytes){check(cudaMalloc(&value,std::max<std::size_t>(bytes,1)));}
    ~Device(){if(value)cudaFree(value);}
};
struct Mapped {
    unsigned char* host=nullptr;
    unsigned char* device=nullptr;
    explicit Mapped(std::size_t bytes) {
        void* allocation=nullptr;
        check(cudaHostAlloc(&allocation,bytes,cudaHostAllocMapped));
        host=static_cast<unsigned char*>(allocation);
        void* alias=nullptr;
        const auto status=cudaHostGetDevicePointer(&alias,host,0);
        if(status!=cudaSuccess){cudaFreeHost(host);host=nullptr;check(status);}
        device=static_cast<unsigned char*>(alias);
        std::memset(host,0xa5,bytes);
    }
    ~Mapped(){if(host)cudaFreeHost(host);}
};
struct Graph {
    cudaGraph_t value=nullptr;
    cudaGraphExec_t executable=nullptr;
    ~Graph(){if(executable)cudaGraphExecDestroy(executable);if(value)cudaGraphDestroy(value);}
};
void await(std::atomic_ref<std::uint32_t> flag,cudaStream_t stream) {
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
    while(!flag.load(std::memory_order_acquire)) {
        // Query submits deferred command batches without joining the stream.
        const auto status=cudaStreamQuery(stream);
        if(status!=cudaErrorNotReady)check(status);
        if(std::chrono::steady_clock::now()>deadline)throw std::runtime_error("Mapped publication timed out");
    }
}
void run(int input_count,int route_count) {
    Stream stream;
    Device input(std::size_t(input_count)*2),routes(std::size_t(route_count)*4);
    const auto payload_bytes=std::size_t(input_count+route_count)*4;
    Mapped mapped(128+payload_bytes+32);
    auto ready=std::atomic_ref<std::uint32_t>(*reinterpret_cast<std::uint32_t*>(mapped.host));
    auto notified=std::atomic_ref<std::uint32_t>(*reinterpret_cast<std::uint32_t*>(mapped.host+64));
    Graph graph;
    check(cudaStreamBeginCapture(stream.value,cudaStreamCaptureModeGlobal));
    mfq::cuda::moe_publish_routes(input.value,static_cast<const std::int32_t*>(routes.value),
        reinterpret_cast<float*>(mapped.device+128),reinterpret_cast<std::int32_t*>(mapped.device+128+std::size_t(input_count)*4),
        reinterpret_cast<std::uint32_t*>(mapped.device),input_count,route_count,stream.value);
    mfq::cuda::signal_mapped_flag(reinterpret_cast<std::uint32_t*>(mapped.device+64),stream.value);
    check(cudaStreamEndCapture(stream.value,&graph.value));
    check(cudaGraphInstantiate(&graph.executable,graph.value,0));
    std::vector<__half> values(input_count);
    std::vector<std::int32_t> ids(route_count);
    for(int epoch=1;epoch<=16;++epoch) {
        for(int i=0;i<input_count;++i)values[i]=__float2half_rn(float((i*17+epoch*31)%1023)/128.f);
        for(int i=0;i<route_count;++i)ids[i]=i*43+epoch*19;
        ready.store(0,std::memory_order_release);notified.store(0,std::memory_order_release);
        if(input_count)check(cudaMemcpyAsync(input.value,values.data(),values.size()*2,cudaMemcpyHostToDevice,stream.value));
        if(route_count)check(cudaMemcpyAsync(routes.value,ids.data(),ids.size()*4,cudaMemcpyHostToDevice,stream.value));
        check(cudaGraphLaunch(graph.executable,stream.value));
        await(ready,stream.value);
        const auto* actual_input=reinterpret_cast<const float*>(mapped.host+128);
        const auto* actual_routes=reinterpret_cast<const std::int32_t*>(mapped.host+128+std::size_t(input_count)*4);
        for(int i=0;i<input_count;++i)if(actual_input[i]!=__half2float(values[i]))
            throw std::runtime_error("Input was published before its payload became visible");
        for(int i=0;i<route_count;++i)if(actual_routes[i]!=ids[i])
            throw std::runtime_error("Routes were published before their payload became visible");
        await(notified,stream.value);check(cudaStreamSynchronize(stream.value));
        for(int i=0;i<32;++i)if(mapped.host[128+payload_bytes+i]!=0xa5)
            throw std::runtime_error("Mapped publication overwrote its destination guard");
    }
}
void run_acquire(int count) {
    Stream stream;
    const auto input_bytes=std::size_t(count)*sizeof(__half);
    const auto output_offset=256+((input_bytes+15)&~std::size_t(15));
    Mapped mapped(output_offset+std::size_t(count)*4+32);
    auto ready=std::atomic_ref<std::uint32_t>(*reinterpret_cast<std::uint32_t*>(mapped.host));
    auto started=std::atomic_ref<std::uint32_t>(*reinterpret_cast<std::uint32_t*>(mapped.host+64));
    auto returned=std::atomic_ref<std::uint32_t>(*reinterpret_cast<std::uint32_t*>(mapped.host+128));
    auto* input=reinterpret_cast<__half*>(mapped.host+256);
    const auto* output=reinterpret_cast<const float*>(mapped.host+output_offset);
    Graph graph;
    check(cudaStreamBeginCapture(stream.value,cudaStreamCaptureModeGlobal));
    mfq::cuda::signal_mapped_flag(reinterpret_cast<std::uint32_t*>(mapped.device+64),stream.value);
    mfq::cuda::wait_mapped_flag(reinterpret_cast<std::uint32_t*>(mapped.device),stream.value);
    mfq::cuda::moe_publish_routes(mapped.device+256,nullptr,reinterpret_cast<float*>(mapped.device+output_offset),nullptr,
        reinterpret_cast<std::uint32_t*>(mapped.device+128),count,0,stream.value);
    check(cudaStreamEndCapture(stream.value,&graph.value));check(cudaGraphInstantiate(&graph.executable,graph.value,0));
    for(int epoch=1;epoch<=16;++epoch) {
        ready.store(0,std::memory_order_release);started.store(0,std::memory_order_release);returned.store(0,std::memory_order_release);
        try {
            check(cudaGraphLaunch(graph.executable,stream.value));await(started,stream.value);
            for(int i=0;i<count;++i)input[i]=__float2half_rn(float((i*19+epoch*13)%1023)/128.f);
            ready.store(1,std::memory_order_release);await(returned,stream.value);
            for(int i=0;i<count;++i)if(output[i]!=__half2float(input[i]))
                throw std::runtime_error("GPU acquired a stale mapped input");
            check(cudaStreamSynchronize(stream.value));
        }catch(...) {
            ready.store(1,std::memory_order_release);
            (void)cudaStreamSynchronize(stream.value);
            throw;
        }
    }
}
}
int main()try {
    int devices=0;
    const auto status=cudaGetDeviceCount(&devices);
    if(status==cudaErrorNoDevice || status==cudaErrorInsufficientDriver || (status==cudaSuccess && !devices)) {
        std::cout<<"SKIP: CUDA device unavailable\n";return 77;
    }
    check(status);int device=0,mapped=0;
    check(cudaGetDevice(&device));check(cudaDeviceGetAttribute(&mapped,cudaDevAttrCanMapHostMemory,device));
    if(!mapped){std::cout<<"SKIP: mapped host memory unavailable\n";return 77;}
    int cases=0;
    for(int input:{0,31,4096,32771})for(int routes:{0,1,10,32,33,80,257}){run(input,routes);++cases;}
    for(int count:{0,1,31,4096,32771})run_acquire(count);
    std::cout<<"Mapped route/input publication and notification: "<<cases<<" geometries, 16 retained Graph replays each PASS\n";
    std::cout<<"Delayed CPU-to-GPU acquire and payload visibility: 5 geometries, 16 retained Graph replays each PASS\n";
    return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
