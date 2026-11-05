#pragma once
#include "../../kernels/gguf.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>

namespace mfq_gguf_dense_schedule {
// Calibration runs only during eager preparation, with the real immutable
// weights and already-allocated activation/output packets. Graph replays keep
// the chosen device work; no timing, allocation or host callback is captured.

// Pack four operator invocations into each calibration graph. This retains
// the original 4 warm + 48 measured invocations per candidate while reducing
// host graph submissions by four; no calibration is captured in serving graphs.
template<std::size_t Candidates,class Launch>
std::size_t choose_schedule(const std::vector<mfq::cuda::gguf::DenseProjection>& projections,
        int device,int64_t columns,int64_t tokens,bool half_output,cudaStream_t stream,
        Launch&& launch,bool aligned_m1=false) {
    static_assert(Candidates>=2);
    std::string key=std::to_string(device)+":"+std::to_string(columns)+":"+
        std::to_string(tokens)+":"+std::to_string(half_output)+":"+std::to_string(aligned_m1);
    for(const auto& p:projections)key+="/"+std::to_string(int(p.type))+":"+
        std::to_string(p.rows)+":"+std::to_string(p.stride);
    static std::mutex mutex;static std::map<std::string,std::size_t> choices;
    const std::lock_guard<std::mutex> lock(mutex);
    if(const auto found=choices.find(key);found!=choices.end())return found->second;
    cudaStreamCaptureStatus status=cudaStreamCaptureStatusNone;
    TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamIsCapturing(stream,&status));
    if(status!=cudaStreamCaptureStatusNone)return 0;
    struct Resources {
        std::array<cudaGraph_t,Candidates> graphs{};
        std::array<cudaGraphExec_t,Candidates> executable{};
        cudaEvent_t begin=nullptr,end=nullptr;
        ~Resources() {
            for(auto graph:executable)if(graph)(void)cudaGraphExecDestroy(graph);
            for(auto graph:graphs)if(graph)(void)cudaGraphDestroy(graph);
            if(begin)(void)cudaEventDestroy(begin);
            if(end)(void)cudaEventDestroy(end);
        }
    } resources;
    constexpr int Packed=4,Replays=4;
    for(std::size_t mode=0;mode<Candidates;++mode) {
        TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        try {for(int repeat=0;repeat<Packed;++repeat)launch(mode);}
        catch(...) {(void)cudaStreamEndCapture(stream,&resources.graphs[mode]);throw;}
        TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamEndCapture(stream,&resources.graphs[mode]));
        TYLOINFER_NATIVE_CUDA_CHECK(cudaGraphInstantiate(&resources.executable[mode],resources.graphs[mode],0));
    }
    for(auto graph:resources.executable)TYLOINFER_NATIVE_CUDA_CHECK(cudaGraphLaunch(graph,stream));
    TYLOINFER_NATIVE_CUDA_CHECK(cudaEventCreate(&resources.begin));
    TYLOINFER_NATIVE_CUDA_CHECK(cudaEventCreate(&resources.end));
    std::array<std::array<float,3>,Candidates> samples{};
    for(int trial=0;trial<3;++trial)for(std::size_t pass=0;pass<Candidates;++pass) {
        const auto mode=(trial+pass)%Candidates;
        TYLOINFER_NATIVE_CUDA_CHECK(cudaEventRecord(resources.begin,stream));
        for(int repeat=0;repeat<Replays;++repeat)
            TYLOINFER_NATIVE_CUDA_CHECK(cudaGraphLaunch(resources.executable[mode],stream));
        TYLOINFER_NATIVE_CUDA_CHECK(cudaEventRecord(resources.end,stream));
        TYLOINFER_NATIVE_CUDA_CHECK(cudaEventSynchronize(resources.end));
        float elapsed=0;TYLOINFER_NATIVE_CUDA_CHECK(cudaEventElapsedTime(&elapsed,resources.begin,resources.end));
        samples[mode][trial]=elapsed/(Packed*Replays);
    }
    for(auto& sample:samples)std::sort(sample.begin(),sample.end());
    std::size_t selected=0;
    for(std::size_t mode=1;mode<Candidates;++mode)
        if(std::isfinite(samples[mode][1]) && samples[mode][1]<samples[selected][1])selected=mode;
    if(selected && samples[selected][1]>=samples[0][1]*.97f)selected=0;
    choices.emplace(std::move(key),selected);
    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        std::fprintf(stderr,"native_gguf_dense_schedule actual_match=1 M=%lld matrices=%zu separate_us=%.6f batch_us=%.6f",
            static_cast<long long>(tokens),projections.size(),1000.*samples[0][1],1000.*samples[1][1]);
        if constexpr(Candidates>2)std::fprintf(stderr," pair_us=%.6f",1000.*samples[2][1]);
        std::fprintf(stderr," selected=%s\n",selected==0?"separate":selected==1?"batch":"pair");
    }
    return selected;
}
template<class Separate,class Batch>
bool choose_batch(const std::vector<mfq::cuda::gguf::DenseProjection>& projections,
        int device,int64_t columns,int64_t tokens,bool half_output,cudaStream_t stream,
        Separate&& separate,Batch&& batch,bool aligned_m1=false) {
    return choose_schedule<2>(projections,device,columns,tokens,half_output,stream,
        [&](std::size_t mode){if(mode)batch();else separate();},aligned_m1)!=0;
}
template<class Separate,class Batch,class Pair>
std::size_t choose_group(const std::vector<mfq::cuda::gguf::DenseProjection>& projections,
        int device,int64_t columns,int64_t tokens,bool half_output,cudaStream_t stream,
        Separate&& separate,Batch&& batch,Pair&& pair) {
    return choose_schedule<3>(projections,device,columns,tokens,half_output,stream,
        [&](std::size_t mode){if(mode==2)pair();else if(mode)batch();else separate();});
}
}
