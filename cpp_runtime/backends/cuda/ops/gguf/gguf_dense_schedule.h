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
template<class Separate,class Batch>
bool choose_batch(const std::vector<mfq::cuda::gguf::DenseProjection>& projections,
        int device,int64_t columns,int64_t tokens,bool half_output,cudaStream_t stream,
        Separate&& separate,Batch&& batch) {
    std::string key=std::to_string(device)+":"+std::to_string(columns)+":"+
        std::to_string(tokens)+":"+std::to_string(half_output);
    for(const auto& p:projections)key+="/"+std::to_string(int(p.type))+":"+
        std::to_string(p.rows)+":"+std::to_string(p.stride);
    static std::mutex mutex;
    static std::map<std::string,bool> choices;
    const std::lock_guard<std::mutex> lock(mutex);
    if(const auto found=choices.find(key);found!=choices.end())return found->second;
    cudaStreamCaptureStatus status=cudaStreamCaptureStatusNone;
    TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamIsCapturing(stream,&status));
    // A caller may skip eager warmup. Keep the original independent operators
    // for that capture and leave calibration for a later eager invocation.
    if(status!=cudaStreamCaptureStatusNone)return false;
    struct Resources {
        cudaGraph_t graphs[2]{};
        cudaGraphExec_t executable[2]{};
        cudaEvent_t begin=nullptr,end=nullptr;
        ~Resources() {
            for(auto graph:executable)if(graph)(void)cudaGraphExecDestroy(graph);
            for(auto graph:graphs)if(graph)(void)cudaGraphDestroy(graph);
            if(begin)(void)cudaEventDestroy(begin);
            if(end)(void)cudaEventDestroy(end);
        }
    } resources;
    for(int mode=0;mode<2;++mode) {
        TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        try {if(mode)batch();else separate();}
        catch(...) {
            (void)cudaStreamEndCapture(stream,&resources.graphs[mode]);
            throw;
        }
        TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamEndCapture(stream,&resources.graphs[mode]));
        TYLOINFER_NATIVE_CUDA_CHECK(cudaGraphInstantiate(&resources.executable[mode],resources.graphs[mode],0));
    }
    for(int repeat=0;repeat<4;++repeat)for(auto graph:resources.executable)
        TYLOINFER_NATIVE_CUDA_CHECK(cudaGraphLaunch(graph,stream));
    TYLOINFER_NATIVE_CUDA_CHECK(cudaEventCreate(&resources.begin));
    TYLOINFER_NATIVE_CUDA_CHECK(cudaEventCreate(&resources.end));
    std::array<float,3> samples[2]{};
    for(int trial=0;trial<3;++trial)for(int pass=0;pass<2;++pass) {
        const int mode=(trial+pass)%2;
        TYLOINFER_NATIVE_CUDA_CHECK(cudaEventRecord(resources.begin,stream));
        for(int repeat=0;repeat<16;++repeat)
            TYLOINFER_NATIVE_CUDA_CHECK(cudaGraphLaunch(resources.executable[mode],stream));
        TYLOINFER_NATIVE_CUDA_CHECK(cudaEventRecord(resources.end,stream));
        TYLOINFER_NATIVE_CUDA_CHECK(cudaEventSynchronize(resources.end));
        float elapsed=0;TYLOINFER_NATIVE_CUDA_CHECK(cudaEventElapsedTime(&elapsed,resources.begin,resources.end));
        samples[mode][trial]=elapsed/16;
    }
    for(auto& sample:samples)std::sort(sample.begin(),sample.end());
    // Require a measurable improvement rather than selecting timing noise.
    const bool selected=std::isfinite(samples[1][1]) && samples[1][1]<samples[0][1]*.97f;
    choices.emplace(std::move(key),selected);
    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1')
        std::fprintf(stderr,"native_gguf_dense_schedule actual_match=1 M=%lld matrices=%zu separate_us=%.6f batch_us=%.6f selected=%s\n",
            static_cast<long long>(tokens),projections.size(),1000.*samples[0][1],1000.*samples[1][1],selected?"batch":"separate");
    return selected;
}
}
