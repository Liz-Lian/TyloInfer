#include "profile.h"
#include "native/cuda/env.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string_view>

bool CudaProfiler::selected(const std::string& name) const {
    if (filter.empty()) return true;

    const std::string_view filter_value(filter);
    size_t begin = 0;
    while (begin <= filter_value.size()) {
        const size_t end = filter_value.find(',', begin);
        const size_t count = end == std::string_view::npos
            ? filter_value.size() - begin
            : end - begin;
        if (filter_value.substr(begin, count) == name) return true;
        if (end == std::string_view::npos) break;
        begin = end + 1;
    }
    return false;
}

void CudaProfiler::reset() {
    for (auto& event : pending) {
        cudaEventDestroy(event.start);
        cudaEventDestroy(event.stop);
    }
    pending.clear();
    stats.clear();
    order.clear();
}

void CudaProfiler::report(const std::string& title) {
    if (!enabled) return;
    if (!pending.empty()) cudaEventSynchronize(pending.back().stop);
    const auto* timeline_setting=mfq::cuda::env_lookup("TYLOINFER_PREFILL_LAYER_TIMELINE");
    if(!pending.empty() && timeline_setting && std::atoi(timeline_setting)!=0) {
        std::ostringstream trace;
        trace<<"profile_timeline_begin "<<title<<'\n';
        for(const auto& event:pending)if(event.name.starts_with("prefill.")) {
            float begin=0,end=0;
            TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&begin,pending.front().start,event.start));
            TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&end,pending.front().start,event.stop));
            trace<<"profile_timeline name="<<event.name<<" start_ms="<<begin<<" end_ms="<<end<<'\n';
        }
        const auto output=trace.str();std::fwrite(output.data(),1,output.size(),stderr);
    }
    for (auto& event : pending) {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, event.start, event.stop);
        stats.at(event.name).ms += static_cast<double>(ms);
        cudaEventDestroy(event.start);
        cudaEventDestroy(event.stop);
    }
    pending.clear();
    std::cerr << "profile " << title << '\n';
    for (const auto& name : order) {
        const auto& stat = stats.at(name);
        std::cerr << "profile_item"
                  << " name=" << name
                  << " calls=" << stat.calls
                  << " cuda_ms=" << stat.ms
                  << " cuda_avg_ms="
                  << (stat.calls ? stat.ms / static_cast<double>(stat.calls) : 0.0)
                  << " wall_ms=" << stat.wall_ms
                  << " wall_avg_ms="
                  << (stat.calls ? stat.wall_ms / static_cast<double>(stat.calls) : 0.0)
                  << '\n';
    }
}

namespace mfq::cuda::internal {

PrefillCudaTimer::PrefillCudaTimer()
    : stream_(mfq_get_current_cuda_stream()) {
    TYLOINFER_CUDA_CHECK(cudaEventCreate(&started_));
    try {
        TYLOINFER_CUDA_CHECK(cudaEventCreate(&finished_));
        TYLOINFER_CUDA_CHECK(cudaEventRecord(started_, stream_));
    } catch (...) {
        if (finished_ != nullptr) cudaEventDestroy(finished_);
        cudaEventDestroy(started_);
        finished_ = nullptr;
        started_ = nullptr;
        throw;
    }
}

PrefillCudaTimer::~PrefillCudaTimer() {
    if (finished_ != nullptr) cudaEventDestroy(finished_);
    if (started_ != nullptr) cudaEventDestroy(started_);
}

cudaEvent_t PrefillCudaTimer::finished_event() const {
    return finished_;
}

double PrefillCudaTimer::elapsed_ms() const {
    TYLOINFER_CUDA_CHECK(cudaEventSynchronize(finished_));
    float elapsed = 0.0f;
    TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&elapsed, started_, finished_));
    return static_cast<double>(elapsed);
}

} // namespace mfq::cuda::internal
