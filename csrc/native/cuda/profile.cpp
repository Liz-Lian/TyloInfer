#include "profile.h"
#include "native/cuda/env.h"

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string_view>

#if defined(TYLOINFER_NATIVE_CUDA_RUNTIME) && defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#endif

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

void report_cuda_memory(
        bool enabled,
        const char * stage) {
    if (!enabled) return;
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    TYLOINFER_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const auto stats = mfq_cuda_memory_stats(mfq_current_cuda_device());
    constexpr double mib = 1024.0 * 1024.0;
    std::cout << "cuda_memory_stage=" << stage
              << " free_mib=" << free_bytes / mib
              << " used_mib=" << (total_bytes - free_bytes) / mib
              << " total_mib=" << total_bytes / mib
              << " allocated_mib=" << stats.allocated_bytes / mib
              << " active_mib=" << stats.active_bytes / mib
              << " reserved_mib=" << stats.reserved_bytes / mib
              << " inactive_split_mib="
              << stats.inactive_split_bytes / mib
              << " requested_mib=" << stats.requested_bytes / mib
              << " allocations=" << stats.allocations
              << " segments=" << stats.segments
              << " retries=" << stats.retries
              << " ooms=" << stats.ooms << "\n";
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    // cudaMemGetInfo does not describe this process's allocation ownership or
    // WDDM budget. Report those separately when diagnosing residency pressure.
    const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    const auto native=context->memory_stats();
    const auto [pool_used,pool_reserved]=context->pool_memory_stats();
    std::cout<<"native_memory_stage="<<stage<<" owned_mib="<<native.allocated/mib
             <<" peak_mib="<<native.peak/mib<<" limit_mib="<<native.limit/mib
             <<" local_usage_mib="<<context->local_memory_usage()/mib
             <<" pool_used_mib="<<pool_used/mib<<" pool_reserved_mib="<<pool_reserved/mib<<"\n";
#ifdef _WIN32
    Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
    cudaDeviceProp properties{};TYLOINFER_CUDA_CHECK(cudaGetDeviceProperties(&properties,mfq_current_cuda_device()));
    if(SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))))for(UINT index=0;;++index) {
        Microsoft::WRL::ComPtr<IDXGIAdapter1> item;
        if(factory->EnumAdapters1(index,&item)==DXGI_ERROR_NOT_FOUND)break;
        DXGI_ADAPTER_DESC1 description{};
        if(FAILED(item->GetDesc1(&description)) || std::memcmp(properties.luid,&description.AdapterLuid,8))continue;
        Microsoft::WRL::ComPtr<IDXGIAdapter3> adapter;
        if(SUCCEEDED(item.As(&adapter)))for(auto segment:{DXGI_MEMORY_SEGMENT_GROUP_LOCAL,DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL}) {
            DXGI_QUERY_VIDEO_MEMORY_INFO memory{};
            if(SUCCEEDED(adapter->QueryVideoMemoryInfo(0,segment,&memory)))
                std::cout<<"wddm_memory_stage="<<stage<<" segment="<<(segment==DXGI_MEMORY_SEGMENT_GROUP_LOCAL?"local":"nonlocal")
                         <<" budget_mib="<<memory.Budget/mib<<" usage_mib="<<memory.CurrentUsage/mib
                         <<" reservation_mib="<<memory.CurrentReservation/mib<<"\n";
        }
        break;
    }
#endif
#endif
}
