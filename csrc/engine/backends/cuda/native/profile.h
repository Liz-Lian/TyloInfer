#pragma once

#include "engine/backends/cuda/native/tensor/backend.h"
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/ops.h"
#endif

#include <chrono>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mfq::cuda {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
using Trace = mfq::ops::Trace;
#else
using Trace = std::vector<std::pair<std::string, mfq_tensor_backend::Tensor>>;
#endif
}

void report_cuda_memory(bool enabled, const char* stage);

struct ProfileStat {
    double ms = 0.0;
    double wall_ms = 0.0;
    int64_t calls = 0;
};

struct CudaProfiler {
    struct PendingEvent {
        std::string name;
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
    };

    bool enabled = false;
    bool graph_events = false;
    std::string filter;
    std::unordered_map<std::string, ProfileStat> stats;
    std::vector<std::string> order;
    std::vector<PendingEvent> pending;

    bool selected(const std::string& name) const;
    void reset();

    class Range {
    public:
        Range(CudaProfiler& profiler, const std::string& name)
            : profiler_(&profiler), name_(name) {
            if (!profiler.enabled || !profiler.selected(name)) return;
            cudaEventCreate(&start_);
            cudaEventCreate(&stop_);
            stream_ = mfq_get_current_cuda_stream().stream();
            if (profiler.graph_events) cudaEventRecordWithFlags(start_, stream_, cudaEventRecordExternal);
            else cudaEventRecord(start_, stream_);
            started_ = std::chrono::steady_clock::now();
        }
        ~Range() {
            if (start_) cudaEventDestroy(start_);
            if (stop_) cudaEventDestroy(stop_);
        }
        Range(const Range&) = delete;
        Range& operator=(const Range&) = delete;
        void finish() {
            if (!start_) return;
            if (profiler_->graph_events) cudaEventRecordWithFlags(stop_, stream_, cudaEventRecordExternal);
            else cudaEventRecord(stop_, stream_);
            auto ended = std::chrono::steady_clock::now();
            auto it = profiler_->stats.find(name_);
            if (it == profiler_->stats.end()) {
                profiler_->order.push_back(name_);
                it = profiler_->stats.emplace(name_, ProfileStat{}).first;
            }
            it->second.wall_ms += std::chrono::duration<double, std::milli>(ended - started_).count();
            it->second.calls += 1;
            profiler_->pending.push_back(PendingEvent{name_, start_, stop_});
            start_ = stop_ = nullptr;
        }
    private:
        CudaProfiler* profiler_;
        std::string name_;
        cudaEvent_t start_ = nullptr, stop_ = nullptr;
        cudaStream_t stream_ = nullptr;
        std::chrono::steady_clock::time_point started_;
    };

    template <typename Fn>
    auto measure(const std::string & name, Fn && fn) -> decltype(fn()) {
        Range range(*this, name);
        auto out = fn();
        range.finish();
        return out;
    }

    void report(const std::string& title);
};

namespace mfq::cuda::internal {

class PrefillCudaTimer {
public:
    PrefillCudaTimer();
    ~PrefillCudaTimer();

    PrefillCudaTimer(const PrefillCudaTimer&) = delete;
    PrefillCudaTimer& operator=(const PrefillCudaTimer&) = delete;

    cudaEvent_t finished_event() const;
    double elapsed_ms() const;

private:
    cudaStream_t stream_ = nullptr;
    cudaEvent_t started_ = nullptr;
    cudaEvent_t finished_ = nullptr;
};

} // namespace mfq::cuda::internal
