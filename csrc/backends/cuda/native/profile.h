#pragma once

#include "backend.h"

#include <chrono>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mfq::cuda {
using Trace = std::vector<std::pair<std::string, mfq_tensor_backend::Tensor>>;
}

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

    template <typename Fn>
    auto measure(const std::string & name, Fn && fn) -> decltype(fn()) {
        if (!enabled || !selected(name)) return fn();
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        auto stream = mfq_get_current_cuda_stream().stream();
        if (graph_events) {
            cudaEventRecordWithFlags(start, stream, cudaEventRecordExternal);
        } else {
            cudaEventRecord(start, stream);
        }
        auto t0 = std::chrono::steady_clock::now();
        auto out = fn();
        if (graph_events) {
            cudaEventRecordWithFlags(stop, stream, cudaEventRecordExternal);
        } else {
            cudaEventRecord(stop, stream);
        }
        auto t1 = std::chrono::steady_clock::now();
        auto it = stats.find(name);
        if (it == stats.end()) {
            order.push_back(name);
            it = stats.emplace(name, ProfileStat{}).first;
        }
        it->second.wall_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        it->second.calls += 1;
        pending.push_back(PendingEvent{name, start, stop});
        return out;
    }

    void report(const std::string& title);
};
