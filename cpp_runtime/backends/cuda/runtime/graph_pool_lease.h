#pragma once
#include "mfq_cuda_context.h"
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>

namespace mfq::cuda {
// Captured intermediate buffers remain in the stream pool after their local
// tensors die. Several retained graphs must share its entire lifetime.
class GraphPoolLease {
    std::shared_ptr<Context> context_;
    cudaStream_t stream_;
    bool owns_;
    GraphPoolLease(std::shared_ptr<Context> context,cudaStream_t stream)
        :context_(std::move(context)),stream_(stream),owns_(!context_->has_graph_pool(stream_)) {
        if(owns_)context_->begin_graph_pool(stream_);
    }
public:
    GraphPoolLease(const GraphPoolLease&)=delete;
    GraphPoolLease& operator=(const GraphPoolLease&)=delete;
    ~GraphPoolLease(){if(owns_)context_->end_graph_pool(stream_);}
    static std::shared_ptr<GraphPoolLease> acquire(std::shared_ptr<Context> context,cudaStream_t stream) {
        using Key=std::pair<std::uintptr_t,std::uintptr_t>;
        static std::mutex mutex;
        static std::map<Key,std::weak_ptr<GraphPoolLease>> owners;
        std::lock_guard lock(mutex);
        for(auto entry=owners.begin();entry!=owners.end();)
            if(entry->second.expired())entry=owners.erase(entry);else ++entry;
        const Key key{reinterpret_cast<std::uintptr_t>(context.get()),reinterpret_cast<std::uintptr_t>(stream)};
        auto& weak=owners[key];auto owner=weak.lock();
        if(!owner){owner=std::shared_ptr<GraphPoolLease>(new GraphPoolLease(std::move(context),stream));weak=owner;}
        return owner;
    }
};
} // namespace mfq::cuda
