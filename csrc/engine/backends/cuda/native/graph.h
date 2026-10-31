#pragma once

#include "context.h"

namespace mfq::cuda {

class Graph final {
public:
    Graph() = default;
    ~Graph() noexcept;

    Graph(Graph&& other) noexcept;
    Graph& operator=(Graph&& other) noexcept;
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;

    void prepare_memory();
    void prepare_memory(
        const std::vector<StreamHandle>& participant_streams);
    void capture_begin();
    // Record in a pool already owned by a decoder or expert-cache lifetime.
    void capture_begin_shared();
    void capture_end();
    void replay();
    void reset() noexcept;
    bool valid() const noexcept { return executable_ != nullptr; }
    cudaGraphExec_t native_executable() const noexcept { return executable_; }
    std::size_t nodes() const;

private:
    int device_ = 0;
    StreamHandle stream_;
    std::shared_ptr<Context> context_;
    std::vector<StreamHandle> pool_streams_;
    std::vector<std::shared_ptr<Context>> pool_contexts_;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t executable_ = nullptr;
    bool shared_pool_ = false;
};

class GraphWarmupScope {
public:
    GraphWarmupScope(std::shared_ptr<Context> context, cudaStream_t stream)
        : context_(std::move(context)), stream_(stream) {
        context_->begin_graph_warmup(stream_);
    }
    ~GraphWarmupScope() { context_->end_graph_warmup(stream_); }
    GraphWarmupScope(const GraphWarmupScope&) = delete;

private:
    std::shared_ptr<Context> context_;
    cudaStream_t stream_;
};

}  // namespace mfq::cuda
