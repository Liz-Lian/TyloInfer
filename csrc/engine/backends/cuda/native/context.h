#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "stream.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mfq::cuda {

void record_gpu_timestamp(uint64_t* timestamps, std::size_t index, cudaStream_t stream);

StreamHandle current_stream(int device = -1);
StreamHandle stream_from_pool(bool high_priority = false, int device = -1);

class StreamGuard final {
public:
    explicit StreamGuard(StreamHandle stream);
    ~StreamGuard() noexcept;

    StreamGuard(const StreamGuard&) = delete;
    StreamGuard& operator=(const StreamGuard&) = delete;

private:
    int device_ = 0;
    std::optional<StreamHandle> previous_;
};

class BlasHandle final {
public:
    explicit BlasHandle(int device);
    ~BlasHandle() noexcept;

    BlasHandle(BlasHandle&& other) noexcept;
    BlasHandle& operator=(BlasHandle&& other) noexcept;
    BlasHandle(const BlasHandle&) = delete;
    BlasHandle& operator=(const BlasHandle&) = delete;

    cublasHandle_t get() const noexcept { return handle_; }
    void set_stream(cudaStream_t stream);

private:
    int device_ = 0;
    cublasHandle_t handle_ = nullptr;
};

struct AllocationBudget;
struct ParallelBatchMatmulContext;
struct DeviceMemoryStats {
    std::size_t limit = 0;
    std::size_t allocated = 0;
    std::size_t peak = 0;
};

class Context final : public std::enable_shared_from_this<Context> {
public:
    explicit Context(int device);
    ~Context() noexcept;

    int device() const noexcept { return device_; }
    Stream& stream() noexcept { return stream_; }
    const Stream& stream() const noexcept { return stream_; }
    BlasHandle& blas() noexcept { return blas_; }
    std::shared_ptr<ParallelBatchMatmulContext> parallel_blas;
    cudaMemPool_t memory_pool() const noexcept { return pool_; }
    bool supports_async_allocations() const noexcept { return async_allocations_; }
    DeviceMemoryStats memory_stats() const noexcept;
    std::pair<std::uint64_t,std::uint64_t> pool_memory_stats() const;
    std::size_t local_memory_usage() const;
    // Process-owned pages, including idle allocator pages and CUDA libraries.
    std::size_t resident_memory_usage() const;
    std::size_t available_memory() const;
    void reclaim_idle_memory();
    // Temporarily reuse already charged expert-cache storage on one ordered
    // compute stream. Graph/resident allocations never enter this workspace.
    void begin_workspace_loan(const std::vector<std::pair<void*,std::size_t>>& regions,
                              cudaStream_t stream);
    void suspend_workspace_loan();
    void end_workspace_loan();
    std::size_t workspace_available() const;
    std::vector<std::pair<void*,std::size_t>> workspace_touched_regions() const;
    void check_memory_limit(std::size_t additional = 0) const;
    void set_memory_limit(std::size_t bytes);

    void* allocate(std::size_t bytes, cudaStream_t stream = nullptr);
    void release(
        void* pointer,
        std::size_t bytes,
        cudaStream_t stream = nullptr) noexcept;
    void begin_graph_pool(cudaStream_t stream);
    void begin_graph_warmup(cudaStream_t stream);
    void end_graph_warmup(cudaStream_t stream) noexcept;
    void begin_graph_capture(cudaStream_t stream);
    void end_graph_capture(cudaStream_t stream) noexcept;
    void end_graph_pool(cudaStream_t stream) noexcept;
    void trim();

private:
    struct GraphPool {
        cudaMemPool_t memory = nullptr;
        bool capturing = false;
        bool warming = false;
        std::map<std::size_t, std::vector<void*>> available;
        std::unordered_set<void*> owned;
    };

    int device_ = 0;
    Stream stream_;
    BlasHandle blas_;
    cudaMemPool_t pool_ = nullptr;
    cudaMemPool_t resident_pool_ = nullptr;
    bool async_allocations_ = false;
    std::shared_ptr<AllocationBudget> allocation_budget_;
    void trim_pool_pages() const;
    mutable std::mutex graph_pool_mutex_;
    // A tensor may use a larger cached graph block and outlive its pool.
    // Account/free the allocation's capacity, not that tensor's logical bytes.
    std::unordered_map<void*,std::size_t> graph_allocation_sizes_;
    struct WorkspaceLoan {
        cudaStream_t stream=nullptr;
        std::thread::id thread;
        bool enabled=true;
        std::map<std::uintptr_t,std::size_t> free;
        std::unordered_map<void*,std::size_t> live;
        std::vector<std::pair<std::uintptr_t,std::size_t>> touched;
    };
    std::unique_ptr<WorkspaceLoan> workspace_loan_;
    std::unordered_map<cudaStream_t, GraphPool> graph_pools_;
};

// One set belongs to one Engine; standalone tensor callers use a thread-local set.
using ContextSet = std::unordered_map<int, std::shared_ptr<Context>>;
class ContextGuard final {
public:
    explicit ContextGuard(ContextSet& contexts);
    ~ContextGuard() noexcept;
    ContextGuard(const ContextGuard&) = delete;
    ContextGuard& operator=(const ContextGuard&) = delete;
private:
    ContextSet* previous_;
    bool changed_;
    std::unordered_map<int, StreamHandle> streams_;
};

std::shared_ptr<Context> default_context(int device = 0);

}  // namespace mfq::cuda
