#pragma once

#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/backends/cuda/native/context.h"
#include "engine/backends/cuda/native/stream.h"
#include "engine/backends/cuda/native/blas.h"
#include "engine/backends/cuda/ops/projection/tq.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

// Execution-owned CUDA resources for TQ.  Constructing the pool is deliberately
// device-free; a device is touched only when its first resource is requested.
struct TqPrefillPlan {
    ::tyloi::DenseTensor host, device;
    tyloi::cuda::Event done;
    ~TqPrefillPlan() noexcept;
};

struct TqPrefillPlanLease {
    std::shared_ptr<TqPrefillPlan> plan;
    cudaStream_t stream = nullptr;
    ~TqPrefillPlanLease() noexcept;
};

class TqGemmResources {
public:
    explicit TqGemmResources(int device);
    ~TqGemmResources() noexcept;

    TqGemmResources(const TqGemmResources&) = delete;
    TqGemmResources& operator=(const TqGemmResources&) = delete;

    bool indexed_mma() const noexcept;
    bool prefill_mma() const noexcept;
    bool host_register() const noexcept;
    int device() const noexcept;
    std::shared_ptr<TqPrefillPlan> prefill_plan(std::int64_t bytes);
    void retain(const MfqCudaStream& stream);
    cublasHandle_t get();
    std::vector<std::int64_t> routes(const ::tyloi::DenseTensor& ids);
    void prepare_graph();
    std::shared_ptr<const TqProjectionGroup> group(std::vector<::tyloi::encoding::TqTensor> weights);
    std::vector<::tyloi::DenseTensor> storage() const;
    tyloi::cuda::BlasResources& blas() noexcept { return blas_; }

private:
    struct RouteReadback;
    struct GraphPool {
        MfqCudaStream stream;
        std::shared_ptr<tyloi::cuda::Context> context;
    };
    int device_;
    bool indexed_mma_ = false;
    bool prefill_mma_ = false;
    bool host_register_ = false;
    mutable std::mutex mutex_;
    tyloi::cuda::BlasResources blas_;
    std::vector<std::shared_ptr<TqPrefillPlan>> prefill_plans_;
    std::vector<std::shared_ptr<RouteReadback>> route_readbacks_;
    std::vector<std::shared_ptr<const TqProjectionGroup>> groups_;
    std::unordered_map<cudaStream_t, GraphPool> graph_pools_;
};

class TqExec {
public:
    TqExec() = default;
    ~TqExec() noexcept;

    TqExec(const TqExec&) = delete;
    TqExec& operator=(const TqExec&) = delete;

    std::shared_ptr<TqGemmResources> get(int device);
    bool empty() const noexcept;
    std::vector<::tyloi::DenseTensor> storage() const;
    void drain();
    // The caller drains before clearing resources.
    void clear();

private:
    mutable std::mutex mutex_;
    std::unordered_map<int, std::shared_ptr<TqGemmResources>> resources_;
};
