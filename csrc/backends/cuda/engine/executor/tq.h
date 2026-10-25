#pragma once

#include "backend.h"
#include "context.h"
#include "native/blas.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

// Execution-owned CUDA resources for TQ.  Constructing the pool is deliberately
// device-free; a device is touched only when its first resource is requested.
struct TqPrefillPlan {
    mfq_tensor_backend::Tensor host, device;
    mfq::cuda::Event done;
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
    std::vector<std::int64_t> routes(const mfq_tensor_backend::Tensor& ids);
    void prepare_graph();
    mfq::cuda::BlasResources& blas() noexcept { return blas_; }

private:
    struct RouteReadback;
    struct GraphPool {
        MfqCudaStream stream;
        std::shared_ptr<mfq::cuda::Context> context;
    };
    int device_;
    bool indexed_mma_ = false;
    bool prefill_mma_ = false;
    bool host_register_ = false;
    std::mutex mutex_;
    mfq::cuda::BlasResources blas_;
    std::vector<std::shared_ptr<TqPrefillPlan>> prefill_plans_;
    std::vector<std::shared_ptr<RouteReadback>> route_readbacks_;
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
    void drain();

private:
    mutable std::mutex mutex_;
    std::unordered_map<int, std::shared_ptr<TqGemmResources>> resources_;
};
