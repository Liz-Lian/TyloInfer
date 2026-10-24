#include "execution/tq.h"

#include "backend.h"
#include "context.h"

#include <algorithm>
#include <exception>
#include <stdexcept>

TqPrefillPlan::~TqPrefillPlan() noexcept {
    try {
        done.synchronize();
    } catch (...) {
    }
}

TqPrefillPlanLease::~TqPrefillPlanLease() noexcept {
    if (!plan) return;
    if (cudaEventRecord(plan->done.get(), stream) != cudaSuccess) {
        (void)cudaStreamSynchronize(stream);
    }
}

TqGemmResources::TqGemmResources(int device) : device_(device) {
    int major = 0;
    TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(
        &major, cudaDevAttrComputeCapabilityMajor, device));
    indexed_mma_ = major >= 8;
    prefill_mma_ = major >= 7;

    int supported = 0;
    const auto registered = cudaDeviceGetAttribute(
        &supported, cudaDevAttrHostRegisterSupported, device);
    if (registered == cudaSuccess) {
        host_register_ = supported != 0;
    } else if (registered == cudaErrorInvalidValue) {
        (void)cudaGetLastError();
    } else {
        TYLOINFER_CUDA_CHECK(registered);
    }
}

TqGemmResources::~TqGemmResources() noexcept = default;

bool TqGemmResources::indexed_mma() const noexcept { return indexed_mma_; }
bool TqGemmResources::prefill_mma() const noexcept { return prefill_mma_; }
bool TqGemmResources::host_register() const noexcept { return host_register_; }
int TqGemmResources::device() const noexcept { return device_; }

std::shared_ptr<TqPrefillPlan> TqGemmResources::prefill_plan(std::int64_t bytes) {
    std::lock_guard<std::mutex> guard(mutex_);
    for (auto it = prefill_plans_.begin(); it != prefill_plans_.end();) {
        if (it->use_count() != 1 || !(*it)->done.ready()) {
            ++it;
            continue;
        }
        if ((*it)->host.numel() >= bytes) {
            mfq_cuda_record_stream((*it)->device,
                mfq_get_current_cuda_stream(device_));
            return *it;
        }
        it = prefill_plans_.erase(it);
    }
    auto plan = std::make_shared<TqPrefillPlan>();
    const auto capacity = (bytes + 4095) & ~std::int64_t(4095);
    plan->host = mfq_tensor_backend::empty(
        {capacity}, mfq_tensor_backend::TensorOptions()
            .dtype(mfq_tensor_backend::kUInt8).pinned_memory(true));
    plan->device = mfq_tensor_backend::empty(
        {capacity}, mfq_tensor_backend::TensorOptions()
            .dtype(mfq_tensor_backend::kUInt8)
            .device(mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, device_)));
    prefill_plans_.push_back(plan);
    return plan;
}

void TqGemmResources::retain(const MfqCudaStream& stream) {
    std::lock_guard<std::mutex> guard(mutex_);
    handles_.try_emplace(stream.stream(), Entry{stream, {}});
}

cublasHandle_t TqGemmResources::get() {
    const auto stream = mfq_get_current_cuda_stream(device_);
    std::lock_guard<std::mutex> guard(mutex_);
    auto found = handles_.find(stream.stream());
    if (found == handles_.end()) {
        found = handles_.emplace(stream.stream(), Entry{stream, {}}).first;
    }
    if (!found->second.handle) {
        auto handle = std::make_unique<mfq::cuda::BlasHandle>(device_);
        handle->set_stream(stream.stream());
        found->second.handle = std::move(handle);
    }
    return found->second.handle->get();
}

TqExec::~TqExec() noexcept {
    try {
        drain();
    } catch (...) {
    }
}

std::shared_ptr<TqGemmResources> TqExec::get(int device) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto found = resources_.find(device);
    if (found != resources_.end()) return found->second;
    auto resources = std::make_shared<TqGemmResources>(device);
    resources_.emplace(device, resources);
    return resources;
}

bool TqExec::empty() const noexcept {
    std::lock_guard<std::mutex> guard(mutex_);
    return resources_.empty();
}

void TqExec::drain() {
    std::vector<std::shared_ptr<TqGemmResources>> resources;
    {
        std::lock_guard<std::mutex> guard(mutex_);
        resources.reserve(resources_.size());
        for (auto& [_, resource] : resources_) resources.push_back(resource);
    }
    std::exception_ptr failure;
    for (const auto& resource : resources) {
        try {
            MfqCudaGuard guard(resource->device());
            TYLOINFER_CUDA_CHECK(cudaDeviceSynchronize());
        } catch (...) {
            if (!failure) failure = std::current_exception();
        }
    }
    if (failure) std::rethrow_exception(failure);
}
