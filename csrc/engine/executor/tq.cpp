#include "engine/executor/tq.h"

#include "engine/backends/cuda/native/tensor/backend.h"
#include "context.h"
#include "engine/backends/cuda/native/memory.h"
#include "engine/backends/cuda/kernels/moe/pipeline.h"
#include "backends/cuda/engine/executor/decode_window.h"

#include <atomic>
#include <climits>

#include <algorithm>
#include <exception>
#include <stdexcept>

namespace tb = mfq_tensor_backend;

struct TqGemmResources::RouteReadback {
    int device;
    MfqCudaStream stream;
    std::mutex mutex;
    std::unique_ptr<mfq::cuda::HostBuffer> host;
    void* mapped=nullptr;
    bool can_map=false;
    RouteReadback(int device,MfqCudaStream stream):device(device),stream(std::move(stream)) {
        int supported=0;
        TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(&supported,cudaDevAttrCanMapHostMemory,device));
        can_map=supported!=0;
    }
};

std::vector<std::int64_t> TqGemmResources::routes(const tb::Tensor& ids) {
    const auto source=ids.contiguous();
    std::vector<std::int64_t> result(source.numel());
    if(!source.numel())return result;
    if(!source.is_cuda() || (source.scalar_type()!=tb::kInt32 && source.scalar_type()!=tb::kInt64)) {
        const auto host=source.cpu().to(tb::kInt64);
        std::copy_n(host.data_ptr<std::int64_t>(),result.size(),result.data());return result;
    }
    const MfqCudaGuard device_guard(source.get_device());
    const auto stream=mfq_get_current_cuda_stream(source.get_device());
    std::shared_ptr<RouteReadback> readback;
    {
        std::lock_guard<std::mutex> guard(mutex_);
        for(const auto& candidate:route_readbacks_)
            if(candidate->device==source.get_device() && candidate->stream.stream()==stream.stream()) {
                readback=candidate;break;
            }
        if(!readback) {
            readback=std::make_shared<RouteReadback>(source.get_device(),stream);
            route_readbacks_.push_back(readback);
        }
    }
    // Keep a stable DMA destination per execution stream. Pageable tensor
    // allocation and GPU integer conversion are unnecessary for this ID list.
    std::lock_guard<std::mutex> guard(readback->mutex);
    const auto required=source.nbytes()+64;
    if(!readback->host || readback->host->size()<required) {
        const auto bytes=readback->host && readback->host->size()<=SIZE_MAX/2?
            std::max(required,readback->host->size()*2):required;
        readback->host=std::make_unique<mfq::cuda::HostBuffer>(bytes,readback->can_map);
        readback->mapped=nullptr;
        if(readback->can_map)TYLOINFER_CUDA_CHECK(cudaHostGetDevicePointer(&readback->mapped,readback->host->data(),0));
    }
    auto* values=static_cast<std::uint8_t*>(readback->host->data())+64;
    if(readback->mapped && source.scalar_type()==tb::kInt32 && source.numel()<=INT_MAX) {
        auto* ready=static_cast<std::uint32_t*>(readback->host->data());
        *static_cast<volatile std::uint32_t*>(ready)=0;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        // Reuse the format-independent route publication already used by
        // DecodeWindow. The GPU publishes IDs plus a system-visible flag;
        // no copy-engine transfer or stream-wide synchronization is needed.
        mfq::cuda::moe_publish_routes(nullptr,source.data_ptr<std::int32_t>(),nullptr,
            reinterpret_cast<std::int32_t*>(readback->mapped)+16,
            static_cast<std::uint32_t*>(readback->mapped),0,static_cast<int>(source.numel()),stream.stream());
        TYLOINFER_CUDA_CHECK(cudaGetLastError());
        mfq::cuda::wait_route_publication(ready,stream.stream());
    }else {
        TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(values,source.data_ptr(),source.nbytes(),cudaMemcpyDeviceToHost,stream.stream()));
        TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
    }
    if(source.scalar_type()==tb::kInt32) {
        std::copy_n(reinterpret_cast<const std::int32_t*>(values),result.size(),result.data());
    }else std::copy_n(reinterpret_cast<const std::int64_t*>(values),result.size(),result.data());
    return result;
}

void TqGemmResources::prepare_graph() {
    const auto stream=mfq_get_current_cuda_stream();
    std::lock_guard<std::mutex> guard(mutex_);
    if(graph_pools_.count(stream.stream()))return;
    const auto context=mfq::cuda::default_context(stream.device());
    context->begin_graph_pool(stream.stream());
    try {graph_pools_.emplace(stream.stream(),GraphPool{stream,context});}
    catch(...) {context->end_graph_pool(stream.stream());throw;}
}

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

TqGemmResources::TqGemmResources(int device) : device_(device), blas_(device) {
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

TqGemmResources::~TqGemmResources() noexcept {
    for (auto& [_, pool] : graph_pools_)
        pool.context->end_graph_pool(pool.stream.stream());
}

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
    blas_.retain(stream);
}

cublasHandle_t TqGemmResources::get() {
    return blas_.get();
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
