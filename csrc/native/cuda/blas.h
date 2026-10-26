#pragma once

#include "backends/cuda/native/tensor/backend.h"
#include "backends/cuda/native/context.h"

#include <memory>
#include <mutex>
#include <unordered_map>

namespace mfq::cuda {

// Per-device, per-stream cuBLAS handles. The owner retains each stream until
// its handle can no longer be used; callers only borrow this resource.
class BlasResources final {
public:
    explicit BlasResources(int device) : device_(device) {}

    BlasResources(const BlasResources&) = delete;
    BlasResources& operator=(const BlasResources&) = delete;

    void retain(const MfqCudaStream& stream) {
        std::lock_guard<std::mutex> guard(mutex_);
        handles_.try_emplace(stream.stream(), Entry{stream, {}});
    }

    cublasHandle_t get(const MfqCudaStream& stream) {
        std::lock_guard<std::mutex> guard(mutex_);
        auto found = handles_.find(stream.stream());
        if (found == handles_.end())
            found = handles_.emplace(stream.stream(), Entry{stream, {}}).first;
        if (!found->second.handle) {
            auto handle = std::make_unique<BlasHandle>(device_);
            handle->set_stream(stream.stream());
            found->second.handle = std::move(handle);
        }
        return found->second.handle->get();
    }

    cublasHandle_t get() { return get(mfq_get_current_cuda_stream(device_)); }

private:
    struct Entry {
        MfqCudaStream stream;
        std::unique_ptr<BlasHandle> handle;
    };

    int device_;
    std::mutex mutex_;
    std::unordered_map<cudaStream_t, Entry> handles_;
};

} // namespace mfq::cuda
