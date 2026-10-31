#pragma once

#include "stream.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>

namespace mfq::cuda {

class Context;

class Buffer final {
public:
    Buffer() = default;
    Buffer(std::shared_ptr<Context> context, std::size_t bytes);
    ~Buffer() noexcept;

    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    void* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return bytes_; }
    explicit operator bool() const noexcept { return data_ != nullptr; }

private:
    void reset() noexcept;

    std::shared_ptr<Context> context_;
    void* data_ = nullptr;
    std::size_t bytes_ = 0;
    StreamHandle stream_;
};

// cudaHostAlloc reports an exhausted pinned-memory limit as
// cudaErrorMemoryAllocation on Linux, but WDDM can reject a large mapped
// allocation with a mapping error instead ("resource already mapped").
// Callers that retry smaller sizes or fall back to pageable memory treat all
// of these as recoverable resource exhaustion.
inline bool pinned_allocation_rejected(cudaError_t status) noexcept {
    return status == cudaErrorMemoryAllocation || status == cudaErrorAlreadyMapped ||
           status == cudaErrorMapBufferObjectFailed;
}

class HostBuffer final {
public:
    explicit HostBuffer(std::size_t bytes, bool mapped = false);
    ~HostBuffer() noexcept;

    HostBuffer(HostBuffer&& other) noexcept;
    HostBuffer& operator=(HostBuffer&& other) noexcept;
    HostBuffer(const HostBuffer&) = delete;
    HostBuffer& operator=(const HostBuffer&) = delete;

    void* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return bytes_; }

private:
    void reset() noexcept;

    void* data_ = nullptr;
    std::size_t bytes_ = 0;
};

}  // namespace mfq::cuda
