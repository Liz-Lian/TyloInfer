#include "memory.h"
#include "context.h"
#include "device.h"
#include "native/error.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace mfq::cuda {

Buffer::Buffer(std::shared_ptr<Context> context, std::size_t bytes)
    : context_(std::move(context)), bytes_(bytes) {
    if (!context_) {
        throw std::invalid_argument("CUDA buffer requires a context");
    }
    stream_ = current_stream(context_->device());
    data_ = context_->allocate(bytes_, stream_.stream());
}

Buffer::~Buffer() noexcept {
    reset();
}

Buffer::Buffer(Buffer&& other) noexcept
    : context_(std::move(other.context_)),
      data_(std::exchange(other.data_, nullptr)),
      bytes_(std::exchange(other.bytes_, 0)),
      stream_(std::exchange(other.stream_, {})) {}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) {
        reset();
        context_ = std::move(other.context_);
        data_ = std::exchange(other.data_, nullptr);
        bytes_ = std::exchange(other.bytes_, 0);
        stream_ = std::exchange(other.stream_, {});
    }
    return *this;
}

void Buffer::reset() noexcept {
    if (context_ && data_ != nullptr) {
        context_->release(data_, bytes_, stream_.stream());
    }
    data_ = nullptr;
    bytes_ = 0;
    stream_ = {};
    context_.reset();
}

HostBuffer::HostBuffer(std::size_t bytes, bool mapped) : bytes_(bytes) {
    if (bytes_ == 0) {
        return;
    }
    const auto flags = mapped ? cudaHostAllocMapped : cudaHostAllocDefault;
    const auto allocation=cudaHostAlloc(&data_,bytes_,flags);
    if(pinned_allocation_rejected(allocation)) {
        (void)cudaGetLastError();
        data_=nullptr;
        throw MfqResourceExhausted("CUDA pinned allocation failed: requested="+std::to_string(bytes_)+
            " error="+cudaGetErrorName(allocation));
    }
    TYLOINFER_NATIVE_CUDA_CHECK(allocation);
}

HostBuffer::~HostBuffer() noexcept {
    reset();
}

HostBuffer::HostBuffer(HostBuffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      bytes_(std::exchange(other.bytes_, 0)) {}

HostBuffer& HostBuffer::operator=(HostBuffer&& other) noexcept {
    if (this != &other) {
        reset();
        data_ = std::exchange(other.data_, nullptr);
        bytes_ = std::exchange(other.bytes_, 0);
    }
    return *this;
}

void HostBuffer::reset() noexcept {
    if (data_ != nullptr) {
        (void)cudaFreeHost(data_);
    }
    data_ = nullptr;
    bytes_ = 0;
}

}  // namespace mfq::cuda
