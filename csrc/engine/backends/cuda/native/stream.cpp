#include "stream.h"
#include "device.h"

#include <utility>

namespace mfq::cuda {

Event::Event(unsigned flags) {
    TYLOINFER_NATIVE_CUDA_CHECK(cudaGetDevice(&device_));
    TYLOINFER_NATIVE_CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags));
}

Event::~Event() noexcept {
    if (event_ != nullptr) {
        internal::on_device_noexcept(device_, [&] { (void)cudaEventDestroy(event_); });
    }
}

Event::Event(Event&& other) noexcept
    : device_(other.device_), event_(std::exchange(other.event_, nullptr)) {}

Event& Event::operator=(Event&& other) noexcept {
    if (this != &other) {
        if (event_ != nullptr) {
            internal::on_device_noexcept(device_, [&] { (void)cudaEventDestroy(event_); });
        }
        device_ = other.device_;
        event_ = std::exchange(other.event_, nullptr);
    }
    return *this;
}

void Event::record(cudaStream_t stream) {
    DeviceGuard guard(device_);
    TYLOINFER_NATIVE_CUDA_CHECK(cudaEventRecord(event_, stream));
}

bool Event::ready() const {
    DeviceGuard guard(device_);
    const auto status = cudaEventQuery(event_);
    if (status == cudaSuccess) {
        return true;
    }
    if (status == cudaErrorNotReady) {
        (void)cudaGetLastError();
        return false;
    }
    TYLOINFER_NATIVE_CUDA_CHECK(status);
    return false;
}

void Event::synchronize() const {
    DeviceGuard guard(device_);
    TYLOINFER_NATIVE_CUDA_CHECK(cudaEventSynchronize(event_));
}

Stream::Stream(int device, unsigned flags) : device_(device) {
    DeviceGuard guard(device_);
    TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, flags));
}

Stream::Stream(int device, unsigned flags, int priority) : device_(device) {
    DeviceGuard guard(device_);
    TYLOINFER_NATIVE_CUDA_CHECK(
        cudaStreamCreateWithPriority(&stream_, flags, priority));
}

Stream::~Stream() noexcept {
    if (stream_ != nullptr) {
        internal::on_device_noexcept(device_, [&] { (void)cudaStreamDestroy(stream_); });
    }
}

Stream::Stream(Stream&& other) noexcept
    : device_(other.device_), stream_(std::exchange(other.stream_, nullptr)) {}

Stream& Stream::operator=(Stream&& other) noexcept {
    if (this != &other) {
        if (stream_ != nullptr) {
            internal::on_device_noexcept(device_, [&] { (void)cudaStreamDestroy(stream_); });
        }
        device_ = other.device_;
        stream_ = std::exchange(other.stream_, nullptr);
    }
    return *this;
}

void Stream::synchronize() const {
    DeviceGuard guard(device_);
    TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void Stream::wait(const Event& event) const {
    DeviceGuard guard(device_);
    TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamWaitEvent(stream_, event.get(), 0));
}

}  // namespace mfq::cuda
