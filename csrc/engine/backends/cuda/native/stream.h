#pragma once

#include <cuda_runtime_api.h>

#include <memory>
#include <utility>

namespace mfq::cuda {

class Event final {
public:
    explicit Event(unsigned flags = cudaEventDisableTiming);
    ~Event() noexcept;

    Event(Event&& other) noexcept;
    Event& operator=(Event&& other) noexcept;
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    cudaEvent_t get() const noexcept { return event_; }
    void record(cudaStream_t stream);
    bool ready() const;
    void synchronize() const;

private:
    int device_ = 0;
    cudaEvent_t event_ = nullptr;
};

class Stream final {
public:
    explicit Stream(int device, unsigned flags = cudaStreamNonBlocking);
    Stream(int device, unsigned flags, int priority);
    ~Stream() noexcept;

    Stream(Stream&& other) noexcept;
    Stream& operator=(Stream&& other) noexcept;
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    int device() const noexcept { return device_; }
    cudaStream_t get() const noexcept { return stream_; }
    void synchronize() const;
    void wait(const Event& event) const;

private:
    int device_ = 0;
    cudaStream_t stream_ = nullptr;
};

// Owning handles retain a Stream or alias a Context-owned Stream.
class StreamHandle final {
public:
    StreamHandle() = default;
    StreamHandle(int device, cudaStream_t stream, std::shared_ptr<Stream> owner = {})
        : device_(device), stream_(stream), owner_(std::move(owner)) {}

    int device_index() const noexcept { return device_; }
    int device() const noexcept { return device_; }
    cudaStream_t stream() const noexcept { return stream_; }
    operator cudaStream_t() const noexcept { return stream_; }
    explicit operator bool() const noexcept { return stream_ != nullptr; }

private:
    int device_ = 0;
    cudaStream_t stream_ = nullptr;
    std::shared_ptr<Stream> owner_;
};

}  // namespace mfq::cuda
