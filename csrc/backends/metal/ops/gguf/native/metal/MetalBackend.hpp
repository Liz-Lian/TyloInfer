#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace splash::metal {

class MetalBuffer {
public:
    MetalBuffer() = default;
    MetalBuffer(std::size_t slot, std::uint64_t bytes, std::uint64_t offset = 0)
        : slot_(slot), bytes_(bytes), offset_(offset) {}
    explicit operator bool() const noexcept { return bytes_ != 0; }
    std::uint64_t sizeBytes() const noexcept { return bytes_; }
    bool sameView(const MetalBuffer& other) const noexcept {
        return slot_ == other.slot_ && bytes_ == other.bytes_ && offset_ == other.offset_;
    }
    std::size_t slot() const noexcept { return slot_; }
    std::uint64_t offset() const noexcept { return offset_; }
private:
    std::size_t slot_ = 0;
    std::uint64_t bytes_ = 0, offset_ = 0;
};

class MetalBackend {
public:
    MetalBuffer view(const MetalBuffer& base, std::uint64_t offset, std::uint64_t bytes) const {
        if (offset > base.sizeBytes() || bytes > base.sizeBytes() - offset)
            throw std::invalid_argument("GGUF buffer view exceeds its allocation");
        return {base.slot(), bytes, base.offset() + offset};
    }
};

struct DispatchSize { std::uint64_t x = 1, y = 1, z = 1; };
struct BufferBinding { std::uint32_t index = 0; MetalBuffer buffer; };
struct BytesBinding { std::uint32_t index = 0; const void* data = nullptr; std::uint64_t sizeBytes = 0; };
struct IndirectGrid { MetalBuffer buffer; std::uint64_t offsetBytes = 0; };
struct ComputeDispatch {
    std::string pipelineName;
    std::vector<BufferBinding> buffers;
    std::vector<BytesBinding> bytes;
    DispatchSize threadgroups, threadsPerThreadgroup;
    IndirectGrid indirectThreadgroups;
};
struct SharedEvent {};
struct HostStepTiming { double pickupSeconds = 0, workSeconds = 0, resumeSeconds = 0; };
enum class HostStepGpu { Idle, Awake };
struct EventStep {
    enum class Kind { Signal, Wait, Host };
    std::size_t before = 0;
    SharedEvent event;
    std::uint64_t value = 0;
    Kind kind = Kind::Signal;
    std::function<void()> work;
    std::function<void(const HostStepTiming&)> timed;
    HostStepGpu gpu = HostStepGpu::Idle;
    std::function<void()> then;
};
struct Command { std::span<const ComputeDispatch> dispatches; std::span<const EventStep> events; };
}
