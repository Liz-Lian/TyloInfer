#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <numeric>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfq {

enum class ScalarType : std::uint8_t {
    boolean,
    uint8,
    int8,
    int16,
    int32,
    int64,
    float16,
    bfloat16,
    float32,
    float64,
    float8_e4m3fn,
};

enum class DeviceType : std::uint8_t {
    cpu,
    cuda,
};

inline constexpr std::size_t kMaximumTensorRank = 8;

constexpr std::size_t scalar_size(ScalarType type) {
    switch (type) {
        case ScalarType::boolean:
        case ScalarType::uint8:
        case ScalarType::int8:
        case ScalarType::float8_e4m3fn:
            return 1;
        case ScalarType::int16:
        case ScalarType::float16:
        case ScalarType::bfloat16:
            return 2;
        case ScalarType::int32:
        case ScalarType::float32:
            return 4;
        case ScalarType::int64:
        case ScalarType::float64:
            return 8;
    }
    return 0;
}

struct Device {
    DeviceType type = DeviceType::cpu;
    std::int32_t index = 0;

    constexpr Device() noexcept = default;
    constexpr explicit Device(DeviceType device_type, std::int32_t device_index = 0) noexcept
        : type(device_type), index(device_index) {}

    constexpr bool is_cpu() const noexcept { return type == DeviceType::cpu; }
    constexpr bool is_cuda() const noexcept { return type == DeviceType::cuda; }
    constexpr bool operator==(const Device&) const noexcept = default;
};

struct TensorView {
    void* data = nullptr;
    std::array<std::int64_t, kMaximumTensorRank> sizes{};
    std::array<std::int64_t, kMaximumTensorRank> strides{};
    std::uint8_t rank = 0;
    ScalarType scalar_type = ScalarType::float32;
    Device device{};

    constexpr bool defined() const noexcept { return data != nullptr; }

    std::int64_t size(std::int64_t dimension) const {
        const auto normalized = normalize_dimension(dimension);
        return sizes[normalized];
    }

    std::int64_t stride(std::int64_t dimension) const {
        const auto normalized = normalize_dimension(dimension);
        return strides[normalized];
    }

    std::int64_t numel() const {
        if (rank == 0) {
            return defined() ? 1 : 0;
        }
        std::int64_t result = 1;
        for (std::size_t index = 0; index < rank; ++index) {
            if (sizes[index] < 0 ||
                (sizes[index] != 0 &&
                 result > std::numeric_limits<std::int64_t>::max() / sizes[index])) {
                throw std::overflow_error("tensor element count overflow");
            }
            result *= sizes[index];
        }
        return result;
    }

    std::size_t nbytes() const {
        const auto elements = numel();
        const auto width = scalar_size(scalar_type);
        if (elements < 0 ||
            static_cast<std::uint64_t>(elements) >
                std::numeric_limits<std::size_t>::max() / width) {
            throw std::overflow_error("tensor byte count overflow");
        }
        return static_cast<std::size_t>(elements) * width;
    }

    bool is_contiguous() const noexcept {
        std::int64_t expected = 1;
        for (std::size_t reverse = rank; reverse > 0; --reverse) {
            const auto index = reverse - 1;
            if (sizes[index] > 1 && strides[index] != expected) {
                return false;
            }
            expected *= sizes[index];
        }
        return true;
    }

    template <typename T>
    T* data_as() const noexcept {
        return static_cast<T*>(data);
    }

private:
    std::size_t normalize_dimension(std::int64_t dimension) const {
        const auto normalized = dimension < 0 ? dimension + rank : dimension;
        if (normalized < 0 || normalized >= rank) {
            throw std::out_of_range("tensor dimension is out of range");
        }
        return static_cast<std::size_t>(normalized);
    }
};

inline TensorView make_contiguous_view(
    void* data,
    std::span<const std::int64_t> shape,
    ScalarType scalar_type,
    Device device) {
    if (shape.size() > kMaximumTensorRank) {
        throw std::invalid_argument("tensor rank exceeds the native CUDA ABI limit");
    }
    TensorView result;
    result.data = data;
    result.rank = static_cast<std::uint8_t>(shape.size());
    result.scalar_type = scalar_type;
    result.device = device;
    std::int64_t stride = 1;
    for (std::size_t reverse = shape.size(); reverse > 0; --reverse) {
        const auto index = reverse - 1;
        if (shape[index] < 0) {
            throw std::invalid_argument("tensor shape cannot contain a negative extent");
        }
        result.sizes[index] = shape[index];
        result.strides[index] = stride;
        if (shape[index] != 0 &&
            stride > std::numeric_limits<std::int64_t>::max() / shape[index]) {
            throw std::overflow_error("tensor stride overflow");
        }
        stride *= shape[index];
    }
    return result;
}

inline TensorView reshape_view(
    const TensorView& source,
    std::span<const std::int64_t> requested_shape) {
    if (!source.is_contiguous()) {
        throw std::invalid_argument("reshape_view requires contiguous storage");
    }
    if (requested_shape.size() > kMaximumTensorRank) {
        throw std::invalid_argument("tensor rank exceeds the native CUDA ABI limit");
    }

    std::vector<std::int64_t> resolved(requested_shape.begin(), requested_shape.end());
    std::int64_t inferred_index = -1;
    std::int64_t known_elements = 1;
    for (std::size_t index = 0; index < resolved.size(); ++index) {
        const auto extent = resolved[index];
        if (extent == -1) {
            if (inferred_index >= 0) {
                throw std::invalid_argument("reshape can infer only one dimension");
            }
            inferred_index = static_cast<std::int64_t>(index);
            continue;
        }
        if (extent < 0) {
            throw std::invalid_argument("reshape extent must be non-negative or -1");
        }
        if (extent != 0 &&
            known_elements > std::numeric_limits<std::int64_t>::max() / extent) {
            throw std::overflow_error("reshape element count overflow");
        }
        known_elements *= extent;
    }

    const auto source_elements = source.numel();
    if (inferred_index >= 0) {
        if (known_elements == 0 || source_elements % known_elements != 0) {
            throw std::invalid_argument("reshape cannot infer a compatible extent");
        }
        resolved[static_cast<std::size_t>(inferred_index)] = source_elements / known_elements;
    } else if (known_elements != source_elements) {
        throw std::invalid_argument("reshape changes the tensor element count");
    }

    return make_contiguous_view(
        source.data,
        resolved,
        source.scalar_type,
        source.device);
}

inline TensorView transpose_view(
    const TensorView& source,
    std::int64_t first_dimension,
    std::int64_t second_dimension) {
    auto result = source;
    auto normalize = [&](std::int64_t dimension) -> std::size_t {
        const auto normalized = dimension < 0 ? dimension + source.rank : dimension;
        if (normalized < 0 || normalized >= source.rank) {
            throw std::out_of_range("transpose dimension is out of range");
        }
        return static_cast<std::size_t>(normalized);
    };
    const auto first = normalize(first_dimension);
    const auto second = normalize(second_dimension);
    std::swap(result.sizes[first], result.sizes[second]);
    std::swap(result.strides[first], result.strides[second]);
    return result;
}

class TensorOptions {
public:
    TensorOptions& dtype(ScalarType value) & noexcept {
        scalar_type_ = value;
        has_scalar_type_ = true;
        return *this;
    }
    TensorOptions dtype(ScalarType value) const & noexcept {
        auto result = *this;
        result.scalar_type_ = value;
        result.has_scalar_type_ = true;
        return result;
    }
    TensorOptions&& dtype(ScalarType value) && noexcept {
        scalar_type_ = value;
        has_scalar_type_ = true;
        return std::move(*this);
    }

    TensorOptions& device(Device value) & noexcept {
        device_ = value;
        has_device_ = true;
        return *this;
    }
    TensorOptions device(Device value) const & noexcept {
        auto result = *this;
        result.device_ = value;
        result.has_device_ = true;
        return result;
    }
    TensorOptions&& device(Device value) && noexcept {
        device_ = value;
        has_device_ = true;
        return std::move(*this);
    }

    TensorOptions& device(DeviceType value) & noexcept {
        device_ = Device{value, 0};
        has_device_ = true;
        return *this;
    }
    TensorOptions device(DeviceType value) const & noexcept {
        return device(Device{value, 0});
    }
    TensorOptions&& device(DeviceType value) && noexcept {
        device_ = Device{value, 0};
        has_device_ = true;
        return std::move(*this);
    }

    TensorOptions& pinned_memory(bool value) & noexcept {
        pinned_ = value;
        has_pinned_ = true;
        return *this;
    }
    TensorOptions pinned_memory(bool value) const & noexcept {
        auto result = *this;
        result.pinned_ = value;
        result.has_pinned_ = true;
        return result;
    }
    TensorOptions&& pinned_memory(bool value) && noexcept {
        pinned_ = value;
        has_pinned_ = true;
        return std::move(*this);
    }

    ScalarType scalar_type() const noexcept { return scalar_type_; }
    const Device& target_device() const noexcept { return device_; }
    bool pinned() const noexcept { return pinned_; }
    bool has_dtype() const noexcept { return has_scalar_type_; }
    bool has_device() const noexcept { return has_device_; }
    bool has_pinned_memory() const noexcept { return has_pinned_; }

private:
    ScalarType scalar_type_ = ScalarType::float32;
    Device device_{};
    bool pinned_ = false;
    bool has_scalar_type_ = false;
    bool has_device_ = false;
    bool has_pinned_ = false;
};

struct TensorStorage {
    std::shared_ptr<void> owner;
    void* base = nullptr;
    std::size_t bytes = 0;
    Device device{};
    bool pinned = false;
    std::function<void(std::uintptr_t)> record_stream;
};

class IntArrayRef {
public:
    IntArrayRef() = default;
    IntArrayRef(std::span<const std::int64_t> values) : values_(values.begin(), values.end()) {}
    IntArrayRef(std::initializer_list<std::int64_t> values) : values_(values) {}
    explicit IntArrayRef(std::vector<std::int64_t> values) : values_(std::move(values)) {}

    std::size_t size() const noexcept { return values_.size(); }
    bool empty() const noexcept { return values_.empty(); }
    const std::int64_t* data() const noexcept { return values_.data(); }
    const std::int64_t& operator[](std::size_t index) const { return values_.at(index); }
    auto begin() const noexcept { return values_.begin(); }
    auto end() const noexcept { return values_.end(); }
    std::vector<std::int64_t> vec() const { return values_; }
    operator std::span<const std::int64_t>() const noexcept { return values_; }

    IntArrayRef slice(std::size_t start, std::size_t length) const {
        if (start > values_.size() || length > values_.size() - start) {
            throw std::out_of_range("shape slice is out of range");
        }
        return IntArrayRef(std::span<const std::int64_t>(values_).subspan(start, length));
    }

    bool operator==(const IntArrayRef&) const noexcept = default;

private:
    std::vector<std::int64_t> values_;
};

inline std::ostream& operator<<(std::ostream& stream, const Device& device) {
    stream << (device.is_cuda() ? "cuda" : "cpu");
    if (device.is_cuda() || device.index != 0) stream << ':' << device.index;
    return stream;
}

inline std::ostream& operator<<(std::ostream& stream, const IntArrayRef& values) {
    stream << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) stream << ", ";
        stream << values[index];
    }
    return stream << ']';
}

class Tensor {
public:
    Tensor() = default;

    Tensor(
        std::shared_ptr<TensorStorage> storage,
        TensorView view,
        std::size_t byte_offset = 0)
        : storage_(std::move(storage)), view_(view), byte_offset_(byte_offset) {
        if (storage_ && storage_->base != nullptr) {
            view_.data = static_cast<std::byte*>(storage_->base) + byte_offset_;
        }
    }

    bool defined() const noexcept { return storage_ != nullptr; }
    std::int64_t dim() const noexcept { return defined() ? view_.rank : 0; }
    std::int64_t numel() const { return defined() ? view_.numel() : 0; }
    std::size_t nbytes() const { return defined() ? view_.nbytes() : 0; }
    std::int64_t size(std::int64_t dimension) const { return view_.size(dimension); }
    std::int64_t stride(std::int64_t dimension) const { return view_.stride(dimension); }
    ScalarType scalar_type() const noexcept { return view_.scalar_type; }
    ScalarType dtype() const noexcept { return scalar_type(); }
    const Device& device() const noexcept { return view_.device; }
    int get_device() const noexcept { return device().index; }
    bool is_cpu() const noexcept { return defined() && device().is_cpu(); }
    bool is_cuda() const noexcept { return defined() && device().is_cuda(); }
    bool is_contiguous() const noexcept { return defined() && view_.is_contiguous(); }
    std::size_t element_size() const noexcept { return scalar_size(scalar_type()); }

    IntArrayRef sizes() const {
        return IntArrayRef(std::span<const std::int64_t>(view_.sizes).first(view_.rank));
    }

    const TensorView& view_descriptor() const noexcept { return view_; }
    std::size_t byte_offset() const noexcept { return byte_offset_; }
    void record_stream(std::uintptr_t stream) const {
        if (storage_ && storage_->record_stream) storage_->record_stream(stream);
    }

    template <typename T>
    T* data_ptr() const noexcept {
        return static_cast<T*>(view_.data);
    }

    void* data_ptr() const noexcept { return view_.data; }

    TensorOptions options() const noexcept {
        return TensorOptions{}.dtype(scalar_type()).device(device()).pinned_memory(
            storage_ && storage_->pinned);
    }

protected:
    std::shared_ptr<TensorStorage> storage_;
    TensorView view_{};
    std::size_t byte_offset_ = 0;
};

}  // namespace mfq
