#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>
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

namespace tyloi {

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

inline constexpr ScalarType kBool = ScalarType::boolean;
inline constexpr ScalarType kUInt8 = ScalarType::uint8;
inline constexpr ScalarType kInt8 = ScalarType::int8;
inline constexpr ScalarType kInt16 = ScalarType::int16;
inline constexpr ScalarType kInt32 = ScalarType::int32;
inline constexpr ScalarType kInt64 = ScalarType::int64;
inline constexpr ScalarType kFloat16 = ScalarType::float16;
inline constexpr ScalarType kHalf = ScalarType::float16;
inline constexpr ScalarType kBFloat16 = ScalarType::bfloat16;
inline constexpr ScalarType kFloat32 = ScalarType::float32;
inline constexpr ScalarType kFloat64 = ScalarType::float64;
inline constexpr ScalarType kFloat8E4M3FN = ScalarType::float8_e4m3fn;
inline constexpr DeviceType kCPU = DeviceType::cpu;
inline constexpr DeviceType kCUDA = DeviceType::cuda;

namespace indexing {

struct NoneType final {};
inline constexpr NoneType None{};

class Slice final {
public:
    Slice() = default;
    Slice(std::int64_t start, std::int64_t end, std::int64_t step = 1)
        : start_(start), end_(end), step_(step), has_start_(true), has_end_(true) {}
    Slice(std::int64_t start, NoneType, std::int64_t step = 1)
        : start_(start), step_(step), has_start_(true) {}

    bool has_start() const noexcept { return has_start_; }
    bool has_end() const noexcept { return has_end_; }
    std::int64_t start() const noexcept { return start_; }
    std::int64_t end() const noexcept { return end_; }
    std::int64_t step() const noexcept { return step_; }

private:
    std::int64_t start_ = 0;
    std::int64_t end_ = 0;
    std::int64_t step_ = 1;
    bool has_start_ = false;
    bool has_end_ = false;
};

}  // namespace indexing

class TensorIndex;
class Tensor;
class TensorImpl;
class DenseTensor;
namespace encoding {
struct Tq;
class TqTensor;
}

// Representation interface: every implementation shares the logical
// shape/dtype metadata, layout identification, storage access and lifetime
// contract below. It never travels through model-side data flow; the public
// currency is the Tensor handle. Numeric entry points stay in ops; no compute
// virtuals.
class TensorImpl {
public:
    virtual ~TensorImpl() = default;
    virtual bool defined() const noexcept = 0;
    virtual std::int64_t dim() const noexcept = 0;
    virtual std::int64_t numel() const = 0;
    // Logical expansion bytes; physical bytes come from backing_nbytes().
    virtual std::size_t nbytes() const = 0;
    virtual std::size_t backing_nbytes() const = 0;
    virtual std::int64_t size(std::int64_t dimension) const = 0;
    virtual ScalarType scalar_type() const noexcept = 0;
    virtual const Device& device() const noexcept = 0;
    virtual bool is_contiguous() const noexcept = 0;
    virtual bool is_encoded() const noexcept = 0;
    // Every buffer this representation requires, in declaration order: values,
    // scales, codebooks, packed payloads. An undefined entry means the weight
    // is not prepared. Callers deduplicate by storage identity.
    //
    // View and cast rules: an encoded representation has no dense view of the
    // values it stands for, so reshape/narrow/cast never apply to it. buffers()
    // enumerates the payload for accounting, residency and keep-alive only.
    // The only path that changes a representation's storage is preparation,
    // which publishes a new one through relocated(); a published weight is
    // never rewritten in place.
    virtual std::vector<const DenseTensor*> buffers() const = 0;
    virtual void record_stream(std::uintptr_t stream) const = 0;

    // Backing buffers that exist, derived from buffers().
    std::vector<std::shared_ptr<TensorStorage>> backings() const;

    // A copy of this representation with every buffer resident on `device`,
    // keeping the encoding. Weight preparation uses this; a published weight
    // is never rewritten in place.
    virtual std::shared_ptr<TensorImpl> relocated(Device device) const = 0;

    ScalarType dtype() const noexcept { return scalar_type(); }
    int get_device() const noexcept { return device().index; }
    bool is_cpu() const noexcept { return defined() && device().is_cpu(); }
    bool is_cuda() const noexcept { return defined() && device().is_cuda(); }
    std::size_t element_size() const noexcept { return scalar_size(scalar_type()); }
    TensorOptions options() const noexcept {
        const auto storages = backings();
        return TensorOptions{}.dtype(scalar_type()).device(device()).pinned_memory(
            !storages.empty() && storages.front()->pinned);
    }
    void require_dense() const {
        if (is_encoded()) throw std::invalid_argument("operation requires a dense tensor");
    }
};

class DenseTensor final : public TensorImpl {
public:
    DenseTensor() = default;

    DenseTensor(
        std::shared_ptr<TensorStorage> storage,
        TensorView view,
        std::size_t byte_offset = 0)
        : storage_(std::move(storage)), view_(view), byte_offset_(byte_offset) {
        if (storage_ && storage_->base != nullptr) {
            view_.data = static_cast<std::byte*>(storage_->base) + byte_offset_;
        }
    }

    bool defined() const noexcept override { return storage_ != nullptr; }
    std::int64_t dim() const noexcept override { return defined() ? view_.rank : 0; }
    std::int64_t numel() const override { return defined() ? view_.numel() : 0; }
    std::size_t nbytes() const override { return defined() ? view_.nbytes() : 0; }
    // Bytes of the backing this tensor keeps alive, not of the view itself.
    std::size_t backing_nbytes() const override { return storage_ ? storage_->bytes : 0; }
    std::int64_t size(std::int64_t dimension) const override { return view_.size(dimension); }
    std::int64_t stride(std::int64_t dimension) const { return view_.stride(dimension); }
    ScalarType scalar_type() const noexcept override { return view_.scalar_type; }
    const Device& device() const noexcept override { return view_.device; }
    bool is_contiguous() const noexcept override { return defined() && view_.is_contiguous(); }
    bool is_encoded() const noexcept override { return false; }
    // DenseTensor is already the concrete representation; the accessor exists
    // so callers can treat a handle and a representation the same way.
    const DenseTensor& dense() const noexcept { return *this; }
    DenseTensor& dense() noexcept { return *this; }
    std::vector<const DenseTensor*> buffers() const override { return {this}; }
    std::shared_ptr<TensorImpl> relocated(Device device) const override;
    const std::shared_ptr<TensorStorage>& storage() const noexcept { return storage_; }
    std::size_t byte_offset() const noexcept { return byte_offset_; }
    void record_stream(std::uintptr_t stream) const override {
        if (storage_ && storage_->record_stream) storage_->record_stream(stream);
    }

    ScalarType dtype() const noexcept { return TensorImpl::dtype(); }
    std::size_t element_size() const noexcept { return TensorImpl::element_size(); }
    TensorOptions options() const noexcept { return TensorImpl::options(); }
    void require_dense() const { TensorImpl::require_dense(); }

    IntArrayRef sizes() const {
        return IntArrayRef(std::span<const std::int64_t>(view_.sizes).first(view_.rank));
    }

    const TensorView& view_descriptor() const { return view_; }
    DenseTensor alias(TensorView view, std::size_t byte_offset) const {
        return DenseTensor(storage_, view, byte_offset);
    }

    template <typename T>
    T* data_ptr() const {
        return static_cast<T*>(view_.data);
    }

    void* data_ptr() const { return view_.data; }

    DenseTensor reshape(std::span<const std::int64_t> shape) const;
    DenseTensor reshape(std::initializer_list<std::int64_t> shape) const {
        return reshape(std::span<const std::int64_t>(shape.begin(), shape.size()));
    }
    DenseTensor view(std::span<const std::int64_t> shape) const { return reshape(shape); }
    DenseTensor view(std::initializer_list<std::int64_t> shape) const { return reshape(shape); }
    DenseTensor transpose(std::int64_t first, std::int64_t second) const;
    DenseTensor permute(std::span<const std::int64_t> order) const;
    DenseTensor permute(std::initializer_list<std::int64_t> order) const {
        return permute(std::span<const std::int64_t>(order.begin(), order.size()));
    }
    DenseTensor unsqueeze(std::int64_t dimension) const;
    DenseTensor squeeze(std::int64_t dimension) const;
    DenseTensor squeeze() const;
    DenseTensor flatten(std::int64_t start = 0, std::int64_t end = -1) const;
    DenseTensor narrow(std::int64_t dimension, std::int64_t start, std::int64_t length) const;
    DenseTensor select(std::int64_t dimension, std::int64_t index) const;
    DenseTensor slice(
        std::int64_t dimension,
        std::int64_t start,
        std::int64_t end,
        std::int64_t step = 1) const;
    DenseTensor index(std::initializer_list<TensorIndex> indices) const;
    DenseTensor operator[](std::int64_t index) const { return select(0, index); }
    DenseTensor index_select(std::int64_t dimension, const DenseTensor& indices) const;
    DenseTensor gather(std::int64_t dimension, const DenseTensor& indices) const;
    DenseTensor expand(std::span<const std::int64_t> shape) const;
    DenseTensor expand(std::initializer_list<std::int64_t> shape) const {
        return expand(std::span<const std::int64_t>(shape.begin(), shape.size()));
    }
    DenseTensor repeat(std::span<const std::int64_t> repeats) const;
    DenseTensor repeat(std::initializer_list<std::int64_t> repeats) const {
        return repeat(std::span<const std::int64_t>(repeats.begin(), repeats.size()));
    }
    DenseTensor repeat_interleave(std::int64_t repeats, std::int64_t dimension) const;
    DenseTensor unfold(
        std::int64_t dimension,
        std::int64_t size,
        std::int64_t step) const;
    std::vector<DenseTensor> split_with_sizes(
        std::span<const std::int64_t> sections,
        std::int64_t dimension = 0) const;
    std::vector<DenseTensor> split_with_sizes(
        std::initializer_list<std::int64_t> sections,
        std::int64_t dimension = 0) const {
        return split_with_sizes(
            std::span<const std::int64_t>(sections.begin(), sections.size()),
            dimension);
    }
    std::vector<DenseTensor> chunk(std::int64_t chunks, std::int64_t dimension = 0) const;
    DenseTensor mean(std::int64_t dimension, bool keep_dimension = false) const;
    DenseTensor mean() const;
    DenseTensor sum(std::int64_t dimension, bool keep_dimension = false) const;
    DenseTensor sum() const;
    DenseTensor max() const;
    DenseTensor amax(std::int64_t dimension, bool keep_dimension = false) const;
    DenseTensor norm() const;
    DenseTensor abs() const;
    DenseTensor square() const;
    DenseTensor exp() const;
    DenseTensor sqrt() const;
    DenseTensor clamp(double minimum, double maximum) const;
    DenseTensor clamp_min(double minimum) const;
    DenseTensor clamp_max(double maximum) const;
    DenseTensor remainder(double divisor) const;
    DenseTensor argmax(std::int64_t dimension, bool keep_dimension = false) const;
    DenseTensor eq(const DenseTensor& other) const;
    DenseTensor eq(double other) const;
    DenseTensor ne(const DenseTensor& other) const;
    DenseTensor ne(double other) const;
    DenseTensor all() const;
    DenseTensor all(std::int64_t dimension, bool keep_dimension = false) const;
    DenseTensor any(std::int64_t dimension, bool keep_dimension = false) const;
    DenseTensor logical_not() const;
    DenseTensor detach() const { return *this; }
    bool equal(const DenseTensor& other) const;
    DenseTensor masked_select(const DenseTensor& mask) const;
    DenseTensor tril(std::int64_t diagonal = 0) const;
    DenseTensor masked_fill(const DenseTensor& mask, double value) const;
    DenseTensor& masked_fill_(const DenseTensor& mask, double value);
    DenseTensor reshape_as(const DenseTensor& other) const { return reshape(other.sizes()); }

    DenseTensor contiguous() const;
    DenseTensor clone() const;
    DenseTensor to(ScalarType type) const;
    DenseTensor to(Device device) const;
    DenseTensor to(DeviceType device) const { return to(Device{device, 0}); }
    DenseTensor to(Device device, bool, bool) const { return to(device); }
    DenseTensor to(const TensorOptions& options, bool = false, bool = false) const {
        const auto destination = options.has_device() ? options.target_device() : device();
        const auto type = options.has_dtype() ? options.scalar_type() : scalar_type();
        auto moved = destination == device()
            ? *this
            : to(destination);
        return type == moved.scalar_type()
            ? moved
            : moved.to(type);
    }
    DenseTensor to(DeviceType device, ScalarType type) const {
        return to(Device{device, 0}, type);
    }
    DenseTensor to(Device device, ScalarType type) const {
        if (device != this->device() && type != scalar_type()) {
            return to(type).to(device);
        }
        auto moved = device == this->device() ? *this : to(device);
        return type == moved.scalar_type() ? moved : moved.to(type);
    }
    DenseTensor cpu() const { return to(Device{DeviceType::cpu, 0}); }

    DenseTensor& copy_(const DenseTensor& source, bool non_blocking = false);
    DenseTensor& zero_();
    DenseTensor& fill_(double value);
    DenseTensor& add_(const DenseTensor& value);
    DenseTensor& scatter_add_(std::int64_t dimension, const DenseTensor& index, const DenseTensor& source);
    DenseTensor& scatter_(std::int64_t dimension, const DenseTensor& index, const DenseTensor& source);
    DenseTensor& index_copy_(std::int64_t dimension, const DenseTensor& index, const DenseTensor& source);
    DenseTensor& index_fill_(std::int64_t dimension, const DenseTensor& index, double value);
    DenseTensor& index_put_(std::initializer_list<TensorIndex> indices, const DenseTensor& value);
    DenseTensor& index_put_(std::initializer_list<TensorIndex> indices, double value);

    template <typename T>
    T item() const {
        if (!defined() || numel() != 1) {
            throw std::invalid_argument("item() requires a defined scalar tensor");
        }
        const DenseTensor host = is_cpu() && is_contiguous() ? *this : cpu().contiguous();
        return scalar_cast<T>(host.data_ptr(), host.scalar_type());
    }

private:
    template <typename T>
    static T scalar_cast(const void* pointer, ScalarType type) {
        switch (type) {
            case ScalarType::boolean: return static_cast<T>(*static_cast<const bool*>(pointer));
            case ScalarType::uint8: return static_cast<T>(*static_cast<const std::uint8_t*>(pointer));
            case ScalarType::int8: return static_cast<T>(*static_cast<const std::int8_t*>(pointer));
            case ScalarType::int16: return static_cast<T>(*static_cast<const std::int16_t*>(pointer));
            case ScalarType::int32: return static_cast<T>(*static_cast<const std::int32_t*>(pointer));
            case ScalarType::int64: return static_cast<T>(*static_cast<const std::int64_t*>(pointer));
            case ScalarType::float32: return static_cast<T>(*static_cast<const float*>(pointer));
            case ScalarType::float64: return static_cast<T>(*static_cast<const double*>(pointer));
            case ScalarType::float16: {
                const auto bits = *static_cast<const std::uint16_t*>(pointer);
                const auto sign = static_cast<std::uint32_t>(bits & 0x8000u) << 16;
                auto exponent = static_cast<std::uint32_t>((bits >> 10) & 0x1fu);
                auto mantissa = static_cast<std::uint32_t>(bits & 0x03ffu);
                std::uint32_t value = 0;
                if (exponent == 0) {
                    if (mantissa != 0) {
                        exponent = 127 - 15 + 1;
                        while ((mantissa & 0x0400u) == 0) {
                            mantissa <<= 1;
                            --exponent;
                        }
                        mantissa &= 0x03ffu;
                        value = sign | (exponent << 23) | (mantissa << 13);
                    } else {
                        value = sign;
                    }
                } else if (exponent == 0x1fu) {
                    value = sign | 0x7f800000u | (mantissa << 13);
                } else {
                    value = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
                }
                float decoded = 0.0f;
                std::memcpy(&decoded, &value, sizeof(decoded));
                return static_cast<T>(decoded);
            }
            case ScalarType::bfloat16: {
                const auto value = static_cast<std::uint32_t>(
                    *static_cast<const std::uint16_t*>(pointer)) << 16;
                float decoded = 0.0f;
                std::memcpy(&decoded, &value, sizeof(decoded));
                return static_cast<T>(decoded);
            }
            case ScalarType::float8_e4m3fn: {
                const auto bits = *static_cast<const std::uint8_t*>(pointer);
                const double sign = (bits & 0x80u) != 0 ? -1.0 : 1.0;
                const auto exponent = static_cast<int>((bits >> 3) & 0x0fu);
                const auto mantissa = static_cast<int>(bits & 0x07u);
                if (exponent == 0) {
                    return static_cast<T>(sign * std::ldexp(
                        static_cast<double>(mantissa) / 8.0, -6));
                }
                if (exponent == 0x0f && mantissa == 0x07) {
                    return static_cast<T>(std::numeric_limits<double>::quiet_NaN());
                }
                return static_cast<T>(sign * std::ldexp(
                    1.0 + static_cast<double>(mantissa) / 8.0,
                    exponent - 7));
            }
        }
        throw std::invalid_argument("unknown tensor scalar type");
    }

    std::shared_ptr<TensorStorage> storage_;
    TensorView view_{};
    std::size_t byte_offset_ = 0;
};

// Public data currency for models, state and ops: a copyable handle over a
// polymorphic representation. Copying a handle shares the implementation and
// its backing, never the data. Callers never dereference or unwrap it.
class Tensor {
public:
    Tensor() = default;
    Tensor(std::shared_ptr<TensorImpl> implementation) noexcept
        : impl_(std::move(implementation)) {}
    // A DenseTensor is already a representation; wrapping it copies only the
    // view metadata and shares the backing.
    Tensor(DenseTensor value)
        : impl_(std::make_shared<DenseTensor>(std::move(value))) {}
    template <typename Impl, typename = std::enable_if_t<
                 std::is_base_of<TensorImpl, Impl>::value &&
                 !std::is_same<Impl, TensorImpl>::value>>
    Tensor(std::shared_ptr<Impl> implementation)
        : impl_(std::move(implementation)) {}
    // Loading hands out owned representations that are read-only by convention;
    // the handle has no separate const type.
    template <typename Impl, typename = std::enable_if_t<
                 std::is_base_of<TensorImpl, Impl>::value>>
    Tensor(std::shared_ptr<const Impl> implementation)
        : impl_(std::const_pointer_cast<TensorImpl>(
              std::shared_ptr<const TensorImpl>(std::move(implementation)))) {}

    bool defined() const noexcept { return impl_ != nullptr && impl_->defined(); }
    explicit operator bool() const noexcept { return defined(); }

    /** Compare view identity; distinct representations match only for identical dense geometry and data. */
    bool same_view(const Tensor& other) const;

    // Concrete representation access stays in loading, storage and ops code.
    const DenseTensor& dense() const;
    DenseTensor& dense();
    // Representation probe: null when the handle holds another representation.
    template <typename Impl>
    const Impl* representation() const noexcept {
        return dynamic_cast<const Impl*>(impl_.get());
    }
    template <typename Impl>
    Impl* representation() noexcept {
        return dynamic_cast<Impl*>(impl_.get());
    }
    const std::shared_ptr<TensorImpl>& implementation() const noexcept { return impl_; }

    std::int64_t dim() const noexcept { return reference().dim(); }
    std::int64_t numel() const { return reference().numel(); }
    std::size_t nbytes() const { return reference().nbytes(); }
    std::size_t backing_nbytes() const { return reference().backing_nbytes(); }
    std::int64_t size(std::int64_t dimension) const { return reference().size(dimension); }
    ScalarType scalar_type() const noexcept { return reference().scalar_type(); }
    const Device& device() const noexcept { return reference().device(); }
    bool is_contiguous() const noexcept { return reference().is_contiguous(); }
    bool is_encoded() const noexcept { return reference().is_encoded(); }
    std::vector<const DenseTensor*> buffers() const { return reference().buffers(); }
    std::vector<std::shared_ptr<TensorStorage>> backings() const { return reference().backings(); }
    void record_stream(std::uintptr_t stream) const { reference().record_stream(stream); }
    Tensor relocated(Device device) const { return Tensor{reference().relocated(device)}; }

    ScalarType dtype() const noexcept { return reference().dtype(); }
    int get_device() const noexcept { return reference().get_device(); }
    bool is_cpu() const noexcept { return reference().is_cpu(); }
    bool is_cuda() const noexcept { return reference().is_cuda(); }
    std::size_t element_size() const noexcept { return reference().element_size(); }
    TensorOptions options() const noexcept { return reference().options(); }
    void require_dense() const { reference().require_dense(); }

private:
    // An empty handle reports the same metadata as an undefined DenseTensor.
    const TensorImpl& reference() const noexcept;

    std::shared_ptr<TensorImpl> impl_;
};

class TensorIndex final {
public:
    enum class Kind : std::uint8_t { integer, slice, tensor };

    TensorIndex(std::int64_t value) : kind_(Kind::integer), integer_(value) {}
    TensorIndex(int value) : TensorIndex(static_cast<std::int64_t>(value)) {}
    TensorIndex(const indexing::Slice& value) : kind_(Kind::slice), slice_(value) {}
    TensorIndex(const DenseTensor& value)
        : kind_(Kind::tensor), tensor_(std::make_shared<DenseTensor>(value)) {}

    Kind kind() const noexcept { return kind_; }
    std::int64_t integer() const noexcept { return integer_; }
    const indexing::Slice& slice() const noexcept { return slice_; }
    const DenseTensor& tensor() const { return *tensor_; }

private:
    Kind kind_ = Kind::integer;
    std::int64_t integer_ = 0;
    indexing::Slice slice_{};
    std::shared_ptr<DenseTensor> tensor_;
};

// View metadata helpers: they share the backing storage and only build a new
// DenseTensor view. DenseTensor members and ops:: both enter through here.
DenseTensor reshape(const DenseTensor& source, std::span<const std::int64_t> shape);
DenseTensor transpose(const DenseTensor& source, std::int64_t first, std::int64_t second);
DenseTensor permute(const DenseTensor& source, std::span<const std::int64_t> order);
DenseTensor unsqueeze(const DenseTensor& source, std::int64_t dimension);
DenseTensor squeeze(const DenseTensor& source, std::int64_t dimension);
DenseTensor squeeze(const DenseTensor& source);
DenseTensor flatten(const DenseTensor& source, std::int64_t start, std::int64_t end);
DenseTensor narrow(const DenseTensor& source, std::int64_t dimension, std::int64_t start, std::int64_t length);
DenseTensor select(const DenseTensor& source, std::int64_t dimension, std::int64_t index);
DenseTensor slice(const DenseTensor& source, std::int64_t dimension, std::int64_t start, std::int64_t end, std::int64_t step);
DenseTensor index(const DenseTensor& source, std::initializer_list<TensorIndex> indices);
DenseTensor expand(const DenseTensor& source, std::span<const std::int64_t> shape);
DenseTensor unfold(const DenseTensor& source, std::int64_t dimension, std::int64_t window, std::int64_t step);
std::vector<DenseTensor> split_with_sizes(const DenseTensor& source, std::span<const std::int64_t> sections, std::int64_t dimension);
std::vector<DenseTensor> chunk(const DenseTensor& source, std::int64_t chunks, std::int64_t dimension);

// Physical bytes of a set of buffers, counted once per backing identity.
// Aliased views of one storage contribute only the first one seen.
std::size_t backing_bytes(std::span<const DenseTensor* const> buffers);

DenseTensor operator+(const DenseTensor& left, const DenseTensor& right);
DenseTensor operator+(const DenseTensor& left, double right);
DenseTensor operator+(double left, const DenseTensor& right);
DenseTensor operator-(const DenseTensor& left, const DenseTensor& right);
DenseTensor operator-(const DenseTensor& left, double right);
DenseTensor operator-(double left, const DenseTensor& right);
DenseTensor operator-(const DenseTensor& value);
DenseTensor operator*(const DenseTensor& left, const DenseTensor& right);
DenseTensor operator*(const DenseTensor& left, double right);
DenseTensor operator*(double left, const DenseTensor& right);
DenseTensor operator/(const DenseTensor& left, const DenseTensor& right);
DenseTensor operator/(const DenseTensor& left, double right);
DenseTensor operator/(double left, const DenseTensor& right);
DenseTensor operator<(const DenseTensor& left, const DenseTensor& right);
DenseTensor operator<(const DenseTensor& left, double right);
DenseTensor operator<=(const DenseTensor& left, const DenseTensor& right);
DenseTensor operator<=(const DenseTensor& left, double right);
DenseTensor operator>(const DenseTensor& left, const DenseTensor& right);
DenseTensor operator>(const DenseTensor& left, double right);
DenseTensor operator>=(const DenseTensor& left, const DenseTensor& right);
DenseTensor operator>=(const DenseTensor& left, double right);
DenseTensor operator&(const DenseTensor& left, const DenseTensor& right);
DenseTensor operator|(const DenseTensor& left, const DenseTensor& right);
DenseTensor operator~(const DenseTensor& value);
DenseTensor operator==(const DenseTensor& left, const DenseTensor& right);
DenseTensor operator==(const DenseTensor& left, double right);
DenseTensor operator!=(const DenseTensor& left, const DenseTensor& right);
DenseTensor operator!=(const DenseTensor& left, double right);

}  // namespace tyloi
