#pragma once

#include "engine/tensor.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <optional>
#include <ostream>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace tyloi::cuda {

// Owned CPU tensors and pinned transfer buffers; tensor views share storage
// and must not double-count it. File/codec buffers and driver memory are
// observed separately through process physical working-set measurements.
inline std::atomic<std::size_t> tensor_host_bytes{0},tensor_host_peak_bytes{0};
inline void charge_tensor_host_bytes(std::size_t bytes) {
    const auto used=tensor_host_bytes.fetch_add(bytes)+bytes;
    auto peak=tensor_host_peak_bytes.load();
    while (peak<used && !tensor_host_peak_bytes.compare_exchange_weak(peak,used)) {}
}

class NoGradGuard final {
public:
    NoGradGuard() = default;
};

DenseTensor contiguous(const DenseTensor& input);
DenseTensor clone(const DenseTensor& input);
DenseTensor cast(const DenseTensor& input, ScalarType type);
DenseTensor cast(const DenseTensor& input, ScalarType type, Device target);
DenseTensor copy(const DenseTensor& input, Device target);
void copy_into(DenseTensor& destination, const DenseTensor& source, bool non_blocking = false);

DenseTensor reshape(const DenseTensor& input, std::span<const std::int64_t> shape);
DenseTensor transpose(const DenseTensor& input, std::int64_t first, std::int64_t second);
DenseTensor permute(const DenseTensor& input, std::span<const std::int64_t> order);
DenseTensor squeeze(const DenseTensor& input, std::int64_t dimension);
DenseTensor squeeze(const DenseTensor& input);
DenseTensor flatten(const DenseTensor& input, std::int64_t start, std::int64_t end);
DenseTensor narrow(const DenseTensor& input, std::int64_t dimension, std::int64_t start, std::int64_t length);
DenseTensor select(const DenseTensor& input, std::int64_t dimension, std::int64_t index);
DenseTensor slice(const DenseTensor& input, std::int64_t dimension, std::int64_t start, std::int64_t end, std::int64_t step);
DenseTensor index(const DenseTensor& input, std::initializer_list<TensorIndex> indices);
DenseTensor unfold(const DenseTensor& input, std::int64_t dimension, std::int64_t window, std::int64_t step);
std::vector<DenseTensor> split_with_sizes(const DenseTensor& input, std::span<const std::int64_t> sections, std::int64_t dimension);
std::vector<DenseTensor> chunk(const DenseTensor& input, std::int64_t chunks, std::int64_t dimension);

DenseTensor binary_cuda(const DenseTensor& left, const DenseTensor& right, int operation);
DenseTensor scalar_binary_cuda(const DenseTensor& left, double right, int operation, bool scalar_first = false);
DenseTensor unary_cuda(const DenseTensor& source, int operation, double first = 0.0, double second = 0.0);
DenseTensor reduce_cuda(
    const DenseTensor& source,
    std::int64_t dimension,
    bool keep_dimension,
    int operation);
void fill_cuda(DenseTensor& destination, double value);
/** Complete host transfers by default; asynchronous callers retain host storage until the stream finishes. */
void copy_cuda(DenseTensor& destination, const DenseTensor& source, bool non_blocking = false);
/** Encode one host value using the native FP16 rounding path. */
std::uint16_t encode_float16(float value);
/** Encode one host value using the native BF16 rounding path. */
std::uint16_t encode_bfloat16(float value);
DenseTensor index_select_cuda(const DenseTensor& source, std::int64_t dimension, const DenseTensor& indices);
DenseTensor gather_cuda(const DenseTensor& source, std::int64_t dimension, const DenseTensor& indices);
DenseTensor repeat_cuda(const DenseTensor& source, std::span<const std::int64_t> repeats);
DenseTensor repeat_interleave_cuda(const DenseTensor& source, std::int64_t repeats, std::int64_t dimension);
DenseTensor masked_select_cuda(const DenseTensor& source, const DenseTensor& mask);
void scatter_cuda(DenseTensor& destination, std::int64_t dimension, const DenseTensor& index, const DenseTensor& source, bool add);
void index_copy_cuda(DenseTensor& destination, std::int64_t dimension, const DenseTensor& index, const DenseTensor& source);
void index_fill_cuda(DenseTensor& destination, std::int64_t dimension, const DenseTensor& index, double value);
void masked_fill_cuda(DenseTensor& destination, const DenseTensor& mask, double value);

DenseTensor empty(std::span<const std::int64_t> shape, const TensorOptions& options = {});
DenseTensor empty(std::initializer_list<std::int64_t> shape, const TensorOptions& options = {});
DenseTensor empty_like(const DenseTensor& source, const TensorOptions* options = nullptr);
DenseTensor zeros(std::span<const std::int64_t> shape, const TensorOptions& options = {});
DenseTensor zeros(std::initializer_list<std::int64_t> shape, const TensorOptions& options = {});
DenseTensor zeros_like(const DenseTensor& source);
DenseTensor ones(std::span<const std::int64_t> shape, const TensorOptions& options = {});
DenseTensor ones(std::initializer_list<std::int64_t> shape, const TensorOptions& options = {});
DenseTensor full(std::span<const std::int64_t> shape, double value, const TensorOptions& options = {});
DenseTensor full(std::initializer_list<std::int64_t> shape, double value, const TensorOptions& options = {});
DenseTensor full_like(const DenseTensor& source, double value);
DenseTensor ones_like(const DenseTensor& source);
DenseTensor from_blob(
    void* data,
    std::span<const std::int64_t> shape,
    const TensorOptions& options = {});
DenseTensor from_blob(
    void* data,
    std::initializer_list<std::int64_t> shape,
    const TensorOptions& options = {});

DenseTensor linear(
    const DenseTensor& input,
    const DenseTensor& weight,
    const std::optional<DenseTensor>& bias = std::nullopt);
DenseTensor scaled_dot_product_attention(
    const DenseTensor& query,
    const DenseTensor& key,
    const DenseTensor& value,
    const std::optional<DenseTensor>& mask,
    double dropout,
    bool causal,
    const std::optional<double>& scale,
    bool enable_grouped_query_attention);

DenseTensor cat(std::span<const DenseTensor> tensors, std::int64_t dimension = 0);
DenseTensor cat(std::initializer_list<DenseTensor> tensors, std::int64_t dimension = 0);
DenseTensor cat(const std::vector<DenseTensor>& tensors, std::int64_t dimension = 0);
DenseTensor stack(std::span<const DenseTensor> tensors, std::int64_t dimension = 0);
DenseTensor stack(std::initializer_list<DenseTensor> tensors, std::int64_t dimension = 0);
DenseTensor stack(const std::vector<DenseTensor>& tensors, std::int64_t dimension = 0);
DenseTensor arange(std::int64_t end, const TensorOptions& options = {});
DenseTensor arange(std::int64_t start, std::int64_t end, const TensorOptions& options = {});
DenseTensor arange(
    std::int64_t start,
    std::int64_t end,
    std::int64_t step,
    const TensorOptions& options = {});
DenseTensor randn(std::span<const std::int64_t> shape, const TensorOptions& options = {});
DenseTensor randn(std::initializer_list<std::int64_t> shape, const TensorOptions& options = {});
DenseTensor randint(
    std::int64_t high,
    std::span<const std::int64_t> shape,
    const TensorOptions& options = {});
DenseTensor randint(
    std::int64_t low,
    std::int64_t high,
    std::span<const std::int64_t> shape,
    const TensorOptions& options = {});
inline DenseTensor randint(
    std::int64_t low,
    std::int64_t high,
    std::initializer_list<std::int64_t> shape,
    const TensorOptions& options = {}) {
    return randint(
        low, high,
        std::span<const std::int64_t>(shape.begin(), shape.size()),
        options);
}
DenseTensor randperm(std::int64_t size, const TensorOptions& options = {});
void manual_seed(std::int64_t seed);

DenseTensor matmul(const DenseTensor& left, const DenseTensor& right);
DenseTensor bmm(const DenseTensor& left, const DenseTensor& right);
DenseTensor baddbmm(const DenseTensor& input, const DenseTensor& left, const DenseTensor& right);
DenseTensor where(const DenseTensor& condition, const DenseTensor& yes, const DenseTensor& no);
DenseTensor where(const DenseTensor& condition, const DenseTensor& yes, double no);
DenseTensor where(const DenseTensor& condition, double yes, const DenseTensor& no);
DenseTensor silu(const DenseTensor& input);
DenseTensor gelu(const DenseTensor& input, const std::string& approximation = "none");
DenseTensor relu(const DenseTensor& input);
DenseTensor sigmoid(const DenseTensor& input);
DenseTensor tanh(const DenseTensor& input);
DenseTensor exp(const DenseTensor& input);
DenseTensor exp2(const DenseTensor& input);
DenseTensor log(const DenseTensor& input);
DenseTensor log1p(const DenseTensor& input);
DenseTensor log2(const DenseTensor& input);
DenseTensor sqrt(const DenseTensor& input);
DenseTensor rsqrt(const DenseTensor& input);
DenseTensor reciprocal(const DenseTensor& input);
DenseTensor sin(const DenseTensor& input);
DenseTensor cos(const DenseTensor& input);
DenseTensor ceil(const DenseTensor& input);
DenseTensor softplus(const DenseTensor& input);
DenseTensor pow(const DenseTensor& input, double exponent);
DenseTensor pow(const DenseTensor& input, const DenseTensor& exponent);
DenseTensor remainder(const DenseTensor& input, double divisor);
DenseTensor clamp(const DenseTensor& input, double minimum, double maximum);
DenseTensor clamp_min(const DenseTensor& input, double minimum);
DenseTensor clamp_max(const DenseTensor& input, double maximum);
DenseTensor softmax(const DenseTensor& input, std::int64_t dimension);
inline DenseTensor softmax(
    const DenseTensor& input,
    std::int64_t dimension,
    ScalarType dtype) {
    return cast(softmax(input, dimension), dtype);
}
DenseTensor log_softmax(const DenseTensor& input, std::int64_t dimension);
DenseTensor logsumexp(const DenseTensor& input, std::int64_t dimension, bool keep_dimension = false);
DenseTensor sum(const DenseTensor& input, std::int64_t dimension, bool keep_dimension = false);
DenseTensor mean(const DenseTensor& input, std::int64_t dimension, bool keep_dimension = false);
DenseTensor argmax(const DenseTensor& input, std::int64_t dimension, bool keep_dimension = false);
std::tuple<DenseTensor, DenseTensor> max(
    const DenseTensor& input,
    std::int64_t dimension,
    bool keep_dimension = false);
DenseTensor dot(const DenseTensor& left, const DenseTensor& right);
DenseTensor isfinite(const DenseTensor& input);
DenseTensor isneginf(const DenseTensor& input);
bool equal(const DenseTensor& left, const DenseTensor& right);
std::tuple<DenseTensor, DenseTensor> topk(
    const DenseTensor& input,
    std::int64_t count,
    std::int64_t dimension = -1,
    bool largest = true,
    bool sorted = true);
std::tuple<DenseTensor, DenseTensor> sort(
    const DenseTensor& input,
    std::int64_t dimension = -1,
    bool descending = false);
DenseTensor layer_norm(
    const DenseTensor& input,
    std::span<const std::int64_t> normalized_shape,
    const DenseTensor& weight,
    const DenseTensor& bias,
    double epsilon);
inline DenseTensor layer_norm(
    const DenseTensor& input,
    std::initializer_list<std::int64_t> normalized_shape,
    const DenseTensor& weight,
    const DenseTensor& bias,
    double epsilon) {
    return layer_norm(
        input,
        std::span<const std::int64_t>(
            normalized_shape.begin(), normalized_shape.size()),
        weight, bias, epsilon);
}
DenseTensor constant_pad_nd(
    const DenseTensor& input,
    std::span<const std::int64_t> padding,
    double value = 0.0);
inline DenseTensor constant_pad_nd(
    const DenseTensor& input,
    std::initializer_list<std::int64_t> padding,
    double value = 0.0) {
    return constant_pad_nd(
        input,
        std::span<const std::int64_t>(padding.begin(), padding.size()),
        value);
}
DenseTensor conv1d(
    const DenseTensor& input,
    const DenseTensor& weight,
    const DenseTensor& bias,
    std::span<const std::int64_t> stride,
    std::span<const std::int64_t> padding,
    std::span<const std::int64_t> dilation,
    std::int64_t groups);
DenseTensor conv2d(
    const DenseTensor& input,
    const DenseTensor& weight,
    const DenseTensor& bias,
    std::span<const std::int64_t> stride,
    std::span<const std::int64_t> padding,
    std::span<const std::int64_t> dilation,
    std::int64_t groups);
DenseTensor avg_pool1d(
    const DenseTensor& input,
    std::span<const std::int64_t> kernel,
    std::span<const std::int64_t> stride,
    std::span<const std::int64_t> padding = {});
DenseTensor cumsum(const DenseTensor& input, std::int64_t dimension);
DenseTensor multinomial(
    const DenseTensor& probabilities,
    std::int64_t samples,
    bool replacement = false);
DenseTensor einsum(const std::string& equation, std::span<const DenseTensor> operands);
inline DenseTensor einsum(
    const std::string& equation,
    std::initializer_list<DenseTensor> operands) {
    return einsum(
        equation,
        std::span<const DenseTensor>(operands.begin(), operands.size()));
}

struct NativeIValue final {
    DenseTensor value;
    bool isTensor() const noexcept { return value.defined(); }
    DenseTensor toTensor() const { return value; }
};

std::vector<char> pickle_save(const DenseTensor& value);
NativeIValue pickle_load(const std::vector<char>& bytes);

template <typename Value>
ScalarType scalar_type_for();

template <> inline ScalarType scalar_type_for<bool>() { return kBool; }
template <> inline ScalarType scalar_type_for<std::uint8_t>() { return kUInt8; }
template <> inline ScalarType scalar_type_for<std::int8_t>() { return kInt8; }
template <> inline ScalarType scalar_type_for<std::int16_t>() { return kInt16; }
template <> inline ScalarType scalar_type_for<std::int32_t>() { return kInt32; }
template <> inline ScalarType scalar_type_for<std::int64_t>() { return kInt64; }
template <> inline ScalarType scalar_type_for<float>() { return kFloat32; }
template <> inline ScalarType scalar_type_for<double>() { return kFloat64; }

template <typename Value>
DenseTensor tensor(std::span<const Value> values, TensorOptions options = {}) {
    if (options.scalar_type() == kFloat32 && !std::is_same_v<Value, float>) {
        options.dtype(scalar_type_for<Value>());
    }
    auto host_options = options;
    const auto target = options.target_device();
    host_options.device(kCPU).pinned_memory(target.is_cuda());
    auto host = clone(from_blob(
        const_cast<Value*>(values.data()),
        {static_cast<std::int64_t>(values.size())},
        host_options));
    return target.is_cuda() ? copy(host, target) : host;
}

template <typename Value>
DenseTensor tensor(const std::vector<Value>& values, TensorOptions options = {}) {
    return tensor(std::span<const Value>(values.data(), values.size()), options);
}

template <typename Value>
DenseTensor tensor(std::initializer_list<Value> values, TensorOptions options = {}) {
    return tensor(std::vector<Value>(values), options);
}

}  // namespace tyloi::cuda
