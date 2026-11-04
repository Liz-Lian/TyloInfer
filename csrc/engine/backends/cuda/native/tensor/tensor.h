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

Tensor contiguous(const Tensor& input);
Tensor clone(const Tensor& input);
Tensor cast(const Tensor& input, ScalarType type);
Tensor cast(const Tensor& input, ScalarType type, Device target);
Tensor copy(const Tensor& input, Device target);
void copy_into(Tensor& destination, const Tensor& source);

Tensor reshape(const Tensor& input, std::span<const std::int64_t> shape);
Tensor transpose(const Tensor& input, std::int64_t first, std::int64_t second);
Tensor permute(const Tensor& input, std::span<const std::int64_t> order);
Tensor unsqueeze(const Tensor& input, std::int64_t dimension);
Tensor squeeze(const Tensor& input, std::int64_t dimension);
Tensor squeeze(const Tensor& input);
Tensor flatten(const Tensor& input, std::int64_t start, std::int64_t end);
Tensor narrow(const Tensor& input, std::int64_t dimension, std::int64_t start, std::int64_t length);
Tensor select(const Tensor& input, std::int64_t dimension, std::int64_t index);
Tensor slice(const Tensor& input, std::int64_t dimension, std::int64_t start, std::int64_t end, std::int64_t step);
Tensor index(const Tensor& input, std::initializer_list<TensorIndex> indices);
Tensor expand(const Tensor& input, std::span<const std::int64_t> shape);
Tensor unfold(const Tensor& input, std::int64_t dimension, std::int64_t window, std::int64_t step);
std::vector<Tensor> split_with_sizes(const Tensor& input, std::span<const std::int64_t> sections, std::int64_t dimension);
std::vector<Tensor> chunk(const Tensor& input, std::int64_t chunks, std::int64_t dimension);

Tensor binary_cuda(const Tensor& left, const Tensor& right, int operation);
Tensor scalar_binary_cuda(const Tensor& left, double right, int operation, bool scalar_first = false);
Tensor index_select_cuda(const Tensor& source, std::int64_t dimension, const Tensor& indices);
Tensor gather_cuda(const Tensor& source, std::int64_t dimension, const Tensor& indices);
Tensor repeat_cuda(const Tensor& source, std::span<const std::int64_t> repeats);
Tensor repeat_interleave_cuda(const Tensor& source, std::int64_t repeats, std::int64_t dimension);
Tensor masked_select_cuda(const Tensor& source, const Tensor& mask);
void scatter_cuda(Tensor& destination, std::int64_t dimension, const Tensor& index, const Tensor& source, bool add);
void index_copy_cuda(Tensor& destination, std::int64_t dimension, const Tensor& index, const Tensor& source);
void index_fill_cuda(Tensor& destination, std::int64_t dimension, const Tensor& index, double value);
void masked_fill_cuda(Tensor& destination, const Tensor& mask, double value);

Tensor empty(std::span<const std::int64_t> shape, const TensorOptions& options = {});
Tensor empty(std::initializer_list<std::int64_t> shape, const TensorOptions& options = {});
Tensor empty_like(const Tensor& source, const TensorOptions* options = nullptr);
Tensor zeros(std::span<const std::int64_t> shape, const TensorOptions& options = {});
Tensor zeros(std::initializer_list<std::int64_t> shape, const TensorOptions& options = {});
Tensor zeros_like(const Tensor& source);
Tensor ones(std::span<const std::int64_t> shape, const TensorOptions& options = {});
Tensor ones(std::initializer_list<std::int64_t> shape, const TensorOptions& options = {});
Tensor full(std::span<const std::int64_t> shape, double value, const TensorOptions& options = {});
Tensor full(std::initializer_list<std::int64_t> shape, double value, const TensorOptions& options = {});
Tensor full_like(const Tensor& source, double value);
Tensor ones_like(const Tensor& source);
Tensor from_blob(
    void* data,
    std::span<const std::int64_t> shape,
    const TensorOptions& options = {});
Tensor from_blob(
    void* data,
    std::initializer_list<std::int64_t> shape,
    const TensorOptions& options = {});

Tensor linear(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias = std::nullopt);
Tensor scaled_dot_product_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& mask,
    double dropout,
    bool causal,
    const std::optional<double>& scale,
    bool enable_grouped_query_attention);

Tensor cat(std::span<const Tensor> tensors, std::int64_t dimension = 0);
Tensor cat(std::initializer_list<Tensor> tensors, std::int64_t dimension = 0);
Tensor cat(const std::vector<Tensor>& tensors, std::int64_t dimension = 0);
Tensor stack(std::span<const Tensor> tensors, std::int64_t dimension = 0);
Tensor stack(std::initializer_list<Tensor> tensors, std::int64_t dimension = 0);
Tensor stack(const std::vector<Tensor>& tensors, std::int64_t dimension = 0);
Tensor arange(std::int64_t end, const TensorOptions& options = {});
Tensor arange(std::int64_t start, std::int64_t end, const TensorOptions& options = {});
Tensor arange(
    std::int64_t start,
    std::int64_t end,
    std::int64_t step,
    const TensorOptions& options = {});
Tensor randn(std::span<const std::int64_t> shape, const TensorOptions& options = {});
Tensor randn(std::initializer_list<std::int64_t> shape, const TensorOptions& options = {});
Tensor randint(
    std::int64_t high,
    std::span<const std::int64_t> shape,
    const TensorOptions& options = {});
Tensor randint(
    std::int64_t low,
    std::int64_t high,
    std::span<const std::int64_t> shape,
    const TensorOptions& options = {});
inline Tensor randint(
    std::int64_t low,
    std::int64_t high,
    std::initializer_list<std::int64_t> shape,
    const TensorOptions& options = {}) {
    return randint(
        low, high,
        std::span<const std::int64_t>(shape.begin(), shape.size()),
        options);
}
Tensor randperm(std::int64_t size, const TensorOptions& options = {});
void manual_seed(std::int64_t seed);

Tensor matmul(const Tensor& left, const Tensor& right);
Tensor bmm(const Tensor& left, const Tensor& right);
Tensor baddbmm(const Tensor& input, const Tensor& left, const Tensor& right);
Tensor where(const Tensor& condition, const Tensor& yes, const Tensor& no);
Tensor where(const Tensor& condition, const Tensor& yes, double no);
Tensor where(const Tensor& condition, double yes, const Tensor& no);
Tensor silu(const Tensor& input);
Tensor gelu(const Tensor& input, const std::string& approximation = "none");
Tensor relu(const Tensor& input);
Tensor sigmoid(const Tensor& input);
Tensor tanh(const Tensor& input);
Tensor exp(const Tensor& input);
Tensor exp2(const Tensor& input);
Tensor log(const Tensor& input);
Tensor log1p(const Tensor& input);
Tensor log2(const Tensor& input);
Tensor sqrt(const Tensor& input);
Tensor rsqrt(const Tensor& input);
Tensor reciprocal(const Tensor& input);
Tensor sin(const Tensor& input);
Tensor cos(const Tensor& input);
Tensor ceil(const Tensor& input);
Tensor softplus(const Tensor& input);
Tensor pow(const Tensor& input, double exponent);
Tensor pow(const Tensor& input, const Tensor& exponent);
Tensor remainder(const Tensor& input, double divisor);
Tensor clamp(const Tensor& input, double minimum, double maximum);
Tensor clamp_min(const Tensor& input, double minimum);
Tensor clamp_max(const Tensor& input, double maximum);
Tensor softmax(const Tensor& input, std::int64_t dimension);
inline Tensor softmax(
    const Tensor& input,
    std::int64_t dimension,
    ScalarType dtype) {
    return cast(softmax(input, dimension), dtype);
}
Tensor log_softmax(const Tensor& input, std::int64_t dimension);
Tensor logsumexp(const Tensor& input, std::int64_t dimension, bool keep_dimension = false);
Tensor sum(const Tensor& input, std::int64_t dimension, bool keep_dimension = false);
Tensor mean(const Tensor& input, std::int64_t dimension, bool keep_dimension = false);
Tensor argmax(const Tensor& input, std::int64_t dimension, bool keep_dimension = false);
std::tuple<Tensor, Tensor> max(
    const Tensor& input,
    std::int64_t dimension,
    bool keep_dimension = false);
Tensor dot(const Tensor& left, const Tensor& right);
Tensor isfinite(const Tensor& input);
Tensor isneginf(const Tensor& input);
bool equal(const Tensor& left, const Tensor& right);
std::tuple<Tensor, Tensor> topk(
    const Tensor& input,
    std::int64_t count,
    std::int64_t dimension = -1,
    bool largest = true,
    bool sorted = true);
std::tuple<Tensor, Tensor> sort(
    const Tensor& input,
    std::int64_t dimension = -1,
    bool descending = false);
Tensor layer_norm(
    const Tensor& input,
    std::span<const std::int64_t> normalized_shape,
    const Tensor& weight,
    const Tensor& bias,
    double epsilon);
inline Tensor layer_norm(
    const Tensor& input,
    std::initializer_list<std::int64_t> normalized_shape,
    const Tensor& weight,
    const Tensor& bias,
    double epsilon) {
    return layer_norm(
        input,
        std::span<const std::int64_t>(
            normalized_shape.begin(), normalized_shape.size()),
        weight, bias, epsilon);
}
Tensor constant_pad_nd(
    const Tensor& input,
    std::span<const std::int64_t> padding,
    double value = 0.0);
inline Tensor constant_pad_nd(
    const Tensor& input,
    std::initializer_list<std::int64_t> padding,
    double value = 0.0) {
    return constant_pad_nd(
        input,
        std::span<const std::int64_t>(padding.begin(), padding.size()),
        value);
}
Tensor conv1d(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    std::span<const std::int64_t> stride,
    std::span<const std::int64_t> padding,
    std::span<const std::int64_t> dilation,
    std::int64_t groups);
Tensor conv2d(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    std::span<const std::int64_t> stride,
    std::span<const std::int64_t> padding,
    std::span<const std::int64_t> dilation,
    std::int64_t groups);
Tensor avg_pool1d(
    const Tensor& input,
    std::span<const std::int64_t> kernel,
    std::span<const std::int64_t> stride,
    std::span<const std::int64_t> padding = {});
Tensor cumsum(const Tensor& input, std::int64_t dimension);
Tensor multinomial(
    const Tensor& probabilities,
    std::int64_t samples,
    bool replacement = false);
Tensor einsum(const std::string& equation, std::span<const Tensor> operands);
inline Tensor einsum(
    const std::string& equation,
    std::initializer_list<Tensor> operands) {
    return einsum(
        equation,
        std::span<const Tensor>(operands.begin(), operands.size()));
}

struct NativeIValue final {
    Tensor value;
    bool isTensor() const noexcept { return value.defined(); }
    Tensor toTensor() const { return value; }
};

std::vector<char> pickle_save(const Tensor& value);
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
Tensor tensor(std::span<const Value> values, TensorOptions options = {}) {
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
Tensor tensor(const std::vector<Value>& values, TensorOptions options = {}) {
    return tensor(std::span<const Value>(values.data(), values.size()), options);
}

template <typename Value>
Tensor tensor(std::initializer_list<Value> values, TensorOptions options = {}) {
    return tensor(std::vector<Value>(values), options);
}

}  // namespace tyloi::cuda
