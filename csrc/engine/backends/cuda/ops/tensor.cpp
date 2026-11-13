#include "engine/backends/cuda/ops/tensor.h"

#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/native/collectives.h"
#include "engine/backends/cuda/native/env.h"
#include "engine/ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_norm_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_activation_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_sampling_ops.h"

#include <optional>
#include <cstring>
#include <tuple>
#include <vector>

namespace tyloi::cuda::ops {

tyloi::DenseTensor where(tyloi::DenseTensor condition, tyloi::DenseTensor yes, tyloi::DenseTensor no) {
    std::optional<MfqCudaGuard> guard;
    if (yes.device().is_cuda()) guard.emplace(yes.device().index);
    return ::tyloi::cuda::where(condition, yes, no);
}

tyloi::DenseTensor sharded_embedding(std::span<const Tensor> shards, tyloi::DenseTensor ids,
                                     std::span<const std::int64_t> used, std::int64_t rows) {
    std::optional<MfqCudaGuard> guard;
    if (ids.device().is_cuda()) guard.emplace(ids.device().index);
    DenseTensor result;
    for (const auto shard : used) {
        if (shard < 0 || static_cast<std::size_t>(shard) >= shards.size())
            throw std::out_of_range("embedding shard index out of range");
        auto mask = (ids >= shard * rows) & (ids < (shard + 1) * rows);
        auto local = ::tyloi::cuda::where(mask, ids - shard * rows, ::tyloi::cuda::zeros_like(ids));
        auto values = ::tyloi::ops::embedding(local, shards[static_cast<std::size_t>(shard)]).dense();
        if (values.sizes().vec() != std::vector<int64_t>({ids.size(0), ids.size(1), ids.size(2), shards[static_cast<std::size_t>(shard)].size(1)}))
            throw std::runtime_error("Qwen4 ngram embedding shape mismatch");
        values = values * mask.unsqueeze(-1).to(values.scalar_type());
        result = result.defined() ? result + values : values;
    }
    return result;
}

std::vector<float> floats(const tyloi::DenseTensor& input) {
    std::optional<MfqCudaGuard> guard;
    if (input.device().is_cuda()) guard.emplace(input.device().index);
    auto host = input.to(kCPU, kFloat32).contiguous();
    const auto* data = host.data_ptr<float>();
    return {data, data + host.numel()};
}

std::vector<std::int64_t> integers(const tyloi::DenseTensor& input) {
    std::optional<MfqCudaGuard> guard;
    if (input.device().is_cuda()) guard.emplace(input.device().index);
    auto host = input.to(kCPU, kInt64).contiguous();
    const auto* data = host.data_ptr<std::int64_t>();
    return {data, data + host.numel()};
}

::tyloi::ops::TopK top_k(const tyloi::DenseTensor& logits, std::int64_t k, bool force_radix) {
    MfqCudaGuard guard(logits.device().index);
    ::tyloi::ops::TopK result;
    result.rows = logits.size(0);
    result.columns = k;
    const auto* option = cuda::env_lookup("TYLOINFER_MTP_RADIX_TOPK");
    if ((force_radix || (option && option[0] == '1' && result.rows <= 32)) && k <= 64) {
        auto host = ::sample_top_k_compact_cuda(logits, k).cpu().contiguous();
        const auto* packets = host.data_ptr<std::int64_t>();
        result.values.reserve(std::size_t(result.rows * k));
        result.indices.reserve(std::size_t(result.rows * k));
        for (std::int64_t row = 0; row < result.rows; ++row) {
            for (std::int64_t column = 0; column < k; ++column) {
                const auto packet = std::uint64_t(packets[row * host.size(1) + column]);
                const auto raw = std::uint32_t(packet >> 32);
                float value;
                std::memcpy(&value, &raw, sizeof(value));
                result.values.push_back(value);
                result.indices.push_back(std::uint32_t(packet));
            }
        }
    } else {
        auto selected = ::tyloi::cuda::topk(logits, k, -1, true, true);
        result.values = floats(std::get<0>(selected));
        result.indices = integers(std::get<1>(selected));
    }
    return result;
}

void decode_commit(tyloi::DenseTensor next, tyloi::DenseTensor generated, tyloi::DenseTensor step,
                   tyloi::DenseTensor input, tyloi::DenseTensor position, tyloi::DenseTensor length) {
    MfqCudaGuard guard(next.device().index);
    decode_graph_commit_cuda(next, generated, step, input, position, length);
}
template <typename Value>
static tyloi::DenseTensor tensor_values(std::span<const Value> values, Device target) {
    return ::tyloi::cuda::tensor(values, TensorOptions().dtype(::tyloi::cuda::scalar_type_for<Value>()).device(target));
}
tyloi::DenseTensor tensor(std::span<const float> values, Device target) { return tensor_values(values, target); }
tyloi::DenseTensor tensor(std::span<const std::int32_t> values, Device target) { return tensor_values(values, target); }
tyloi::DenseTensor tensor(std::span<const std::int64_t> values, Device target) { return tensor_values(values, target); }
tyloi::DenseTensor tensor(std::span<const std::uint8_t> values, Device target) { return tensor_values(values, target); }
tyloi::DenseTensor from_blob(void* data, std::span<const std::int64_t> shape, const TensorOptions& options) {
    return ::tyloi::cuda::from_blob(data, shape, options);
}
tyloi::DenseTensor gelu(tyloi::DenseTensor input, bool approximate) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    return ::tyloi::cuda::gelu(input, approximate ? "tanh" : "none");
}
tyloi::DenseTensor silu(tyloi::DenseTensor input) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    return ::tyloi::cuda::silu(input);
}
tyloi::DenseTensor relu_squared(tyloi::DenseTensor input) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    if (input.is_cpu()) return ::tyloi::cuda::relu(input).square();
    return relu_squared_cuda(::tyloi::cuda::contiguous(input));
}
tyloi::DenseTensor softcap(tyloi::DenseTensor input, double cap) {
    if (!(cap > 0.0)) return input;
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    auto scaled = ::tyloi::cuda::scalar_binary_cuda(input, cap, 3);
    return ::tyloi::cuda::scalar_binary_cuda(::tyloi::cuda::tanh(scaled), cap, 2);
}
tyloi::DenseTensor scale(tyloi::DenseTensor input, double factor) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    return ::tyloi::cuda::scalar_binary_cuda(input, factor, 2);
}
tyloi::DenseTensor divide(tyloi::DenseTensor input, double divisor) {
    std::optional<MfqCudaGuard> guard;
    if (input.device().is_cuda()) guard.emplace(input.device().index);
    return input / divisor;
}
tyloi::DenseTensor copy(tyloi::DenseTensor input, Device target) {
    return target.is_cuda() ? Executor::current().collectives().copy(input, target.index)
                            : ::tyloi::cuda::copy(input, target);
}
void copy_into(tyloi::DenseTensor destination, tyloi::DenseTensor source, bool non_blocking) {
    std::optional<MfqCudaGuard> guard;
    if (destination.device().is_cuda()) guard.emplace(destination.device().index);
    else if (source.device().is_cuda()) guard.emplace(source.device().index);
    ::tyloi::cuda::copy_into(destination, source, non_blocking);
}
tyloi::DenseTensor cast(tyloi::DenseTensor input, ScalarType type) { return ::tyloi::cuda::cast(input, type); }
tyloi::DenseTensor cast(tyloi::DenseTensor input, ScalarType type, Device target) { return ::tyloi::cuda::cast(input, type, target); }
tyloi::DenseTensor contiguous(tyloi::DenseTensor input) { return ::tyloi::cuda::contiguous(input); }
tyloi::DenseTensor clone(tyloi::DenseTensor input) { return ::tyloi::cuda::clone(input); }
tyloi::DenseTensor cat(std::span<const tyloi::DenseTensor> inputs, std::int64_t dim) {
    std::optional<MfqCudaGuard> guard;
    if (inputs.front().device().is_cuda()) guard.emplace(inputs.front().device().index);
    return ::tyloi::cuda::cat(inputs, dim);
}
tyloi::DenseTensor zeros(std::span<const std::int64_t> shape, const TensorOptions& options) {
    std::optional<MfqCudaGuard> guard;
    if (options.target_device().is_cuda()) guard.emplace(options.target_device().index);
    return ::tyloi::cuda::zeros(shape, options);
}
tyloi::DenseTensor index_select(tyloi::DenseTensor input, std::int64_t dim, tyloi::DenseTensor indices) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    return ::tyloi::cuda::index_select_cuda(input, dim, indices);
}
tyloi::DenseTensor sum(tyloi::DenseTensor input, std::int64_t dim, bool keepdim) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    return ::tyloi::cuda::sum(input, dim, keepdim);
}
void index_copy(tyloi::DenseTensor input, std::int64_t dim, tyloi::DenseTensor indices, tyloi::DenseTensor source) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    ::tyloi::cuda::index_copy_cuda(input, dim, indices, source);
}
tyloi::DenseTensor arange(std::int64_t first, std::int64_t last, Device device) {
    std::optional<MfqCudaGuard> guard;
    if (device.is_cuda()) guard.emplace(device.index);
    return ::tyloi::cuda::arange(first, last, ::tyloi::TensorOptions().device(device).dtype(::tyloi::kInt64));
}
tyloi::DenseTensor full(std::span<const std::int64_t> shape, std::int64_t value, Device device) {
    std::optional<MfqCudaGuard> guard;
    if (device.is_cuda()) guard.emplace(device.index);
    return ::tyloi::cuda::full(shape, value, ::tyloi::TensorOptions().device(device).dtype(::tyloi::kInt64));
}
tyloi::DenseTensor add(tyloi::DenseTensor input, std::int64_t offset) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    return ::tyloi::cuda::scalar_binary_cuda(input, static_cast<double>(offset), 0);
}
tyloi::DenseTensor add(tyloi::DenseTensor left, tyloi::DenseTensor right) {
    std::optional<MfqCudaGuard> guard;
    if (left.is_cuda()) guard.emplace(left.device().index);
    return ::tyloi::cuda::binary_cuda(left, right, 0);
}
tyloi::DenseTensor multiply(tyloi::DenseTensor left, tyloi::DenseTensor right) {
    std::optional<MfqCudaGuard> guard;
    if (left.is_cuda()) guard.emplace(left.device().index);
    return ::tyloi::cuda::binary_cuda(left, right, 2);
}
tyloi::DenseTensor matmul(tyloi::DenseTensor left, tyloi::DenseTensor right) {
    std::optional<MfqCudaGuard> guard;
    if (left.is_cuda()) guard.emplace(left.device().index);
    return ::tyloi::cuda::matmul(left, right);
}
tyloi::DenseTensor add_residual(tyloi::DenseTensor residual, tyloi::DenseTensor value, bool configured) {
    std::optional<MfqCudaGuard> guard;
    if (residual.is_cuda()) guard.emplace(residual.device().index);
    if (configured) {
        if (Executor::current().fp32_residual()) {
            residual = residual.to(::tyloi::kFloat32);
            value = value.to(::tyloi::kFloat32);
        } else if (value.scalar_type() != residual.scalar_type()) {
            value = value.to(residual.scalar_type()).contiguous();
        }
    }
    if (residual.is_cpu()) return residual + value;
    return acc_cuda(residual, value);
}

tyloi::DenseTensor mean(tyloi::DenseTensor input, std::int64_t dim, bool keepdim) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    return ::tyloi::cuda::mean(input, dim, keepdim);
}
tyloi::DenseTensor amax(tyloi::DenseTensor input, std::int64_t dim, bool keepdim) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    return ::tyloi::cuda::reduce_cuda(input, dim, keepdim, 2);
}
tyloi::DenseTensor argmax(tyloi::DenseTensor input, std::int64_t dim, bool keepdim) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    return ::tyloi::cuda::argmax(input, dim, keepdim);
}
tyloi::DenseTensor all(tyloi::DenseTensor input, std::int64_t dim, bool keepdim) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    return ::tyloi::cuda::reduce_cuda(input, dim, keepdim, 4);
}
tyloi::DenseTensor any(tyloi::DenseTensor input, std::int64_t dim, bool keepdim) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    return ::tyloi::cuda::reduce_cuda(input, dim, keepdim, 5);
}
tyloi::DenseTensor abs(tyloi::DenseTensor input) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    return ::tyloi::cuda::unary_cuda(input, 0);
}
tyloi::DenseTensor square(tyloi::DenseTensor input) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    return ::tyloi::cuda::unary_cuda(input, 1);
}
tyloi::DenseTensor exp(tyloi::DenseTensor input) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    return ::tyloi::cuda::exp(input);
}
tyloi::DenseTensor sqrt(tyloi::DenseTensor input) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    return ::tyloi::cuda::sqrt(input);
}
tyloi::DenseTensor logical_not(tyloi::DenseTensor input) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    return ::tyloi::cuda::unary_cuda(input, 8);
}
tyloi::DenseTensor clamp(tyloi::DenseTensor input, double minimum, double maximum) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    return ::tyloi::cuda::clamp(input, minimum, maximum);
}
tyloi::DenseTensor clamp_min(tyloi::DenseTensor input, double minimum) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    return ::tyloi::cuda::clamp_min(input, minimum);
}
tyloi::DenseTensor clamp_max(tyloi::DenseTensor input, double maximum) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    return ::tyloi::cuda::clamp_max(input, maximum);
}
tyloi::DenseTensor remainder(tyloi::DenseTensor input, double divisor) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    return ::tyloi::cuda::remainder(input, divisor);
}
tyloi::DenseTensor eq(tyloi::DenseTensor left, tyloi::DenseTensor right) {
    std::optional<MfqCudaGuard> guard;
    if (left.is_cuda()) guard.emplace(left.device().index);
    return ::tyloi::cuda::binary_cuda(left, right, 10);
}
tyloi::DenseTensor eq(tyloi::DenseTensor input, double other) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    return ::tyloi::cuda::scalar_binary_cuda(input, other, 10);
}
tyloi::DenseTensor ne(tyloi::DenseTensor left, tyloi::DenseTensor right) {
    std::optional<MfqCudaGuard> guard;
    if (left.is_cuda()) guard.emplace(left.device().index);
    return ::tyloi::cuda::binary_cuda(left, right, 11);
}
tyloi::DenseTensor ne(tyloi::DenseTensor input, double other) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    return ::tyloi::cuda::scalar_binary_cuda(input, other, 11);
}
bool equal(const tyloi::DenseTensor& left, const tyloi::DenseTensor& right) {
    std::optional<MfqCudaGuard> guard;
    if (left.is_cuda()) guard.emplace(left.device().index);
    return ::tyloi::cuda::equal(left, right);
}
tyloi::DenseTensor masked_select(tyloi::DenseTensor input, tyloi::DenseTensor mask) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    return ::tyloi::cuda::masked_select_cuda(input, mask);
}
tyloi::DenseTensor gather(tyloi::DenseTensor input, std::int64_t dim, tyloi::DenseTensor indices) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    return ::tyloi::cuda::gather_cuda(input, dim, indices);
}
tyloi::DenseTensor repeat(tyloi::DenseTensor input, std::span<const std::int64_t> repeats) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    return ::tyloi::cuda::repeat_cuda(input, repeats);
}
tyloi::DenseTensor repeat_interleave(tyloi::DenseTensor input, std::int64_t repeats, std::int64_t dim) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    return ::tyloi::cuda::repeat_interleave_cuda(input, repeats, dim);
}
tyloi::DenseTensor tril(tyloi::DenseTensor input, std::int64_t diagonal) {
    if (input.dim() < 2) throw std::invalid_argument("tril requires at least two dimensions");
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    const auto options = ::tyloi::TensorOptions{}.dtype(::tyloi::kInt64).device(input.device());
    auto rows = ::tyloi::cuda::arange(input.size(-2), options).unsqueeze(-1);
    auto columns = ::tyloi::cuda::arange(input.size(-1), options).unsqueeze(0);
    return ::tyloi::cuda::where(columns <= rows + diagonal, input, 0.0);
}
tyloi::DenseTensor masked_fill(tyloi::DenseTensor input, tyloi::DenseTensor mask, double value) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    auto result = ::tyloi::cuda::clone(input);
    ::tyloi::cuda::masked_fill_cuda(result, mask, value);
    return result;
}
void masked_fill_inplace(tyloi::DenseTensor input, tyloi::DenseTensor mask, double value) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    ::tyloi::cuda::masked_fill_cuda(input, mask, value);
}
void fill(tyloi::DenseTensor input, double value) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    ::tyloi::cuda::fill_cuda(input, value);
}
void scatter_add(tyloi::DenseTensor input, std::int64_t dim, tyloi::DenseTensor indices, tyloi::DenseTensor source) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    ::tyloi::cuda::scatter_cuda(input, dim, indices, source, true);
}
void scatter(tyloi::DenseTensor input, std::int64_t dim, tyloi::DenseTensor indices, tyloi::DenseTensor source) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    ::tyloi::cuda::scatter_cuda(input, dim, indices, source, false);
}
void index_fill(tyloi::DenseTensor input, std::int64_t dim, tyloi::DenseTensor indices, double value) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    ::tyloi::cuda::index_fill_cuda(input, dim, indices, value);
}

} // namespace tyloi::cuda::ops
