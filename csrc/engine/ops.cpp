#include "engine/ops.h"

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/ops/norm/rms.h"
#include "engine/backends/cuda/ops/norm/layer.h"
#include "engine/backends/cuda/ops/vision/attention.h"
#include "engine/backends/cuda/ops/ffn/ffn.h"
#include "engine/backends/cuda/ops/projection/projection.h"
#include "engine/backends/cuda/ops/tensor.h"
#include "engine/backends/cuda/ops/attention/attention.h"
#include "engine/backends/cuda/ops/attention/kv.h"
#include "engine/backends/cuda/ops/rope/rope.h"
#include "engine/backends/cuda/native/profile.h"
#include "engine/backends/cuda/native/device.h"
#endif

namespace tyloi::ops {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
namespace {
void device(Device value) {
    if (!value.is_cpu() && !value.is_cuda())
        throw std::invalid_argument("unsupported tensor device");
    if (value.is_cuda()) (void)cuda::Executor::current();
}
void input(const Tensor& value) {
    if (!value.defined()) throw std::invalid_argument("undefined tensor");
    value.require_dense();
    device(value.device());
}
void compatible(const Tensor& left, const Tensor& right) {
    input(left);
    input(right);
    if (left.device() != right.device())
        throw std::invalid_argument("tensor devices do not match");
}
void check_weight(const weights::Weights& value, weights::Layout layout) {
    if (value.backend() != DeviceType::cuda || value.layout() != layout)
        throw std::invalid_argument("weight backend or layout does not match operation");
    if (value.size() == 0 || (layout == weights::Layout::matrix && value.size() != 1))
        throw std::invalid_argument("weight has invalid size");
    const auto count = layout == weights::Layout::group ? value.size() : 1;
    for (std::size_t i = 0; i < count; ++i) {
        const auto placement = value.placement(i);
        if (layout != weights::Layout::experts) {
            const auto shape = value.shape(i);
            if (shape.rows <= 0 || shape.cols <= 0)
                throw std::invalid_argument("weight has invalid shape");
        }
        if (placement.devices.empty() && !placement.caller)
            throw std::invalid_argument("weight has no execution placement");
        for (const auto& place : placement.devices) {
            if ((!place.is_cpu() && !place.is_cuda()) || place.index < 0 ||
                (place.is_cpu() && place.index != 0))
                throw std::invalid_argument("invalid weight execution device");
        }
        if (placement.caller && *placement.caller != DeviceType::cpu &&
            *placement.caller != DeviceType::cuda)
            throw std::invalid_argument("invalid caller-selected weight device type");
    }
}
void require_cuda(const Tensor& value) {
    if (!value.device().is_cuda()) throw std::invalid_argument("operation requires a CUDA tensor");
}
bool encoded(std::initializer_list<const Tensor*> values) {
    for (const auto* value : values) if (value->is_encoded()) return true;
    return false;
}
}

AttentionResult attention(const AttentionInput& request, const state::KvView& kv,
                          const AttentionConfig& config) {
    compatible(request.query, request.key);
    compatible(request.query, request.value);
    input(request.positions);
    input(request.write_positions);
    if (request.sequence_lengths) input(*request.sequence_lengths);
    if (request.attention_mask) input(*request.attention_mask);
    if (request.query.dim() != 4 || request.key.dim() != 4 || request.value.dim() != 4 ||
        request.key.sizes() != request.value.sizes() ||
        request.query.size(0) != request.key.size(0) ||
        request.query.size(2) != request.key.size(2) ||
        request.query.size(3) != request.key.size(3) || request.key.size(1) <= 0 ||
        request.query.size(1) <= 0 || request.query.size(3) <= 0)
        throw std::invalid_argument("attention query, key and value geometry is incompatible");
    for (const Tensor* tensor : {&kv.visible_key, &kv.visible_value, &kv.physical_key,
                                  &kv.physical_value, &kv.key_chunk_ptrs,
                                  &kv.value_chunk_ptrs, &kv.page_table})
        if (tensor->defined()) compatible(request.query, *tensor);
    require_cuda(request.query);
    return cuda::ops::attention(request, kv, config);
}

Tensor attention(Tensor query, Tensor key, Tensor value,
                 const std::optional<Tensor>& mask, double scale) {
    compatible(query, key);
    compatible(query, value);
    if (mask) compatible(query, *mask);
    if (query.dim() != 4 || key.dim() != 4 || value.dim() != 4 ||
        query.size(3) != key.size(3) || key.size(2) != value.size(2))
        throw std::invalid_argument("attention query, key and value geometry is incompatible");
    const auto broadcasts = [](std::int64_t a, std::int64_t b) {
        return a == b || a == 1 || b == 1;
    };
    for (int axis = 0; axis < 2; ++axis) {
        const auto q = query.size(axis), k = key.size(axis), v = value.size(axis);
        if (!broadcasts(q, k) || !broadcasts(q > k ? q : k, v))
            throw std::invalid_argument("attention query, key and value geometry is incompatible");
    }
    require_cuda(query);
    return cuda::ops::attention(std::move(query), std::move(key), std::move(value), mask, scale);
}

Tensor rope(Tensor value, Tensor positions, const RopeTable& table, bool bf16, bool grid) {
    compatible(value, table.cos);
    compatible(value, table.sin);
    table.sections.require_dense(); table.empty_sections.require_dense();
    table.interleaved_order.require_dense(); table.interleaved_inverse.require_dense();
    input(positions);
    if (value.dim() == 0 || table.rotary_dim <= 0 || table.rotary_dim % 2 != 0 ||
        table.rotary_dim > value.size(-1))
        throw std::invalid_argument("invalid RoPE rotary dimension");
    require_cuda(value);
    return cuda::ops::rope(std::move(value), std::move(positions), table, bf16, grid);
}
std::optional<Tensor> norm_rope_kv(const AttentionInput& request, const state::KvView& kv,
    const RopeTable& table, const Tensor& q_norm, const Tensor& k_norm, double eps, double offset) {
    if (!request.query.is_cuda()) return std::nullopt;
    if (encoded({&request.query, &request.key, &request.value, &request.positions, &request.write_positions,
                 &table.cos, &table.sin, &table.sections, &table.empty_sections, &table.interleaved_order,
                 &table.interleaved_inverse, &q_norm, &k_norm, &kv.visible_key, &kv.visible_value,
                 &kv.physical_key, &kv.physical_value, &kv.key_chunk_ptrs, &kv.value_chunk_ptrs, &kv.page_table}) ||
        (request.sequence_lengths && request.sequence_lengths->is_encoded()) ||
        (request.attention_mask && request.attention_mask->is_encoded())) return std::nullopt;
    return cuda::ops::norm_rope_kv(request, kv, table, q_norm, k_norm, eps, offset);
}
std::optional<Tensor> rope_kv(const AttentionInput& request, const state::KvView& kv,
                               const RopeTable& table) {
    if (!request.query.is_cuda()) return std::nullopt;
    if (encoded({&request.query, &request.key, &request.value, &request.positions, &request.write_positions,
                 &table.cos, &table.sin, &table.sections, &table.empty_sections, &table.interleaved_order,
                 &table.interleaved_inverse, &kv.visible_key, &kv.visible_value, &kv.physical_key,
                 &kv.physical_value, &kv.key_chunk_ptrs, &kv.value_chunk_ptrs, &kv.page_table}) ||
        (request.sequence_lengths && request.sequence_lengths->is_encoded()) ||
        (request.attention_mask && request.attention_mask->is_encoded())) return std::nullopt;
    return cuda::ops::rope_kv(request, kv, table);
}

std::pair<Tensor, Tensor> append_kv(state::KvView kv, Tensor key, Tensor value,
                                    Tensor positions, std::int64_t start,
                                    bool contiguous_prefill_prefix) {
    compatible(key, value);
    compatible(key, positions);
    if (key.dim() != 4 || value.dim() != 4 || key.sizes() != value.sizes())
        throw std::invalid_argument("KV write key and value geometry is incompatible");
    for (const Tensor* tensor : {&kv.physical_key, &kv.physical_value, &kv.key_chunk_ptrs,
                                  &kv.value_chunk_ptrs, &kv.page_table})
        if (tensor->defined()) compatible(key, *tensor);
    // A zero batch skips the unpaged, unquantized index writes.
    if (kv.paged() || kv.quantization.enabled() || positions.dim() != 2 || positions.size(0) != 0)
        require_cuda(key);
    return cuda::ops::append_kv(std::move(kv), std::move(key), std::move(value),
                                std::move(positions), start, contiguous_prefill_prefix);
}

Tensor add(Tensor left, Tensor right) {
    compatible(left, right);
    require_cuda(left);
    return cuda::ops::add(std::move(left), std::move(right));
}
Tensor multiply(Tensor left, Tensor right) {
    compatible(left, right);
    require_cuda(left);
    return cuda::ops::multiply(std::move(left), std::move(right));
}
Tensor matmul(Tensor left, Tensor right) {
    compatible(left, right);
    require_cuda(left);
    return cuda::ops::matmul(std::move(left), std::move(right));
}
Tensor transpose(Tensor value, std::int64_t first, std::int64_t second) {
    input(value);
    return cuda::ops::transpose(std::move(value), first, second);
}
Tensor add_residual(Tensor residual, Tensor value) {
    compatible(residual, value);
    require_cuda(residual);
    return cuda::ops::add_residual(std::move(residual), std::move(value));
}
std::optional<ResidualNorms> residual_norms(Tensor residual, Tensor attention, const Tensor& attention_weight,
                                            const Tensor& dense_weight, const Tensor& router_weight,
                                            const Tensor& expert_weight, double eps) {
    if (!residual.is_cuda()) return std::nullopt;
    if (encoded({&residual, &attention, &attention_weight, &dense_weight, &router_weight, &expert_weight}))
        return std::nullopt;
    return cuda::ops::residual_norms(std::move(residual), std::move(attention), attention_weight,
                                     dense_weight, router_weight, expert_weight, eps);
}
Tensor merge_norms(Tensor dense, Tensor experts, Tensor residual, const Tensor& dense_weight,
                   const Tensor& expert_weight, const Tensor& output_weight,
                   const Tensor& scale, double eps) {
    for (const Tensor* value : std::initializer_list<const Tensor*>{
             &experts, &residual, &dense_weight, &expert_weight, &output_weight, &scale})
        compatible(dense, *value);
    require_cuda(dense);
    return cuda::ops::merge_norms(std::move(dense), std::move(experts), std::move(residual),
                                  dense_weight, expert_weight, output_weight, scale, eps);
}
struct ProfileRange::Impl {
    std::optional<cuda::DeviceGuard> guard;
    std::optional<CudaProfiler::Range> range;
    Impl(CudaProfiler& profiler, const char* name, std::optional<Device> device) {
        if (device && device->is_cuda()) guard.emplace(device->index);
        range.emplace(profiler, name);
    }
};
ProfileRange::ProfileRange(const char* name, std::optional<Device> device) {
    auto& profiler = cuda::Executor::current().profiler();
    if (profiler.enabled && profiler.selected(name))
        impl_ = std::make_unique<Impl>(profiler, name, device);
}
ProfileRange::~ProfileRange() = default;
void ProfileRange::finish() { if (impl_) impl_->range->finish(); }
SelectedExperts topk(Tensor logits, std::int64_t count, const MoeRouting& routing) {
    input(logits);
    require_cuda(logits);
    return cuda::ops::topk(std::move(logits), count, routing);
}
Tensor expert_scale(Tensor weights, Tensor ids, const Tensor& scales) {
    compatible(weights, ids);
    compatible(weights, scales);
    require_cuda(weights);
    return cuda::ops::expert_scale(std::move(weights), std::move(ids), scales);
}
Tensor weighted_reduce(Tensor pairs, Tensor weights) {
    compatible(pairs, weights);
    require_cuda(pairs);
    return cuda::ops::weighted_reduce(std::move(pairs), std::move(weights));
}
RoutedGeglu routed_geglu(Tensor value, Tensor ids, const weights::Weights& gate_up,
                         const weights::Weights& down, bool trace) {
    check_weight(gate_up, weights::Layout::experts);
    check_weight(down, weights::Layout::experts);
    compatible(value, ids);
    if (gate_up.size() == 0 || down.size() == 0 || value.dim() != 2 || ids.dim() != 2 || value.size(0) != ids.size(0) ||
        gate_up.size() != down.size() || gate_up.shape().cols != value.size(-1) ||
        gate_up.shape().rows % 2 != 0 || gate_up.shape().rows / 2 != down.shape().cols)
        throw std::invalid_argument("routed GEGLU input, routes, and expert banks are incompatible");
    require_cuda(value);
    return cuda::ops::routed_geglu(std::move(value), std::move(ids), gate_up, down, trace);
}

Tensor rms_norm(Tensor value, const Tensor& weight, double eps, double offset, ScalarType output_type) {
    input(value);
    input(weight);
    if (value.device() != weight.device() || value.dim() == 0 || weight.dim() == 0 ||
        value.size(-1) <= 0 || weight.numel() != value.size(-1) ||
        weight.size(-1) != value.size(-1))
        throw std::invalid_argument("rms_norm input and weight are incompatible");
    if (output_type != ScalarType::float32 && output_type != ScalarType::float16 &&
        output_type != ScalarType::bfloat16)
        throw std::invalid_argument("unsupported rms_norm output type");
    require_cuda(value);
    return cuda::ops::rms_norm(std::move(value), weight, eps, offset, output_type);
}
Tensor layer_norm(Tensor value, const Tensor& weight, const Tensor& bias, double epsilon) {
    compatible(value, weight);
    compatible(value, bias);
    if (value.dim() == 0 || weight.dim() != 1 || bias.dim() != 1 ||
        weight.size(0) != value.size(-1) || bias.size(0) != value.size(-1))
        throw std::invalid_argument("layer_norm input and weights are incompatible");
    require_cuda(value);
    return cuda::ops::layer_norm(std::move(value), weight, bias, epsilon);
}
Tensor gelu(Tensor value, bool approximate) {
    input(value);
    require_cuda(value);
    return cuda::ops::gelu(std::move(value), approximate);
}
Tensor silu(Tensor value) {
    input(value);
    require_cuda(value);
    return cuda::ops::silu(std::move(value));
}
Tensor relu_squared(Tensor value) {
    input(value);
    require_cuda(value);
    return cuda::ops::relu_squared(std::move(value));
}
Tensor axial_rope(Tensor value, const std::vector<std::int32_t>& positions, double theta) {
    input(value);
    require_cuda(value);
    return cuda::ops::axial_rope(std::move(value), positions, theta);
}
Tensor segmented_attention(Tensor query, Tensor key, Tensor value,
                           const std::vector<std::int32_t>& lengths) {
    compatible(query, key);
    compatible(query, value);
    require_cuda(query);
    return cuda::ops::segmented_attention(std::move(query), std::move(key), std::move(value), lengths);
}
std::array<Tensor, 3> qkv_norm(std::array<Tensor, 3> values, std::array<Tensor, 3> weights,
                               double eps, double offset, ScalarType output_type) {
    if (output_type != ScalarType::float32 && output_type != ScalarType::bfloat16)
        throw std::invalid_argument("unsupported qkv_norm output type");
    for (std::size_t i = 0; i < values.size(); ++i) {
        input(values[i]);
        if (weights[i].defined()) {
            compatible(values[i], weights[i]);
            if (values[i].dim() == 0 || weights[i].dim() == 0 ||
                values[i].size(-1) <= 0 || weights[i].numel() != values[i].size(-1))
                throw std::invalid_argument("qkv_norm input and weight are incompatible");
        }
    }
    for (std::size_t i = 0; i < values.size(); ++i)
        if (weights[i].defined()) require_cuda(values[i]);
    return cuda::ops::qkv_norm(std::move(values), std::move(weights), eps, offset, output_type);
}
Tensor softcap(Tensor value, double cap) {
    input(value);
    if (cap > 0.0) require_cuda(value);
    return cuda::ops::softcap(std::move(value), cap);
}
Tensor scale(Tensor value, double factor) {
    input(value);
    require_cuda(value);
    return cuda::ops::scale(std::move(value), factor);
}
Tensor embedding(Tensor ids, const weights::Weights& weight) {
    check_weight(weight, weights::Layout::matrix);
    input(ids);
    return cuda::ops::embedding(std::move(ids), weight);
}
Tensor project(Tensor value, const weights::Weights& weight) {
    check_weight(weight, weights::Layout::matrix);
    input(value);
    return cuda::ops::project(std::move(value), weight);
}
Tensor linear(Tensor value, const weights::Weights& weight, const Tensor& bias) {
    check_weight(weight, weights::Layout::matrix);
    input(value);
    if (bias.defined()) input(bias);
    return cuda::ops::linear(std::move(value), weight, bias);
}
Tensor affine(Tensor value, const weights::Weights& weight, const Tensor& bias) {
    check_weight(weight, weights::Layout::matrix);
    input(value);
    if (bias.defined()) input(bias);
    return cuda::ops::affine(std::move(value), weight, bias);
}
Tensor mlp(Tensor value, const weights::Weights& gate_up, const weights::Weights& down,
           MlpActivation activation, double swiglu_limit) {
    check_weight(gate_up, weights::Layout::group);
    check_weight(down, weights::Layout::matrix);
    input(value);
    return cuda::ops::mlp(std::move(value), gate_up, down, activation, swiglu_limit);
}
std::vector<Tensor> project_group(Tensor value, const weights::Weights& group) {
    check_weight(group, weights::Layout::group);
    input(value);
    return cuda::ops::project_group(std::move(value), group);
}
Tensor sigmoid_project(Tensor value, Tensor gate, const weights::Weights& weight) {
    check_weight(weight, weights::Layout::matrix);
    compatible(value, gate);
    return cuda::ops::sigmoid_project(std::move(value), std::move(gate), weight);
}
Tensor tensor(std::span<const float> values, Device target) {
    device(target);
    return cuda::ops::tensor(values, target);
}
Tensor tensor(std::span<const std::int32_t> values, Device target) {
    device(target);
    return cuda::ops::tensor(values, target);
}
Tensor tensor(std::span<const std::int64_t> values, Device target) {
    device(target);
    return cuda::ops::tensor(values, target);
}
Tensor tensor(std::span<const std::uint8_t> values, Device target) {
    device(target);
    return cuda::ops::tensor(values, target);
}
Tensor from_blob(void* data, std::span<const std::int64_t> shape, const TensorOptions& options) {
    device(options.target_device());
    return cuda::ops::from_blob(data, shape, options);
}
Tensor copy(Tensor value, Device target) {
    input(value);
    device(target);
    return cuda::ops::copy(std::move(value), target);
}
void copy_into(Tensor destination, Tensor source) {
    input(destination);
    input(source);
    cuda::ops::copy_into(std::move(destination), std::move(source));
}
Tensor cast(Tensor value, ScalarType type) {
    input(value);
    return cuda::ops::cast(std::move(value), type);
}
Tensor cast(Tensor value, ScalarType type, Device target) {
    input(value);
    device(target);
    return cuda::ops::cast(std::move(value), type, target);
}
Tensor contiguous(Tensor value) {
    input(value);
    return cuda::ops::contiguous(std::move(value));
}
Tensor clone(Tensor value) {
    input(value);
    return cuda::ops::clone(std::move(value));
}
Tensor reshape(Tensor value, std::span<const std::int64_t> shape) {
    input(value);
    return cuda::ops::reshape(std::move(value), shape);
}
Tensor expand(Tensor value, std::span<const std::int64_t> shape) {
    input(value);
    return cuda::ops::expand(std::move(value), shape);
}
Tensor cat(std::span<const Tensor> values, std::int64_t dim) {
    if (values.empty()) throw std::invalid_argument("cat requires at least one tensor");
    input(values.front());
    for (const auto& value : values.subspan(1)) compatible(values.front(), value);
    return cuda::ops::cat(values, dim);
}
Tensor zeros(std::span<const std::int64_t> shape, const TensorOptions& options) {
    device(options.target_device());
    return cuda::ops::zeros(shape, options);
}
Tensor select(Tensor value, std::int64_t dim, std::int64_t index) {
    input(value);
    return cuda::ops::select(std::move(value), dim, index);
}
Tensor unsqueeze(Tensor value, std::int64_t dim) {
    input(value);
    return cuda::ops::unsqueeze(std::move(value), dim);
}
Tensor slice(Tensor value, std::int64_t dim, std::int64_t start, std::int64_t end, std::int64_t step) {
    input(value);
    return cuda::ops::slice(std::move(value), dim, start, end, step);
}
Tensor index_select(Tensor value, std::int64_t dim, Tensor indices) {
    compatible(value, indices);
    require_cuda(value);
    return cuda::ops::index_select(std::move(value), dim, std::move(indices));
}
Tensor sum(Tensor value, std::int64_t dim, bool keepdim) {
    input(value);
    require_cuda(value);
    return cuda::ops::sum(std::move(value), dim, keepdim);
}
void index_copy(Tensor value, std::int64_t dim, Tensor indices, Tensor source) {
    compatible(value, indices);
    compatible(value, source);
    require_cuda(value);
    cuda::ops::index_copy(std::move(value), dim, std::move(indices), std::move(source));
}
std::vector<Tensor> chunk(Tensor value, std::int64_t count, std::int64_t dim) {
    input(value);
    return cuda::ops::chunk(std::move(value), count, dim);
}
Tensor arange(std::int64_t first, std::int64_t last, Device target) {
    device(target);
    return cuda::ops::arange(first, last, target);
}
Tensor full(std::span<const std::int64_t> shape, std::int64_t value, Device target) {
    device(target);
    return cuda::ops::full(shape, value, target);
}
Tensor add(Tensor value, std::int64_t offset) {
    input(value);
    require_cuda(value);
    return cuda::ops::add(std::move(value), offset);
}
#endif
} // namespace tyloi::ops
