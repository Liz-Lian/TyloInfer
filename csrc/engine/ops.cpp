#include "engine/ops.h"

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/ops/norm/rms.h"
#include "engine/backends/cuda/ops/norm/layer.h"
#include "engine/backends/cuda/ops/vision/attention.h"
#include "engine/backends/cuda/ops/ffn/ffn.h"
#include "engine/backends/cuda/ops/projection/projection.h"
#include "engine/backends/cuda/ops/tensor.h"
#include "engine/backends/cuda/native/tensor/tensor.h"
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
void input(const DenseTensor& value) {
    if (!value.defined()) throw std::invalid_argument("undefined tensor");
    value.require_dense();
    device(value.device());
}
void compatible(const DenseTensor& left, const DenseTensor& right) {
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
void require_cuda(const DenseTensor& value) {
    if (!value.device().is_cuda()) throw std::invalid_argument("operation requires a CUDA tensor");
}
bool encoded(std::initializer_list<const DenseTensor*> values) {
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
    for (const DenseTensor* tensor : {&kv.visible_key, &kv.visible_value, &kv.physical_key,
                                  &kv.physical_value, &kv.key_chunk_ptrs,
                                  &kv.value_chunk_ptrs, &kv.page_table})
        if (tensor->defined()) compatible(request.query, *tensor);
    require_cuda(request.query);
    return cuda::ops::attention(request, kv, config);
}

DenseTensor attention(DenseTensor query, DenseTensor key, DenseTensor value,
                 const std::optional<DenseTensor>& mask, double scale) {
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

DenseTensor rope(DenseTensor value, DenseTensor positions, const RopeTable& table, bool bf16, bool grid) {
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
std::optional<DenseTensor> norm_rope_kv(const AttentionInput& request, const state::KvView& kv,
    const RopeTable& table, const DenseTensor& q_norm, const DenseTensor& k_norm, double eps, double offset) {
    if (!request.query.is_cuda()) return std::nullopt;
    if (encoded({&request.query, &request.key, &request.value, &request.positions, &request.write_positions,
                 &table.cos, &table.sin, &table.sections, &table.empty_sections, &table.interleaved_order,
                 &table.interleaved_inverse, &q_norm, &k_norm, &kv.visible_key, &kv.visible_value,
                 &kv.physical_key, &kv.physical_value, &kv.key_chunk_ptrs, &kv.value_chunk_ptrs, &kv.page_table}) ||
        (request.sequence_lengths && request.sequence_lengths->is_encoded()) ||
        (request.attention_mask && request.attention_mask->is_encoded())) return std::nullopt;
    return cuda::ops::norm_rope_kv(request, kv, table, q_norm, k_norm, eps, offset);
}
std::optional<DenseTensor> rope_kv(const AttentionInput& request, const state::KvView& kv,
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

std::pair<DenseTensor, DenseTensor> append_kv(state::KvView kv, DenseTensor key, DenseTensor value,
                                    DenseTensor positions, std::int64_t start,
                                    bool contiguous_prefill_prefix) {
    compatible(key, value);
    compatible(key, positions);
    if (key.dim() != 4 || value.dim() != 4 || key.sizes() != value.sizes())
        throw std::invalid_argument("KV write key and value geometry is incompatible");
    for (const DenseTensor* tensor : {&kv.physical_key, &kv.physical_value, &kv.key_chunk_ptrs,
                                  &kv.value_chunk_ptrs, &kv.page_table})
        if (tensor->defined()) compatible(key, *tensor);
    // A zero batch skips the unpaged, unquantized index writes.
    if (kv.paged() || kv.quantization.enabled() || positions.dim() != 2 || positions.size(0) != 0)
        require_cuda(key);
    return cuda::ops::append_kv(std::move(kv), std::move(key), std::move(value),
                                std::move(positions), start, contiguous_prefill_prefix);
}

DenseTensor add(DenseTensor left, DenseTensor right) {
    compatible(left, right);
    require_cuda(left);
    return cuda::ops::add(std::move(left), std::move(right));
}
DenseTensor multiply(DenseTensor left, DenseTensor right) {
    compatible(left, right);
    require_cuda(left);
    return cuda::ops::multiply(std::move(left), std::move(right));
}
DenseTensor matmul(DenseTensor left, DenseTensor right) {
    compatible(left, right);
    require_cuda(left);
    return cuda::ops::matmul(std::move(left), std::move(right));
}
DenseTensor transpose(DenseTensor value, std::int64_t first, std::int64_t second) {
    input(value);
    return tyloi::transpose(std::move(value), first, second);
}
DenseTensor add_residual(DenseTensor residual, DenseTensor value) {
    compatible(residual, value);
    require_cuda(residual);
    return cuda::ops::add_residual(std::move(residual), std::move(value));
}
std::optional<ResidualNorms> residual_norms(DenseTensor residual, DenseTensor attention, const DenseTensor& attention_weight,
                                            const DenseTensor& dense_weight, const DenseTensor& router_weight,
                                            const DenseTensor& expert_weight, double eps) {
    if (!residual.is_cuda()) return std::nullopt;
    if (encoded({&residual, &attention, &attention_weight, &dense_weight, &router_weight, &expert_weight}))
        return std::nullopt;
    return cuda::ops::residual_norms(std::move(residual), std::move(attention), attention_weight,
                                     dense_weight, router_weight, expert_weight, eps);
}
DenseTensor merge_norms(DenseTensor dense, DenseTensor experts, DenseTensor residual, const DenseTensor& dense_weight,
                   const DenseTensor& expert_weight, const DenseTensor& output_weight,
                   const DenseTensor& scale, double eps) {
    for (const DenseTensor* value : std::initializer_list<const DenseTensor*>{
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
SelectedExperts topk(DenseTensor logits, std::int64_t count, const MoeRouting& routing) {
    input(logits);
    require_cuda(logits);
    return cuda::ops::topk(std::move(logits), count, routing);
}
DenseTensor expert_scale(DenseTensor weights, DenseTensor ids, const DenseTensor& scales) {
    compatible(weights, ids);
    compatible(weights, scales);
    require_cuda(weights);
    return cuda::ops::expert_scale(std::move(weights), std::move(ids), scales);
}
DenseTensor weighted_reduce(DenseTensor pairs, DenseTensor weights) {
    compatible(pairs, weights);
    require_cuda(pairs);
    return cuda::ops::weighted_reduce(std::move(pairs), std::move(weights));
}
RoutedGeglu routed_geglu(DenseTensor value, DenseTensor ids, const weights::Weights& gate_up,
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

DenseTensor rms_norm(DenseTensor value, const DenseTensor& weight, double eps, double offset, ScalarType output_type) {
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
DenseTensor layer_norm(DenseTensor value, const DenseTensor& weight, const DenseTensor& bias, double epsilon) {
    compatible(value, weight);
    compatible(value, bias);
    if (value.dim() == 0 || weight.dim() != 1 || bias.dim() != 1 ||
        weight.size(0) != value.size(-1) || bias.size(0) != value.size(-1))
        throw std::invalid_argument("layer_norm input and weights are incompatible");
    require_cuda(value);
    return cuda::ops::layer_norm(std::move(value), weight, bias, epsilon);
}
DenseTensor gelu(DenseTensor value, bool approximate) {
    input(value);
    require_cuda(value);
    return cuda::ops::gelu(std::move(value), approximate);
}
DenseTensor silu(DenseTensor value) {
    input(value);
    require_cuda(value);
    return cuda::ops::silu(std::move(value));
}
DenseTensor relu_squared(DenseTensor value) {
    input(value);
    require_cuda(value);
    return cuda::ops::relu_squared(std::move(value));
}
DenseTensor axial_rope(DenseTensor value, const std::vector<std::int32_t>& positions, double theta) {
    input(value);
    require_cuda(value);
    return cuda::ops::axial_rope(std::move(value), positions, theta);
}
DenseTensor segmented_attention(DenseTensor query, DenseTensor key, DenseTensor value,
                           const std::vector<std::int32_t>& lengths) {
    compatible(query, key);
    compatible(query, value);
    require_cuda(query);
    return cuda::ops::segmented_attention(std::move(query), std::move(key), std::move(value), lengths);
}
std::array<DenseTensor, 3> qkv_norm(std::array<DenseTensor, 3> values, std::array<DenseTensor, 3> weights,
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
DenseTensor softcap(DenseTensor value, double cap) {
    input(value);
    if (cap > 0.0) require_cuda(value);
    return cuda::ops::softcap(std::move(value), cap);
}
DenseTensor scale(DenseTensor value, double factor) {
    input(value);
    require_cuda(value);
    return cuda::ops::scale(std::move(value), factor);
}
DenseTensor embedding(DenseTensor ids, const weights::Weights& weight) {
    check_weight(weight, weights::Layout::matrix);
    input(ids);
    return cuda::ops::embedding(std::move(ids), weight);
}
DenseTensor project(DenseTensor value, const weights::Weights& weight) {
    check_weight(weight, weights::Layout::matrix);
    input(value);
    return cuda::ops::project(std::move(value), weight);
}
DenseTensor linear(DenseTensor value, const weights::Weights& weight, const DenseTensor& bias) {
    check_weight(weight, weights::Layout::matrix);
    input(value);
    if (bias.defined()) input(bias);
    return cuda::ops::linear(std::move(value), weight, bias);
}
DenseTensor affine(DenseTensor value, const weights::Weights& weight, const DenseTensor& bias) {
    check_weight(weight, weights::Layout::matrix);
    input(value);
    if (bias.defined()) input(bias);
    return cuda::ops::affine(std::move(value), weight, bias);
}
DenseTensor mlp(DenseTensor value, const weights::Weights& gate_up, const weights::Weights& down,
           MlpActivation activation, double swiglu_limit) {
    check_weight(gate_up, weights::Layout::group);
    check_weight(down, weights::Layout::matrix);
    input(value);
    return cuda::ops::mlp(std::move(value), gate_up, down, activation, swiglu_limit);
}
std::vector<DenseTensor> project_group(DenseTensor value, const weights::Weights& group) {
    check_weight(group, weights::Layout::group);
    input(value);
    return cuda::ops::project_group(std::move(value), group);
}
DenseTensor sigmoid_project(DenseTensor value, DenseTensor gate, const weights::Weights& weight) {
    check_weight(weight, weights::Layout::matrix);
    compatible(value, gate);
    return cuda::ops::sigmoid_project(std::move(value), std::move(gate), weight);
}
DenseTensor tensor(std::span<const float> values, Device target) {
    device(target);
    return cuda::ops::tensor(values, target);
}
DenseTensor tensor(std::span<const std::int32_t> values, Device target) {
    device(target);
    return cuda::ops::tensor(values, target);
}
DenseTensor tensor(std::span<const std::int64_t> values, Device target) {
    device(target);
    return cuda::ops::tensor(values, target);
}
DenseTensor tensor(std::span<const std::uint8_t> values, Device target) {
    device(target);
    return cuda::ops::tensor(values, target);
}
DenseTensor from_blob(void* data, std::span<const std::int64_t> shape, const TensorOptions& options) {
    device(options.target_device());
    return cuda::ops::from_blob(data, shape, options);
}
DenseTensor copy(DenseTensor value, Device target) {
    input(value);
    device(target);
    return cuda::ops::copy(std::move(value), target);
}
void copy_into(DenseTensor destination, DenseTensor source) {
    input(destination);
    input(source);
    cuda::ops::copy_into(std::move(destination), std::move(source));
}
DenseTensor cast(DenseTensor value, ScalarType type) {
    input(value);
    return cuda::ops::cast(std::move(value), type);
}
DenseTensor cast(DenseTensor value, ScalarType type, Device target) {
    input(value);
    device(target);
    return cuda::ops::cast(std::move(value), type, target);
}
DenseTensor contiguous(DenseTensor value) {
    input(value);
    return cuda::ops::contiguous(std::move(value));
}
DenseTensor clone(DenseTensor value) {
    input(value);
    return cuda::ops::clone(std::move(value));
}
DenseTensor reshape(DenseTensor value, std::span<const std::int64_t> shape) {
    input(value);
    return tyloi::reshape(std::move(value), shape);
}
DenseTensor expand(DenseTensor value, std::span<const std::int64_t> shape) {
    input(value);
    return tyloi::expand(std::move(value), shape);
}
DenseTensor cat(std::span<const DenseTensor> values, std::int64_t dim) {
    if (values.empty()) throw std::invalid_argument("cat requires at least one tensor");
    input(values.front());
    for (const auto& value : values.subspan(1)) compatible(values.front(), value);
    return cuda::ops::cat(values, dim);
}
DenseTensor zeros(std::span<const std::int64_t> shape, const TensorOptions& options) {
    device(options.target_device());
    return cuda::ops::zeros(shape, options);
}
DenseTensor select(DenseTensor value, std::int64_t dim, std::int64_t index) {
    input(value);
    return tyloi::select(std::move(value), dim, index);
}
DenseTensor unsqueeze(DenseTensor value, std::int64_t dim) {
    input(value);
    return tyloi::unsqueeze(std::move(value), dim);
}
DenseTensor slice(DenseTensor value, std::int64_t dim, std::int64_t start, std::int64_t end, std::int64_t step) {
    input(value);
    return tyloi::slice(std::move(value), dim, start, end, step);
}
DenseTensor index_select(DenseTensor value, std::int64_t dim, DenseTensor indices) {
    compatible(value, indices);
    require_cuda(value);
    return cuda::ops::index_select(std::move(value), dim, std::move(indices));
}
DenseTensor sum(DenseTensor value, std::int64_t dim, bool keepdim) {
    input(value);
    require_cuda(value);
    return cuda::ops::sum(std::move(value), dim, keepdim);
}
void index_copy(DenseTensor value, std::int64_t dim, DenseTensor indices, DenseTensor source) {
    compatible(value, indices);
    compatible(value, source);
    require_cuda(value);
    cuda::ops::index_copy(std::move(value), dim, std::move(indices), std::move(source));
}
std::vector<DenseTensor> chunk(DenseTensor value, std::int64_t count, std::int64_t dim) {
    input(value);
    return tyloi::chunk(std::move(value), count, dim);
}
DenseTensor arange(std::int64_t first, std::int64_t last, Device target) {
    device(target);
    return cuda::ops::arange(first, last, target);
}
DenseTensor full(std::span<const std::int64_t> shape, std::int64_t value, Device target) {
    device(target);
    return cuda::ops::full(shape, value, target);
}
DenseTensor add(DenseTensor value, std::int64_t offset) {
    input(value);
    require_cuda(value);
    return cuda::ops::add(std::move(value), offset);
}
DenseTensor mean(DenseTensor value, std::int64_t dim, bool keepdim) {
    input(value);
    require_cuda(value);
    return cuda::ops::mean(std::move(value), dim, keepdim);
}
DenseTensor amax(DenseTensor value, std::int64_t dim, bool keepdim) {
    input(value);
    require_cuda(value);
    return cuda::ops::amax(std::move(value), dim, keepdim);
}
DenseTensor argmax(DenseTensor value, std::int64_t dim, bool keepdim) {
    input(value);
    require_cuda(value);
    return cuda::ops::argmax(std::move(value), dim, keepdim);
}
DenseTensor all(DenseTensor value, std::int64_t dim, bool keepdim) {
    input(value);
    require_cuda(value);
    return cuda::ops::all(std::move(value), dim, keepdim);
}
DenseTensor any(DenseTensor value, std::int64_t dim, bool keepdim) {
    input(value);
    require_cuda(value);
    return cuda::ops::any(std::move(value), dim, keepdim);
}
DenseTensor abs(DenseTensor value) {
    input(value);
    require_cuda(value);
    return cuda::ops::abs(std::move(value));
}
DenseTensor square(DenseTensor value) {
    input(value);
    require_cuda(value);
    return cuda::ops::square(std::move(value));
}
DenseTensor exp(DenseTensor value) {
    input(value);
    require_cuda(value);
    return cuda::ops::exp(std::move(value));
}
DenseTensor sqrt(DenseTensor value) {
    input(value);
    require_cuda(value);
    return cuda::ops::sqrt(std::move(value));
}
DenseTensor logical_not(DenseTensor value) {
    input(value);
    require_cuda(value);
    return cuda::ops::logical_not(std::move(value));
}
DenseTensor clamp(DenseTensor value, double minimum, double maximum) {
    input(value);
    require_cuda(value);
    return cuda::ops::clamp(std::move(value), minimum, maximum);
}
DenseTensor clamp_min(DenseTensor value, double minimum) {
    input(value);
    require_cuda(value);
    return cuda::ops::clamp_min(std::move(value), minimum);
}
DenseTensor clamp_max(DenseTensor value, double maximum) {
    input(value);
    require_cuda(value);
    return cuda::ops::clamp_max(std::move(value), maximum);
}
DenseTensor remainder(DenseTensor value, double divisor) {
    input(value);
    require_cuda(value);
    return cuda::ops::remainder(std::move(value), divisor);
}
DenseTensor eq(DenseTensor left, DenseTensor right) {
    compatible(left, right);
    require_cuda(left);
    return cuda::ops::eq(std::move(left), std::move(right));
}
DenseTensor eq(DenseTensor value, double other) {
    input(value);
    require_cuda(value);
    return cuda::ops::eq(std::move(value), other);
}
DenseTensor ne(DenseTensor left, DenseTensor right) {
    compatible(left, right);
    require_cuda(left);
    return cuda::ops::ne(std::move(left), std::move(right));
}
DenseTensor ne(DenseTensor value, double other) {
    input(value);
    require_cuda(value);
    return cuda::ops::ne(std::move(value), other);
}
bool equal(const DenseTensor& left, const DenseTensor& right) {
    compatible(left, right);
    require_cuda(left);
    return cuda::ops::equal(left, right);
}
DenseTensor masked_select(DenseTensor value, DenseTensor mask) {
    compatible(value, mask);
    require_cuda(value);
    return cuda::ops::masked_select(std::move(value), std::move(mask));
}
DenseTensor gather(DenseTensor value, std::int64_t dim, DenseTensor indices) {
    compatible(value, indices);
    require_cuda(value);
    return cuda::ops::gather(std::move(value), dim, std::move(indices));
}
DenseTensor repeat(DenseTensor value, std::span<const std::int64_t> repeats) {
    input(value);
    require_cuda(value);
    return cuda::ops::repeat(std::move(value), repeats);
}
DenseTensor repeat_interleave(DenseTensor value, std::int64_t repeats, std::int64_t dim) {
    input(value);
    require_cuda(value);
    return cuda::ops::repeat_interleave(std::move(value), repeats, dim);
}
DenseTensor tril(DenseTensor value, std::int64_t diagonal) {
    input(value);
    require_cuda(value);
    return cuda::ops::tril(std::move(value), diagonal);
}
DenseTensor masked_fill(DenseTensor value, DenseTensor mask, double fill_value) {
    compatible(value, mask);
    require_cuda(value);
    return cuda::ops::masked_fill(std::move(value), std::move(mask), fill_value);
}
void masked_fill_inplace(DenseTensor value, DenseTensor mask, double fill_value) {
    compatible(value, mask);
    require_cuda(value);
    cuda::ops::masked_fill_inplace(std::move(value), std::move(mask), fill_value);
}
void fill(DenseTensor value, double fill_value) {
    input(value);
    if (value.is_cuda()) {
        cuda::ops::fill(std::move(value), fill_value);
        return;
    }
    if (!value.is_contiguous())
        throw std::invalid_argument("CPU fill requires a contiguous tensor");
    if (fill_value == 0.0) {
        std::memset(value.data_ptr(), 0, value.nbytes());
        return;
    }
    switch (value.scalar_type()) {
        case ScalarType::boolean:
            std::fill_n(value.data_ptr<bool>(), value.numel(), fill_value != 0.0);
            break;
        case ScalarType::uint8:
            std::fill_n(value.data_ptr<std::uint8_t>(), value.numel(), static_cast<std::uint8_t>(fill_value));
            break;
        case ScalarType::int8:
            std::fill_n(value.data_ptr<std::int8_t>(), value.numel(), static_cast<std::int8_t>(fill_value));
            break;
        case ScalarType::int16:
            std::fill_n(value.data_ptr<std::int16_t>(), value.numel(), static_cast<std::int16_t>(fill_value));
            break;
        case ScalarType::int32:
            std::fill_n(value.data_ptr<std::int32_t>(), value.numel(), static_cast<std::int32_t>(fill_value));
            break;
        case ScalarType::int64:
            std::fill_n(value.data_ptr<std::int64_t>(), value.numel(), static_cast<std::int64_t>(fill_value));
            break;
        case ScalarType::float16:
            std::fill_n(value.data_ptr<std::uint16_t>(), value.numel(),
                cuda::encode_float16(static_cast<float>(fill_value)));
            break;
        case ScalarType::bfloat16:
            std::fill_n(value.data_ptr<std::uint16_t>(), value.numel(),
                cuda::encode_bfloat16(static_cast<float>(fill_value)));
            break;
        case ScalarType::float32:
            std::fill_n(value.data_ptr<float>(), value.numel(), static_cast<float>(fill_value));
            break;
        case ScalarType::float64:
            std::fill_n(value.data_ptr<double>(), value.numel(), fill_value);
            break;
        default:
            throw std::invalid_argument("CPU fill for this scalar type is not implemented yet");
    }
}
void scatter_add(DenseTensor value, std::int64_t dim, DenseTensor indices, DenseTensor source) {
    compatible(value, indices);
    compatible(value, source);
    require_cuda(value);
    cuda::ops::scatter_add(std::move(value), dim, std::move(indices), std::move(source));
}
void scatter(DenseTensor value, std::int64_t dim, DenseTensor indices, DenseTensor source) {
    compatible(value, indices);
    compatible(value, source);
    require_cuda(value);
    cuda::ops::scatter(std::move(value), dim, std::move(indices), std::move(source));
}
void index_fill(DenseTensor value, std::int64_t dim, DenseTensor indices, double fill_value) {
    compatible(value, indices);
    require_cuda(value);
    cuda::ops::index_fill(std::move(value), dim, std::move(indices), fill_value);
}
#endif
} // namespace tyloi::ops
