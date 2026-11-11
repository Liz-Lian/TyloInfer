#include "engine/ops.h"

#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/ops/norm/rms.h"
#include "engine/backends/cuda/ops/norm/layer.h"
#include "engine/backends/cuda/ops/vision/attention.h"
#include "engine/backends/cuda/ops/ffn/dense.h"
#include "engine/backends/cuda/ops/ffn/routed.h"
#include "engine/backends/cuda/ops/ffn/residual.h"
#include "engine/backends/cuda/ops/projection/projection.h"
#include "engine/backends/cuda/ops/tensor.h"
#include "engine/backends/cuda/native/tensor/tensor.h"
#include "engine/backends/cuda/ops/attention/attention.h"
#include "engine/backends/cuda/ops/attention/kv.h"
#include "engine/backends/cuda/ops/rope/rope.h"
#include "engine/backends/cuda/ops/rope/tail.h"
#include "engine/backends/cuda/native/profile.h"
#include "engine/backends/cuda/native/device.h"
#include "engine/backends/cuda/kernels/mfq_cuda_sampling_ops.h"
#include <cmath>
#include <cstring>
#include <exception>
#include <iostream>
#endif

namespace tyloi::ops {
#ifdef TYLOINFER_NATIVE_CUDA
namespace {
const DenseTensor& dense(const Tensor& value) {
    const auto* result = value.representation<DenseTensor>();
    if (!result || result->is_encoded()) throw std::invalid_argument("operation requires a dense tensor");
    return *result;
}
DenseTensor& dense(Tensor& value) {
    auto* result = value.representation<DenseTensor>();
    if (!result || result->is_encoded()) throw std::invalid_argument("operation requires a dense tensor");
    return *result;
}
Tensor owner(DenseTensor value) {
    return std::make_shared<DenseTensor>(std::move(value));
}
std::vector<Tensor> owners(std::vector<DenseTensor> values) {
    std::vector<Tensor> result;
    result.reserve(values.size());
    for (auto& value : values) result.push_back(owner(std::move(value)));
    return result;
}
void device(Device value) {
    if (!value.is_cpu() && !value.is_cuda())
        throw std::invalid_argument("unsupported tensor device");
    if (value.is_cuda()) (void)cuda::Executor::current();
}
void input(const Tensor& value_tensor) {
    const auto& value = dense(value_tensor);
    if (!value.defined()) throw std::invalid_argument("undefined tensor");
    value.require_dense();
    device(value.device());
}
void compatible(const Tensor& left_tensor, const Tensor& right_tensor) {
    const auto& left = ::tyloi::ops::dense(left_tensor);
    const auto& right = ::tyloi::ops::dense(right_tensor);
    input(left);
    input(right);
    if (left.device() != right.device())
        throw std::invalid_argument("tensor devices do not match");
}
void residual_inputs(const Tensor& residual, const Tensor& value) {
    compatible(residual, value);
    if (residual.dim() != 3 || value.dim() != 3 ||
        residual.size(0) != value.size(0) || residual.size(1) != value.size(1) || residual.size(2) != value.size(2))
        throw std::invalid_argument("residual inputs must have matching [batch,tokens,width] shapes");
}
void require_cuda(const Tensor& value_tensor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    if (!value.device().is_cuda()) throw std::invalid_argument("operation requires a CUDA tensor");
}
bool encoded(std::initializer_list<const Tensor> values) {
    for (const auto& value : values)
        if (value.defined() && value.is_encoded()) return true;
    return false;
}
}

AttentionResult attention(const AttentionInput& request, const state::KvView& kv,
                          const AttentionConfig& config) {
    compatible(request.query, request.key);
    compatible(request.query, request.value);
    input(request.positions);
    input(request.write_positions);
    if (request.sequence_lengths.defined()) input(request.sequence_lengths);
    if (request.attention_mask.defined()) input(request.attention_mask);
    if (request.query.dim() != 4 || request.key.dim() != 4 || request.value.dim() != 4 ||
        dense(request.key).sizes() != dense(request.value).sizes() ||
        request.query.size(0) != request.key.size(0) ||
        request.query.size(2) != request.key.size(2) ||
        request.query.size(3) != request.key.size(3) || request.key.size(1) <= 0 ||
        request.query.size(1) <= 0 || request.query.size(3) <= 0)
        throw std::invalid_argument("attention query, key and value geometry is incompatible");
    for (const auto& tensor : {kv.visible_key, kv.visible_value, kv.physical_key,
                                kv.physical_value, kv.key_chunk_ptrs,
                                kv.value_chunk_ptrs, kv.page_table})
        if (tensor.defined()) compatible(request.query, tensor);
    require_cuda(request.query);
    return cuda::ops::attention(request, kv, config);
}

std::int64_t attention_parts(std::int64_t planned_len, Device target) {
    device(target);
    if (!target.is_cuda()) throw std::invalid_argument("attention parts require a CUDA device");
    return cuda::ops::attention_parts(planned_len);
}

Tensor attention(const Tensor& query_tensor, const Tensor& key_tensor, const Tensor& value_tensor,
                 const Tensor& mask, double scale) {
    const auto& query = dense(query_tensor);
    const auto& key = dense(key_tensor);
    const auto& value = dense(value_tensor);
    compatible(query, key);
    compatible(query, value);
    if (mask.defined()) compatible(query, mask);
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
    const std::optional<DenseTensor> dense_mask = mask ? std::optional<DenseTensor>(dense(mask)) : std::nullopt;
    return owner(cuda::ops::attention(query, key, value, dense_mask, scale));
}

Tensor sparse_attention(const Tensor& query_tensor, const Tensor& cache_tensor,
                        const Tensor& indices_tensor, const Tensor& mask_tensor,
                        const Tensor& sinks_tensor, double scale_value) {
    const auto& query = dense(query_tensor);
    const auto& cache = dense(cache_tensor);
    const auto& indices = dense(indices_tensor);
    const auto& mask = dense(mask_tensor);
    const auto& sinks = dense(sinks_tensor);
    compatible(query, cache);
    compatible(query, indices);
    compatible(query, mask);
    compatible(query, sinks);
    require_cuda(query);
    if (query.dim() != 4 || cache.dim() != 3 || indices.dim() != 3 ||
        mask.sizes() != indices.sizes() || sinks.dim() != 1 ||
        query.size(0) != cache.size(0) || query.size(0) != indices.size(0) ||
        query.size(2) != indices.size(1) || query.size(3) != cache.size(2))
        throw std::invalid_argument("sparse attention geometry is incompatible");
    return owner(cuda::ops::sparse_attention(
        query, cache, indices, mask, sinks, scale_value));
}

TailRopeTable tail_rope_table(std::int64_t max_positions, double base, std::int64_t compress_ratio,
        Device target, double compress_base, std::int64_t original_positions,
        double factor, double beta_fast, double beta_slow) {
    device(target);
    if (!target.is_cuda() || target.index < 0)
        throw std::invalid_argument("tail RoPE table requires a CUDA device");
    return cuda::ops::tail_rope_table(max_positions, base, compress_ratio, target,
        compress_base, original_positions, factor, beta_fast, beta_slow);
}

Tensor tail_rope(const Tensor& value, const Tensor& positions,
        const TailRopeTable& table, bool inverse) {
    const auto& x = dense(value);
    const auto& pos = dense(positions);
    device(x.device());
    if (!x.device().is_cuda() || x.device().index < 0)
        throw std::invalid_argument("tail RoPE requires a CUDA device");
    if (!table.cos || !(inverse ? table.negative_sin : table.sin))
        throw std::invalid_argument("tail RoPE table requires cosine and sine");
    return owner(cuda::ops::tail_rope(x, pos, table, inverse));
}

RopeTable rope_table(std::int64_t positions, std::int64_t dimension, double base, Device target,
                     std::int64_t frequency_dim, std::int64_t active_pairs) {
    device(target);
    if (!target.is_cuda()) throw std::invalid_argument("RoPE table requires a CUDA device");
    if (positions <= 0 || dimension <= 0 || dimension % 2 != 0 ||
        !std::isfinite(base) || base <= 0.0 || active_pairs > dimension / 2)
        throw std::invalid_argument("invalid RoPE table parameters");
    return cuda::ops::rope_table(positions, dimension, base, target, frequency_dim, active_pairs);
}

Tensor rope(const Tensor& value_tensor, const Tensor& positions_tensor, const RopeTable& table, bool bf16, bool grid) {
    const auto& value = dense(value_tensor);
    const auto& positions = dense(positions_tensor);
    if (!table.cos || !table.sin) throw std::invalid_argument("RoPE table requires cosine and sine");
    compatible(value, table.cos);
    compatible(value, table.sin);
    for (const auto& item : {table.sections, table.empty_sections,
                             table.interleaved_order, table.interleaved_inverse})
        if (item.defined()) (void)dense(item);
    input(positions);
    if (value.dim() == 0 || table.rotary_dim <= 0 || table.rotary_dim % 2 != 0 ||
        table.rotary_dim > value.size(-1))
        throw std::invalid_argument("invalid RoPE rotary dimension");
    require_cuda(value);
    return owner(cuda::ops::rope(value, positions, table, bf16, grid));
}
Tensor norm_rope_kv(const AttentionInput& request, const state::KvView& kv,
    const RopeTable& table, const Tensor& q_norm, const Tensor& k_norm, double eps, double offset) {
    if (!request.query.is_cuda()) return {};
    if (encoded({request.query, request.key, request.value, request.positions, request.write_positions,
                 table.cos, table.sin, table.sections, table.empty_sections, table.interleaved_order,
                 table.interleaved_inverse, q_norm, k_norm, kv.visible_key, kv.visible_value,
                 kv.physical_key, kv.physical_value, kv.key_chunk_ptrs, kv.value_chunk_ptrs, kv.page_table,
                 request.sequence_lengths, request.attention_mask})) return {};
    auto result = cuda::ops::norm_rope_kv(request, kv, table, dense(q_norm), dense(k_norm), eps, offset);
    return result ? owner(std::move(*result)) : Tensor{};
}
Tensor rope_kv(const AttentionInput& request, const state::KvView& kv,
                               const RopeTable& table) {
    if (!request.query.is_cuda()) return {};
    if (encoded({request.query, request.key, request.value, request.positions, request.write_positions,
                 table.cos, table.sin, table.sections, table.empty_sections, table.interleaved_order,
                 table.interleaved_inverse, kv.visible_key, kv.visible_value, kv.physical_key,
                 kv.physical_value, kv.key_chunk_ptrs, kv.value_chunk_ptrs, kv.page_table,
                 request.sequence_lengths, request.attention_mask})) return {};
    auto result = cuda::ops::rope_kv(request, kv, table);
    return result ? owner(std::move(*result)) : Tensor{};
}

std::pair<Tensor, Tensor> append_kv(state::KvView kv, const Tensor& key_tensor, const Tensor& value_tensor,
                                    const Tensor& positions_tensor, std::int64_t start,
                                    bool contiguous_prefill_prefix) {
    const auto& key = dense(key_tensor);
    const auto& value = dense(value_tensor);
    const auto& positions = dense(positions_tensor);
    compatible(key, value);
    compatible(key, positions);
    if (key.dim() != 4 || value.dim() != 4 || key.sizes() != value.sizes())
        throw std::invalid_argument("KV write key and value geometry is incompatible");
    for (const auto& tensor : {kv.physical_key, kv.physical_value, kv.key_chunk_ptrs,
                                kv.value_chunk_ptrs, kv.page_table})
        if (tensor.defined()) compatible(key, tensor);
    // A zero batch skips the unpaged, unquantized index writes.
    if (kv.paged() || kv.quantization.enabled() || positions.dim() != 2 || positions.size(0) != 0)
        require_cuda(key);
    auto result = cuda::ops::append_kv(std::move(kv), key, value, positions, start, contiguous_prefill_prefix);
    return {owner(std::move(result.first)), owner(std::move(result.second))};
}

Tensor add(const Tensor& left_tensor, const Tensor& right_tensor) {
    const auto& left = ::tyloi::ops::dense(left_tensor);
    const auto& right = ::tyloi::ops::dense(right_tensor);
    compatible(left, right);
    require_cuda(left);
    return owner(cuda::ops::add(left, right));
}
Tensor multiply(const Tensor& left_tensor, const Tensor& right_tensor) {
    const auto& left = ::tyloi::ops::dense(left_tensor);
    const auto& right = ::tyloi::ops::dense(right_tensor);
    compatible(left, right);
    require_cuda(left);
    return owner(cuda::ops::multiply(left, right));
}
Tensor matmul(const Tensor& left_tensor, const Tensor& right_tensor) {
    const auto& left = ::tyloi::ops::dense(left_tensor);
    const auto& right = ::tyloi::ops::dense(right_tensor);
    compatible(left, right);
    require_cuda(left);
    return owner(cuda::ops::matmul(left, right));
}
Tensor transpose(const Tensor& value_tensor, std::int64_t first, std::int64_t second) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(tyloi::transpose(value, first, second));
}
Tensor add_residual(const Tensor& residual_tensor, const Tensor& value_tensor, ResidualMode mode) {
    const auto& residual = ::tyloi::ops::dense(residual_tensor);
    const auto& value = ::tyloi::ops::dense(value_tensor);
    compatible(residual, value);
    require_cuda(residual);
    return owner(cuda::ops::add_residual(residual, value, mode == ResidualMode::configured));
}

std::array<Tensor, 2> residual_norm(const Tensor& residual, const Tensor& value,
        const Tensor& weight, double eps, double offset, bool bf16) {
    residual_inputs(residual, value);
    compatible(residual, weight);
    if (weight.numel() != residual.size(2))
        throw std::invalid_argument("residual norm weight has the wrong width");
    auto result = cuda::ops::residual_norm(dense(residual), dense(value), weight, eps, offset, bf16);
    return {owner(std::move(result[0])), owner(std::move(result[1]))};
}
std::array<Tensor, 2> recurrent_norm(const Tensor& residual, const Tensor& value,
        const Tensor& weight, double eps, double offset) {
    residual_inputs(residual, value);
    compatible(residual, weight);
    if (weight.numel() != residual.size(2))
        throw std::invalid_argument("residual norm weight has the wrong width");
    auto result = cuda::ops::recurrent_norm(dense(residual), dense(value), weight, eps, offset);
    return {owner(std::move(result[0])), owner(std::move(result[1]))};
}
Tensor residual_add(const Tensor& residual, const Tensor& value) {
    residual_inputs(residual, value);
    return owner(cuda::ops::residual_add(dense(residual), dense(value)));
}
Tensor recurrent_add(const Tensor& residual, const Tensor& value) {
    residual_inputs(residual, value);
    return owner(cuda::ops::recurrent_add(dense(residual), dense(value)));
}
std::optional<ResidualNorms> residual_norms(const Tensor& residual_tensor, const Tensor& attention_tensor, const Tensor& attention_weight_tensor,
                                            const Tensor& dense_weight_tensor, const Tensor& router_weight_tensor,
                                            const Tensor& expert_weight_tensor, double eps) {
    if (!residual_tensor.is_cuda() ||
        encoded({residual_tensor, attention_tensor, attention_weight_tensor,
                 dense_weight_tensor, router_weight_tensor, expert_weight_tensor})) return std::nullopt;
    const auto& residual = ::tyloi::ops::dense(residual_tensor);
    const auto& attention = ::tyloi::ops::dense(attention_tensor);
    const auto& attention_weight = ::tyloi::ops::dense(attention_weight_tensor);
    const auto& dense_weight = ::tyloi::ops::dense(dense_weight_tensor);
    const auto& router_weight = ::tyloi::ops::dense(router_weight_tensor);
    const auto& expert_weight = ::tyloi::ops::dense(expert_weight_tensor);
    return cuda::ops::residual_norms(residual, attention, attention_weight,
                                     dense_weight, router_weight, expert_weight, eps);
}
Tensor merge_norms(const Tensor& dense_tensor, const Tensor& experts_tensor, const Tensor& residual_tensor, const Tensor& dense_weight_tensor,
                   const Tensor& expert_weight_tensor, const Tensor& output_weight_tensor,
                   const Tensor& scale_tensor, double eps) {
    const auto& dense = ::tyloi::ops::dense(dense_tensor);
    const auto& experts = ::tyloi::ops::dense(experts_tensor);
    const auto& residual = ::tyloi::ops::dense(residual_tensor);
    const auto& dense_weight = ::tyloi::ops::dense(dense_weight_tensor);
    const auto& expert_weight = ::tyloi::ops::dense(expert_weight_tensor);
    const auto& output_weight = ::tyloi::ops::dense(output_weight_tensor);
    const auto& scale = ::tyloi::ops::dense(scale_tensor);
    for (const DenseTensor* value : std::initializer_list<const DenseTensor*>{
             &experts, &residual, &dense_weight, &expert_weight, &output_weight, &scale})
        compatible(dense, *value);
    require_cuda(dense);
    return owner(cuda::ops::merge_norms(dense, experts, residual,
                                  dense_weight, expert_weight, output_weight, scale, eps));
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
struct ProfileCapture::Impl {
    CudaProfiler& profiler;
    bool previous, enabled, selected;
    std::string report_name;
    Impl(CudaProfiler& profiler, bool selected, std::string report_name)
        : profiler(profiler), previous(profiler.enabled), enabled(selected || profiler.enabled),
          selected(selected), report_name(std::move(report_name)) {
        if (selected) {
            profiler.reset();
            profiler.enabled = true;
        }
    }
    ~Impl() {
        if (selected && !report_name.empty() && std::uncaught_exceptions() == 0) {
            try {
                profiler.report(report_name);
                profiler.reset();
            } catch (const std::exception& error) {
                std::cerr << "prefill_profile_error " << error.what() << '\n';
            }
        }
        profiler.enabled = previous;
    }
};
ProfileCapture::ProfileCapture(bool selected)
    : impl_(selected ? std::make_unique<Impl>(cuda::Executor::current().profiler(), true, std::string{}) : nullptr) {}
ProfileCapture::ProfileCapture(bool selected, const std::string& report_name)
    : impl_(std::make_unique<Impl>(cuda::Executor::current().profiler(), selected, report_name)) {}
ProfileCapture::~ProfileCapture() = default;
void ProfileCapture::report(const std::string& name) {
    if (impl_ && impl_->selected) impl_->profiler.report(name);
}
void ProfileCapture::resume() {
    if (impl_) impl_->profiler.enabled = impl_->enabled;
}
SelectedExperts topk(const Tensor& logits_tensor, std::int64_t count, const MoeRouting& routing,
                     const Tensor& bias) {
    const auto& logits = ::tyloi::ops::dense(logits_tensor);
    input(logits);
    require_cuda(logits);
    const auto* selection_bias = bias.defined() ? &dense(bias) : nullptr;
    if (selection_bias) compatible(logits, bias);
    return cuda::ops::topk(logits, count, routing, selection_bias);
}
void expert_scale(Tensor& weights_tensor, const Tensor& ids_tensor, const Tensor& scales_tensor) {
    auto& weights = dense(weights_tensor);
    const auto& ids = dense(ids_tensor);
    const auto& scales = dense(scales_tensor);
    compatible(weights, ids);
    compatible(weights, scales);
    require_cuda(weights);
    cuda::ops::expert_scale(weights, ids, scales);
}
Tensor weighted_reduce(const Tensor& pairs_tensor, const Tensor& weights_tensor) {
    const auto& pairs = ::tyloi::ops::dense(pairs_tensor);
    const auto& weights = ::tyloi::ops::dense(weights_tensor);
    compatible(pairs, weights);
    require_cuda(pairs);
    return owner(cuda::ops::weighted_reduce(pairs, weights));
}
RoutedGeglu routed_geglu(const Tensor& value_tensor, const Tensor& ids_tensor,
                         const Tensor& gate_up, const Tensor& down,
                         bool trace) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    const auto& ids = ::tyloi::ops::dense(ids_tensor);
    compatible(value, ids);
    if (!gate_up.defined() || !down.defined() || gate_up.dim() != 3 || down.dim() != 3 ||
        value.dim() != 2 || ids.dim() != 2 || value.size(0) != ids.size(0) ||
        value.scalar_type() != kFloat16 || ids.scalar_type() != kInt32 ||
        value.size(0) <= 0 || value.size(1) <= 0 || ids.size(1) <= 0 ||
        gate_up.size(0) <= 0 || gate_up.size(1) <= 0 || down.size(1) <= 0 ||
        gate_up.size(0) != down.size(0) || gate_up.size(2) != value.size(-1) ||
        gate_up.size(1) % 2 != 0 || gate_up.size(1) / 2 != down.size(2))
        throw std::invalid_argument("routed GEGLU input, routes, and expert banks are incompatible");
    require_cuda(value);
    return cuda::ops::routed_geglu(value, ids, gate_up, down, trace);
}

Tensor rms_norm(const Tensor& value_tensor, const Tensor& weight_tensor, double eps, double offset, ScalarType output_type) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    const auto& weight = ::tyloi::ops::dense(weight_tensor);
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
    return owner(cuda::ops::rms_norm(value, weight, eps, offset, output_type));
}
Tensor layer_norm(const Tensor& value_tensor, const Tensor& weight_tensor, const Tensor& bias_tensor, double epsilon) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    const auto& weight = ::tyloi::ops::dense(weight_tensor);
    const auto& bias = ::tyloi::ops::dense(bias_tensor);
    compatible(value, weight);
    compatible(value, bias);
    if (value.dim() == 0 || weight.dim() != 1 || bias.dim() != 1 ||
        weight.size(0) != value.size(-1) || bias.size(0) != value.size(-1))
        throw std::invalid_argument("layer_norm input and weights are incompatible");
    require_cuda(value);
    return owner(cuda::ops::layer_norm(value, weight, bias, epsilon));
}
Tensor gelu(const Tensor& value_tensor, bool approximate) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::gelu(value, approximate));
}
Tensor silu(const Tensor& value_tensor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::silu(value));
}
Tensor relu_squared(const Tensor& value_tensor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::relu_squared(value));
}
Tensor axial_rope(const Tensor& value_tensor, const std::vector<std::int32_t>& positions, double theta) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::axial_rope(value, positions, theta));
}
Tensor segmented_attention(const Tensor& query_tensor, const Tensor& key_tensor, const Tensor& value_tensor,
                           const std::vector<std::int32_t>& lengths) {
    const auto& query = ::tyloi::ops::dense(query_tensor);
    const auto& key = ::tyloi::ops::dense(key_tensor);
    const auto& value = ::tyloi::ops::dense(value_tensor);
    compatible(query, key);
    compatible(query, value);
    require_cuda(query);
    return owner(cuda::ops::segmented_attention(query, key, value, lengths));
}
std::array<Tensor, 3> qkv_norm(std::array<Tensor, 3> values, std::array<Tensor, 3> weights,
                               double eps, double offset, ScalarType output_type) {
    if (output_type != ScalarType::float32 && output_type != ScalarType::bfloat16)
        throw std::invalid_argument("unsupported qkv_norm output type");
    std::array<const DenseTensor*, 3> dense_weights{};
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!values[i].defined()) throw std::invalid_argument("qkv_norm requires all values");
        input(values[i]);
        if (weights[i].defined()) {
            dense_weights[i] = &dense(weights[i]);
            compatible(values[i], weights[i]);
            if (values[i].dim() == 0 || weights[i].dim() == 0 ||
                values[i].size(-1) <= 0 || weights[i].numel() != values[i].size(-1))
                throw std::invalid_argument("qkv_norm input and weight are incompatible");
        }
    }
    for (std::size_t i = 0; i < values.size(); ++i)
        if (weights[i].defined()) require_cuda(values[i]);
    (void)cuda::Executor::current();
    std::array<DenseTensor, 3> dense_values{dense(values[0]), dense(values[1]), dense(values[2])};
    auto result = cuda::ops::qkv_norm(std::move(dense_values), dense_weights, eps, offset, output_type);
    return {owner(std::move(result[0])), owner(std::move(result[1])), owner(std::move(result[2]))};
}
Tensor softcap(const Tensor& value_tensor, double cap) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    if (cap > 0.0) require_cuda(value);
    return owner(cuda::ops::softcap(value, cap));
}
Tensor scale(const Tensor& value_tensor, double factor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::scale(value, factor));
}
Tensor divide(const Tensor& value_tensor, double divisor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(cuda::ops::divide(value, divisor));
}
Tensor embedding(const Tensor& ids_tensor, const ::tyloi::Tensor& weight) {
    const auto& ids = ::tyloi::ops::dense(ids_tensor);
    input(ids);
    return owner(cuda::ops::embedding(ids, weight));
}
Tensor project(const Tensor& value_tensor, const ::tyloi::Tensor& weight) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(cuda::ops::project(value, weight));
}
Tensor project(const Tensor& value, const Tensor& weight, const Tensor& ids) {
    compatible(value, ids);
    return owner(cuda::ops::project(dense(value), weight, dense(ids)));
}
Tensor low_rank(const Tensor& value_tensor, const Tensor& output_a, const Tensor& output_b,
                std::int64_t groups, bool groupwise, bool profile) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(cuda::ops::low_rank(value, output_a, output_b, groups, groupwise, profile));
}
Tensor linear(const Tensor& value_tensor, const ::tyloi::Tensor& weight, const Tensor& bias) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    const auto* bias_dense = bias.defined() ? &::tyloi::ops::dense(bias) : nullptr;
    if (bias_dense) input(*bias_dense);
    return owner(cuda::ops::linear(value, weight, bias_dense));
}
Tensor affine(const Tensor& value_tensor, const ::tyloi::Tensor& weight, const Tensor& bias) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    const auto* bias_dense = bias.defined() ? &::tyloi::ops::dense(bias) : nullptr;
    if (bias_dense) input(*bias_dense);
    return owner(cuda::ops::affine(value, weight, bias_dense));
}
Tensor mlp(const Tensor& value_tensor, std::span<const Tensor> gate_up, const ::tyloi::Tensor& down,
           MlpActivation activation, double swiglu_limit) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(cuda::ops::mlp(value, gate_up, down, activation, swiglu_limit));
}
Tensor bf16_mlp(const Tensor& value, std::span<const Tensor> gate_up, const Tensor& down,
        MlpActivation activation, double swiglu_limit) {
    input(value);
    return owner(cuda::ops::bf16_mlp(dense(value), gate_up, down, activation, swiglu_limit));
}
Tensor additive_mlp(const Tensor& value,
        std::span<const Tensor> gate_up, const Tensor& down, MlpActivation activation, double limit,
        std::span<const Tensor> extra_gate_up, const Tensor& extra_down,
        MlpActivation extra_activation, double extra_limit) {
    input(value);
    return owner(cuda::ops::additive_mlp(dense(value), gate_up, down, activation, limit,
        extra_gate_up, extra_down, extra_activation, extra_limit));
}
std::optional<Tensor> mlp_residual(const Tensor& value, const Tensor& residual,
        std::span<const Tensor> gate_up, const Tensor& down, MlpActivation activation,
        double swiglu_limit, bool recurrent) {
    compatible(value, residual);
    if (value.dim() != 2 || value.sizes() != residual.sizes())
        throw std::invalid_argument("fused MLP residual inputs must have matching [rows,width] shapes");
    auto result = cuda::ops::mlp_residual(dense(value), dense(residual), gate_up, down,
        activation, swiglu_limit, recurrent);
    if (!result) return std::nullopt;
    return owner(std::move(*result));
}
std::vector<Tensor> project_group(const Tensor& value_tensor, std::span<const Tensor> group) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owners(cuda::ops::project_group(value, group));
}
bool grouped(std::span<const Tensor> weights) {
    return cuda::ops::grouped(weights);
}
std::vector<Tensor> project_group(const Tensor& value_tensor, std::span<const Tensor> weights,
                                  std::size_t prefix) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owners(cuda::ops::project_group(value, weights, prefix));
}
Tensor sigmoid_project(const Tensor& value_tensor, const Tensor& gate_tensor, const ::tyloi::Tensor& weight) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    const auto& gate = ::tyloi::ops::dense(gate_tensor);
    compatible(value, gate);
    return owner(cuda::ops::sigmoid_project(value, gate, weight));
}
Tensor tensor(std::span<const float> values, Device target) {
    device(target);
    return owner(cuda::ops::tensor(values, target));
}
Tensor tensor(std::span<const std::int32_t> values, Device target) {
    device(target);
    return owner(cuda::ops::tensor(values, target));
}
Tensor tensor(std::span<const std::int64_t> values, Device target) {
    device(target);
    return owner(cuda::ops::tensor(values, target));
}
Tensor tensor(std::span<const std::uint8_t> values, Device target) {
    device(target);
    return owner(cuda::ops::tensor(values, target));
}
Tensor from_blob(void* data, std::span<const std::int64_t> shape, const TensorOptions& options) {
    device(options.target_device());
    return owner(cuda::ops::from_blob(data, shape, options));
}
Tensor copy(const Tensor& value_tensor, Device target) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    device(target);
    return owner(cuda::ops::copy(value, target));
}
void copy_into(Tensor destination_tensor, const Tensor& source_tensor) {
    auto& destination = ::tyloi::ops::dense(destination_tensor);
    const auto& source = ::tyloi::ops::dense(source_tensor);
    input(destination);
    input(source);
    cuda::ops::copy_into(destination, source);
}
Tensor cast(const Tensor& value_tensor, ScalarType type) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(cuda::ops::cast(value, type));
}
Tensor cast(const Tensor& value_tensor, ScalarType type, Device target) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    device(target);
    return owner(cuda::ops::cast(value, type, target));
}
Tensor contiguous(const Tensor& value_tensor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(cuda::ops::contiguous(value));
}
Tensor clone(const Tensor& value_tensor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(cuda::ops::clone(value));
}
Tensor reshape(const Tensor& value_tensor, std::span<const std::int64_t> shape) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(tyloi::reshape(value, shape));
}
Tensor expand(const Tensor& value_tensor, std::span<const std::int64_t> shape) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(tyloi::expand(value, shape));
}
Tensor cat(std::span<const Tensor> values, std::int64_t dim) {
    if (values.empty()) throw std::invalid_argument("cat requires at least one tensor");
    std::vector<DenseTensor> dense_values;
    dense_values.reserve(values.size());
    for (const auto& value : values) {
        if (!value.defined()) throw std::invalid_argument("cat requires defined tensors");
        if (dense_values.empty()) input(value);
        else compatible(dense_values.front(), value);
        dense_values.push_back(dense(value));
    }
    return owner(cuda::ops::cat(dense_values, dim));
}
Tensor zeros(std::span<const std::int64_t> shape, const TensorOptions& options) {
    device(options.target_device());
    return owner(cuda::ops::zeros(shape, options));
}
Tensor empty(std::span<const std::int64_t> shape, const TensorOptions& options) {
    device(options.target_device());
    return owner(cuda::empty(shape, options));
}
Tensor sample_greedy(const Tensor& logits_tensor) {
    const auto& logits = dense(logits_tensor);
    input(logits);
    require_cuda(logits);
    return owner(::sample_greedy_cuda(logits));
}

std::vector<float> floats(const Tensor& tensor) {
    const auto& value = dense(tensor);
    input(value);
    return cuda::ops::floats(value);
}

std::vector<std::int64_t> integers(const Tensor& tensor) {
    const auto& value = dense(tensor);
    input(value);
    return cuda::ops::integers(value);
}

TopK top_k(const Tensor& tensor, std::int64_t k, bool force_radix) {
    const auto& logits = dense(tensor);
    input(logits);
    require_cuda(logits);
    if (logits.dim() != 2 || k < 1 || k > logits.size(1))
        throw std::invalid_argument("invalid top-k sampling geometry");
    return cuda::ops::top_k(logits, k, force_radix);
}

void decode_commit(const Tensor& next, Tensor& generated, Tensor& step,
                   Tensor& input_tensor, Tensor& position, Tensor& length) {
    input(next);
    require_cuda(next);
    const std::array<const Tensor*, 6> operands{&next, &generated, &step, &input_tensor, &position, &length};
    for (const auto* value : operands) compatible(next, *value);
    for (const auto* value : operands) {
        const auto& item = dense(*value);
        if (item.scalar_type() != kInt64 || !item.is_contiguous())
            throw std::invalid_argument("decode commit requires contiguous int64 tensors");
    }
    if (dense(next).numel() != 1 || dense(generated).numel() < 1 || dense(step).numel() != 1 ||
        dense(input_tensor).numel() != 1 || dense(position).numel() != 1 || dense(length).numel() != 1)
        throw std::invalid_argument("invalid decode commit geometry");
    cuda::ops::decode_commit(dense(next), dense(generated), dense(step), dense(input_tensor), dense(position), dense(length));
}
Tensor sample_stochastic(const Tensor& logits_tensor, const Tensor& random_tensor,
                                          double temperature, std::int64_t top_k, double top_p) {
    const auto& logits = dense(logits_tensor);
    const auto& random = dense(random_tensor);
    compatible(logits, random);
    require_cuda(logits);
    if (top_k > 0)
        return owner(::sample_top_k_top_p_cuda(logits, random, temperature, top_k, top_p));
    return owner(::sample_softmax_cuda(logits, random, temperature));
}
void add_token_counts(Tensor& counts_tensor, const Tensor& ids_tensor) {
    auto& counts = dense(counts_tensor);
    const auto& ids = dense(ids_tensor);
    compatible(counts, ids);
    require_cuda(counts);
    ::sample_token_counts_add_cuda(counts, ids);
}
void apply_penalties(Tensor& logits_tensor, const Tensor& counts_tensor,
                     double presence, double frequency, double repetition) {
    auto& logits = dense(logits_tensor);
    const auto& counts = dense(counts_tensor);
    compatible(logits, counts);
    require_cuda(logits);
    ::sample_apply_penalties_cuda(logits, counts, presence, frequency, repetition);
}
Tensor constrain(const Tensor& logits_tensor, const MfqTokenConstraintPtr& constraint) {
    const auto& logits = dense(logits_tensor);
    input(logits);
    auto masked = logits.to(kCPU, kFloat32).contiguous();
    constraint->apply(masked.data_ptr<float>(), static_cast<std::size_t>(masked.numel()));
    return owner(std::move(masked));
}
std::int64_t item(const Tensor& value_tensor) {
    const auto& value = dense(value_tensor);
    input(value);
    return value.item<std::int64_t>();
}
Tensor select(const Tensor& value_tensor, std::int64_t dim, std::int64_t index) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(tyloi::select(value, dim, index));
}
Tensor unsqueeze(const Tensor& value_tensor, std::int64_t dim) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(tyloi::unsqueeze(value, dim));
}
Tensor slice(const Tensor& value_tensor, std::int64_t dim, std::int64_t start, std::int64_t end, std::int64_t step) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(tyloi::slice(value, dim, start, end, step));
}
Tensor index_select(const Tensor& value_tensor, std::int64_t dim, const Tensor& indices_tensor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    const auto& indices = ::tyloi::ops::dense(indices_tensor);
    compatible(value, indices);
    require_cuda(value);
    return owner(cuda::ops::index_select(value, dim, indices));
}
Tensor sum(const Tensor& value_tensor, std::int64_t dim, bool keepdim) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::sum(value, dim, keepdim));
}
void index_copy(Tensor& value_tensor, std::int64_t dim, const Tensor& indices_tensor, const Tensor& source_tensor) {
    auto& value = ::tyloi::ops::dense(value_tensor);
    const auto& indices = ::tyloi::ops::dense(indices_tensor);
    const auto& source = ::tyloi::ops::dense(source_tensor);
    compatible(value, indices);
    compatible(value, source);
    require_cuda(value);
    cuda::ops::index_copy(value, dim, indices, source);
}
std::vector<Tensor> chunk(const Tensor& value_tensor, std::int64_t count, std::int64_t dim) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owners(tyloi::chunk(value, count, dim));
}
Tensor arange(std::int64_t first, std::int64_t last, Device target) {
    device(target);
    return owner(cuda::ops::arange(first, last, target));
}
Tensor full(std::span<const std::int64_t> shape, std::int64_t value, Device target) {
    device(target);
    return owner(cuda::ops::full(shape, value, target));
}
Tensor add(const Tensor& value_tensor, std::int64_t offset) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::add(value, offset));
}
Tensor mean(const Tensor& value_tensor, std::int64_t dim, bool keepdim) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::mean(value, dim, keepdim));
}
Tensor amax(const Tensor& value_tensor, std::int64_t dim, bool keepdim) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::amax(value, dim, keepdim));
}
Tensor argmax(const Tensor& value_tensor, std::int64_t dim, bool keepdim) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::argmax(value, dim, keepdim));
}
Tensor all(const Tensor& value_tensor, std::int64_t dim, bool keepdim) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::all(value, dim, keepdim));
}
Tensor any(const Tensor& value_tensor, std::int64_t dim, bool keepdim) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::any(value, dim, keepdim));
}
Tensor abs(const Tensor& value_tensor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::abs(value));
}
Tensor square(const Tensor& value_tensor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::square(value));
}
Tensor exp(const Tensor& value_tensor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::exp(value));
}
Tensor sqrt(const Tensor& value_tensor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::sqrt(value));
}
Tensor logical_not(const Tensor& value_tensor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::logical_not(value));
}
Tensor clamp(const Tensor& value_tensor, double minimum, double maximum) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::clamp(value, minimum, maximum));
}
Tensor clamp_min(const Tensor& value_tensor, double minimum) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::clamp_min(value, minimum));
}
Tensor clamp_max(const Tensor& value_tensor, double maximum) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::clamp_max(value, maximum));
}
Tensor remainder(const Tensor& value_tensor, double divisor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::remainder(value, divisor));
}
Tensor eq(const Tensor& left_tensor, const Tensor& right_tensor) {
    const auto& left = ::tyloi::ops::dense(left_tensor);
    const auto& right = ::tyloi::ops::dense(right_tensor);
    compatible(left, right);
    require_cuda(left);
    return owner(cuda::ops::eq(left, right));
}
Tensor eq(const Tensor& value_tensor, double other) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::eq(value, other));
}
Tensor ne(const Tensor& left_tensor, const Tensor& right_tensor) {
    const auto& left = ::tyloi::ops::dense(left_tensor);
    const auto& right = ::tyloi::ops::dense(right_tensor);
    compatible(left, right);
    require_cuda(left);
    return owner(cuda::ops::ne(left, right));
}
Tensor ne(const Tensor& value_tensor, double other) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::ne(value, other));
}
bool equal(const Tensor& left_tensor, const Tensor& right_tensor) {
    const auto& left = ::tyloi::ops::dense(left_tensor);
    const auto& right = ::tyloi::ops::dense(right_tensor);
    compatible(left, right);
    require_cuda(left);
    return cuda::ops::equal(left, right);
}
Tensor masked_select(const Tensor& value_tensor, const Tensor& mask_tensor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    const auto& mask = ::tyloi::ops::dense(mask_tensor);
    compatible(value, mask);
    require_cuda(value);
    return owner(cuda::ops::masked_select(value, mask));
}
Tensor gather(const Tensor& value_tensor, std::int64_t dim, const Tensor& indices_tensor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    const auto& indices = ::tyloi::ops::dense(indices_tensor);
    compatible(value, indices);
    require_cuda(value);
    return owner(cuda::ops::gather(value, dim, indices));
}
Tensor repeat(const Tensor& value_tensor, std::span<const std::int64_t> repeats) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::repeat(value, repeats));
}
Tensor repeat_interleave(const Tensor& value_tensor, std::int64_t repeats, std::int64_t dim) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::repeat_interleave(value, repeats, dim));
}
Tensor tril(const Tensor& value_tensor, std::int64_t diagonal) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::tril(value, diagonal));
}
Tensor masked_fill(const Tensor& value_tensor, const Tensor& mask_tensor, double fill_value) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    const auto& mask = ::tyloi::ops::dense(mask_tensor);
    compatible(value, mask);
    require_cuda(value);
    return owner(cuda::ops::masked_fill(value, mask, fill_value));
}
void masked_fill_inplace(Tensor& value_tensor, const Tensor& mask_tensor, double fill_value) {
    auto& value = ::tyloi::ops::dense(value_tensor);
    const auto& mask = ::tyloi::ops::dense(mask_tensor);
    compatible(value, mask);
    require_cuda(value);
    cuda::ops::masked_fill_inplace(value, mask, fill_value);
}
void fill(Tensor& value_tensor, double fill_value) {
    auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    if (value.is_cuda()) {
        cuda::ops::fill(value, fill_value);
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
void scatter_add(Tensor& value_tensor, std::int64_t dim, const Tensor& indices_tensor, const Tensor& source_tensor) {
    auto& value = ::tyloi::ops::dense(value_tensor);
    const auto& indices = ::tyloi::ops::dense(indices_tensor);
    const auto& source = ::tyloi::ops::dense(source_tensor);
    compatible(value, indices);
    compatible(value, source);
    require_cuda(value);
    cuda::ops::scatter_add(value, dim, indices, source);
}
void scatter(Tensor& value_tensor, std::int64_t dim, const Tensor& indices_tensor, const Tensor& source_tensor) {
    auto& value = ::tyloi::ops::dense(value_tensor);
    const auto& indices = ::tyloi::ops::dense(indices_tensor);
    const auto& source = ::tyloi::ops::dense(source_tensor);
    compatible(value, indices);
    compatible(value, source);
    require_cuda(value);
    cuda::ops::scatter(value, dim, indices, source);
}
void index_fill(Tensor& value_tensor, std::int64_t dim, const Tensor& indices_tensor, double fill_value) {
    auto& value = ::tyloi::ops::dense(value_tensor);
    const auto& indices = ::tyloi::ops::dense(indices_tensor);
    compatible(value, indices);
    require_cuda(value);
    cuda::ops::index_fill(value, dim, indices, fill_value);
}
#endif
} // namespace tyloi::ops
