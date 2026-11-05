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
#include "engine/backends/cuda/kernels/mfq_cuda_sampling_ops.h"
#endif

namespace tyloi::ops {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
namespace {
const DenseTensor& dense(const Tensor& value) {
    const auto* result = dynamic_cast<const DenseTensor*>(&value);
    if (!result || result->is_encoded()) throw std::invalid_argument("operation requires a dense tensor");
    return *result;
}
DenseTensor& dense(Tensor& value) {
    auto* result = dynamic_cast<DenseTensor*>(&value);
    if (!result || result->is_encoded()) throw std::invalid_argument("operation requires a dense tensor");
    return *result;
}
std::shared_ptr<Tensor> owner(DenseTensor value) {
    return std::make_shared<DenseTensor>(std::move(value));
}
std::vector<std::shared_ptr<Tensor>> owners(std::vector<DenseTensor> values) {
    std::vector<std::shared_ptr<Tensor>> result;
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
void require_cuda(const Tensor& value_tensor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    if (!value.device().is_cuda()) throw std::invalid_argument("operation requires a CUDA tensor");
}
bool encoded(std::initializer_list<const Tensor*> values) {
    for (const auto* value : values)
        if (value && (!dynamic_cast<const DenseTensor*>(value) || value->is_encoded())) return true;
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
        dense(request.key).sizes() != dense(request.value).sizes() ||
        request.query.size(0) != request.key.size(0) ||
        request.query.size(2) != request.key.size(2) ||
        request.query.size(3) != request.key.size(3) || request.key.size(1) <= 0 ||
        request.query.size(1) <= 0 || request.query.size(3) <= 0)
        throw std::invalid_argument("attention query, key and value geometry is incompatible");
    for (const Tensor* tensor : {kv.visible_key.get(), kv.visible_value.get(), kv.physical_key.get(),
                                  kv.physical_value.get(), kv.key_chunk_ptrs.get(),
                                  kv.value_chunk_ptrs.get(), kv.page_table.get()})
        if (tensor && tensor->defined()) compatible(request.query, *tensor);
    require_cuda(request.query);
    return cuda::ops::attention(request, kv, config);
}

std::shared_ptr<Tensor> attention(const Tensor& query_tensor, const Tensor& key_tensor, const Tensor& value_tensor,
                 const Tensor* mask, double scale) {
    const auto& query = dense(query_tensor);
    const auto& key = dense(key_tensor);
    const auto& value = dense(value_tensor);
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
    const std::optional<DenseTensor> dense_mask = mask ? std::optional<DenseTensor>(dense(*mask)) : std::nullopt;
    return owner(cuda::ops::attention(query, key, value, dense_mask, scale));
}

std::shared_ptr<Tensor> rope(const Tensor& value_tensor, const Tensor& positions_tensor, const RopeTable& table, bool bf16, bool grid) {
    const auto& value = dense(value_tensor);
    const auto& positions = dense(positions_tensor);
    if (!table.cos || !table.sin) throw std::invalid_argument("RoPE table requires cosine and sine");
    compatible(value, *table.cos);
    compatible(value, *table.sin);
    for (const auto* item : {table.sections.get(), table.empty_sections.get(),
                             table.interleaved_order.get(), table.interleaved_inverse.get()})
        if (item) (void)dense(*item);
    input(positions);
    if (value.dim() == 0 || table.rotary_dim <= 0 || table.rotary_dim % 2 != 0 ||
        table.rotary_dim > value.size(-1))
        throw std::invalid_argument("invalid RoPE rotary dimension");
    require_cuda(value);
    return owner(cuda::ops::rope(value, positions, table, bf16, grid));
}
std::shared_ptr<Tensor> norm_rope_kv(const AttentionInput& request, const state::KvView& kv,
    const RopeTable& table, const Tensor& q_norm, const Tensor& k_norm, double eps, double offset) {
    if (!request.query.is_cuda()) return nullptr;
    if (encoded({&request.query, &request.key, &request.value, &request.positions, &request.write_positions,
                 table.cos.get(), table.sin.get(), table.sections.get(), table.empty_sections.get(), table.interleaved_order.get(),
                 table.interleaved_inverse.get(), &q_norm, &k_norm, kv.visible_key.get(), kv.visible_value.get(),
                 kv.physical_key.get(), kv.physical_value.get(), kv.key_chunk_ptrs.get(), kv.value_chunk_ptrs.get(), kv.page_table.get(),
                 request.sequence_lengths, request.attention_mask})) return nullptr;
    auto result = cuda::ops::norm_rope_kv(request, kv, table, dense(q_norm), dense(k_norm), eps, offset);
    return result ? owner(std::move(*result)) : nullptr;
}
std::shared_ptr<Tensor> rope_kv(const AttentionInput& request, const state::KvView& kv,
                               const RopeTable& table) {
    if (!request.query.is_cuda()) return nullptr;
    if (encoded({&request.query, &request.key, &request.value, &request.positions, &request.write_positions,
                 table.cos.get(), table.sin.get(), table.sections.get(), table.empty_sections.get(), table.interleaved_order.get(),
                 table.interleaved_inverse.get(), kv.visible_key.get(), kv.visible_value.get(), kv.physical_key.get(),
                 kv.physical_value.get(), kv.key_chunk_ptrs.get(), kv.value_chunk_ptrs.get(), kv.page_table.get(),
                 request.sequence_lengths, request.attention_mask})) return nullptr;
    auto result = cuda::ops::rope_kv(request, kv, table);
    return result ? owner(std::move(*result)) : nullptr;
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> append_kv(state::KvView kv, const Tensor& key_tensor, const Tensor& value_tensor,
                                    const Tensor& positions_tensor, std::int64_t start,
                                    bool contiguous_prefill_prefix) {
    const auto& key = dense(key_tensor);
    const auto& value = dense(value_tensor);
    const auto& positions = dense(positions_tensor);
    compatible(key, value);
    compatible(key, positions);
    if (key.dim() != 4 || value.dim() != 4 || key.sizes() != value.sizes())
        throw std::invalid_argument("KV write key and value geometry is incompatible");
    for (const Tensor* tensor : {kv.physical_key.get(), kv.physical_value.get(), kv.key_chunk_ptrs.get(),
                                  kv.value_chunk_ptrs.get(), kv.page_table.get()})
        if (tensor && tensor->defined()) compatible(key, *tensor);
    // A zero batch skips the unpaged, unquantized index writes.
    if (kv.paged() || kv.quantization.enabled() || positions.dim() != 2 || positions.size(0) != 0)
        require_cuda(key);
    auto result = cuda::ops::append_kv(std::move(kv), key, value, positions, start, contiguous_prefill_prefix);
    return {owner(std::move(result.first)), owner(std::move(result.second))};
}

std::shared_ptr<Tensor> add(const Tensor& left_tensor, const Tensor& right_tensor) {
    const auto& left = ::tyloi::ops::dense(left_tensor);
    const auto& right = ::tyloi::ops::dense(right_tensor);
    compatible(left, right);
    require_cuda(left);
    return owner(cuda::ops::add(left, right));
}
std::shared_ptr<Tensor> multiply(const Tensor& left_tensor, const Tensor& right_tensor) {
    const auto& left = ::tyloi::ops::dense(left_tensor);
    const auto& right = ::tyloi::ops::dense(right_tensor);
    compatible(left, right);
    require_cuda(left);
    return owner(cuda::ops::multiply(left, right));
}
std::shared_ptr<Tensor> matmul(const Tensor& left_tensor, const Tensor& right_tensor) {
    const auto& left = ::tyloi::ops::dense(left_tensor);
    const auto& right = ::tyloi::ops::dense(right_tensor);
    compatible(left, right);
    require_cuda(left);
    return owner(cuda::ops::matmul(left, right));
}
std::shared_ptr<Tensor> transpose(const Tensor& value_tensor, std::int64_t first, std::int64_t second) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(tyloi::transpose(value, first, second));
}
std::shared_ptr<Tensor> add_residual(const Tensor& residual_tensor, const Tensor& value_tensor) {
    const auto& residual = ::tyloi::ops::dense(residual_tensor);
    const auto& value = ::tyloi::ops::dense(value_tensor);
    compatible(residual, value);
    require_cuda(residual);
    return owner(cuda::ops::add_residual(residual, value));
}
std::optional<ResidualNorms> residual_norms(const Tensor& residual_tensor, const Tensor& attention_tensor, const Tensor& attention_weight_tensor,
                                            const Tensor& dense_weight_tensor, const Tensor& router_weight_tensor,
                                            const Tensor& expert_weight_tensor, double eps) {
    if (!residual_tensor.is_cuda() ||
        encoded({&residual_tensor, &attention_tensor, &attention_weight_tensor,
                 &dense_weight_tensor, &router_weight_tensor, &expert_weight_tensor})) return std::nullopt;
    const auto& residual = ::tyloi::ops::dense(residual_tensor);
    const auto& attention = ::tyloi::ops::dense(attention_tensor);
    const auto& attention_weight = ::tyloi::ops::dense(attention_weight_tensor);
    const auto& dense_weight = ::tyloi::ops::dense(dense_weight_tensor);
    const auto& router_weight = ::tyloi::ops::dense(router_weight_tensor);
    const auto& expert_weight = ::tyloi::ops::dense(expert_weight_tensor);
    return cuda::ops::residual_norms(residual, attention, attention_weight,
                                     dense_weight, router_weight, expert_weight, eps);
}
std::shared_ptr<Tensor> merge_norms(const Tensor& dense_tensor, const Tensor& experts_tensor, const Tensor& residual_tensor, const Tensor& dense_weight_tensor,
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
SelectedExperts topk(const Tensor& logits_tensor, std::int64_t count, const MoeRouting& routing) {
    const auto& logits = ::tyloi::ops::dense(logits_tensor);
    input(logits);
    require_cuda(logits);
    return cuda::ops::topk(logits, count, routing);
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
std::shared_ptr<Tensor> weighted_reduce(const Tensor& pairs_tensor, const Tensor& weights_tensor) {
    const auto& pairs = ::tyloi::ops::dense(pairs_tensor);
    const auto& weights = ::tyloi::ops::dense(weights_tensor);
    compatible(pairs, weights);
    require_cuda(pairs);
    return owner(cuda::ops::weighted_reduce(pairs, weights));
}
RoutedGeglu routed_geglu(const Tensor& value_tensor, const Tensor& ids_tensor,
                         const tyloi::cuda::Experts& gate_up, const tyloi::cuda::Experts& down,
                         bool trace) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    const auto& ids = ::tyloi::ops::dense(ids_tensor);
    compatible(value, ids);
    if (gate_up.size() == 0 || down.size() == 0 || value.dim() != 2 || ids.dim() != 2 || value.size(0) != ids.size(0) ||
        gate_up.size() != down.size() || gate_up.shape().cols != value.size(-1) ||
        gate_up.shape().rows % 2 != 0 || gate_up.shape().rows / 2 != down.shape().cols)
        throw std::invalid_argument("routed GEGLU input, routes, and expert banks are incompatible");
    require_cuda(value);
    return cuda::ops::routed_geglu(value, ids, gate_up, down, trace);
}

std::shared_ptr<Tensor> rms_norm(const Tensor& value_tensor, const Tensor& weight_tensor, double eps, double offset, ScalarType output_type) {
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
std::shared_ptr<Tensor> layer_norm(const Tensor& value_tensor, const Tensor& weight_tensor, const Tensor& bias_tensor, double epsilon) {
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
std::shared_ptr<Tensor> gelu(const Tensor& value_tensor, bool approximate) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::gelu(value, approximate));
}
std::shared_ptr<Tensor> silu(const Tensor& value_tensor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::silu(value));
}
std::shared_ptr<Tensor> relu_squared(const Tensor& value_tensor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::relu_squared(value));
}
std::shared_ptr<Tensor> axial_rope(const Tensor& value_tensor, const std::vector<std::int32_t>& positions, double theta) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::axial_rope(value, positions, theta));
}
std::shared_ptr<Tensor> segmented_attention(const Tensor& query_tensor, const Tensor& key_tensor, const Tensor& value_tensor,
                           const std::vector<std::int32_t>& lengths) {
    const auto& query = ::tyloi::ops::dense(query_tensor);
    const auto& key = ::tyloi::ops::dense(key_tensor);
    const auto& value = ::tyloi::ops::dense(value_tensor);
    compatible(query, key);
    compatible(query, value);
    require_cuda(query);
    return owner(cuda::ops::segmented_attention(query, key, value, lengths));
}
std::array<std::shared_ptr<Tensor>, 3> qkv_norm(std::array<const Tensor*, 3> values, std::array<const Tensor*, 3> weights,
                               double eps, double offset, ScalarType output_type) {
    if (output_type != ScalarType::float32 && output_type != ScalarType::bfloat16)
        throw std::invalid_argument("unsupported qkv_norm output type");
    std::array<const DenseTensor*, 3> dense_weights{};
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!values[i]) throw std::invalid_argument("qkv_norm requires all values");
        input(*values[i]);
        if (weights[i]) dense_weights[i] = &dense(*weights[i]);
        if (weights[i] && weights[i]->defined()) {
            compatible(*values[i], *weights[i]);
            if (values[i]->dim() == 0 || weights[i]->dim() == 0 ||
                values[i]->size(-1) <= 0 || weights[i]->numel() != values[i]->size(-1))
                throw std::invalid_argument("qkv_norm input and weight are incompatible");
        }
    }
    for (std::size_t i = 0; i < values.size(); ++i)
        if (weights[i] && weights[i]->defined()) require_cuda(*values[i]);
    (void)cuda::Executor::current();
    std::array<DenseTensor, 3> dense_values{dense(*values[0]), dense(*values[1]), dense(*values[2])};
    auto result = cuda::ops::qkv_norm(std::move(dense_values), dense_weights, eps, offset, output_type);
    return {owner(std::move(result[0])), owner(std::move(result[1])), owner(std::move(result[2]))};
}
std::shared_ptr<Tensor> softcap(const Tensor& value_tensor, double cap) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    if (cap > 0.0) require_cuda(value);
    return owner(cuda::ops::softcap(value, cap));
}
std::shared_ptr<Tensor> scale(const Tensor& value_tensor, double factor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::scale(value, factor));
}
std::shared_ptr<Tensor> embedding(const Tensor& ids_tensor, const ::tyloi::Tensor& weight) {
    const auto& ids = ::tyloi::ops::dense(ids_tensor);
    input(ids);
    return owner(cuda::ops::embedding(ids, weight));
}
std::shared_ptr<Tensor> project(const Tensor& value_tensor, const ::tyloi::Tensor& weight) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(cuda::ops::project(value, weight));
}
std::shared_ptr<Tensor> linear(const Tensor& value_tensor, const ::tyloi::Tensor& weight, const Tensor* bias_tensor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    const auto* bias = bias_tensor ? &::tyloi::ops::dense(*bias_tensor) : nullptr;
    if (bias && bias->defined()) input(*bias);
    return owner(cuda::ops::linear(value, weight, bias));
}
std::shared_ptr<Tensor> affine(const Tensor& value_tensor, const ::tyloi::Tensor& weight, const Tensor* bias_tensor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    const auto* bias = bias_tensor ? &::tyloi::ops::dense(*bias_tensor) : nullptr;
    if (bias && bias->defined()) input(*bias);
    return owner(cuda::ops::affine(value, weight, bias));
}
std::shared_ptr<Tensor> mlp(const Tensor& value_tensor, const ::QuantLinearGroup& gate_up, const ::tyloi::Tensor& down,
           MlpActivation activation, double swiglu_limit) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(cuda::ops::mlp(value, gate_up, down, activation, swiglu_limit));
}
std::vector<std::shared_ptr<Tensor>> project_group(const Tensor& value_tensor, const ::QuantLinearGroup& group) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owners(cuda::ops::project_group(value, group));
}
std::shared_ptr<Tensor> sigmoid_project(const Tensor& value_tensor, const Tensor& gate_tensor, const ::tyloi::Tensor& weight) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    const auto& gate = ::tyloi::ops::dense(gate_tensor);
    compatible(value, gate);
    return owner(cuda::ops::sigmoid_project(value, gate, weight));
}
std::shared_ptr<Tensor> tensor(std::span<const float> values, Device target) {
    device(target);
    return owner(cuda::ops::tensor(values, target));
}
std::shared_ptr<Tensor> tensor(std::span<const std::int32_t> values, Device target) {
    device(target);
    return owner(cuda::ops::tensor(values, target));
}
std::shared_ptr<Tensor> tensor(std::span<const std::int64_t> values, Device target) {
    device(target);
    return owner(cuda::ops::tensor(values, target));
}
std::shared_ptr<Tensor> tensor(std::span<const std::uint8_t> values, Device target) {
    device(target);
    return owner(cuda::ops::tensor(values, target));
}
std::shared_ptr<Tensor> from_blob(void* data, std::span<const std::int64_t> shape, const TensorOptions& options) {
    device(options.target_device());
    return owner(cuda::ops::from_blob(data, shape, options));
}
std::shared_ptr<Tensor> copy(const Tensor& value_tensor, Device target) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    device(target);
    return owner(cuda::ops::copy(value, target));
}
void copy_into(Tensor& destination_tensor, const Tensor& source_tensor) {
    auto& destination = ::tyloi::ops::dense(destination_tensor);
    const auto& source = ::tyloi::ops::dense(source_tensor);
    input(destination);
    input(source);
    cuda::ops::copy_into(destination, source);
}
std::shared_ptr<Tensor> cast(const Tensor& value_tensor, ScalarType type) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(cuda::ops::cast(value, type));
}
std::shared_ptr<Tensor> cast(const Tensor& value_tensor, ScalarType type, Device target) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    device(target);
    return owner(cuda::ops::cast(value, type, target));
}
std::shared_ptr<Tensor> contiguous(const Tensor& value_tensor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(cuda::ops::contiguous(value));
}
std::shared_ptr<Tensor> clone(const Tensor& value_tensor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(cuda::ops::clone(value));
}
std::shared_ptr<Tensor> reshape(const Tensor& value_tensor, std::span<const std::int64_t> shape) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(tyloi::reshape(value, shape));
}
std::shared_ptr<Tensor> expand(const Tensor& value_tensor, std::span<const std::int64_t> shape) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(tyloi::expand(value, shape));
}
std::shared_ptr<Tensor> cat(std::span<const Tensor* const> values, std::int64_t dim) {
    if (values.empty()) throw std::invalid_argument("cat requires at least one tensor");
    std::vector<DenseTensor> dense_values;
    dense_values.reserve(values.size());
    for (const auto* value : values) {
        if (!value) throw std::invalid_argument("cat requires non-null tensors");
        if (dense_values.empty()) input(*value);
        else compatible(dense_values.front(), *value);
        dense_values.push_back(dense(*value));
    }
    return owner(cuda::ops::cat(dense_values, dim));
}
std::shared_ptr<Tensor> zeros(std::span<const std::int64_t> shape, const TensorOptions& options) {
    device(options.target_device());
    return owner(cuda::ops::zeros(shape, options));
}
std::shared_ptr<Tensor> empty(std::span<const std::int64_t> shape, const TensorOptions& options) {
    device(options.target_device());
    return owner(cuda::empty(shape, options));
}
std::shared_ptr<Tensor> sample_greedy(const Tensor& logits_tensor) {
    const auto& logits = dense(logits_tensor);
    input(logits);
    require_cuda(logits);
    return owner(::sample_greedy_cuda(logits));
}
std::shared_ptr<Tensor> sample_stochastic(const Tensor& logits_tensor, const Tensor& random_tensor,
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
std::shared_ptr<Tensor> constrain(const Tensor& logits_tensor, const MfqTokenConstraintPtr& constraint) {
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
std::shared_ptr<Tensor> select(const Tensor& value_tensor, std::int64_t dim, std::int64_t index) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(tyloi::select(value, dim, index));
}
std::shared_ptr<Tensor> unsqueeze(const Tensor& value_tensor, std::int64_t dim) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(tyloi::unsqueeze(value, dim));
}
std::shared_ptr<Tensor> slice(const Tensor& value_tensor, std::int64_t dim, std::int64_t start, std::int64_t end, std::int64_t step) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(tyloi::slice(value, dim, start, end, step));
}
std::shared_ptr<Tensor> index_select(const Tensor& value_tensor, std::int64_t dim, const Tensor& indices_tensor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    const auto& indices = ::tyloi::ops::dense(indices_tensor);
    compatible(value, indices);
    require_cuda(value);
    return owner(cuda::ops::index_select(value, dim, indices));
}
std::shared_ptr<Tensor> sum(const Tensor& value_tensor, std::int64_t dim, bool keepdim) {
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
std::vector<std::shared_ptr<Tensor>> chunk(const Tensor& value_tensor, std::int64_t count, std::int64_t dim) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owners(tyloi::chunk(value, count, dim));
}
std::shared_ptr<Tensor> arange(std::int64_t first, std::int64_t last, Device target) {
    device(target);
    return owner(cuda::ops::arange(first, last, target));
}
std::shared_ptr<Tensor> full(std::span<const std::int64_t> shape, std::int64_t value, Device target) {
    device(target);
    return owner(cuda::ops::full(shape, value, target));
}
std::shared_ptr<Tensor> add(const Tensor& value_tensor, std::int64_t offset) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::add(value, offset));
}
std::shared_ptr<Tensor> mean(const Tensor& value_tensor, std::int64_t dim, bool keepdim) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::mean(value, dim, keepdim));
}
std::shared_ptr<Tensor> amax(const Tensor& value_tensor, std::int64_t dim, bool keepdim) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::amax(value, dim, keepdim));
}
std::shared_ptr<Tensor> argmax(const Tensor& value_tensor, std::int64_t dim, bool keepdim) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::argmax(value, dim, keepdim));
}
std::shared_ptr<Tensor> all(const Tensor& value_tensor, std::int64_t dim, bool keepdim) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::all(value, dim, keepdim));
}
std::shared_ptr<Tensor> any(const Tensor& value_tensor, std::int64_t dim, bool keepdim) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::any(value, dim, keepdim));
}
std::shared_ptr<Tensor> abs(const Tensor& value_tensor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::abs(value));
}
std::shared_ptr<Tensor> square(const Tensor& value_tensor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::square(value));
}
std::shared_ptr<Tensor> exp(const Tensor& value_tensor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::exp(value));
}
std::shared_ptr<Tensor> sqrt(const Tensor& value_tensor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::sqrt(value));
}
std::shared_ptr<Tensor> logical_not(const Tensor& value_tensor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::logical_not(value));
}
std::shared_ptr<Tensor> clamp(const Tensor& value_tensor, double minimum, double maximum) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::clamp(value, minimum, maximum));
}
std::shared_ptr<Tensor> clamp_min(const Tensor& value_tensor, double minimum) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::clamp_min(value, minimum));
}
std::shared_ptr<Tensor> clamp_max(const Tensor& value_tensor, double maximum) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::clamp_max(value, maximum));
}
std::shared_ptr<Tensor> remainder(const Tensor& value_tensor, double divisor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::remainder(value, divisor));
}
std::shared_ptr<Tensor> eq(const Tensor& left_tensor, const Tensor& right_tensor) {
    const auto& left = ::tyloi::ops::dense(left_tensor);
    const auto& right = ::tyloi::ops::dense(right_tensor);
    compatible(left, right);
    require_cuda(left);
    return owner(cuda::ops::eq(left, right));
}
std::shared_ptr<Tensor> eq(const Tensor& value_tensor, double other) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::eq(value, other));
}
std::shared_ptr<Tensor> ne(const Tensor& left_tensor, const Tensor& right_tensor) {
    const auto& left = ::tyloi::ops::dense(left_tensor);
    const auto& right = ::tyloi::ops::dense(right_tensor);
    compatible(left, right);
    require_cuda(left);
    return owner(cuda::ops::ne(left, right));
}
std::shared_ptr<Tensor> ne(const Tensor& value_tensor, double other) {
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
std::shared_ptr<Tensor> masked_select(const Tensor& value_tensor, const Tensor& mask_tensor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    const auto& mask = ::tyloi::ops::dense(mask_tensor);
    compatible(value, mask);
    require_cuda(value);
    return owner(cuda::ops::masked_select(value, mask));
}
std::shared_ptr<Tensor> gather(const Tensor& value_tensor, std::int64_t dim, const Tensor& indices_tensor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    const auto& indices = ::tyloi::ops::dense(indices_tensor);
    compatible(value, indices);
    require_cuda(value);
    return owner(cuda::ops::gather(value, dim, indices));
}
std::shared_ptr<Tensor> repeat(const Tensor& value_tensor, std::span<const std::int64_t> repeats) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::repeat(value, repeats));
}
std::shared_ptr<Tensor> repeat_interleave(const Tensor& value_tensor, std::int64_t repeats, std::int64_t dim) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::repeat_interleave(value, repeats, dim));
}
std::shared_ptr<Tensor> tril(const Tensor& value_tensor, std::int64_t diagonal) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::tril(value, diagonal));
}
std::shared_ptr<Tensor> masked_fill(const Tensor& value_tensor, const Tensor& mask_tensor, double fill_value) {
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
