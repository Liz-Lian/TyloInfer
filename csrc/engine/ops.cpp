#include "engine/ops.h"

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/ops/norm/rms.h"
#include "engine/backends/cuda/ops/ffn/ffn.h"
#include "engine/backends/cuda/ops/projection/projection.h"
#include "engine/backends/cuda/ops/tensor.h"
#include "engine/backends/cuda/ops/attention/attention.h"
#include "engine/backends/cuda/ops/attention/kv.h"
#include "engine/backends/cuda/ops/rope/rope.h"
#include "engine/backends/cuda/native/profile.h"
#endif

namespace mfq::ops {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
namespace {
void device(Device value) {
    if (!value.is_cpu() && !value.is_cuda())
        throw std::invalid_argument("unsupported tensor device");
    if (value.is_cuda()) (void)cuda::Executor::current();
}
void input(const Tensor& value) {
    if (!value.defined()) throw std::invalid_argument("undefined tensor");
    device(value.device());
}
void compatible(const Tensor& left, const Tensor& right) {
    input(left);
    input(right);
    if (left.device() != right.device())
        throw std::invalid_argument("tensor devices do not match");
}
void require_cuda(const Tensor& value) {
    if (!value.device().is_cuda()) throw std::invalid_argument("operation requires a CUDA tensor");
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

Tensor rope(Tensor value, Tensor positions, const RopeTable& table, bool bf16, bool grid) {
    compatible(value, table.cos);
    compatible(value, table.sin);
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
    return cuda::ops::norm_rope_kv(request, kv, table, q_norm, k_norm, eps, offset);
}
std::optional<Tensor> rope_kv(const AttentionInput& request, const state::KvView& kv,
                               const RopeTable& table) {
    if (!request.query.is_cuda()) return std::nullopt;
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
ResidualNorms residual_norms(Tensor residual, Tensor attention, const Tensor& attention_weight,
                             const Tensor& dense_weight, const Tensor& router_weight,
                             const Tensor& expert_weight, double eps) {
    for (const Tensor* value : std::initializer_list<const Tensor*>{
             &attention, &attention_weight, &dense_weight, &router_weight, &expert_weight})
        compatible(residual, *value);
    require_cuda(residual);
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
    CudaProfiler::Range range;
    Impl(CudaProfiler& profiler, const char* name) : range(profiler, name) {}
};
ProfileRange::ProfileRange(const char* name) {
    auto& profiler = cuda::Executor::current().profiler();
    if (profiler.enabled && profiler.selected(name))
        impl_ = std::make_unique<Impl>(profiler, name);
}
ProfileRange::~ProfileRange() = default;
void ProfileRange::finish() { if (impl_) impl_->range.finish(); }
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
RoutedGeglu routed_geglu(Tensor value, Tensor ids, const weights::ExpertBank& gate_up,
                         const weights::ExpertBank& down, const ProjectionOptions& options,
                         bool trace) {
    compatible(value, ids);
    if (value.dim() != 2 || ids.dim() != 2 || value.size(0) != ids.size(0) ||
        gate_up.count() != down.count() || gate_up.cols() != value.size(-1) ||
        gate_up.rows() % 2 != 0 || gate_up.rows() / 2 != down.cols())
        throw std::invalid_argument("routed GEGLU input, routes, and expert banks are incompatible");
    require_cuda(value);
    return cuda::ops::routed_geglu(std::move(value), std::move(ids), gate_up, down, options, trace);
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
Tensor embedding(Tensor ids, const weights::Matrix& weight) {
    input(ids);
    return cuda::ops::embedding(std::move(ids), weight);
}
Tensor project(Tensor value, const weights::Matrix& weight, const ProjectionOptions& options) {
    input(value);
    return cuda::ops::project(std::move(value), weight, options);
}
Tensor mlp(Tensor value, const weights::MatrixGroup& gate_up, const weights::Matrix& down,
           MlpActivation activation, double swiglu_limit, const FfnMode& mode) {
    input(value);
    return cuda::ops::mlp(std::move(value), gate_up, down, activation, swiglu_limit, mode);
}
std::vector<Tensor> project(Tensor value, const weights::MatrixGroup& group,
                            const ProjectionOptions& options) {
    input(value);
    return cuda::ops::project(std::move(value), group, options);
}
Tensor sigmoid_project(Tensor value, Tensor gate, const weights::Matrix& weight,
                       const ProjectionOptions& options) {
    compatible(value, gate);
    return cuda::ops::sigmoid_project(std::move(value), std::move(gate), weight, options);
}
Tensor copy(Tensor value, Device target) {
    input(value);
    device(target);
    return cuda::ops::copy(std::move(value), target);
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
Tensor slice(Tensor value, std::int64_t dim, std::int64_t start, std::int64_t end, std::int64_t step) {
    input(value);
    return cuda::ops::slice(std::move(value), dim, start, end, step);
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
} // namespace mfq::ops
