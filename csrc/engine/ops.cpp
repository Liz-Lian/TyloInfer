#include "engine/ops.h"

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/ops/norm/rms.h"
#include "engine/backends/cuda/ops/ffn/ffn.h"
#include "engine/backends/cuda/ops/projection/projection.h"
#include "engine/backends/cuda/ops/tensor.h"
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
}

Tensor add(Tensor left, Tensor right) {
    compatible(left, right);
    return cuda::ops::add(std::move(left), std::move(right));
}
Tensor multiply(Tensor left, Tensor right) {
    compatible(left, right);
    return cuda::ops::multiply(std::move(left), std::move(right));
}
Tensor matmul(Tensor left, Tensor right) {
    compatible(left, right);
    return cuda::ops::matmul(std::move(left), std::move(right));
}
Tensor transpose(Tensor value, std::int64_t first, std::int64_t second) {
    input(value);
    return cuda::ops::transpose(std::move(value), first, second);
}
Tensor add_residual(Tensor residual, Tensor value) {
    compatible(residual, value);
    return cuda::ops::add_residual(std::move(residual), std::move(value));
}
ResidualNorms residual_norms(Tensor residual, Tensor attention, const Tensor& attention_weight,
                             const Tensor& dense_weight, const Tensor& router_weight,
                             const Tensor& expert_weight, double eps) {
    for (const Tensor* value : std::initializer_list<const Tensor*>{
             &attention, &attention_weight, &dense_weight, &router_weight, &expert_weight})
        compatible(residual, *value);
    return cuda::ops::residual_norms(std::move(residual), std::move(attention), attention_weight,
                                     dense_weight, router_weight, expert_weight, eps);
}
Tensor merge_norms(Tensor dense, Tensor experts, Tensor residual, const Tensor& dense_weight,
                   const Tensor& expert_weight, const Tensor& output_weight,
                   const Tensor& scale, double eps) {
    for (const Tensor* value : std::initializer_list<const Tensor*>{
             &experts, &residual, &dense_weight, &expert_weight, &output_weight, &scale})
        compatible(dense, *value);
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
    return cuda::ops::topk(std::move(logits), count, routing);
}
Tensor expert_scale(Tensor weights, Tensor ids, const Tensor& scales) {
    compatible(weights, ids);
    compatible(weights, scales);
    return cuda::ops::expert_scale(std::move(weights), std::move(ids), scales);
}
Tensor weighted_reduce(Tensor pairs, Tensor weights) {
    compatible(pairs, weights);
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
    return cuda::ops::routed_geglu(std::move(value), std::move(ids), gate_up, down, options, trace);
}

Tensor rms_norm(Tensor value, const Tensor& weight, double eps, double offset, ScalarType output_type) {
    input(value);
    input(weight);
    if (value.device() != weight.device() || value.dim() == 0 || weight.dim() == 0 ||
        value.size(-1) <= 0 || weight.numel() != value.size(-1) ||
        weight.size(-1) != value.size(-1))
        throw std::invalid_argument("rms_norm input and weight are incompatible");
    if (output_type != ScalarType::float32 && output_type != ScalarType::float16)
        throw std::invalid_argument("unsupported rms_norm output type");
    return cuda::ops::rms_norm(std::move(value), weight, eps, offset, output_type);
}
Tensor softcap(Tensor value, double cap) {
    input(value);
    return cuda::ops::softcap(std::move(value), cap);
}
Tensor scale(Tensor value, double factor) {
    input(value);
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
Tensor copy(Tensor value, Device target) {
    input(value);
    device(target);
    return cuda::ops::copy(std::move(value), target);
}
Tensor cast(Tensor value, ScalarType type) {
    input(value);
    return cuda::ops::cast(std::move(value), type);
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
    return cuda::ops::add(std::move(value), offset);
}
#endif
} // namespace mfq::ops
