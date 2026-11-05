#pragma once

#include "engine/tensor.h"
#include "engine/state/kv_view.h"
#include "engine/weights/weights.h"

#include <cstdint>
#include <array>
#include <initializer_list>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// An ordered set of prepared projection weights.
struct QuantLinearGroup;
namespace tyloi::cuda { struct Experts; }

namespace tyloi::ops {

// In the native CUDA runtime, computation requires CUDA unless stated below.
// CUDA execution and prepared-weight operations require an execution scope.
// qkv_norm and append_kv also require a scope for their CPU passthrough cases.
using Trace = std::vector<std::pair<std::string, DenseTensor>>;

// Query, key and value are [B,H,T,D]; KV may use fewer heads.
struct AttentionInput {
    DenseTensor query, key, value, positions, write_positions;
    std::optional<DenseTensor> sequence_lengths, attention_mask;
    std::int64_t cache_position = 0, planned_kv_length = 0, decode_attention_parts = 0;
};
struct AttentionConfig {
    bool bf16 = false, sliding = false;
    std::int64_t window = 0;
    double scale = 0.0;
};
// token_major selects [B,T,H,D]; otherwise value is [B,H,T,D].
struct AttentionResult { DenseTensor value; bool token_major = false; };
AttentionResult attention(const AttentionInput& input, const state::KvView& kv,
                          const AttentionConfig& config);
DenseTensor attention(DenseTensor query, DenseTensor key, DenseTensor value,
                 const std::optional<DenseTensor>& additive_mask, double scale);
// Writes the supplied backing and returns visible KV; paged prefixes may be gathered.
// An unpaged, unquantized zero-batch write can use CPU tensors.
std::pair<DenseTensor, DenseTensor> append_kv(state::KvView kv, DenseTensor key, DenseTensor value,
                                    DenseTensor positions, std::int64_t start,
                                    bool contiguous_prefill_prefix = false);

struct RopeTable {
    DenseTensor cos, sin, sections, empty_sections, interleaved_order, interleaved_inverse;
    std::int64_t rotary_dim = 0;
};

DenseTensor rope(DenseTensor input, DenseTensor positions, const RopeTable& table, bool bf16 = false,
            bool grid_mrope_positions = false);
// Unsupported fusions return nullopt without writing; success writes the original KV and returns query.
// Consumed data must share query's device; input.positions is converted to that device.
std::optional<DenseTensor> norm_rope_kv(const AttentionInput&, const state::KvView&, const RopeTable&,
                                   const DenseTensor& q_norm, const DenseTensor& k_norm, double eps, double offset);
std::optional<DenseTensor> rope_kv(const AttentionInput&, const state::KvView&, const RopeTable&);

// Normalize over the last dimension, preserving shape; F32 output is the default.
DenseTensor rms_norm(DenseTensor input, const DenseTensor& weight, double eps, double offset,
                ScalarType output_type = ScalarType::float32);
DenseTensor layer_norm(DenseTensor input, const DenseTensor& weight, const DenseTensor& bias, double epsilon);
DenseTensor gelu(DenseTensor input, bool approximate = false);
DenseTensor silu(DenseTensor input);
DenseTensor relu_squared(DenseTensor input);
DenseTensor axial_rope(DenseTensor input, const std::vector<std::int32_t>& positions, double theta);
DenseTensor segmented_attention(DenseTensor query, DenseTensor key, DenseTensor value,
                           const std::vector<std::int32_t>& lengths);
// Undefined weights pass their value through, including CPU values; other outputs use F32 or BF16.
std::array<DenseTensor, 3> qkv_norm(std::array<DenseTensor, 3> values, std::array<DenseTensor, 3> weights,
                               double eps, double offset, ScalarType output_type);
DenseTensor add(DenseTensor left, DenseTensor right);
DenseTensor multiply(DenseTensor left, DenseTensor right);
DenseTensor matmul(DenseTensor left, DenseTensor right);
DenseTensor transpose(DenseTensor input, std::int64_t first, std::int64_t second);
DenseTensor add_residual(DenseTensor residual, DenseTensor value);

struct ResidualNorms {
    DenseTensor residual, dense, router, experts;
};
// A declined fusion returns nullopt without computing or writing.
// Inputs are flattened to two dimensions, preserving the last; results have the flattened shape.
std::optional<ResidualNorms> residual_norms(DenseTensor residual, DenseTensor attention, const DenseTensor& attention_weight,
                                            const DenseTensor& dense_weight, const DenseTensor& router_weight,
                                            const DenseTensor& expert_weight, double eps);
DenseTensor merge_norms(DenseTensor dense, DenseTensor experts, DenseTensor residual, const DenseTensor& dense_weight,
                   const DenseTensor& expert_weight, const DenseTensor& output_weight,
                   const DenseTensor& scale, double eps);

// A CUDA target selects the measurement device; by default the current execution scope is used.
class ProfileRange {
public:
    explicit ProfileRange(const char* name, std::optional<Device> device = {});
    ~ProfileRange();
    void finish();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
template <class Fn>
auto measure(const char* name, std::optional<Device> device, Fn&& operation) -> decltype(operation()) {
    ProfileRange range(name, device);
    auto result = operation();
    range.finish();
    return result;
}
template <class Fn>
auto measure(const char* name, Fn&& operation) -> decltype(operation()) {
    return measure(name, std::nullopt, std::forward<Fn>(operation));
}
// Preserve dtype; a cap that is not positive returns the input handle.
DenseTensor softcap(DenseTensor input, double cap);
DenseTensor scale(DenseTensor input, double factor);
// Prepared weights determine supported host, CUDA and sharded projection/embedding paths.
DenseTensor embedding(DenseTensor ids, const ::tyloi::Tensor& weight);
DenseTensor project(DenseTensor input, const ::tyloi::Tensor& weight);
// linear uses dense weight dtype for dense projection and bias; otherwise it adds
// bias in the projection result dtype. It restores input dtype and makes contiguous.
// affine adds bias in the projection result dtype without restoring input dtype.
DenseTensor linear(DenseTensor input, const ::tyloi::Tensor& weight, const DenseTensor& bias);
DenseTensor affine(DenseTensor input, const ::tyloi::Tensor& weight, const DenseTensor& bias);
std::vector<DenseTensor> project_group(DenseTensor input, const ::QuantLinearGroup& group);
// Computes project(input * sigmoid(gate)).
DenseTensor sigmoid_project(DenseTensor input, DenseTensor gate, const ::tyloi::Tensor& weight);
// Copy, cast, views and allocation support CPU as well as CUDA, subject to dtype support.
// Numeric reductions, elementwise ops and in-place writers require CUDA unless stated.
DenseTensor mean(DenseTensor input, std::int64_t dim, bool keepdim = false);
DenseTensor amax(DenseTensor input, std::int64_t dim, bool keepdim = false);
DenseTensor argmax(DenseTensor input, std::int64_t dim, bool keepdim = false);
DenseTensor all(DenseTensor input, std::int64_t dim, bool keepdim = false);
DenseTensor any(DenseTensor input, std::int64_t dim, bool keepdim = false);
DenseTensor abs(DenseTensor input);
DenseTensor square(DenseTensor input);
DenseTensor exp(DenseTensor input);
DenseTensor sqrt(DenseTensor input);
DenseTensor logical_not(DenseTensor input);
DenseTensor clamp(DenseTensor input, double minimum, double maximum);
DenseTensor clamp_min(DenseTensor input, double minimum);
DenseTensor clamp_max(DenseTensor input, double maximum);
DenseTensor remainder(DenseTensor input, double divisor);
DenseTensor eq(DenseTensor left, DenseTensor right);
DenseTensor eq(DenseTensor input, double other);
DenseTensor ne(DenseTensor left, DenseTensor right);
DenseTensor ne(DenseTensor input, double other);
bool equal(const DenseTensor& left, const DenseTensor& right);
DenseTensor masked_select(DenseTensor input, DenseTensor mask);
DenseTensor gather(DenseTensor input, std::int64_t dim, DenseTensor indices);
DenseTensor repeat(DenseTensor input, std::span<const std::int64_t> repeats);
inline DenseTensor repeat(DenseTensor input, std::initializer_list<std::int64_t> repeats) {
    return repeat(std::move(input), std::span<const std::int64_t>(repeats.begin(), repeats.size()));
}
DenseTensor repeat_interleave(DenseTensor input, std::int64_t repeats, std::int64_t dim);
DenseTensor tril(DenseTensor input, std::int64_t diagonal = 0);
DenseTensor masked_fill(DenseTensor input, DenseTensor mask, double value);
// Fill supports CPU and CUDA; CUDA requires an execution scope.
void fill(DenseTensor input, double value);
void masked_fill_inplace(DenseTensor input, DenseTensor mask, double value);
void scatter_add(DenseTensor input, std::int64_t dim, DenseTensor indices, DenseTensor source);
void scatter(DenseTensor input, std::int64_t dim, DenseTensor indices, DenseTensor source);
void index_fill(DenseTensor input, std::int64_t dim, DenseTensor indices, double value);
// Copies typed values into owned storage on target.
DenseTensor tensor(std::span<const float> values, Device target);
DenseTensor tensor(std::span<const std::int32_t> values, Device target);
DenseTensor tensor(std::span<const std::int64_t> values, Device target);
DenseTensor tensor(std::span<const std::uint8_t> values, Device target);
// The caller keeps external data alive until its last consumer completes.
DenseTensor from_blob(void* data, std::span<const std::int64_t> shape, const TensorOptions& options);
inline DenseTensor from_blob(void* data, std::initializer_list<std::int64_t> shape, const TensorOptions& options) {
    return from_blob(data, std::span<const std::int64_t>(shape.begin(), shape.size()), options);
}
DenseTensor copy(DenseTensor input, Device target);
// Writes destination's existing backing; source and destination may use different devices.
void copy_into(DenseTensor destination, DenseTensor source);
DenseTensor cast(DenseTensor input, ScalarType type);
// Targeted cast uses native tensor conversion; CUDA copy uses execution collectives.
DenseTensor cast(DenseTensor input, ScalarType type, Device target);
DenseTensor contiguous(DenseTensor input);
DenseTensor clone(DenseTensor input);
DenseTensor reshape(DenseTensor input, std::span<const std::int64_t> shape);
inline DenseTensor reshape(DenseTensor input, std::initializer_list<std::int64_t> shape) {
    return ops::reshape(std::move(input), std::span<const std::int64_t>(shape.begin(), shape.size()));
}
DenseTensor expand(DenseTensor input, std::span<const std::int64_t> shape);
inline DenseTensor expand(DenseTensor input, std::initializer_list<std::int64_t> shape) {
    return ops::expand(std::move(input), std::span<const std::int64_t>(shape.begin(), shape.size()));
}
DenseTensor cat(std::span<const DenseTensor> inputs, std::int64_t dim);
inline DenseTensor cat(std::initializer_list<DenseTensor> inputs, std::int64_t dim) {
    return cat(std::span<const DenseTensor>(inputs.begin(), inputs.size()), dim);
}
DenseTensor zeros(std::span<const std::int64_t> shape, const TensorOptions& options);
inline DenseTensor zeros(std::initializer_list<std::int64_t> shape, const TensorOptions& options) {
    return zeros(std::span<const std::int64_t>(shape.begin(), shape.size()), options);
}
DenseTensor select(DenseTensor input, std::int64_t dim, std::int64_t index);
DenseTensor unsqueeze(DenseTensor input, std::int64_t dim);
DenseTensor slice(DenseTensor input, std::int64_t dim, std::int64_t start, std::int64_t end, std::int64_t step = 1);
DenseTensor index_select(DenseTensor input, std::int64_t dim, DenseTensor indices);
DenseTensor sum(DenseTensor input, std::int64_t dim, bool keepdim = false);
// Modifies the input's backing storage.
void index_copy(DenseTensor input, std::int64_t dim, DenseTensor indices, DenseTensor source);
std::vector<DenseTensor> chunk(DenseTensor input, std::int64_t count, std::int64_t dim);
// Position tensors are I64.
DenseTensor arange(std::int64_t first, std::int64_t last, Device device);
DenseTensor full(std::span<const std::int64_t> shape, std::int64_t value, Device device);
DenseTensor add(DenseTensor input, std::int64_t offset);

// MLP activation selected by the model config. relu_squared consumes a single
// Up projection, so it has no gate branch and no SwiGLU clamp.
enum class MlpActivation { silu_gated, gelu_gated, relu_squared };

// Prepared weights determine supported host and sharded MLP paths.
DenseTensor mlp(DenseTensor input, const ::QuantLinearGroup& gate_up, const ::tyloi::Tensor& down,
           MlpActivation activation, double swiglu_limit);

enum class RouterActivation { softmax, sigmoid, sqrt_softplus };
struct MoeRouting {
    RouterActivation activation = RouterActivation::softmax;
    bool normalize = false;
    bool delayed_softmax = false;
    double scale = 1.0;
};

struct SelectedExperts { DenseTensor ids, weights; };
SelectedExperts topk(DenseTensor logits, std::int64_t count, const MoeRouting& routing);
// Scale the selected weights in place and return the same storage.
DenseTensor expert_scale(DenseTensor weights, DenseTensor ids, const DenseTensor& scales);
DenseTensor weighted_reduce(DenseTensor pairs, DenseTensor weights);
// Traced intermediates are F32 snapshots taken before subsequent projections.
struct RoutedGeglu { DenseTensor pairs, gate_up, hidden; };
RoutedGeglu routed_geglu(DenseTensor input, DenseTensor ids,
                         const tyloi::cuda::Experts& gate_up, const tyloi::cuda::Experts& down,
                         bool trace);

struct YarnScaling {
    double factor=1, beta_fast=32, beta_slow=1;
    std::int64_t original_context=0;
    bool enabled() const { return factor>1; }
};

} // namespace tyloi::ops
