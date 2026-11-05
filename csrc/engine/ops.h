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
    const Tensor &query, &key, &value, &positions, &write_positions;
    const Tensor *sequence_lengths = nullptr, *attention_mask = nullptr;
    std::int64_t cache_position = 0, planned_kv_length = 0, decode_attention_parts = 0;
};
struct AttentionConfig {
    bool bf16 = false, sliding = false;
    std::int64_t window = 0;
    double scale = 0.0;
};
// token_major selects [B,T,H,D]; otherwise value is [B,H,T,D].
struct AttentionResult { std::shared_ptr<Tensor> value; bool token_major = false; };
AttentionResult attention(const AttentionInput& input, const state::KvView& kv,
                          const AttentionConfig& config);
std::shared_ptr<Tensor> attention(const Tensor& query, const Tensor& key, const Tensor& value,
                 const Tensor* additive_mask, double scale);
// Writes the supplied backing and returns visible KV; paged prefixes may be gathered.
// An unpaged, unquantized zero-batch write can use CPU tensors.
std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> append_kv(state::KvView kv, const Tensor& key, const Tensor& value,
                                    const Tensor& positions, std::int64_t start,
                                    bool contiguous_prefill_prefix = false);

struct RopeTable {
    std::shared_ptr<const Tensor> cos, sin, sections, empty_sections, interleaved_order, interleaved_inverse;
    std::int64_t rotary_dim = 0;
};

std::shared_ptr<Tensor> rope(const Tensor& input, const Tensor& positions, const RopeTable& table, bool bf16 = false,
            bool grid_mrope_positions = false);
// Unsupported fusions return nullptr without writing; success writes the original KV and returns query.
// Consumed data must share query's device; input.positions is converted to that device.
std::shared_ptr<Tensor> norm_rope_kv(const AttentionInput&, const state::KvView&, const RopeTable&,
                                   const Tensor& q_norm, const Tensor& k_norm, double eps, double offset);
std::shared_ptr<Tensor> rope_kv(const AttentionInput&, const state::KvView&, const RopeTable&);

// Normalize over the last dimension, preserving shape; F32 output is the default.
std::shared_ptr<Tensor> rms_norm(const Tensor& input, const Tensor& weight, double eps, double offset,
                ScalarType output_type = ScalarType::float32);
std::shared_ptr<Tensor> layer_norm(const Tensor& input, const Tensor& weight, const Tensor& bias, double epsilon);
std::shared_ptr<Tensor> gelu(const Tensor& input, bool approximate = false);
std::shared_ptr<Tensor> silu(const Tensor& input);
std::shared_ptr<Tensor> relu_squared(const Tensor& input);
std::shared_ptr<Tensor> axial_rope(const Tensor& input, const std::vector<std::int32_t>& positions, double theta);
std::shared_ptr<Tensor> segmented_attention(const Tensor& query, const Tensor& key, const Tensor& value,
                           const std::vector<std::int32_t>& lengths);
// Null or undefined weights pass their value through, including CPU values; other outputs use F32 or BF16.
std::array<std::shared_ptr<Tensor>, 3> qkv_norm(std::array<const Tensor*, 3> values, std::array<const Tensor*, 3> weights,
                               double eps, double offset, ScalarType output_type);
std::shared_ptr<Tensor> add(const Tensor& left, const Tensor& right);
std::shared_ptr<Tensor> multiply(const Tensor& left, const Tensor& right);
std::shared_ptr<Tensor> matmul(const Tensor& left, const Tensor& right);
std::shared_ptr<Tensor> transpose(const Tensor& input, std::int64_t first, std::int64_t second);
std::shared_ptr<Tensor> add_residual(const Tensor& residual, const Tensor& value);

struct ResidualNorms {
    std::shared_ptr<Tensor> residual, dense, router, experts;
};
// A declined fusion returns nullopt without computing or writing.
// Inputs are flattened to two dimensions, preserving the last; results have the flattened shape.
std::optional<ResidualNorms> residual_norms(const Tensor& residual, const Tensor& attention, const Tensor& attention_weight,
                                            const Tensor& dense_weight, const Tensor& router_weight,
                                            const Tensor& expert_weight, double eps);
std::shared_ptr<Tensor> merge_norms(const Tensor& dense, const Tensor& experts, const Tensor& residual, const Tensor& dense_weight,
                   const Tensor& expert_weight, const Tensor& output_weight,
                   const Tensor& scale, double eps);

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
std::shared_ptr<Tensor> softcap(const Tensor& input, double cap);
std::shared_ptr<Tensor> scale(const Tensor& input, double factor);
// Prepared weights determine supported host, CUDA and sharded projection/embedding paths.
std::shared_ptr<Tensor> embedding(const Tensor& ids, const ::tyloi::Tensor& weight);
std::shared_ptr<Tensor> project(const Tensor& input, const ::tyloi::Tensor& weight);
// linear uses dense weight dtype for dense projection and bias; otherwise it adds
// bias in the projection result dtype. It restores input dtype and makes contiguous.
// affine adds bias in the projection result dtype without restoring input dtype.
std::shared_ptr<Tensor> linear(const Tensor& input, const ::tyloi::Tensor& weight, const Tensor& bias);
std::shared_ptr<Tensor> affine(const Tensor& input, const ::tyloi::Tensor& weight, const Tensor& bias);
std::vector<std::shared_ptr<Tensor>> project_group(const Tensor& input, const ::QuantLinearGroup& group);
// Computes project(input * sigmoid(gate)).
std::shared_ptr<Tensor> sigmoid_project(const Tensor& input, const Tensor& gate, const ::tyloi::Tensor& weight);
// Copy, cast, views and allocation support CPU as well as CUDA, subject to dtype support.
// Numeric reductions, elementwise ops and in-place writers require CUDA unless stated.
std::shared_ptr<Tensor> mean(const Tensor& input, std::int64_t dim, bool keepdim = false);
std::shared_ptr<Tensor> amax(const Tensor& input, std::int64_t dim, bool keepdim = false);
std::shared_ptr<Tensor> argmax(const Tensor& input, std::int64_t dim, bool keepdim = false);
std::shared_ptr<Tensor> all(const Tensor& input, std::int64_t dim, bool keepdim = false);
std::shared_ptr<Tensor> any(const Tensor& input, std::int64_t dim, bool keepdim = false);
std::shared_ptr<Tensor> abs(const Tensor& input);
std::shared_ptr<Tensor> square(const Tensor& input);
std::shared_ptr<Tensor> exp(const Tensor& input);
std::shared_ptr<Tensor> sqrt(const Tensor& input);
std::shared_ptr<Tensor> logical_not(const Tensor& input);
std::shared_ptr<Tensor> clamp(const Tensor& input, double minimum, double maximum);
std::shared_ptr<Tensor> clamp_min(const Tensor& input, double minimum);
std::shared_ptr<Tensor> clamp_max(const Tensor& input, double maximum);
std::shared_ptr<Tensor> remainder(const Tensor& input, double divisor);
std::shared_ptr<Tensor> eq(const Tensor& left, const Tensor& right);
std::shared_ptr<Tensor> eq(const Tensor& input, double other);
std::shared_ptr<Tensor> ne(const Tensor& left, const Tensor& right);
std::shared_ptr<Tensor> ne(const Tensor& input, double other);
bool equal(const Tensor& left, const Tensor& right);
std::shared_ptr<Tensor> masked_select(const Tensor& input, const Tensor& mask);
std::shared_ptr<Tensor> gather(const Tensor& input, std::int64_t dim, const Tensor& indices);
std::shared_ptr<Tensor> repeat(const Tensor& input, std::span<const std::int64_t> repeats);
inline std::shared_ptr<Tensor> repeat(const Tensor& input, std::initializer_list<std::int64_t> repeats) {
    return repeat(input, std::span<const std::int64_t>(repeats.begin(), repeats.size()));
}
std::shared_ptr<Tensor> repeat_interleave(const Tensor& input, std::int64_t repeats, std::int64_t dim);
std::shared_ptr<Tensor> tril(const Tensor& input, std::int64_t diagonal = 0);
std::shared_ptr<Tensor> masked_fill(const Tensor& input, const Tensor& mask, double value);
// Fill supports CPU and CUDA; CUDA requires an execution scope.
void fill(Tensor& input, double value);
void masked_fill_inplace(Tensor& input, const Tensor& mask, double value);
void scatter_add(Tensor& input, std::int64_t dim, const Tensor& indices, const Tensor& source);
void scatter(Tensor& input, std::int64_t dim, const Tensor& indices, const Tensor& source);
void index_fill(Tensor& input, std::int64_t dim, const Tensor& indices, double value);
// Copies typed values into owned storage on target.
std::shared_ptr<Tensor> tensor(std::span<const float> values, Device target);
std::shared_ptr<Tensor> tensor(std::span<const std::int32_t> values, Device target);
std::shared_ptr<Tensor> tensor(std::span<const std::int64_t> values, Device target);
std::shared_ptr<Tensor> tensor(std::span<const std::uint8_t> values, Device target);
// The caller keeps external data alive until its last consumer completes.
std::shared_ptr<Tensor> from_blob(void* data, std::span<const std::int64_t> shape, const TensorOptions& options);
inline std::shared_ptr<Tensor> from_blob(void* data, std::initializer_list<std::int64_t> shape, const TensorOptions& options) {
    return from_blob(data, std::span<const std::int64_t>(shape.begin(), shape.size()), options);
}
std::shared_ptr<Tensor> copy(const Tensor& input, Device target);
// Writes destination's existing backing; source and destination may use different devices.
void copy_into(Tensor& destination, const Tensor& source);
std::shared_ptr<Tensor> cast(const Tensor& input, ScalarType type);
// Targeted cast uses native tensor conversion; CUDA copy uses execution collectives.
std::shared_ptr<Tensor> cast(const Tensor& input, ScalarType type, Device target);
std::shared_ptr<Tensor> contiguous(const Tensor& input);
std::shared_ptr<Tensor> clone(const Tensor& input);
std::shared_ptr<Tensor> reshape(const Tensor& input, std::span<const std::int64_t> shape);
inline std::shared_ptr<Tensor> reshape(const Tensor& input, std::initializer_list<std::int64_t> shape) {
    return ops::reshape(input, std::span<const std::int64_t>(shape.begin(), shape.size()));
}
std::shared_ptr<Tensor> expand(const Tensor& input, std::span<const std::int64_t> shape);
inline std::shared_ptr<Tensor> expand(const Tensor& input, std::initializer_list<std::int64_t> shape) {
    return ops::expand(input, std::span<const std::int64_t>(shape.begin(), shape.size()));
}
std::shared_ptr<Tensor> cat(std::span<const Tensor* const> inputs, std::int64_t dim);
inline std::shared_ptr<Tensor> cat(std::initializer_list<const Tensor*> inputs, std::int64_t dim) {
    return cat(std::span<const Tensor* const>(inputs.begin(), inputs.size()), dim);
}
std::shared_ptr<Tensor> zeros(std::span<const std::int64_t> shape, const TensorOptions& options);
inline std::shared_ptr<Tensor> zeros(std::initializer_list<std::int64_t> shape, const TensorOptions& options) {
    return zeros(std::span<const std::int64_t>(shape.begin(), shape.size()), options);
}
std::shared_ptr<Tensor> select(const Tensor& input, std::int64_t dim, std::int64_t index);
std::shared_ptr<Tensor> unsqueeze(const Tensor& input, std::int64_t dim);
std::shared_ptr<Tensor> slice(const Tensor& input, std::int64_t dim, std::int64_t start, std::int64_t end, std::int64_t step = 1);
std::shared_ptr<Tensor> index_select(const Tensor& input, std::int64_t dim, const Tensor& indices);
std::shared_ptr<Tensor> sum(const Tensor& input, std::int64_t dim, bool keepdim = false);
// Modifies the input's backing storage.
void index_copy(Tensor& input, std::int64_t dim, const Tensor& indices, const Tensor& source);
std::vector<std::shared_ptr<Tensor>> chunk(const Tensor& input, std::int64_t count, std::int64_t dim);
// Position tensors are I64.
std::shared_ptr<Tensor> arange(std::int64_t first, std::int64_t last, Device device);
std::shared_ptr<Tensor> full(std::span<const std::int64_t> shape, std::int64_t value, Device device);
std::shared_ptr<Tensor> add(const Tensor& input, std::int64_t offset);

// MLP activation selected by the model config. relu_squared consumes a single
// Up projection, so it has no gate branch and no SwiGLU clamp.
enum class MlpActivation { silu_gated, gelu_gated, relu_squared };

// Prepared weights determine supported host and sharded MLP paths.
std::shared_ptr<Tensor> mlp(const Tensor& input, const ::QuantLinearGroup& gate_up, const ::tyloi::Tensor& down,
           MlpActivation activation, double swiglu_limit);

enum class RouterActivation { softmax, sigmoid, sqrt_softplus };
struct MoeRouting {
    RouterActivation activation = RouterActivation::softmax;
    bool normalize = false;
    bool delayed_softmax = false;
    double scale = 1.0;
};

struct SelectedExperts { std::shared_ptr<Tensor> ids, weights; };
SelectedExperts topk(const Tensor& logits, std::int64_t count, const MoeRouting& routing);
// Scale the selected weights in place.
void expert_scale(Tensor& weights, const Tensor& ids, const Tensor& scales);
std::shared_ptr<Tensor> weighted_reduce(const Tensor& pairs, const Tensor& weights);
// Traced intermediates are F32 snapshots taken before subsequent projections.
struct RoutedGeglu { std::shared_ptr<Tensor> pairs, gate_up, hidden; };
RoutedGeglu routed_geglu(const Tensor& input, const Tensor& ids,
                         const tyloi::cuda::Experts& gate_up, const tyloi::cuda::Experts& down,
                         bool trace);

struct YarnScaling {
    double factor=1, beta_fast=32, beta_slow=1;
    std::int64_t original_context=0;
    bool enabled() const { return factor>1; }
};

} // namespace tyloi::ops
