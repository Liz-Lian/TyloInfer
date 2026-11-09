#pragma once

#include "engine/tensor.h"
#include "engine/requests/constraint.h"
#include "engine/state/kv_view.h"

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


namespace tyloi::ops {

// In the native CUDA runtime, computation requires CUDA unless stated below.
// CUDA execution and prepared-weight operations require an execution scope.
// qkv_norm and append_kv also require a scope for their CPU passthrough cases.
using Trace = std::vector<std::pair<std::string, std::shared_ptr<Tensor>>>;

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
std::int64_t attention_parts(std::int64_t planned_len, Device device);
Tensor attention(const Tensor& query, const Tensor& key, const Tensor& value,
                 const Tensor& additive_mask, double scale);
// Query [B,H,T,D], shared KV [B,S,D], indices and mask [B,T,K], sinks [H].
// The backend owns execution scratch.
Tensor sparse_attention(const Tensor& query, const Tensor& cache,
                        const Tensor& indices, const Tensor& mask,
                        const Tensor& sinks, double scale);
// Writes the supplied backing and returns visible KV; paged prefixes may be gathered.
// An unpaged, unquantized zero-batch write can use CPU tensors.
std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> append_kv(state::KvView kv, const Tensor& key, const Tensor& value,
                                    const Tensor& positions, std::int64_t start,
                                    bool contiguous_prefill_prefix = false);

struct RopeTable {
    std::shared_ptr<const Tensor> cos, sin, sections, empty_sections, interleaved_order, interleaved_inverse;
    std::int64_t rotary_dim = 0;
};

struct TailRopeTable {
    std::shared_ptr<const Tensor> cos, sin, negative_sin;
};

TailRopeTable tail_rope_table(std::int64_t max_positions, double base, std::int64_t compress_ratio,
    Device device, double compress_base = 0.0, std::int64_t original_positions = 0,
    double factor = 1.0, double beta_fast = 32.0, double beta_slow = 1.0);
std::shared_ptr<Tensor> tail_rope(const Tensor& input, const Tensor& positions,
    const TailRopeTable& table, bool inverse = false);

// CUDA table with [positions, dimension / 2] FP32 cosine and sine tensors.
// Nonpositive frequency_dim uses dimension; negative active_pairs uses all pairs.
RopeTable rope_table(std::int64_t positions, std::int64_t dimension, double base, Device device,
                     std::int64_t frequency_dim = 0, std::int64_t active_pairs = -1);
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
Tensor add(const Tensor& left, const Tensor& right);
Tensor multiply(const Tensor& left, const Tensor& right);
Tensor matmul(const Tensor& left, const Tensor& right);
Tensor transpose(const Tensor& input, std::int64_t first, std::int64_t second);
enum class ResidualMode { preserve, configured };
// configured applies the execution residual policy; preserve skips input conversion.
Tensor add_residual(const Tensor& residual, const Tensor& value,
                    ResidualMode mode = ResidualMode::preserve);

// Residual and value are [batch,tokens,width]. Both results preserve that shape.
// Recurrent normalization retains its existing precision and accumulation policy.
std::array<Tensor, 2> residual_norm(const Tensor& residual, const Tensor& value,
    const Tensor& weight, double eps, double offset, bool bf16);
std::array<Tensor, 2> recurrent_norm(const Tensor& residual, const Tensor& value,
    const Tensor& weight, double eps, double offset);
Tensor residual_add(const Tensor& residual, const Tensor& value);
Tensor recurrent_add(const Tensor& residual, const Tensor& value);

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
class ProfileCapture {
public:
    explicit ProfileCapture(bool selected);
    // Report on normal exit; unselected scopes still restore external profiling state.
    ProfileCapture(bool selected, const std::string& report_name);
    ~ProfileCapture();
    void report(const std::string& name);
    void resume();
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
// Scalar division preserves shape and dtype.
std::shared_ptr<Tensor> divide(const Tensor& input, double divisor);
// Prepared weights determine supported host, CUDA and sharded projection/embedding paths.
Tensor embedding(const Tensor& ids, const ::tyloi::Tensor& weight);
Tensor project(const Tensor& input, const ::tyloi::Tensor& weight);
// Select one weight bank per route; keep the route axis in the result.
Tensor project(const Tensor& input, const Tensor& weight, const Tensor& ids);
// Input starts with [batch,tokens]; each group has output_a.size(1) values.
// Two projections preserve those axes and return output_b.size(0) channels.
Tensor low_rank(const Tensor& input, const Tensor& output_a, const Tensor& output_b,
                std::int64_t groups, bool groupwise, bool profile);
// linear uses dense weight dtype for dense projection and bias; otherwise it adds
// bias in the projection result dtype. It restores input dtype and makes contiguous.
// affine adds bias in the projection result dtype without restoring input dtype.
std::shared_ptr<Tensor> linear(const Tensor& input, const ::tyloi::Tensor& weight, const Tensor* bias = nullptr);
std::shared_ptr<Tensor> affine(const Tensor& input, const ::tyloi::Tensor& weight, const Tensor* bias = nullptr);
std::vector<std::shared_ptr<Tensor>> project_group(const Tensor& input, std::span<const std::shared_ptr<const Tensor>> group);
// Capability depends on prepared weights and the current execution policy.
bool grouped(std::span<const Tensor> weights);
// Ordered projections; combine the first prefix outputs into one. The backend
// may group compatible weights while retaining separate unsupported projections.
std::vector<Tensor> project_group(const Tensor& input, std::span<const Tensor> weights,
                                  std::size_t prefix);
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
std::shared_ptr<Tensor> empty(std::span<const std::int64_t> shape, const TensorOptions& options);
inline std::shared_ptr<Tensor> empty(std::initializer_list<std::int64_t> shape, const TensorOptions& options) {
    return empty(std::span<const std::int64_t>(shape.begin(), shape.size()), options);
}
std::shared_ptr<Tensor> sample_greedy(const Tensor& logits);
void decode_commit(const Tensor& next, Tensor& generated, Tensor& step,
                   Tensor& input, Tensor& position, Tensor& length);
std::shared_ptr<Tensor> sample_stochastic(const Tensor& logits, const Tensor& random,
                                          double temperature, std::int64_t top_k, double top_p);
void add_token_counts(Tensor& counts, const Tensor& ids);
void apply_penalties(Tensor& logits, const Tensor& counts, double presence, double frequency, double repetition);
std::shared_ptr<Tensor> constrain(const Tensor& logits, const MfqTokenConstraintPtr& constraint);
std::int64_t item(const Tensor& value);
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
std::shared_ptr<Tensor> mlp(const Tensor& input, std::span<const std::shared_ptr<const Tensor>> gate_up, const ::tyloi::Tensor& down,
           MlpActivation activation, double swiglu_limit);
// BF16 MLP retains its source-precision projection and activation path.
Tensor bf16_mlp(const Tensor& input, std::span<const Tensor> gate_up, const Tensor& down,
                MlpActivation activation, double swiglu_limit);
// Add two prepared MLP branches using the configured parallel and accumulation policy.
Tensor additive_mlp(const Tensor& input,
    std::span<const Tensor> gate_up, const Tensor& down, MlpActivation activation, double limit,
    std::span<const Tensor> extra_gate_up, const Tensor& extra_down,
    MlpActivation extra_activation, double extra_limit);
// Flattened [rows,width] inputs; a declined fusion does not compute or write.
std::optional<Tensor> mlp_residual(const Tensor& input, const Tensor& residual,
    std::span<const Tensor> gate_up, const Tensor& down, MlpActivation activation,
    double swiglu_limit, bool recurrent = false);

enum class RouterActivation { softmax, sigmoid, sqrt_softplus };
struct MoeRouting {
    RouterActivation activation = RouterActivation::softmax;
    bool normalize = false;
    bool delayed_softmax = false;
    double scale = 1.0;
};

struct SelectedExperts { Tensor ids, weights; };
SelectedExperts topk(const Tensor& logits, std::int64_t count, const MoeRouting& routing,
                     const Tensor& bias = {});
// Scale the selected weights in place.
void expert_scale(Tensor& weights, const Tensor& ids, const Tensor& scales);
std::shared_ptr<Tensor> weighted_reduce(const Tensor& pairs, const Tensor& weights);
// SwiGLU routed experts, reduced in FP32 across the selected routes.
Tensor experts(const Tensor& input, const Tensor& ids, const Tensor& weights,
               std::span<const Tensor> gate_up, const Tensor& down, std::int64_t width);
// Full MoE value: the routed SwiGLU result plus a scalar-gated shared expert.
// `shared` contains Gate, Up and Down in that order. `shared_rows` preserves
// shared projection row calibration; nonpositive values use the full input.
Tensor moe(const Tensor& input, const Tensor& ids, const Tensor& weights,
           std::span<const Tensor> gate_up, const Tensor& down,
           std::span<const Tensor> shared, const Tensor& shared_gate,
           std::int64_t width, std::int64_t shared_rows = 0);
// Input is CUDA F16 [rows,in], ids are CUDA I32 [rows,routes], and banks are
// [experts,2*hidden,in] and [experts,out,hidden]. Pairs are [rows,routes,out].
// Traced intermediates are F32 snapshots taken before subsequent projections.
struct RoutedGeglu { std::shared_ptr<Tensor> pairs, gate_up, hidden; };
RoutedGeglu routed_geglu(const Tensor& input, const Tensor& ids,
                         const Tensor& gate_up, const Tensor& down,
                         bool trace);

struct YarnScaling {
    double factor=1, beta_fast=32, beta_slow=1;
    std::int64_t original_context=0;
    bool enabled() const { return factor>1; }
};

} // namespace tyloi::ops
