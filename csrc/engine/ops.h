#pragma once

#include "engine/tensor.h"
#include "engine/state/kv_view.h"
#include "engine/weights/matrix.h"
#include "engine/weights/moe/bank.h"

#include <cstdint>
#include <array>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfq::ops {

// In the native CUDA runtime, computation requires CUDA unless stated below.
// CUDA execution and prepared-weight operations require an execution scope.
// qkv_norm and append_kv also require a scope for their CPU passthrough cases.
using Trace = std::vector<std::pair<std::string, Tensor>>;

// Query, key and value are [B,H,T,D]; KV may use fewer heads.
struct AttentionInput {
    Tensor query, key, value, positions, write_positions;
    std::optional<Tensor> sequence_lengths, attention_mask;
    std::int64_t cache_position = 0, planned_kv_length = 0, decode_attention_parts = 0;
};
struct AttentionConfig {
    bool bf16 = false, sliding = false;
    std::int64_t window = 0;
    double scale = 0.0;
};
// token_major selects [B,T,H,D]; otherwise value is [B,H,T,D].
struct AttentionResult { Tensor value; bool token_major = false; };
AttentionResult attention(const AttentionInput& input, const state::KvView& kv,
                          const AttentionConfig& config);
// Writes the supplied backing and returns visible KV; paged prefixes may be gathered.
// An unpaged, unquantized zero-batch write can use CPU tensors.
std::pair<Tensor, Tensor> append_kv(state::KvView kv, Tensor key, Tensor value,
                                    Tensor positions, std::int64_t start,
                                    bool contiguous_prefill_prefix = false);

struct FfnMode {
    bool materialized = false;
    bool unfused = false;
    bool serial = false;
    bool batch_serial = false;
    int tile = 0;
    int tiles = 1;
    bool reuse_input = true;
    bool projection_major = false;
};

struct ProjectionOptions {
    bool serial = false;
    bool projection_major = false;
};

struct RopeTable {
    Tensor cos, sin, sections, empty_sections, interleaved_order, interleaved_inverse;
    std::int64_t rotary_dim = 0;
};

Tensor rope(Tensor input, Tensor positions, const RopeTable& table, bool bf16 = false,
            bool grid_mrope_positions = false);
// Unsupported fusions return nullopt without writing; success writes the original KV and returns query.
// Consumed data must share query's device; input.positions is converted to that device.
std::optional<Tensor> norm_rope_kv(const AttentionInput&, const state::KvView&, const RopeTable&,
                                   const Tensor& q_norm, const Tensor& k_norm, double eps, double offset);
std::optional<Tensor> rope_kv(const AttentionInput&, const state::KvView&, const RopeTable&);

// Normalize over the last dimension, preserving shape; F32 output is the default.
Tensor rms_norm(Tensor input, const Tensor& weight, double eps, double offset,
                ScalarType output_type = ScalarType::float32);
// Undefined weights pass their value through, including CPU values; other outputs use F32 or BF16.
std::array<Tensor, 3> qkv_norm(std::array<Tensor, 3> values, std::array<Tensor, 3> weights,
                               double eps, double offset, ScalarType output_type);
Tensor add(Tensor left, Tensor right);
Tensor multiply(Tensor left, Tensor right);
Tensor matmul(Tensor left, Tensor right);
Tensor transpose(Tensor input, std::int64_t first, std::int64_t second);
Tensor add_residual(Tensor residual, Tensor value);

struct ResidualNorms {
    Tensor residual, dense, router, experts;
};
ResidualNorms residual_norms(Tensor residual, Tensor attention, const Tensor& attention_weight,
                             const Tensor& dense_weight, const Tensor& router_weight,
                             const Tensor& expert_weight, double eps);
Tensor merge_norms(Tensor dense, Tensor experts, Tensor residual, const Tensor& dense_weight,
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
Tensor softcap(Tensor input, double cap);
Tensor scale(Tensor input, double factor);
// Prepared weights determine supported host, CUDA and sharded projection/embedding paths.
Tensor embedding(Tensor ids, const weights::Matrix& weight);
Tensor project(Tensor input, const weights::Matrix& weight, const ProjectionOptions& options = {});
std::vector<Tensor> project(Tensor input, const weights::MatrixGroup& group,
                            const ProjectionOptions& options = {});
// Computes project(input * sigmoid(gate)).
Tensor sigmoid_project(Tensor input, Tensor gate, const weights::Matrix& weight,
                       const ProjectionOptions& options = {});
// Copy, cast, views and allocation support CPU as well as CUDA, subject to dtype support.
Tensor copy(Tensor input, Device target);
Tensor cast(Tensor input, ScalarType type);
// Targeted cast uses native tensor conversion; CUDA copy uses execution collectives.
Tensor cast(Tensor input, ScalarType type, Device target);
Tensor contiguous(Tensor input);
Tensor clone(Tensor input);
Tensor reshape(Tensor input, std::span<const std::int64_t> shape);
Tensor slice(Tensor input, std::int64_t dim, std::int64_t start, std::int64_t end, std::int64_t step = 1);
std::vector<Tensor> chunk(Tensor input, std::int64_t count, std::int64_t dim);
// Position tensors are I64.
Tensor arange(std::int64_t first, std::int64_t last, Device device);
Tensor full(std::span<const std::int64_t> shape, std::int64_t value, Device device);
Tensor add(Tensor input, std::int64_t offset);

enum class GatedActivation { silu, gelu };

// MLP activation selected by the model config. relu_squared consumes a single
// Up projection, so it has no gate branch and no SwiGLU clamp.
enum class MlpActivation { silu_gated, gelu_gated, relu_squared };

// Prepared weights determine supported host and sharded MLP paths.
Tensor mlp(Tensor input, const weights::MatrixGroup& gate_up, const weights::Matrix& down,
           MlpActivation activation, double swiglu_limit, const FfnMode& mode);

// Canonical gate/up -> activation product -> down graph. Native operations may
// replace a contiguous part of this graph with an equivalent fused operation.
template <class Tensor, class Fused, class FusedProjection, class Project, class Activate,
          class Down, class FusedDown>
Tensor gated_mlp(Tensor input, bool geglu, double swiglu_limit, Fused fused,
                 FusedProjection fused_projection, Project project, Activate activate, Down down,
                 FusedDown fused_down) {
    const auto activation = geglu ? GatedActivation::gelu : GatedActivation::silu;
    const double limit = geglu ? 0.0 : swiglu_limit;
    if (auto output = fused(input, activation, limit))
        return std::move(*output);
    if (auto hidden = fused_projection(input, activation, limit))
        return down(std::move(*hidden));
    auto parts = project(std::move(input));
    if (parts.size() != 2)
        throw std::runtime_error("gated MLP requires gate and up projections");
    if (auto output = fused_down(parts[0], parts[1], activation, limit))
        return std::move(*output);
    return down(activate(std::move(parts[0]), std::move(parts[1]), activation, limit));
}

// Independent low/high neuron branches may run concurrently. Scheduling is
// native; selecting both results and merging them is part of the model graph.
template <class Branch, class Parallel, class Add>
auto additive_branches(Branch branch, Parallel parallel, Add add) {
    using Tensor = decltype(branch(0));
    std::vector<Tensor> outputs;
    if (!parallel(branch, outputs)) {
        outputs.push_back(branch(0));
        outputs.push_back(branch(1));
    }
    if (outputs.size() != 2)
        throw std::runtime_error("additive FFN requires two branch outputs");
    return add(std::move(outputs[0]), std::move(outputs[1]));
}

template <class Tensor, class Up, class Activate, class Down>
auto mlp(Tensor hidden, Up up, Activate activate, Down down) {
    return down(activate(up(std::move(hidden))));
}

enum class RouterActivation { softmax, sigmoid, sqrt_softplus };
struct MoeRouting {
    RouterActivation activation = RouterActivation::softmax;
    bool normalize = false;
    bool delayed_softmax = false;
    double scale = 1.0;
};

struct SelectedExperts { Tensor ids, weights; };
SelectedExperts topk(Tensor logits, std::int64_t count, const MoeRouting& routing);
// Scale the selected weights in place and return the same storage.
Tensor expert_scale(Tensor weights, Tensor ids, const Tensor& scales);
Tensor weighted_reduce(Tensor pairs, Tensor weights);
// Traced intermediates are F32 snapshots taken before subsequent projections.
struct RoutedGeglu { Tensor pairs, gate_up, hidden; };
RoutedGeglu routed_geglu(Tensor input, Tensor ids, const weights::ExpertBank& gate_up,
                         const weights::ExpertBank& down, const ProjectionOptions& options,
                         bool trace);

// Both branches consume the same input. Native routing may prepare an expert
// cache before shared computation, but never changes the selected experts.
template <class Router, class Select, class Shared, class Experts, class Combine>
auto mixture_of_experts(Router router, Select select, Shared shared, Experts experts,
                        Combine combine) {
    auto selected = select(router());
    auto shared_output = shared(selected);
    auto routed = experts(selected);
    return combine(std::move(routed), std::move(shared_output), selected);
}

template <class Project, class Activate, class Down, class Reduce>
auto routed_experts(Project project, Activate activate, Down down, Reduce reduce) {
    auto projections = project();
    auto hidden = activate(std::move(projections));
    auto pairs = down(std::move(hidden));
    return reduce(std::move(pairs));
}

// Fused reduction may include shared gating and addition. Otherwise the same
// graph is evaluated explicitly, including the ungated shared-expert variant.
template <class Tensor, class Fused, class Reduce, class Gate, class Add, class GatedAdd>
auto combine_experts(Tensor pairs, Tensor shared, bool reduced, bool ungated, Fused fused,
                     Reduce reduce, Gate gate, Add add, GatedAdd gated_add) {
    if (!reduced)
        if (auto result = fused(pairs, shared))
            return std::move(*result);
    auto routed = reduced ? std::move(pairs) : reduce(std::move(pairs));
    if (ungated)
        return add(std::move(routed), std::move(shared));
    auto logits = gate();
    return gated_add(std::move(routed), std::move(shared), std::move(logits));
}

template <class Tensor, class AddNorm, class FeedForward>
auto residual_ffn(Tensor residual, Tensor attention, AddNorm add_norm, FeedForward feed_forward) {
    auto prepared = add_norm(std::move(residual), std::move(attention));
    return feed_forward(std::move(prepared[0]), std::move(prepared[1]));
}

template <class Tensor, class Fused, class FeedForward, class Add>
auto feed_forward(Tensor residual, Tensor normalized, Fused fused, FeedForward project, Add add) {
    if (auto output = fused(normalized, residual))
        return std::move(*output);
    auto output = project(std::move(normalized));
    return add(std::move(residual), std::move(output));
}

// Operators may fuse adjacent arithmetic, but the model owns their order and
// the residual branch. All values keep their backend's native tensor type.
template <class Tensor, class Norm, class Attention, class Tail>
auto attention_layer(Tensor hidden, Norm normalize, Attention attention, Tail tail) {
    auto normalized = normalize(hidden);
    auto projected = attention(std::move(normalized));
    return tail(std::move(hidden), std::move(projected));
}

struct YarnScaling {
    double factor=1, beta_fast=32, beta_slow=1;
    std::int64_t original_context=0;
    bool enabled() const { return factor>1; }
};

} // namespace mfq::ops
