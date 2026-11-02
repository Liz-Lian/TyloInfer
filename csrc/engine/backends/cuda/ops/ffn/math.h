#pragma once

#include <stdexcept>
#include <utility>
#include <vector>

namespace mfq::cuda {

enum class GatedActivation { silu, gelu };

// MLP activation selected by the model config. relu_squared consumes a single
// Up projection, so it has no gate branch and no SwiGLU clamp.
enum class MlpActivation { silu_gated, gelu_gated, relu_squared };

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

} // namespace mfq::cuda
