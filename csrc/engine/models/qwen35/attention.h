#pragma once

#include <utility>

namespace mfq::models::qwen35 {

template <class Tensor> struct LinearProjections {
    Tensor qkv, qk, value, output_gate, alpha, beta;
};

// Convolution/QK normalization can be one fused native operator. GDN updates
// its native recurrent state; the model owns its placement in the graph.
template <class Tensor, class Project, class Gates, class Convolve, class Recur, class Normalize,
          class Output>
Tensor linear_attention(Tensor normalized, Project project, Gates gates, Convolve convolve,
                        Recur recur, Normalize normalize, Output output) {
    auto projected = project(std::move(normalized));
    auto recurrence_gates = gates(projected);
    auto qkv = convolve(projected);
    auto hidden = recur(std::move(qkv), std::move(recurrence_gates));
    auto normed = normalize(std::move(hidden));
    return output(std::move(normed), std::move(projected.output_gate));
}

} // namespace mfq::models::qwen35
