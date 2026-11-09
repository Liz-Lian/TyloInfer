#include "engine/models/common/ffn.h"
#include "engine/backends/cuda/ops/ffn/ffn.h"

#include <stdexcept>

#ifdef TYLOINFER_NATIVE_CUDA
namespace tyloi::models {
namespace {

Tensor feed_forward(const cuda::Ffn& weights, const Tensor& input, bool bf16) {
    if (weights.is_moe && !bf16) return weights.forward(input.dense());
    if (!weights.down) throw std::runtime_error("FFN down projection is missing");
    if (bf16)
        return ops::bf16_mlp(input, weights.gate_up, weights.down,
            weights.activation, weights.swiglu_limit);
    if (weights.important_neurons) {
        const auto& extra = *weights.important_neurons;
        if (!extra.down) throw std::runtime_error("FFN down projection is missing");
        return ops::additive_mlp(input, weights.gate_up, weights.down,
            weights.activation, weights.swiglu_limit, extra.gate_up, extra.down,
            extra.activation, extra.swiglu_limit);
    }
    return ops::mlp(input, weights.gate_up, weights.down,
        weights.activation, weights.swiglu_limit);
}

} // namespace

Tensor full_ffn(const cuda::Ffn& weights, const Tensor& norm,
    const Tensor& residual, const Tensor& attention_output,
    double epsilon, double offset, bool bf16, bool trace,
    ops::Trace* trace_output) {
    const auto batch = residual.size(0), tokens = residual.size(1), width = residual.size(2);
    auto pair = ops::residual_norm(residual, attention_output, norm, epsilon, offset, bf16);
    auto input = ops::reshape(pair[1], {batch * tokens, width});
    if (trace && trace_output)
        trace_output->emplace_back("qwen.ffn_norm", ops::clone(ops::contiguous(
            ops::cast(ops::reshape(input, {batch, tokens, width}), ScalarType::float32))));
    if (!weights.is_moe && !bf16 && !weights.important_neurons) {
        auto flat_residual = ops::reshape(pair[0], {batch * tokens, width});
        if (auto fused = ops::mlp_residual(input, flat_residual, weights.gate_up,
                weights.down, weights.activation, weights.swiglu_limit))
            return ops::reshape(*fused, {batch, tokens, width});
    }
    auto output = ops::reshape(feed_forward(weights, input, bf16), {batch, tokens, width});
    if (trace && trace_output)
        trace_output->emplace_back("qwen.ffn_output", ops::clone(ops::contiguous(
            ops::cast(output, ScalarType::float32))));
    return ops::residual_add(pair[0], output);
}

Tensor linear_ffn(const cuda::Ffn& weights, const Tensor& norm,
    const Tensor& residual, const Tensor& attention_output,
    double epsilon, double offset) {
    const auto batch = residual.size(0), tokens = residual.size(1), width = residual.size(2);
    auto pair = ops::recurrent_norm(residual, attention_output, norm, epsilon, offset);
    auto input = ops::reshape(pair[1], {batch * tokens, width});
    if (!weights.is_moe && !weights.important_neurons) {
        auto flat_residual = ops::reshape(pair[0], {batch * tokens, width});
        if (auto fused = ops::mlp_residual(input, flat_residual, weights.gate_up,
                weights.down, weights.activation, weights.swiglu_limit, true))
            return ops::reshape(*fused, {batch, tokens, width});
    }
    auto output = ops::reshape(feed_forward(weights, input, false), {batch, tokens, width});
    return ops::recurrent_add(pair[0], output);
}

} // namespace tyloi::models
#endif
