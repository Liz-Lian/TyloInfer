#include "executor.h"
#include "engine/backends/cuda/ops/ffn/dense.h"
#include "engine/backends/cuda/ops/ffn/mlp.h"
#include "engine/backends/cuda/ops/ffn/residual.h"

#include <utility>

namespace mfq::cuda {

using Tensor = mfq_tensor_backend::Tensor;

Tensor Executor::full_ffn(
    const Ffn& ffn, const Tensor& ffn_norm, const FullAttention& attention,
    Tensor residual, Tensor output, const Ffn::Mode& mode, Trace* trace_output) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    auto scope = enter(mfq_current_cuda_device());
#endif
    const auto project_ = projector(mode);
    const auto& projection_ = options_.projection;
    const auto& attention_ = options_.attention;
    const auto& dense_ = options_.dense;
    if (!ffn.is_moe) {
        Dense op{dense(ffn), {}, ffn_norm, attention.rms_norm_eps, attention.norm_weight_offset,
            attention.official_bf16, attention.attention_output_gate};
        if (ffn.important_neurons) op.important.emplace(dense(*ffn.important_neurons));
        const auto& parallel = projection_;
        auto options = dense_;
        options.mlp.nvq_fusion = parallel.nvq_fusion;
        options.bf16_norm = attention_.fused_norm;
        return op.forward(project_, profiler_, mmq_, *tq_, branches_, collectives_, parallel, options,
            mode.serial, std::move(residual), std::move(output), trace_output);
    }
    const int64_t B = residual.size(0), T = residual.size(1), H = residual.size(2);
    auto& profiler = profiler_;
    auto record_trace = [&](const char* name, const Tensor& value) {
        if (attention.attention_output_gate && trace_output)
            trace_output->emplace_back(name, value.reshape({B, T, -1})
                .to(mfq_tensor_backend::kFloat32).contiguous().clone());
    };
    auto pair = full_norm(profiler, ffn_norm, attention.rms_norm_eps, attention.norm_weight_offset,
                          attention.official_bf16, attention_.fused_norm,
                          dense_.fp32_residual, std::move(residual), std::move(output));
    residual = pair[0];
    auto input = pair[1].reshape({B * T, H});
    record_trace("qwen.ffn_norm", input);
    auto mlp = dense(ffn);
    auto ff = attention.official_bf16
        ? mlp.bf16(project_, profiler, dense_.bf16_swiglu, input)
              .reshape({B, T, H}).to(mfq_tensor_backend::kBFloat16).contiguous()
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        : ffn.forward(input, mode).reshape({B, T, H});
#else
        : ffn.forward(input, *this, mode).reshape({B, T, H});
#endif
    record_trace("qwen.ffn_output", ff);
    return full_add(profiler, dense_.fp32_residual, dense_.bf16_residual,
                    std::move(residual), std::move(ff));
}

Tensor Executor::linear_ffn(
    const Ffn& ffn, const Tensor& ffn_norm, const LinearAttention& attention,
    Tensor residual, Tensor output, const Ffn::Mode& mode) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    auto scope = enter(mfq_current_cuda_device());
#endif
    const auto project_ = projector(mode);
    const auto& projection_ = options_.projection;
    const auto& dense_ = options_.dense;
    if (!ffn.is_moe) {
        Dense op{dense(ffn), {}, ffn_norm, attention.rms_norm_eps,
            attention.norm_weight_offset, false, false, true};
        if (ffn.important_neurons) op.important.emplace(dense(*ffn.important_neurons));
        const auto& parallel = projection_;
        auto options = dense_;
        options.mlp.nvq_fusion = parallel.nvq_fusion;
        return op.forward(project_, profiler_, mmq_, *tq_, branches_, collectives_, parallel, options,
            mode.serial, std::move(residual), std::move(output));
    }
    const int64_t B = residual.size(0), T = residual.size(1), H = residual.size(2);
    auto& profiler = profiler_;
    auto pair = linear_norm(profiler, ffn_norm, attention.rms_norm_eps, attention.norm_weight_offset,
                            dense_.fp32_residual, std::move(residual), std::move(output));
    residual = pair[0];
    auto input = pair[1].reshape({B * T, H});
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    auto ff = ffn.forward(input, mode).reshape({B, T, H});
#else
    auto ff = ffn.forward(input, *this, mode).reshape({B, T, H});
#endif
    return linear_add(profiler, dense_.fp32_residual, std::move(residual), std::move(ff));
}

} // namespace mfq::cuda
