#include "mlp.h"
#include "engine/backends/cuda/native/collectives.h"
#include "engine/backends/cuda/kernels/mfq_cuda_activation_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_moe_ops.h"

#include <algorithm>
#include <array>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

using Tensor = mfq_tensor_backend::Tensor;
using ActivationKind = mfq::models::MlpActivation;

Tensor dense_routed_projection(const Tensor &bank, const Tensor &input, const Tensor &ids,
                               std::size_t gather_budget_bytes) {
    TYLOINFER_RUNTIME_CHECK(bank.is_cuda() && bank.is_contiguous() && bank.dim() == 3,
                      "dense routed expert bank must be contiguous CUDA [experts,out,in]");
    TYLOINFER_RUNTIME_CHECK(ids.is_cuda() && ids.dim() == 2, "dense routed ids must be [tokens,routes]");
    const int64_t rows = ids.size(0), routes = ids.size(1);
    const int64_t output = bank.size(1), width = bank.size(2);
    TYLOINFER_RUNTIME_CHECK(input.size(0) == rows && input.size(-1) == width,
                      "dense routed input shape does not match the expert bank");
    TYLOINFER_RUNTIME_CHECK(input.dim() == 2 || (input.dim() == 3 && input.size(1) == routes),
                      "dense routed activation shape does not match the route plan");
    auto project = [&](const Tensor &chunk_ids, const Tensor &chunk_input) {
        const int64_t count = chunk_ids.size(0);
        auto selected =
            bank.index_select(0, chunk_ids.reshape({-1}).to(mfq_tensor_backend::kInt64))
                .reshape({count, routes, output, width});
        auto source = chunk_input.dim() == 2
                          ? chunk_input.unsqueeze(1).expand({count, routes, width})
                          : chunk_input;
        return mfq_tensor_backend::matmul(selected, source.to(bank.scalar_type()).unsqueeze(-1))
            .squeeze(-1);
    };
    // Every route copies its whole expert matrix, so a long prompt would gather
    // tokens * routes full matrices at once; bound each copy by row chunks.
    const auto row_bytes = static_cast<std::size_t>(routes * output * width) * bank.element_size();
    const auto rows_per_chunk = gather_budget_bytes / std::max<std::size_t>(1, row_bytes);
    const auto chunk_rows = static_cast<int64_t>(std::max<std::size_t>(1, rows_per_chunk));
    if (rows <= chunk_rows)
        return project(ids, input);
    std::vector<Tensor> parts;
    parts.reserve(static_cast<std::size_t>((rows + chunk_rows - 1) / chunk_rows));
    for (int64_t start = 0; start < rows; start += chunk_rows) {
        const auto count = std::min(chunk_rows, rows - start);
        parts.push_back(project(ids.narrow(0, start, count), input.narrow(0, start, count)));
    }
    return mfq_tensor_backend::cat(parts, 0);
}

namespace mfq::cuda {

Tensor expert_activation(ActivationKind activation, Tensor gate, Tensor up, double limit) {
    if (activation == ActivationKind::relu_squared) {
        TYLOINFER_RUNTIME_CHECK(!gate.defined(), "relu-squared experts have no gate projection");
        return relu_squared_cuda(up.contiguous());
    }
    TYLOINFER_RUNTIME_CHECK(gate.defined(), "gated experts require a gate projection");
    if (activation == ActivationKind::gelu_gated)
        return gelu_mul_cuda(gate.contiguous(), up.contiguous());
    if (limit > 0.0) {
        auto clipped_gate = mfq_tensor_backend::clamp_max(
            gate.to(mfq_tensor_backend::kFloat32), limit);
        auto clipped_up = mfq_tensor_backend::clamp(
            up.to(mfq_tensor_backend::kFloat32), -limit, limit);
        return (mfq_tensor_backend::silu(clipped_gate) * clipped_up)
            .to(mfq_tensor_backend::kFloat16).contiguous();
    }
    return (up * mfq_tensor_backend::silu(gate)).contiguous();
}

bool Mlp::parallel() const {
    if (gate_up.layers.size() != 2 || !down.tensor_parallel() ||
        down.tensor_parallel_axis != TensorParallelAxis::Input) {
        return false;
    }
    const auto &gate = gate_up.layers[0];
    const auto &up = gate_up.layers[1];
    if (!gate.tensor_parallel() || !up.tensor_parallel() ||
        gate.tensor_parallel_axis != TensorParallelAxis::Output ||
        up.tensor_parallel_axis != TensorParallelAxis::Output ||
        gate.tensor_parallel_shards.size() != up.tensor_parallel_shards.size() ||
        gate.tensor_parallel_shards.size() != down.tensor_parallel_shards.size()) {
        return false;
    }
    for (size_t index = 0; index < gate.tensor_parallel_shards.size(); ++index) {
        const auto &gate_shard = gate.tensor_parallel_shards[index];
        const auto &up_shard = up.tensor_parallel_shards[index];
        const auto &down_shard = down.tensor_parallel_shards[index];
        if (gate_shard.device != up_shard.device || gate_shard.device != down_shard.device ||
            gate_shard.output_begin != up_shard.output_begin ||
            gate_shard.output_end != up_shard.output_end ||
            gate_shard.output_begin != down_shard.input_begin ||
            gate_shard.output_end != down_shard.input_end) {
            return false;
        }
    }
    return true;
}

mfq_tensor_backend::Tensor Mlp::sharded(
        CudaProfiler& profiler, KlMmqState& mmq, TqExec& tq, Collectives& collectives,
        int primary, bool peer_first, bool reduce_to_primary, bool fp16_reduce,
        mfq_tensor_backend::Tensor xh) const {
    auto shape = xh.sizes().vec();
    auto flat = xh.reshape({-1, xh.size(-1)});
    const size_t shard_count = down.tensor_parallel_shards.size();
    std::vector<mfq_tensor_backend::Tensor> partials(shard_count);
    for (size_t launch_position = 0; launch_position < shard_count; ++launch_position) {
        const size_t index = parallel_index(peer_first, launch_position, shard_count);
        const auto &gate_shard = gate_up.layers[0].tensor_parallel_shards[index];
        const auto &up_shard = gate_up.layers[1].tensor_parallel_shards[index];
        const auto &down_shard = down.tensor_parallel_shards[index];
        MfqCudaGuard guard(gate_shard.device);
        auto local_x = collectives.copy(flat, gate_shard.device);
        using Tensor = mfq_tensor_backend::Tensor;
        auto unfused = [](const auto &...) { return std::optional<Tensor>{}; };
        partials[index] = mfq::models::gated_mlp(
            local_x, geglu, swiglu_limit, unfused, unfused,
            [&](Tensor input) {
                auto gate =
                    run_quant_linear_shard(profiler, mmq, tq, gate_shard, input);
                auto up =
                    run_quant_linear_shard(profiler, mmq, tq, up_shard, input);
                return std::array<Tensor, 2>{std::move(gate), std::move(up)};
            },
            [&](Tensor gate_output, Tensor up_output, auto activation_kind, double limit) {
                mfq_tensor_backend::Tensor activation;
                if (activation_kind == mfq::models::GatedActivation::gelu) {
                    activation = gelu_mul_cuda(gate_output.contiguous(), up_output.contiguous());
                } else if (limit > 0.0) {
                    auto clipped_gate = mfq_tensor_backend::clamp_max(
                        gate_output.to(mfq_tensor_backend::kFloat32), limit);
                    auto clipped_up = mfq_tensor_backend::clamp(
                        up_output.to(mfq_tensor_backend::kFloat32), -limit, limit);
                    activation = (mfq_tensor_backend::silu(clipped_gate) * clipped_up)
                                     .to(mfq_tensor_backend::kFloat16)
                                     .contiguous();
                } else {
                    activation = (mfq_tensor_backend::silu(gate_output) * up_output).contiguous();
                }
                return activation;
            },
            [&](Tensor activation) {
                return run_quant_linear_shard(profiler, mmq, tq, down_shard,
                                              activation);
            },
            unfused);
    }
    auto output = collectives.sum(std::move(partials), primary,
            reduce_to_primary, fp16_reduce);
    shape.back() = output.size(-1);
    return output.reshape(shape);
}

mfq_tensor_backend::Tensor Mlp::forward(
        const Projector& project, CudaProfiler& profiler, const Options& options,
        mfq_tensor_backend::Tensor xh) const {
    using Tensor = mfq_tensor_backend::Tensor;
    using Activation = mfq::models::GatedActivation;
    if (activation == ActivationKind::relu_squared) {
        TYLOINFER_RUNTIME_CHECK(!parallel(),
                          "relu-squared MLP does not support tensor parallelism");
        auto parts = profiler.measure("ffn.up", [&]() { return project.run(gate_up, xh); });
        TYLOINFER_RUNTIME_CHECK(parts.size() == 1,
                          "relu-squared FFN requires exactly one Up projection");
        auto hidden = profiler.measure("ffn.relu_squared", [&]() {
            return relu_squared_cuda(parts[0].contiguous());
        });
        return profiler.measure("ffn.down",
                                [&]() { return project.run(down, hidden); });
    }
    return mfq::models::gated_mlp(
        xh, geglu, swiglu_limit,
        [&](const Tensor &input, Activation activation, double limit) -> std::optional<Tensor> {
            if (activation != Activation::silu || limit > 0.0)
                return {};
            if (mfq::mfq_compat_enabled && options.nvq_fusion && input.numel() / input.size(-1) == 1 &&
                gate_up.nvq_prefix2 && gate_up.layers.size() == 2 && gate_up.layers[0].is_nvq() &&
                gate_up.layers[1].is_nvq() && down.is_nvq() && gate_up.outs.size() == 2 &&
                gate_up.outs[0] == gate_up.outs[1] && gate_up.outs[0] == down.nvq.neuron_len &&
                (down.nvq.gs == 24 || down.nvq.gs == 28 || down.nvq.gs == 32)) {
                auto shape = input.sizes().vec();
                shape.back() = down.nvq.out;
                return nvq_ffn_swiglu_down(profiler, gate_up.layers[0].nvq, gate_up.layers[1].nvq,
                                           down.nvq, input.reshape({-1, input.size(-1)}))
                    .reshape(shape);
            }
            return {};
        },
        [&](const Tensor &input, Activation activation, double limit) -> std::optional<Tensor> {
            if(limit<=0.0 && activation==Activation::silu && gate_up.tq.defined() &&
                    gate_up.outs.size()==2 && gate_up.outs[0]==gate_up.outs[1])
                return profiler.measure("ffn.gate_up_swiglu",[&] {return project.swiglu(gate_up, input);});
            if (!mfq::mfq_compat_enabled || limit > 0.0 || !gate_up.nint_grouped || !gate_up.nint.split_w.empty())
                return {};
            const auto rows = input.numel() / input.size(-1);
            if (activation == Activation::gelu && options.gelu_fusion && rows == 1)
                return profiler.measure("ffn.gate_up_geglu",
                                        [&] { return project.geglu(gate_up, input); });
            if (activation == Activation::silu && options.silu_fusion && rows >= 1 &&
                rows <= 6 && gate_up.outs.size() == 2 && gate_up.outs[0] == gate_up.outs[1])
                return profiler.measure("ffn.gate_up_swiglu",
                                        [&] { return project.swiglu(gate_up, input); });
            return {};
        },
        [&](Tensor input) {
            return profiler.measure("ffn.gate_up",
                                    [&] { return project.run(gate_up, input); });
        },
        [&](Tensor gate, Tensor up, Activation activation, double limit) {
            if (activation == Activation::gelu)
                return profiler.measure(
                    "ffn.geglu", [&] { return gelu_mul_cuda(gate.contiguous(), up.contiguous()); });
            if (limit > 0.0)
                return profiler.measure("ffn.swiglu_clamped", [&] {
                    auto g =
                        mfq_tensor_backend::clamp_max(gate.to(mfq_tensor_backend::kFloat32), limit);
                    auto u = mfq_tensor_backend::clamp(up.to(mfq_tensor_backend::kFloat32), -limit,
                                                       limit);
                    return (mfq_tensor_backend::silu(g) * u)
                        .to(mfq_tensor_backend::kFloat16)
                        .contiguous();
                });
            return profiler.measure(
                "ffn.swiglu", [&] { return (up * mfq_tensor_backend::silu(gate)).contiguous(); });
        },
        [&](Tensor hidden) {
            return profiler.measure("ffn.down", [&] { return project.run(down, hidden); });
        },
        [&](const Tensor &gate, const Tensor &up, Activation activation,
            double limit) -> std::optional<Tensor> {
            if (activation != Activation::silu || limit > 0.0 || down.is_dense() || down.is_mxfp8())
                return {};
            return profiler.measure("ffn.down",
                                    [&] { return project.gated(down, up, gate, InputGateMode::Silu); });
        });
}

mfq_tensor_backend::Tensor Mlp::bf16(
        const Projector& project, CudaProfiler& profiler, bool fused, Tensor input) const {
    using Activation = mfq::models::GatedActivation;
    const auto unfused = [](const auto&...) { return std::optional<Tensor>{}; };
    return mfq::models::gated_mlp(
        std::move(input), false, 0.0, unfused, unfused,
        [&](Tensor input) {
            return profiler.measure("full.minicpmo45_ffn_gate_up", [&] {
                return project.run(gate_up, input);
            });
        },
        [&](Tensor gate, Tensor up, Activation, double) {
            gate = gate.to(mfq_tensor_backend::kBFloat16).contiguous();
            up = up.to(mfq_tensor_backend::kBFloat16).contiguous();
            return profiler.measure("full.minicpmo45_ffn_swiglu", [&] {
                return fused ? silu_mul_cuda(gate, up)
                             : (mfq_tensor_backend::silu(gate) * up).contiguous();
            });
        },
        [&](Tensor activation) {
            return profiler.measure("full.minicpmo45_ffn_down", [&] {
                return project.run(down, activation).to(mfq_tensor_backend::kBFloat16).contiguous();
            });
        }, unfused);
}

mfq_tensor_backend::Tensor Mlp::kld(
        const Projector& project, CudaProfiler& profiler, KlMmqState& mmq,
        mfq_tensor_backend::Tensor xh) const {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    TYLOINFER_RUNTIME_CHECK(!parallel() &&
                          activation == ActivationKind::silu_gated && !geglu &&
                          swiglu_limit <= 0.0,
                      "FP32-output IN diagnostic requires a local dense SiLU FFN");
    auto parts = profiler.measure("ffn.gate_up", [&]() { return project.run(gate_up, xh); });
    return profiler.measure("ffn.down.fp32_output", [&]() {
        return nint_kld(profiler, mmq, down, parts[1], parts[0], InputGateMode::Silu);
    });
#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}

bool Mlp::can_fuse(const Tensor& x, const Tensor& residual) const {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (activation != ActivationKind::silu_gated || geglu || swiglu_limit > 0.0 ||
        parallel() || !x.is_cuda() || !residual.is_cuda() ||
        x.scalar_type() != mfq_tensor_backend::kFloat16 ||
        residual.scalar_type() != mfq_tensor_backend::kFloat16 || !x.is_contiguous() ||
        !residual.is_contiguous() || x.dim() < 1 || residual.dim() < 1 ||
        x.numel() / x.size(-1) != 1 || residual.numel() / residual.size(-1) != 1 ||
        gate_up.layers.size() != 2 || !gate_up.nvq_prefix2 || !gate_up.layers[0].is_nvq() ||
        !gate_up.layers[1].is_nvq() || !down.is_nvq() || gate_up.layers[0].tensor_parallel() ||
        gate_up.layers[1].tensor_parallel() || down.tensor_parallel() || gate_up.outs.size() != 2 ||
        gate_up.outs[0] != gate_up.outs[1] || gate_up.outs[0] != down.nvq.neuron_len ||
        x.size(-1) != gate_up.layers[0].nvq.neuron_len || residual.size(-1) != down.nvq.out) {
        return false;
    }
    return nvq_fused_residual_format(down.nvq.kernel_format);
#else
    return false;
#endif
}

Tensor Mlp::fused(CudaProfiler& profiler, Tensor x, Tensor residual) const {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    TYLOINFER_RUNTIME_CHECK(can_fuse(x, residual),
                      "FFN fused residual requires a compatible single-token NVQ FFN");
    auto shape = x.sizes().vec();
    shape.back() = down.nvq.out;
    auto output =
        nvq_ffn_swiglu_down(profiler, gate_up.layers[0].nvq, gate_up.layers[1].nvq, down.nvq,
                            x.reshape({1, x.size(-1)}), residual.reshape({1, residual.size(-1)}));
    return output.reshape(shape);
#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}

} // namespace mfq::cuda
