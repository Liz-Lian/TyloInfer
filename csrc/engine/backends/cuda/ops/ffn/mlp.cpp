#include "mlp.h"
#include "engine/ops.h"
#include "engine/backends/cuda/native/collectives.h"
#include "engine/backends/cuda/kernels/mfq_cuda_activation_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_moe_ops.h"

#include <algorithm>
#include <array>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

using DenseTensor = ::tyloi::DenseTensor;
using ActivationKind = tyloi::ops::MlpActivation;

DenseTensor dense_routed_projection(const DenseTensor &bank, const DenseTensor &input, const DenseTensor &ids,
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
    auto project = [&](const DenseTensor &chunk_ids, const DenseTensor &chunk_input) {
        const int64_t count = chunk_ids.size(0);
        auto selected =
            bank.index_select(0, chunk_ids.reshape({-1}).to(::tyloi::kInt64))
                .reshape({count, routes, output, width});
        auto source = chunk_input.dim() == 2
                          ? chunk_input.unsqueeze(1).expand({count, routes, width})
                          : chunk_input;
        return ::tyloi::cuda::matmul(selected, source.to(bank.scalar_type()).unsqueeze(-1))
            .squeeze(-1);
    };
    // Every route copies its whole expert matrix, so a long prompt would gather
    // tokens * routes full matrices at once; bound each copy by row chunks.
    const auto row_bytes = static_cast<std::size_t>(routes * output * width) * bank.element_size();
    const auto rows_per_chunk = gather_budget_bytes / std::max<std::size_t>(1, row_bytes);
    const auto chunk_rows = static_cast<int64_t>(std::max<std::size_t>(1, rows_per_chunk));
    if (rows <= chunk_rows)
        return project(ids, input);
    std::vector<DenseTensor> parts;
    parts.reserve(static_cast<std::size_t>((rows + chunk_rows - 1) / chunk_rows));
    for (int64_t start = 0; start < rows; start += chunk_rows) {
        const auto count = std::min(chunk_rows, rows - start);
        parts.push_back(project(ids.narrow(0, start, count), input.narrow(0, start, count)));
    }
    return ::tyloi::cuda::cat(parts, 0);
}

namespace tyloi::cuda {


DenseTensor expert_activation(ActivationKind activation, DenseTensor gate, DenseTensor up, double limit) {
    if (activation == ActivationKind::relu_squared) {
        TYLOINFER_RUNTIME_CHECK(!gate.defined(), "relu-squared experts have no gate projection");
        return relu_squared_cuda(up.contiguous());
    }
    TYLOINFER_RUNTIME_CHECK(gate.defined(), "gated experts require a gate projection");
    if (activation == ActivationKind::gelu_gated)
        return gelu_mul_cuda(gate.contiguous(), up.contiguous());
    if (limit > 0.0) {
        auto clipped_gate = ::tyloi::cuda::clamp_max(
            gate.to(::tyloi::kFloat32), limit);
        auto clipped_up = ::tyloi::cuda::clamp(
            up.to(::tyloi::kFloat32), -limit, limit);
        return (::tyloi::cuda::silu(clipped_gate) * clipped_up)
            .to(::tyloi::kFloat16).contiguous();
    }
    return (up * ::tyloi::cuda::silu(gate)).contiguous();
}

void Mlp::check(const DenseTensor& input, int primary) const {
    check_projection(gate_up, input);
    const bool relu = activation == ActivationKind::relu_squared;
    if (gate_up.outs.size() != (relu ? 1u : 2u))
        throw std::invalid_argument("MLP has an invalid number of projections");
    auto width = gate_up.outs[0];
    if (!relu) {
        const auto up = gate_up.outs[1];
        if (width != up && (geglu || (width != 1 && up != 1)))
            throw std::invalid_argument("MLP gate and up widths do not match");
        width = std::max(width, up);
    }
    auto device = input.device();
    if (!gate_up.layers.empty()) {
        const auto& first = gate_up.layers.front();
        if (first.tensor_parallel()) device = ::tyloi::Device(::tyloi::kCUDA, primary);
        for (const auto& layer : gate_up.layers) {
            const auto local_device = layer.tensor_parallel()
                ? ::tyloi::Device(::tyloi::kCUDA, primary) : input.device();
            if (local_device != device)
                throw std::invalid_argument("MLP gate and up devices do not match");
        }
    }
    check_projection_device(down, device);
    check_projection_width(down, width);
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

::tyloi::DenseTensor Mlp::sharded(
        CudaProfiler& profiler, KlMmqState& mmq, TqExec& tq, Collectives& collectives,
        int primary, bool peer_first, bool reduce_to_primary, bool fp16_reduce,
        ::tyloi::DenseTensor xh) const {
    auto shape = xh.sizes().vec();
    auto flat = xh.reshape({-1, xh.size(-1)});
    const size_t shard_count = down.tensor_parallel_shards.size();
    std::vector<::tyloi::DenseTensor> partials(shard_count);
    for (size_t launch_position = 0; launch_position < shard_count; ++launch_position) {
        const size_t index = parallel_index(peer_first, launch_position, shard_count);
        const auto &gate_shard = gate_up.layers[0].tensor_parallel_shards[index];
        const auto &up_shard = gate_up.layers[1].tensor_parallel_shards[index];
        const auto &down_shard = down.tensor_parallel_shards[index];
        MfqCudaGuard guard(gate_shard.device);
        auto local_x = collectives.copy(flat, gate_shard.device);
        auto gate_output = run_quant_linear_shard(profiler, mmq, tq, gate_shard, local_x);
        auto up_output = run_quant_linear_shard(profiler, mmq, tq, up_shard, local_x);
        ::tyloi::DenseTensor hidden;
        if (geglu) {
            hidden = gelu_mul_cuda(gate_output.contiguous(), up_output.contiguous());
        } else if (swiglu_limit > 0.0) {
            auto clipped_gate = ::tyloi::cuda::clamp_max(
                gate_output.to(::tyloi::kFloat32), swiglu_limit);
            auto clipped_up = ::tyloi::cuda::clamp(
                up_output.to(::tyloi::kFloat32), -swiglu_limit, swiglu_limit);
            hidden = (::tyloi::cuda::silu(clipped_gate) * clipped_up)
                         .to(::tyloi::kFloat16).contiguous();
        } else {
            hidden = (::tyloi::cuda::silu(gate_output) * up_output).contiguous();
        }
        gate_output = {};
        up_output = {};
        partials[index] = run_quant_linear_shard(profiler, mmq, tq, down_shard, hidden);
    }
    auto output = collectives.sum(std::move(partials), primary,
            reduce_to_primary, fp16_reduce);
    shape.back() = output.size(-1);
    return output.reshape(shape);
}

::tyloi::DenseTensor Mlp::forward(
        const Projector& project, CudaProfiler& profiler, const Options& options,
        ::tyloi::DenseTensor xh) const {
    using DenseTensor = ::tyloi::DenseTensor;
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
    const double limit = geglu ? 0.0 : swiglu_limit;
    if (!geglu && !(limit > 0.0) && tyloi::mfq_compat_enabled && options.nvq_fusion &&
        xh.numel() / xh.size(-1) == 1 && gate_up.nvq_prefix2 && gate_up.layers.size() == 2 &&
        gate_up.layers[0].is_nvq() && gate_up.layers[1].is_nvq() && down.is_nvq() &&
        gate_up.outs.size() == 2 && gate_up.outs[0] == gate_up.outs[1] &&
        gate_up.outs[0] == down.nvq.neuron_len &&
        (down.nvq.gs == 24 || down.nvq.gs == 28 || down.nvq.gs == 32)) {
        auto shape = xh.sizes().vec();
        shape.back() = down.nvq.out;
        return nvq_ffn_swiglu_down(profiler, gate_up.layers[0].nvq, gate_up.layers[1].nvq,
                                   down.nvq, xh.reshape({-1, xh.size(-1)}))
            .reshape(shape);
    }
    std::optional<DenseTensor> fused_hidden;
    if (limit <= 0.0 && !geglu && gate_up.tq.defined() &&
        gate_up.outs.size() == 2 && gate_up.outs[0] == gate_up.outs[1])
        fused_hidden = profiler.measure("ffn.gate_up_swiglu", [&] { return project.swiglu(gate_up, xh); });
    else if (tyloi::mfq_compat_enabled && !(limit > 0.0) && gate_up.nint_grouped && gate_up.nint.split_w.empty()) {
        const auto rows = xh.numel() / xh.size(-1);
        if (geglu && options.gelu_fusion && rows == 1)
            fused_hidden = profiler.measure("ffn.gate_up_geglu", [&] { return project.geglu(gate_up, xh); });
        else if (!geglu && options.silu_fusion && rows >= 1 &&
                 rows <= 6 && gate_up.outs.size() == 2 && gate_up.outs[0] == gate_up.outs[1])
            fused_hidden = profiler.measure("ffn.gate_up_swiglu", [&] { return project.swiglu(gate_up, xh); });
    }
    if (fused_hidden)
        return profiler.measure("ffn.down", [&] { return project.run(down, *fused_hidden); });
    auto parts = profiler.measure("ffn.gate_up", [&] { return project.run(gate_up, xh); });
    if (parts.size() != 2)
        throw std::runtime_error("gated MLP requires gate and up projections");
    if (!geglu && !(limit > 0.0) && !down.is_dense() && !down.is_mxfp8())
        return profiler.measure("ffn.down", [&] {
            return project.gated(down, parts[1], parts[0], InputGateMode::Silu);
        });
    auto gate = std::move(parts[0]);
    auto up = std::move(parts[1]);
    DenseTensor hidden;
    if (geglu)
        hidden = profiler.measure("ffn.geglu", [&] { return gelu_mul_cuda(gate.contiguous(), up.contiguous()); });
    else if (limit > 0.0)
        hidden = profiler.measure("ffn.swiglu_clamped", [&] {
            auto g = ::tyloi::cuda::clamp_max(gate.to(::tyloi::kFloat32), limit);
            auto u = ::tyloi::cuda::clamp(up.to(::tyloi::kFloat32), -limit, limit);
            return (::tyloi::cuda::silu(g) * u).to(::tyloi::kFloat16).contiguous();
        });
    else
        hidden = profiler.measure("ffn.swiglu", [&] { return (up * ::tyloi::cuda::silu(gate)).contiguous(); });
    gate = {};
    up = {};
    return profiler.measure("ffn.down", [&] { return project.run(down, hidden); });
}

::tyloi::DenseTensor Mlp::bf16(
        const Projector& project, CudaProfiler& profiler, bool fused, DenseTensor input) const {
    auto parts = profiler.measure("full.minicpmo45_ffn_gate_up", [&] {
        return project.run(gate_up, input);
    });
    input = {};
    if (parts.size() != 2)
        throw std::runtime_error("gated MLP requires gate and up projections");
    auto gate = std::move(parts[0]);
    auto up = std::move(parts[1]);
    gate = gate.to(::tyloi::kBFloat16).contiguous();
    up = up.to(::tyloi::kBFloat16).contiguous();
    auto hidden = profiler.measure("full.minicpmo45_ffn_swiglu", [&] {
        return fused ? silu_mul_cuda(gate, up)
                     : (::tyloi::cuda::silu(gate) * up).contiguous();
    });
    gate = {};
    up = {};
    return profiler.measure("full.minicpmo45_ffn_down", [&] {
        return project.run(down, hidden).to(::tyloi::kBFloat16).contiguous();
    });
}

::tyloi::DenseTensor Mlp::kld(
        const Projector& project, CudaProfiler& profiler, KlMmqState& mmq,
        ::tyloi::DenseTensor xh) const {
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

bool Mlp::can_fuse(const DenseTensor& x, const DenseTensor& residual) const {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (activation != ActivationKind::silu_gated || geglu || swiglu_limit > 0.0 ||
        parallel() || !x.is_cuda() || !residual.is_cuda() ||
        x.scalar_type() != ::tyloi::kFloat16 ||
        residual.scalar_type() != ::tyloi::kFloat16 || !x.is_contiguous() ||
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

DenseTensor Mlp::fused(CudaProfiler& profiler, DenseTensor x, DenseTensor residual) const {
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

} // namespace tyloi::cuda
