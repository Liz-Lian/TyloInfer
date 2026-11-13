#include "dense.h"


#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/kernels/mfq_cuda_norm_ops.h"
#include "engine/backends/cuda/native/branches.h"
#include "engine/backends/cuda/native/collectives.h"

#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tyloi::cuda::ops {
namespace {

Mlp make_mlp(const Projector& project, std::span<const ::tyloi::Tensor> gate_up,
        const ::tyloi::Tensor& down, tyloi::ops::MlpActivation activation,
        double swiglu_limit) {
    return {project.group(gate_up), down, activation,
            activation == tyloi::ops::MlpActivation::gelu_gated, swiglu_limit};
}

DenseTensor run_mlp(const Mlp& op, DenseTensor input, Executor& executor,
        const Projector& project) {
    auto& profiler = executor.profiler();
    const auto& options = executor.options();
    const auto& projection = options.projection;
    if (op.parallel())
        return profiler.measure("ffn.tensor_parallel", [&] {
            return op.sharded(profiler, executor.mmq(), *executor.tq(), executor.collectives(),
                project.primary_device(), projection.peer_first, projection.reduce_to_primary,
                projection.fp16_reduce, std::move(input));
        });
    auto mlp_options = options.dense.mlp;
    mlp_options.nvq_fusion = projection.nvq_fusion;
    return op.forward(project, profiler, mlp_options, std::move(input));
}

} // namespace

DenseTensor mlp(DenseTensor input, std::span<const ::tyloi::Tensor> gate_up,
        const ::tyloi::Tensor& down, tyloi::ops::MlpActivation activation,
        double swiglu_limit) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    auto& executor = Executor::current();
    const auto project = executor.projector(Executor::current_ffn());
    const auto weights = make_mlp(project, gate_up, down, activation, swiglu_limit);
    weights.check(input, project.primary_device());
    if (input.scalar_type() != ::tyloi::kFloat16)
        input = executor.profiler().measure("ffn.input_cast", [&] {
            return input.to(::tyloi::kFloat16);
        });
    return run_mlp(weights, std::move(input), executor, project);
}

DenseTensor bf16_mlp(DenseTensor input, std::span<const ::tyloi::Tensor> gate_up,
        const ::tyloi::Tensor& down, tyloi::ops::MlpActivation activation,
        double swiglu_limit) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    auto& executor = Executor::current();
    const auto project = executor.projector(Executor::current_ffn());
    const auto weights = make_mlp(project, gate_up, down, activation, swiglu_limit);
    weights.check(input, project.primary_device());
    return weights.bf16(project, executor.profiler(), executor.options().dense.bf16_swiglu,
                        std::move(input)).to(::tyloi::kBFloat16).contiguous();
}

DenseTensor additive_mlp(DenseTensor input, std::span<const ::tyloi::Tensor> gate_up,
        const ::tyloi::Tensor& down, tyloi::ops::MlpActivation activation,
        double swiglu_limit, std::span<const ::tyloi::Tensor> extra_gate_up,
        const ::tyloi::Tensor& extra_down, tyloi::ops::MlpActivation extra_activation,
        double extra_limit) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    auto& executor = Executor::current();
    const auto project = executor.projector(Executor::current_ffn());
    const auto weights = make_mlp(project, gate_up, down, activation, swiglu_limit);
    const auto extra_weights = make_mlp(project, extra_gate_up, extra_down,
                                        extra_activation, extra_limit);
    weights.check(input, project.primary_device());
    extra_weights.check(input, project.primary_device());

    auto& profiler = executor.profiler();
    if (input.scalar_type() != ::tyloi::kFloat16)
        input = profiler.measure("ffn.input_cast", [&] { return input.to(::tyloi::kFloat16); });
    const auto rows = input.numel() / input.size(-1);
    const auto& options = executor.options();
    const bool fp32 = rows >= 16 && executor.mmq().mode == KlMmqMode::Fp16 &&
        options.dense.important_fp32;
    auto run_branch = [&](size_t index) {
        const auto& op = index == 0 ? weights : extra_weights;
        return fp32 ? op.kld(project, profiler, executor.mmq(), input)
                    : run_mlp(op, input, executor, project);
    };
    std::vector<DenseTensor> outputs;
    const auto& projection = options.projection;
    const bool ran_parallel = !fp32 && decode_branches(projection.parallel,
        Executor::current_ffn().serial, rows) && options.dense.important_parallel &&
        executor.branches().run(2, run_branch, outputs);
    if (!ran_parallel) {
        outputs.push_back(run_branch(0));
        outputs.push_back(run_branch(1));
    }
    if (outputs.size() != 2)
        throw std::runtime_error("additive FFN requires two branch outputs");
    auto first = std::move(outputs[0]);
    auto second = std::move(outputs[1]);
    if (fp32)
        return profiler.measure("ffn.in.combine.fp32", [&] {
            return (first + second).to(::tyloi::kFloat16).contiguous();
        });
    return profiler.measure("ffn.in.combine", [&] {
        return acc_cuda(first.contiguous(), second.contiguous());
    });
}

std::optional<DenseTensor> mlp_residual(DenseTensor input, DenseTensor residual,
        std::span<const ::tyloi::Tensor> gate_up, const ::tyloi::Tensor& down,
        tyloi::ops::MlpActivation activation, double swiglu_limit, bool recurrent) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    auto& executor = Executor::current();
    const auto project = executor.projector(Executor::current_ffn());
    const auto weights = make_mlp(project, gate_up, down, activation, swiglu_limit);
    const auto& options = executor.options();
    if (options.dense.fp32_residual || !options.projection.nvq_fusion ||
        !weights.can_fuse(input, residual))
        return std::nullopt;
    return executor.profiler().measure(
        recurrent ? "linear.ffn_down_residual" : "full.ffn_down_residual",
        [&] { return weights.fused(executor.profiler(), std::move(input), std::move(residual)); });
}

} // namespace tyloi::cuda::ops
