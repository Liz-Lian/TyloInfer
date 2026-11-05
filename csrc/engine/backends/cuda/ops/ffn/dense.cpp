#include "dense.h"
#include "engine/ops.h"
#include "residual.h"
#include "engine/backends/cuda/native/branches.h"
#include "engine/backends/cuda/native/collectives.h"
#include "engine/backends/cuda/kernels/mfq_cuda_norm_ops.h"

namespace tyloi::cuda {


namespace {
void record(Trace* trace, const char* name, const DenseTensor& value) {
    if (trace) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        trace->emplace_back(name, std::make_shared<DenseTensor>(value.to(::tyloi::kFloat32).contiguous().clone()));
#else
        trace->emplace_back(name, value.to(::tyloi::kFloat32).contiguous().clone());
#endif
    }
}
} // namespace

static DenseTensor run_mlp(const Mlp& op, DenseTensor input, const Projector& project,
        CudaProfiler& profiler, KlMmqState& mmq, TqExec& tq, Collectives& collectives,
        const Projector::Options& parallel, const Dense::Options& options) {
    if (op.parallel())
        return profiler.measure("ffn.tensor_parallel", [&] {
            return op.sharded(profiler, mmq, tq, collectives, project.primary_device(),
                parallel.peer_first, parallel.reduce_to_primary,
                parallel.fp16_reduce, input);
        });
    return op.forward(project, profiler, options.mlp, std::move(input));
}

DenseTensor Dense::forward(const Projector& project, CudaProfiler& profiler,
        KlMmqState& mmq, TqExec& tq, Branches& branches, Collectives& collectives,
        const Projector::Options& parallel, const Options& options, bool serial,
        DenseTensor residual, DenseTensor input, Trace* trace_output) const {
    const auto batch = residual.size(0), tokens = residual.size(1), width = residual.size(2);
    auto pair = linear
        ? linear_norm(profiler, norm, eps, offset, options.fp32_residual, std::move(residual), std::move(input))
        : full_norm(profiler, norm, eps, offset, bf16, options.bf16_norm,
            options.fp32_residual, std::move(residual), std::move(input));
    residual = pair[0];
    input = pair[1].reshape({batch * tokens, width});
    if (trace) record(trace_output, "qwen.ffn_norm", input.reshape({batch, tokens, width}));
    if (!bf16 && !important && !options.fp32_residual &&
        options.mlp.nvq_fusion && mlp.can_fuse(input, residual.reshape({batch * tokens, width}))) {
        return profiler.measure(linear ? "linear.ffn_down_residual" : "full.ffn_down_residual", [&] {
            return mlp.fused(profiler, input, residual.reshape({batch * tokens, width}))
                .reshape({batch, tokens, width});
        });
    }
    DenseTensor output;
    if (bf16) {
        output = mlp.bf16(project, profiler, options.bf16_swiglu, input)
            .to(::tyloi::kBFloat16).contiguous();
    } else {
        if (input.scalar_type() != ::tyloi::kFloat16)
            input = profiler.measure("ffn.input_cast", [&] { return input.to(::tyloi::kFloat16); });
        if (important) {
            const bool fp32 = batch * tokens >= 16 && mmq.mode == KlMmqMode::Fp16 &&
                options.important_fp32;
            auto run_branch = [&](size_t index) {
                    const auto& op = index == 0 ? mlp : *important;
                    return fp32 ? op.kld(project, profiler, mmq, input)
                                : run_mlp(op, input, project, profiler, mmq, tq, collectives, parallel, options);
            };
            std::vector<DenseTensor> outputs;
            const bool ran_parallel = !fp32 && decode_branches(parallel.parallel, serial, batch * tokens) &&
                options.important_parallel && branches.run(2, run_branch, outputs);
            if (!ran_parallel) {
                outputs.push_back(run_branch(0));
                outputs.push_back(run_branch(1));
            }
            if (outputs.size() != 2)
                throw std::runtime_error("additive FFN requires two branch outputs");
            auto first = std::move(outputs[0]);
            auto second = std::move(outputs[1]);
            if (fp32)
                output = profiler.measure("ffn.in.combine.fp32", [&] {
                    return (first + second).to(::tyloi::kFloat16).contiguous();
                });
            else
                output = profiler.measure("ffn.in.combine", [&] {
                    return acc_cuda(first.contiguous(), second.contiguous());
                });
        } else {
            output = run_mlp(mlp, input, project, profiler, mmq, tq, collectives, parallel, options);
        }
    }
    output = output.reshape({batch, tokens, width});
    if (trace) record(trace_output, "qwen.ffn_output", output);
    return linear
        ? linear_add(profiler, options.fp32_residual, std::move(residual), std::move(output))
        : full_add(profiler, options.fp32_residual, options.bf16_residual,
            std::move(residual), std::move(output));
}

} // namespace tyloi::cuda
