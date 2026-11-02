#include "executor.h"

#include "engine/backends/cuda/native/branches.h"
#include "engine/backends/cuda/native/collectives.h"
#include "engine/backends/cuda/native/context.h"
#include "engine/backends/cuda/kernels/mfq_cuda_sampling_ops.h"
#include "engine/backends/cuda/kernels/gr_prefill_blas.h"
#include "engine/backends/cuda/ops/norm/rms.h"

#include <iostream>
#include <type_traits>

namespace mfq::cuda {

namespace tb = mfq_tensor_backend;
using Tensor = tb::Tensor;

namespace {

void record(Trace* trace, std::string_view name, const Tensor& value) {
    if (trace && !name.empty())
        trace->emplace_back(std::string(name), value.to(tb::kFloat32).contiguous().clone());
}

std::vector<std::vector<size_t>> releases(std::span<const Task> tasks, const Inputs& inputs) {
    std::vector<bool> ready;
    for (const auto& value : inputs.values) ready.push_back(value.defined());
    const auto read = [&](size_t slot) {
        TYLOINFER_RUNTIME_CHECK(slot < ready.size() && ready[slot], "task input is unavailable");
    };
    const auto write = [&](size_t slot) {
        TYLOINFER_RUNTIME_CHECK(slot < ready.size(), "task output slot is out of range");
        ready[slot] = true;
    };
    for (const auto& task : tasks) {
        std::visit([&](const auto& op) {
            using Op = std::decay_t<decltype(op)>;
            if constexpr (std::is_same_v<Op, task::Replay>) {
                TYLOINFER_RUNTIME_CHECK(op.state < inputs.linear.size() &&
                    inputs.linear[op.state].captured != nullptr,
                    "replay task inputs are incomplete");
            } else {
                if constexpr (std::is_same_v<Op, task::Concat>) {
                    read(op.first);
                    read(op.second);
                } else {
                    read(op.input);
                }
                if constexpr (std::is_same_v<Op, task::Dense>) read(op.residual);
                if constexpr (std::is_same_v<Op, task::Attention>) {
                    TYLOINFER_RUNTIME_CHECK(inputs.rope && op.state < inputs.kv.size() &&
                        op.residual != op.output, "attention task inputs are incomplete");
                    write(op.residual);
                }
                if constexpr (std::is_same_v<Op, task::Linear>) {
                    TYLOINFER_RUNTIME_CHECK(op.state < inputs.linear.size() &&
                        op.residual != op.output, "linear task inputs are incomplete");
                    write(op.residual);
                }
                if constexpr (std::is_same_v<Op, task::FullFfn> ||
                              std::is_same_v<Op, task::LinearFfn>) read(op.residual);
                write(op.output);
            }
        }, task.op);
    }
    if (inputs.output) read(*inputs.output);
    std::vector<bool> needed(ready.size(), false);
    if (inputs.output) needed[*inputs.output] = true;
    std::vector<std::vector<size_t>> result(tasks.size());
    for (size_t index = tasks.size(); index-- > 0;) {
        const auto write = [&](size_t slot) {
            needed[slot] = false;
            result[index].push_back(slot);
        };
        const auto read = [&](size_t slot) {
            if (!needed[slot]) result[index].push_back(slot);
            needed[slot] = true;
        };
        std::visit([&](const auto& op) {
            using Op = std::decay_t<decltype(op)>;
            if constexpr (std::is_same_v<Op, task::Replay>) {
                return;
            } else {
                write(op.output);
                if constexpr (std::is_same_v<Op, task::Attention> ||
                              std::is_same_v<Op, task::Linear>) write(op.residual);
                if constexpr (std::is_same_v<Op, task::Concat>) {
                    read(op.first);
                    read(op.second);
                } else {
                    read(op.input);
                }
                if constexpr (std::is_same_v<Op, task::Dense> ||
                              std::is_same_v<Op, task::FullFfn> ||
                              std::is_same_v<Op, task::LinearFfn>) read(op.residual);
            }
        }, tasks[index].op);
    }
    return result;
}

} // namespace

std::unique_ptr<Executor> Executor::fork() const {
    return std::make_unique<Executor>(profiler_, mmq_, tq_, branches_, collectives_, routes_,
                                      options_, ffn_options_, primary_);
}

Projector Executor::projector(const Ffn::Mode& mode) {
    return {profiler_, mmq_, tq_, branches_, collectives_, options_.projection,
            primary_, mode.serial, mode.projection_major};
}

AttentionScratch& Executor::attention_scratch(std::size_t index) {
    if (scratch_.size() <= index) scratch_.resize(index + 1);
    return scratch_[index];
}

Tensor Executor::run(std::span<const Task> tasks, Inputs inputs, const Mode& mode) {
    const auto release = releases(tasks, inputs);
    auto options = options_;
    options.dense.mlp.nvq_fusion = options.projection.nvq_fusion;
    options.dense.bf16_norm = options.attention.fused_norm;
    const auto scratch_size = inputs.scratch_offset + tasks.size();
    if (scratch_.size() < scratch_size) scratch_.resize(scratch_size);
    const auto project = projector(mode.ffn);
    for (size_t index = 0; index < tasks.size(); ++index) {
        const auto& task = tasks[index];
        std::visit([&](const auto& op) {
            using Op = std::decay_t<decltype(op)>;
            if constexpr (std::is_same_v<Op, task::Replay>) {
                const auto& view = inputs.linear[op.state];
                op.weights.replay(view.state, *view.captured, op.tokens);
            } else {
                auto* trace = task.scope >= 0 && task.scope == inputs.trace_scope ? inputs.trace : nullptr;
                auto& values = inputs.values;
                Tensor input, second;
                if constexpr (std::is_same_v<Op, task::Concat>) {
                    input = values[op.first];
                    second = values[op.second];
                } else {
                    input = values[op.input];
                }
                if constexpr (std::is_same_v<Op, task::Dense> ||
                              std::is_same_v<Op, task::FullFfn> ||
                              std::is_same_v<Op, task::LinearFfn>) second = values[op.residual];
                for (auto slot : release[index]) values[slot] = {};
                auto compute = [&]() -> Tensor {
                    if constexpr (std::is_same_v<Op, task::Norm>) {
                        return qwen_rms_norm(input.reshape({-1, input.size(-1)}).to(tb::kFloat32),
                            op.weight, op.eps, op.offset).reshape(input.sizes());
                    } else if constexpr (std::is_same_v<Op, task::Embed>) {
                        return quant_embedding_lookup(op.weight, std::move(input));
                    } else if constexpr (std::is_same_v<Op, task::Softcap>) {
                        return op.cap > 0.0 ? tb::tanh(input / op.cap) * op.cap : std::move(input);
                    } else if constexpr (std::is_same_v<Op, task::Concat>) {
                        return tb::cat({input, second}, op.axis);
                    } else if constexpr (std::is_same_v<Op, task::Project>) {
                        return project.run(op.weight, std::move(input));
                    } else if constexpr (std::is_same_v<Op, task::Attention>) {
                        auto output = op.weights.forward(project, profiler_, options.attention,
                            std::move(input), inputs.attention, inputs.kv[op.state],
                            scratch_[inputs.scratch_offset + index],
                            *inputs.rope, trace);
                        values[op.residual] = std::move(output[0]);
                        return std::move(output[1]);
                    } else if constexpr (std::is_same_v<Op, task::Linear>) {
                        auto output = op.weights.forward(project, profiler_, inputs.linear_options,
                            std::move(input), inputs.linear[op.state].state,
                            inputs.linear[op.state].captured);
                        values[op.residual] = std::move(output[0]);
                        return std::move(output[1]);
                    } else if constexpr (std::is_same_v<Op, task::Dense>) {
                        return op.weights.forward(project, profiler_, mmq_, tq_, branches_, collectives_,
                            options.projection, options.dense, mode.ffn.serial, std::move(second), std::move(input), trace);
                    } else if constexpr (std::is_same_v<Op, task::FullFfn>) {
                        return full_ffn(op.weights, op.norm, op.attention,
                            std::move(second), std::move(input), mode.ffn, trace);
                    } else if constexpr (std::is_same_v<Op, task::LinearFfn>) {
                        return linear_ffn(op.weights, op.norm, op.attention,
                            std::move(second), std::move(input), mode.ffn);
                    }
                };
                values[op.output] = task.profile.empty()
                    ? compute() : profiler_.measure(std::string(task.profile), compute);
                record(trace, task.trace, values[op.output]);
            }
        }, task.op);
    }
    return inputs.output ? std::move(inputs.values[*inputs.output]) : Tensor{};
}

void Executor::prepare(std::int64_t vocabulary) {
    namespace tb=mfq_tensor_backend;
    const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    const auto before=context->local_memory_usage();
    {
        const auto gpu=tb::TensorOptions().device(tb::kCUDA);
        auto random=tb::full({1},.37,gpu.dtype(tb::kFloat32));
        auto counts=tb::zeros({vocabulary},gpu.dtype(tb::kInt32));
        for(auto dtype:{tb::kFloat16,tb::kFloat32}) {
            auto logits=tb::zeros({1,vocabulary},gpu.dtype(dtype));
            // Lazy CUDA modules and sampler workspaces must be charged before
            // expert arenas consume the remaining device budget. Exercise both
            // compact and fallback sampling without touching request RNG/state.
            for(int k:{20,64,65})if(k<=vocabulary)
                (void)sample_top_k_top_p_cuda(logits,random,.7,k,.8);
            (void)sample_softmax_cuda(logits,random,.7);
            (void)sample_apply_penalties_cuda(logits,counts,.2,.05,1.0);
            if(vocabulary>=64)(void)sample_top_k_compact_cuda(logits,64);
        }
#if TYLOINFER_ENABLE_MFQ_COMPAT
        prepare_gr_prefill_blas_runtime();
#endif
    }
    context->reclaim_idle_memory();
    std::cerr<<"runtime_sampling_prepare vocab="<<vocabulary<<" local_before="<<before
        <<" local_after="<<context->local_memory_usage()<<"\n";
}

} // namespace mfq::cuda
