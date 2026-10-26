#include "run.h"

#include "native/branches.h"
#include "native/collectives.h"
#include "ops/norm/rms.h"
#include "engine/executor/options.h"

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
            write(op.output);
        }, task.op);
    }
    read(inputs.output);
    std::vector<bool> needed(ready.size(), false);
    needed[inputs.output] = true;
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
            write(op.output);
            if constexpr (std::is_same_v<Op, task::Attention>) write(op.residual);
            if constexpr (std::is_same_v<Op, task::Concat>) {
                read(op.first);
                read(op.second);
            } else {
                read(op.input);
            }
            if constexpr (std::is_same_v<Op, task::Dense>) read(op.residual);
        }, tasks[index].op);
    }
    return result;
}

} // namespace

Tensor Executor::run(std::span<const Task> tasks, Inputs inputs) {
    const auto release = releases(tasks, inputs);
    auto options = options_;
    options.projection.primary = primary_;
    options.projection.moe = runtime_options::moe(options.projection.moe);
    options.projection.nint_group_quantize =
        !serial_ || runtime_options::nint_group_quantize().value_or(true);
    options.dense.mlp.nvq_fusion = options.projection.nvq_fusion;
    options.dense.bf16_norm = options.attention.fused_norm;
    if (scratch_.size() != tasks.size()) scratch_.resize(tasks.size());
    const Projector project(profiler_, mmq_, tq_, branches_, collectives_,
        serial_, projection_major_, options.projection);
    for (size_t index = 0; index < tasks.size(); ++index) {
        const auto& task = tasks[index];
        auto* trace = task.scope >= 0 && task.scope == inputs.trace_scope ? inputs.trace : nullptr;
        std::visit([&](const auto& op) {
            using Op = std::decay_t<decltype(op)>;
            auto& values = inputs.values;
            Tensor input, second;
            if constexpr (std::is_same_v<Op, task::Concat>) {
                input = values[op.first];
                second = values[op.second];
            } else {
                input = values[op.input];
            }
            if constexpr (std::is_same_v<Op, task::Dense>) second = values[op.residual];
            for (auto slot : release[index]) values[slot] = {};
            if constexpr (std::is_same_v<Op, task::Norm>) {
                values[op.output] = qwen_rms_norm(input.reshape({-1, input.size(-1)}).to(tb::kFloat32),
                    op.weight, op.eps, op.offset).reshape(input.sizes());
            } else if constexpr (std::is_same_v<Op, task::Concat>) {
                values[op.output] = tb::cat({input, second}, op.axis);
            } else if constexpr (std::is_same_v<Op, task::Project>) {
                values[op.output] = project.run(op.weight, std::move(input));
            } else if constexpr (std::is_same_v<Op, task::Attention>) {
                auto output = op.weights.forward(project, profiler_, options.attention,
                    std::move(input), inputs.attention, inputs.kv[op.state], scratch_[index],
                    *inputs.rope, trace);
                values[op.residual] = std::move(output[0]);
                values[op.output] = std::move(output[1]);
            } else if constexpr (std::is_same_v<Op, task::Dense>) {
                values[op.output] = op.weights.forward(project, profiler_, mmq_, tq_, branches_, collectives_,
                    options.projection, options.dense, serial_, std::move(second), std::move(input), trace);
            }
            record(trace, task.trace, values[op.output]);
        }, task.op);
    }
    return std::move(inputs.values[inputs.output]);
}

} // namespace mfq::cuda
