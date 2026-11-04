#include "projection.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/backends/cuda/native/branches.h"
#include "engine/backends/cuda/native/collectives.h"
#include "engine/backends/cuda/executor/tq.h"
#include "engine/backends/cuda/kernels/mfq_cuda_activation_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_quant_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_nint_route_hint.h"

#include <algorithm>
#include <optional>
#include <utility>

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/ops/projection/bias.h"
#include "engine/ops.h"

tyloi::DenseTensor tyloi::cuda::ops::embedding(tyloi::DenseTensor ids, const tyloi::weights::Weights& weight) {
    std::optional<MfqCudaGuard> guard;
    if (ids.is_cuda()) guard.emplace(ids.device().index);
    return quant_embedding_lookup(as_quant_linear(weight), ids);
}

tyloi::DenseTensor tyloi::cuda::ops::project(tyloi::DenseTensor input, const tyloi::weights::Weights& weight) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    return tyloi::cuda::Executor::current().projector(tyloi::cuda::Executor::current_ffn()).run(
        as_quant_linear(weight), input);
}

tyloi::DenseTensor tyloi::cuda::ops::linear(tyloi::DenseTensor input, const tyloi::weights::Weights& weight,
                                  const tyloi::DenseTensor& bias) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    return biased(Executor::current().projector(Executor::current_ffn()), as_quant_linear(weight),
                  bias, input);
}

tyloi::DenseTensor tyloi::cuda::ops::affine(tyloi::DenseTensor input, const tyloi::weights::Weights& weight,
                                    const tyloi::DenseTensor& bias) {
    auto output = project(std::move(input), weight);
    if (!bias.defined()) return output;
    if (output.device() != bias.device())
        throw std::invalid_argument("tensor devices do not match");
    std::optional<MfqCudaGuard> guard;
    if (output.is_cuda()) guard.emplace(output.device().index);
    auto native = output;
    return native + bias.to(native.scalar_type());
}

std::vector<tyloi::DenseTensor> tyloi::cuda::ops::project_group(tyloi::DenseTensor input,
    const tyloi::weights::Weights& group) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    auto parts = tyloi::cuda::Executor::current().projector(tyloi::cuda::Executor::current_ffn()).run(
        as_quant_linear_group(group), input);
    if (parts.size() == group.size()) {
        for (std::size_t i = 0; i < parts.size(); ++i)
            if (parts[i].size(-1) != group.shape(i).rows)
                throw std::runtime_error("grouped projection output boundaries mismatch");
        return parts;
    }
    std::vector<tyloi::DenseTensor> result;
    result.reserve(group.size());
    std::size_t index = 0;
    for (const auto& part : parts) {
        int64_t offset = 0;
        while (offset < part.size(-1) && index < group.size()) {
            const auto width = group.shape(index).rows;
            if (width <= 0 || width > part.size(-1) - offset)
                throw std::runtime_error("grouped projection output boundaries mismatch");
            result.push_back(part.narrow(-1, offset, width));
            offset += width;
            ++index;
        }
        if (offset != part.size(-1))
            throw std::runtime_error("grouped projection output boundaries mismatch");
    }
    if (index != group.size())
        throw std::runtime_error("grouped projection output count mismatch");
    return result;
}

tyloi::DenseTensor tyloi::cuda::ops::sigmoid_project(tyloi::DenseTensor input, tyloi::DenseTensor gate,
    const tyloi::weights::Weights& weight) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    return tyloi::cuda::Executor::current().projector(tyloi::cuda::Executor::current_ffn()).gated(
        as_quant_linear(weight), input,
        gate, InputGateMode::Sigmoid);
}
#endif

::tyloi::DenseTensor run_quant_linear_shard(
        CudaProfiler& profiler,
        KlMmqState& kl_mmq,
        TqExec& tq,
        const QuantLinearShard & shard,
        ::tyloi::DenseTensor x,
        MfqOptional<::tyloi::DenseTensor> gate,
        InputGateMode gate_mode) {
    MfqCudaGuard guard(shard.device);
    if (shard.kind == QuantLinearKind::Tq) {
        if (gate.has_value()) {
            TYLOINFER_RUNTIME_CHECK(gate_mode == InputGateMode::Sigmoid || gate_mode == InputGateMode::Silu,
                "TQ input gate mode must be sigmoid or SiLU");
            auto local_gate = gate.value().to(x.scalar_type());
            x = x * (gate_mode == InputGateMode::Sigmoid
                ? ::tyloi::cuda::sigmoid(local_gate)
                : ::tyloi::cuda::silu(local_gate));
        }
        return profiler.measure("tq.tensor_parallel", [&] {
            const auto resources=tq.get(shard.device);
            return tyloi::cuda::tq::project(shard.tq,&resources->blas(),std::move(x));
        });
    }
    if ((tyloi::mfq_compat_enabled && shard.kind == QuantLinearKind::Mxfp4Sq) ||
            (tyloi::mfq_compat_enabled && shard.kind == QuantLinearKind::Fp8Sq)) {
        if (gate.has_value()) {
            TYLOINFER_RUNTIME_CHECK(
                gate_mode == InputGateMode::Sigmoid || gate_mode == InputGateMode::Silu,
                "SQ input gate mode must be sigmoid or SiLU");
        }
        if (shard.kind == QuantLinearKind::Mxfp4Sq)
            check_device(shard.mxfp4_sq, x);
        else
            check_device(shard.fp8_sq, x);
        if (gate.has_value()) {
            x = x * (gate_mode == InputGateMode::Sigmoid
                ? ::tyloi::cuda::sigmoid(gate.value())
                : ::tyloi::cuda::silu(gate.value()));
        }
        return (tyloi::mfq_compat_enabled && shard.kind == QuantLinearKind::Mxfp4Sq)
            ? tyloi::cuda::sq_matmul(shard.mxfp4_sq, x)
            : tyloi::cuda::sq_matmul(shard.fp8_sq, x);
    }
    if ((tyloi::mfq_compat_enabled && shard.kind == QuantLinearKind::Nint)) {
        if (!gate.has_value() || gate_mode == InputGateMode::Sigmoid || gate_mode == InputGateMode::Silu)
            check_device(shard.nint, x);
        return run_nint_linear(profiler, kl_mmq, shard.nint, x, gate, gate_mode);
    }
    if ((tyloi::mfq_compat_enabled && shard.kind == QuantLinearKind::Nvq)) {
        if (!gate.has_value() || gate_mode == InputGateMode::Sigmoid || gate_mode == InputGateMode::Silu)
            check_device(shard.nvq, x);
        return run_nvq_linear(profiler, kl_mmq, shard.nvq, x, gate, gate_mode);
    }
    if (shard.kind == QuantLinearKind::Mxfp4) {
        TYLOINFER_RUNTIME_CHECK(
            !gate.has_value(),
            "MXFP4 tensor-parallel linear does not support input gating");
        check_device(shard.mxfp4, x);
        return mxfp4_matmul(shard.mxfp4, x);
    }
    if (shard.kind == QuantLinearKind::Dense) {
        if (gate.has_value()) {
            TYLOINFER_RUNTIME_CHECK(
                gate_mode == InputGateMode::Sigmoid || gate_mode == InputGateMode::Silu,
                "dense input gate mode must be sigmoid or SiLU");
        }
        check_device(shard.dense, x);
        auto local = x.to(shard.dense.scalar_type());
        if (gate.has_value()) {
            auto local_gate = gate.value().to(local.scalar_type());
            local = gate_mode == InputGateMode::Sigmoid
                ? local * ::tyloi::cuda::sigmoid(local_gate)
                : local * ::tyloi::cuda::silu(local_gate);
        }
        return dense_projection(std::move(local), shard.dense);
    }
    TYLOINFER_RUNTIME_CHECK(
        !gate.has_value(),
        "MXFP8 tensor-parallel linear does not support input gating");
    check_device(shard.mxfp8, x);
    return mxfp8_matmul(profiler, shard.mxfp8, x);
}

namespace tyloi::cuda {

using ::tyloi::indexing::Slice;

::tyloi::DenseTensor Projector::low_rank(::tyloi::DenseTensor attention,
                                               const QuantLinear &output_a,
                                               const QuantLinear &output_b, int64_t groups,
                                               bool groupwise_output, bool profile) const {
    const auto batch = attention.size(0);
    const auto tokens = attention.size(1);
    const auto rows = batch * tokens;
    const auto rank = output_a.out() / groups;
    auto grouped = attention.contiguous()
                       .reshape({rows, groups, output_a.neuron_len()})
                       .to(::tyloi::kFloat16);
    const auto measure = [&](const char *name, auto &&operation) {
        return profile ? profiler_.measure(name, operation) : operation();
    };
    if (output_a.is_tq()) {
        auto low_rank = measure("dsv4.output_a", [&] {
            const auto resources = tq_.get(grouped.get_device());
            return tyloi::cuda::tq::groupwise(output_a.tq,&resources->blas(), grouped, groups);
        });
        return measure("dsv4.output_b", [&] {
            return run(output_b, low_rank).reshape({batch, tokens, output_b.out()});
        });
    }
    if (groupwise_output && output_a.is_nint() && output_a.nint.bits == 8 &&
        output_a.nint.gs == 48 && output_a.nint.out == groups * rank) {
        auto low_rank = measure("dsv4.output_a", [&]() {
            return nint_matmul_groupwise_u8(profiler_, output_a.nint, grouped, groups);
        });
        return measure("dsv4.output_b", [&]() {
            return run(output_b, low_rank).reshape({batch, tokens, output_b.out()});
        });
    }
    if (groupwise_output && output_a.is_mxfp8() && output_a.out() == groups * rank) {
        auto low_rank = measure("dsv4.output_a", [&]() {
            return groupwise(output_a, grouped, groups);
        });
        return measure("dsv4.output_b", [&]() {
            return run(output_b, low_rank).reshape({batch, tokens, output_b.out()});
        });
    }
    auto expanded = measure("dsv4.output_a", [&]() {
        return run(output_a, grouped.reshape({rows * groups, grouped.size(-1)}))
            .reshape({rows, groups, groups, rank});
    });
    std::vector<::tyloi::DenseTensor> diagonal;
    diagonal.reserve(static_cast<std::size_t>(groups));
    for (std::int64_t group = 0; group < groups; ++group) {
        diagonal.push_back(expanded.index({Slice(), group, group, Slice()}));
    }
    auto low_rank = ::tyloi::cuda::stack(diagonal, 1)
                        .reshape({rows, output_a.out()})
                        .to(::tyloi::kFloat16)
                        .contiguous();
    return measure("dsv4.output_b", [&]() {
        return run(output_b, low_rank).reshape({batch, tokens, output_b.out()});
    });
}

std::vector<::tyloi::DenseTensor>
Projector::grouped(
        const QuantLinearProjectionRefs& projections,
        ::tyloi::DenseTensor x) const {
    TYLOINFER_RUNTIME_CHECK(
        tensor_parallel_output_projections_compatible(projections),
        "tensor-parallel output projections are incompatible");
    for (const auto* projection : projections) check_projection(*projection, x);
    auto output_shape = x.sizes().vec();
    auto flat = x.reshape({-1, x.size(-1)});
    const size_t shard_count =
        projections.front()->tensor_parallel_shards.size();
    if (projection_major_) {
        std::vector<::tyloi::DenseTensor> local_inputs(shard_count);
        for (size_t launch_position = 0;
             launch_position < shard_count; ++launch_position) {
            const size_t shard = parallel_index(
                options_.peer_first, launch_position, shard_count);
            const int device =
                projections.front()->tensor_parallel_shards[shard].device;
            MfqCudaGuard guard(device);
            local_inputs[shard] = collectives_.copy(flat, device);
        }

        const int primary = primary_;
        std::vector<::tyloi::DenseTensor> result;
        result.reserve(projections.size());
        for (const auto * projection : projections) {
            std::vector<::tyloi::DenseTensor> local_outputs(shard_count);
            for (size_t launch_position = 0;
                 launch_position < shard_count; ++launch_position) {
                const size_t shard = parallel_index(
                    options_.peer_first, launch_position, shard_count);
                const auto & weight =
                    projection->tensor_parallel_shards[shard];
                MfqCudaGuard guard(weight.device);
                local_outputs[shard] = run_quant_linear_shard(
                    profiler_, mmq_, tq_,
                    weight, local_inputs[shard]);
            }
            MfqCudaGuard primary_guard(primary);
            std::vector<::tyloi::DenseTensor> gathered;
            gathered.reserve(shard_count);
            for (auto & output : local_outputs) {
                gathered.push_back(collectives_.copy(output, primary));
            }
            auto combined = ::tyloi::cuda::cat(
                gathered, -1).contiguous();
            auto shape = output_shape;
            shape.back() = combined.size(-1);
            result.push_back(combined.reshape(shape));
        }
        return result;
    }
    std::vector<std::vector<::tyloi::DenseTensor>>
        local_outputs(projections.size());
    for (auto & outputs : local_outputs) {
        outputs.resize(shard_count);
    }
    for (size_t launch_position = 0;
         launch_position < shard_count; ++launch_position) {
        const size_t shard = parallel_index(
            options_.peer_first, launch_position, shard_count);
        const int device =
            projections.front()->tensor_parallel_shards[shard].device;
        MfqCudaGuard guard(device);
        // All projections consume the same immutable activation. Transfer it
        // once per rank, then keep the independent local projection work on
        // that rank before gathering each output.
        auto local_x = collectives_.copy(flat, device);
        for (size_t projection = 0;
                projection < projections.size(); ++projection) {
            local_outputs[projection][shard] =
                run_quant_linear_shard(
                    profiler_, mmq_, tq_,
                    projections[projection]
                        ->tensor_parallel_shards[shard], local_x);
        }
    }

    const int primary = primary_;
    MfqCudaGuard primary_guard(primary);
    std::vector<::tyloi::DenseTensor> result;
    result.reserve(projections.size());
    for (auto & projection_outputs : local_outputs) {
        std::vector<::tyloi::DenseTensor> gathered;
        gathered.reserve(shard_count);
        for (auto & output : projection_outputs) {
            gathered.push_back(collectives_.copy(output, primary));
        }
        auto combined = ::tyloi::cuda::cat(
            gathered, -1).contiguous();
        auto shape = output_shape;
        shape.back() = combined.size(-1);
        result.push_back(combined.reshape(shape));
    }
    return result;
}

::tyloi::DenseTensor Projector::sharded(
        const QuantLinear& weight,
        ::tyloi::DenseTensor x,
        MfqOptional<::tyloi::DenseTensor> gate,
        InputGateMode gate_mode) const {
    TYLOINFER_RUNTIME_CHECK(
        weight.tensor_parallel(),
        "tensor-parallel linear has no shards");
    TYLOINFER_RUNTIME_CHECK(
        weight.tensor_parallel_axis == TensorParallelAxis::Output ||
        weight.tensor_parallel_axis == TensorParallelAxis::Input,
        "tensor-parallel linear has an invalid axis");
    std::vector<::tyloi::DenseTensor> local_outputs(
        weight.tensor_parallel_shards.size());
    for (size_t launch_position = 0;
         launch_position < weight.tensor_parallel_shards.size();
         ++launch_position) {
        const size_t index = parallel_index(
            options_.peer_first, launch_position, weight.tensor_parallel_shards.size());
        const auto & shard = weight.tensor_parallel_shards[index];
        MfqCudaGuard guard(shard.device);
        ::tyloi::DenseTensor local_x = x;
        ::tyloi::DenseTensor local_gate;
        if (weight.tensor_parallel_axis == TensorParallelAxis::Input) {
            local_x = x.narrow(
                -1, shard.input_begin,
                shard.input_end - shard.input_begin);
            if (gate.has_value()) {
                local_gate = gate.value().narrow(
                    -1, shard.input_begin,
                    shard.input_end - shard.input_begin);
            }
        } else if (gate.has_value()) {
            local_gate = gate.value();
        }
        local_x = collectives_.copy(local_x, shard.device);
        if (gate.has_value()) {
            local_gate =
                collectives_.copy(local_gate, shard.device);
        }
        if (weight.is_mxfp8() &&
                weight.tensor_parallel_axis == TensorParallelAxis::Input) {
            TYLOINFER_RUNTIME_CHECK(
                !gate.has_value(),
                "MXFP8 input-axis tensor parallelism does not support gating");
            check_device(shard.mxfp8, local_x);
            local_outputs[index] =
                mxfp8_matmul_f32(
                    profiler_, shard.mxfp8, local_x);
        } else {
            local_outputs[index] =
                run_quant_linear_shard(
                    profiler_, mmq_, tq_,
                    shard, local_x,
                    gate.has_value()
                        ? MfqOptional<::tyloi::DenseTensor>(
                            local_gate)
                        : mfq_nullopt,
                    gate_mode);
        }
    }

    const int primary = primary_;
    MfqCudaGuard primary_guard(primary);
    if (weight.tensor_parallel_axis == TensorParallelAxis::Output) {
        std::vector<::tyloi::DenseTensor> gathered;
        gathered.reserve(local_outputs.size());
        for (auto & output : local_outputs) {
            gathered.push_back(
                collectives_.copy(output, primary));
        }
        return ::tyloi::cuda::cat(gathered, -1).contiguous();
    }

    auto reduced = collectives_.sum(std::move(local_outputs), primary_,
            options_.reduce_to_primary, options_.fp16_reduce);
    return weight.is_mxfp8()
        ? reduced.to(x.scalar_type()).contiguous()
        : reduced;
}

::tyloi::DenseTensor Projector::run(
        const QuantLinear& weight,
        ::tyloi::DenseTensor x) const {
    check_projection(weight, x);
    if(weight.is_tq() && !weight.tensor_parallel())return profiler_.measure("tq.linear",[&] {
        const auto resources=tq_.get(x.get_device());
        return tyloi::cuda::tq::project(weight.tq,&resources->blas(),std::move(x));
    });
    if (weight.tensor_parallel() || (tyloi::mfq_compat_enabled && weight.is_nint()) || (tyloi::mfq_compat_enabled && weight.is_nvq())) {
        auto shape = x.sizes().vec();
        auto flat = x.reshape({-1, x.size(-1)});
#if TYLOINFER_ENABLE_MFQ_COMPAT
        auto y = weight.tensor_parallel()
            ? sharded(weight, flat, mfq_nullopt, InputGateMode::None)
            : (tyloi::mfq_compat_enabled && weight.is_nint()) ? run_nint_linear(
                  profiler_, mmq_, weight.nint, flat)
                        : run_nvq_linear(
                  profiler_, mmq_, weight.nvq, flat);
#else
        auto y = sharded(weight, flat, mfq_nullopt, InputGateMode::None);
#endif
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
    if (weight.is_mxfp4()) {
        auto shape = x.sizes().vec();
        auto y = mxfp4_matmul(
            weight.mxfp4, x.reshape({-1, x.size(-1)}));
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
    if ((tyloi::mfq_compat_enabled && weight.is_mxfp4_sq())) return sq_matmul(weight.mxfp4_sq, x);
    if ((tyloi::mfq_compat_enabled && weight.is_fp8_sq())) return sq_matmul(weight.fp8_sq, x);
    if (weight.is_dense()) return dense_projection(x, weight.dense);
    auto shape = x.sizes().vec();
    auto y = mxfp8_matmul(
        profiler_, weight.mxfp8, x.reshape({-1, x.size(-1)}));
    shape.back() = y.size(-1);
    return y.reshape(shape);
}

std::vector<::tyloi::DenseTensor> Projector::run(
        const Projections& projections,
        ::tyloi::DenseTensor x) const {
    for (const auto& weight : projections.weights) check_projection(weight, x);
    const auto join = [&projections](std::vector<::tyloi::DenseTensor> outputs) {
        if (projections.prefix <= 1 || outputs.empty()) return outputs;
        std::vector<::tyloi::DenseTensor> prefix(
            outputs.begin(), outputs.begin() + projections.prefix);
        auto combined = ::tyloi::cuda::cat(prefix, -1);
        outputs.erase(outputs.begin(), outputs.begin() + projections.prefix);
        outputs.insert(outputs.begin(), std::move(combined));
        return outputs;
    };
    const auto run_separate = [&] {
        std::vector<::tyloi::DenseTensor> outputs;
        outputs.reserve(projections.weights.size());
        for (const auto & weight : projections.weights) {
            outputs.push_back(run(weight, x));
        }
        return join(std::move(outputs));
    };

    if (projections.group.tq.defined() && x.is_cuda()) {
        auto grouped = run(projections.group, x);
        std::vector<::tyloi::DenseTensor> outputs(
            projections.weights.size());
        for (size_t i = 0, j = 0; i < projections.indices.size(); ++i) {
            outputs[projections.indices[i]] = std::move(grouped[j++]);
            if (projections.joined && i == 0) i += projections.prefix - 1;
        }
        for (size_t i = 0; i < projections.weights.size(); ++i) {
            if (projections.joined && i > 0 && i < projections.prefix) {
                continue;
            }
            if (!outputs[i].defined()) outputs[i] = run(projections.weights[i], x);
        }
        if (projections.joined) {
            outputs.erase(
                outputs.begin() + 1,
                outputs.begin() + projections.prefix);
            return outputs;
        }
        return join(std::move(outputs));
    }

    if (projections.group.nint_grouped && x.is_cuda() &&
            mmq_.mode == KlMmqMode::Default &&
            x.numel() / x.size(-1) <= 8) {
        auto outputs = run(projections.group, x);
        return projections.joined ? outputs : join(std::move(outputs));
    }
    return run_separate();
}

::tyloi::DenseTensor Projector::groupwise(
        const QuantLinear& weight,
        ::tyloi::DenseTensor grouped,
        int64_t groups) const {
    TYLOINFER_RUNTIME_CHECK(
        weight.is_mxfp8(),
        "groupwise MXFP8 projection requires an MXFP8 tensor");
    if (!weight.tensor_parallel()) {
        return mxfp8_groupwise_matmul(
            profiler_, weight.mxfp8, grouped, groups);
    }
    TYLOINFER_RUNTIME_CHECK(
        weight.tensor_parallel_axis == TensorParallelAxis::Input,
        "groupwise MXFP8 tensor parallelism requires input-axis shards");
    std::vector<::tyloi::DenseTensor> partials;
    partials.reserve(weight.tensor_parallel_shards.size());
    for (const auto & shard : weight.tensor_parallel_shards) {
        TYLOINFER_RUNTIME_CHECK(
            shard.kind == QuantLinearKind::Mxfp8,
            "groupwise MXFP8 tensor-parallel shard kind mismatch");
        MfqCudaGuard guard(shard.device);
        auto local = grouped.narrow(
            -1, shard.input_begin,
            shard.input_end - shard.input_begin);
        local = collectives_.copy(local, shard.device);
        partials.push_back(mxfp8_groupwise_matmul_f32(
            profiler_, shard.mxfp8, local, groups));
    }
    return collectives_.sum(std::move(partials), primary_,
            options_.reduce_to_primary, options_.fp16_reduce)
        .to(grouped.scalar_type()).contiguous();
}

::tyloi::DenseTensor Projector::gated(
        const QuantLinear& weight,
        ::tyloi::DenseTensor x,
        ::tyloi::DenseTensor gate,
        InputGateMode mode) const {
    if (!x.defined() || x.dim() < 1)
        throw std::invalid_argument("projection input must have a last dimension");
    check_projection_device(weight, x.device());
    auto width = x.size(-1);
    if ((!weight.tensor_parallel() || weight.tensor_parallel_axis == TensorParallelAxis::Output) &&
        !weight.is_nint() && !weight.is_nvq() &&
        gate.defined() && gate.dim() > 0 && width == 1)
        width = gate.size(-1);
    check_projection_width(weight, width);
    if (weight.tensor_parallel() || (tyloi::mfq_compat_enabled && weight.is_nint()) || (tyloi::mfq_compat_enabled && weight.is_nvq())) {
        auto shape = x.sizes().vec();
        auto flat = x.reshape({-1, x.size(-1)});
        auto flat_gate = gate.reshape({-1, gate.size(-1)});
#if TYLOINFER_ENABLE_MFQ_COMPAT
        auto y = weight.tensor_parallel()
            ? sharded(weight, flat, flat_gate, mode)
            : (tyloi::mfq_compat_enabled && weight.is_nint()) ? run_nint_linear(
                  profiler_, mmq_,
                  weight.nint, flat, flat_gate, mode)
                        : run_nvq_linear(
                  profiler_, mmq_,
                  weight.nvq, flat, flat_gate, mode);
#else
        auto y = sharded(weight, flat, flat_gate, mode);
#endif
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
    if (weight.is_dense()) {
        TYLOINFER_RUNTIME_CHECK(mode == InputGateMode::Sigmoid || mode == InputGateMode::Silu,
            "dense input gate mode must be sigmoid or SiLU");
        // Match the existing dense shard path, including dtype rounding.
        auto local = x.to(weight.dense.scalar_type());
        auto local_gate = gate.to(weight.dense.scalar_type());
        auto gated = mode == InputGateMode::Sigmoid
            ? local * ::tyloi::cuda::sigmoid(local_gate)
            : local * ::tyloi::cuda::silu(local_gate);
        return dense_projection(gated, weight.dense);
    }
    TYLOINFER_RUNTIME_CHECK(mode == InputGateMode::Sigmoid || mode == InputGateMode::Silu,
        "input gate mode must be sigmoid or SiLU");
    auto local_gate = gate.to(x.scalar_type());
    auto gated = mode == InputGateMode::Sigmoid
        ? x * ::tyloi::cuda::sigmoid(local_gate)
        : x * ::tyloi::cuda::silu(local_gate);
    return run(weight, gated);
}

std::vector<::tyloi::DenseTensor> Projector::run(
        const QuantLinearGroup& group,
        ::tyloi::DenseTensor x) const {
    check_projection(group, x);
    if(group.tq.defined())return profiler_.measure("tq.parallel",[&] {
        const auto resources=tq_.get(x.get_device());
        return group.tq.forward(&resources->blas(),std::move(x));
    });
    const bool default_mmq =
        mmq_.mode == KlMmqMode::Default;
    if (!x.is_cuda()) {
        TYLOINFER_RUNTIME_CHECK(
            !group.nint_grouped,
            "CPU dense offload requires separate compact linear weights");
        std::vector<::tyloi::DenseTensor> result;
        result.reserve(group.layers.size());
        for (const auto & layer : group.layers) {
            result.push_back(run(layer, x));
        }
        return result;
    }
    if (options_.grouped &&
            group.tensor_parallel_output_compatible()) {
        return grouped(group.tensor_parallel_output_projections(), x);
    }
    if (tyloi::mfq_compat_enabled && group.nint_grouped) {
        return run(group.nint, x);
    }
    if (tyloi::mfq_compat_enabled && default_mmq && group.nvq_prefix2 && options_.nvq_fusion) {
        if (nvq_pair_compatible(group.layers[0].nvq, group.layers[1].nvq)) {
            check_device(group.layers[0].nvq, x);
            check_device(group.layers[1].nvq, x);
        }
        auto shape = x.sizes().vec();
        const auto flat =
            x.reshape({-1, x.size(-1)});
        std::vector<::tyloi::DenseTensor> branch_outputs;
        const bool parallel =
            group.decode_branch_parallel &&
            decode_branches(
                options_.parallel, serial_, flat.size(0)) &&
            group.layers.size() > 2 &&
            branches_.run(
                group.layers.size() - 1,
                [&](size_t branch) {
                    if (branch == 0) {
                        return nvq_matmul_multi2(
                            profiler_, group.layers[0].nvq,
                            group.layers[1].nvq, flat);
                    }
                    return run(group.layers[branch + 1], x);
                },
                branch_outputs);
        auto combined = parallel
            ? branch_outputs[0]
            : nvq_matmul_multi2(
                profiler_, group.layers[0].nvq,
                group.layers[1].nvq, flat);
        auto pair = combined.split_with_sizes({group.outs[0], group.outs[1]}, -1);
        std::vector<::tyloi::DenseTensor> result;
        result.reserve(group.layers.size());
        for (size_t i = 0; i < 2; ++i) {
            auto part_shape = shape;
            part_shape.back() = group.outs[i];
            result.push_back(pair[i].reshape(part_shape));
        }
        for (size_t i = 2; i < group.layers.size(); ++i) {
            result.push_back(
                parallel
                    ? branch_outputs[i - 1]
                    : run(group.layers[i], x));
        }
        return result;
    }
    std::vector<::tyloi::DenseTensor> result;
    if (default_mmq && group.decode_branch_parallel &&
            decode_branches(
                options_.parallel, serial_, x.numel() / x.size(-1)) &&
            branches_.run(
                group.layers.size(),
                [&](size_t index) {
                    return run(group.layers[index], x);
                },
                result)) {
        return result;
    }
    result.reserve(group.layers.size());
    for (const auto & layer : group.layers) {
        result.push_back(run(layer, x));
    }
    return result;
}

::tyloi::DenseTensor Projector::swiglu(
        const QuantLinearGroup& group,
        ::tyloi::DenseTensor x) const {
    check_projection(group, x);
    if(group.tq.defined() && group.outs.size()==2 && group.outs[0]==group.outs[1])
        return profiler_.measure("tq.swiglu",[&] {
            const auto resources=tq_.get(x.get_device());
            return group.tq.swiglu(&resources->blas(),std::move(x));
        });
    const bool default_mmq =
        mmq_.mode == KlMmqMode::Default;
    if (tyloi::mfq_compat_enabled && default_mmq && group.nint_grouped && group.nint.split_w.empty() &&
            x.numel() / x.size(-1) >= 1 && x.numel() / x.size(-1) <= 6) {
        return swiglu(group.nint, x);
    }
    if (tyloi::mfq_compat_enabled && default_mmq && group.nvq_prefix2 && group.layers.size() == 2 &&
            options_.nvq_fusion) {
        auto shape = x.sizes().vec();
        auto y = nvq_matmul_swiglu(
            profiler_, group.layers[0].nvq, group.layers[1].nvq,
            x.reshape({-1, x.size(-1)}));
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
    if (group.outs.size() != 2 || group.outs[0] != group.outs[1]) {
        throw std::runtime_error("SwiGLU requires equal gate/up output widths");
    }
    auto parts = run(group, x);
    return ::tyloi::cuda::silu(parts[0]) * parts[1];
}

::tyloi::DenseTensor Projector::geglu(
        const QuantLinearGroup& group,
        ::tyloi::DenseTensor x) const {
    check_projection(group, x);
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (mmq_.mode == KlMmqMode::Default &&
            group.nint_grouped && group.nint.split_w.empty() &&
            x.numel() / x.size(-1) == 1) {
        return geglu(group.nint, x);
    }
#endif

    if (group.outs.size() != 2 || group.outs[0] != group.outs[1]) {
        throw std::runtime_error("GeGLU requires equal gate/up output widths");
    }
    auto parts = run(group, x);
    return gelu_mul_cuda(parts[0].contiguous(), parts[1].contiguous());
}

#if TYLOINFER_ENABLE_MFQ_COMPAT
std::vector<::tyloi::DenseTensor> Projector::run(
        const NintLinearGroup& group, ::tyloi::DenseTensor x) const {
    auto shape = x.sizes().vec();
    std::vector<::tyloi::DenseTensor> parts;
    auto xf = x.reshape({-1, x.size(-1)});
    if (group.w.q8_zero && xf.size(0) > 64 &&
            group.projection_w.size() == group.outs.size()) {
        for (const auto& weight : group.projection_w) check_device(weight, xf);
        parts.reserve(group.projection_w.size());
        for (const auto & projection : group.projection_w) {
            parts.push_back(run_nint_linear(
                profiler_, mmq_, projection, xf));
        }
    } else if (!group.split_w.empty()) {
        for (const auto& weight : group.split_w) check_device(weight, xf);
        parts.reserve(group.outs.size());
        std::vector<int> shared_groups;
        if(serial_ && runtime_options::nint_group_quantize().value_or(true) &&
                mmq_.mode==KlMmqMode::Default && xf.is_contiguous() &&
                xf.size(0)>=1 && xf.size(0)<=6 &&
                (xf.scalar_type()==::tyloi::kFloat16 || xf.scalar_type()==::tyloi::kFloat32)) {
            bool compatible=true;
            for(const auto& weight:group.split_w) {
                compatible=compatible && !weight.q8_zero && weight.neuron_len==xf.size(1) &&
                    (weight.gs==24 || weight.gs==28 || weight.gs==48);
                if(std::find(shared_groups.begin(),shared_groups.end(),weight.gs)==shared_groups.end())
                    shared_groups.push_back(weight.gs);
            }
            if(!compatible || shared_groups.size()<2)shared_groups.clear();
        }
        NintInputQuantizationScope prepared(xf,shared_groups);
        std::vector<::tyloi::DenseTensor> grouped_outputs;
        const bool parallel =
            decode_branches(
                options_.parallel, serial_, xf.size(0)) &&
            branches_.run(
                group.split_w.size(),
                [&](size_t index) {
                    return run_nint_linear(
                        profiler_, mmq_, group.split_w[index], xf);
                },
                grouped_outputs);
        for (size_t i = 0; i < group.split_w.size(); ++i) {
            auto y = parallel
                ? grouped_outputs[i]
                : run_nint_linear(
                      profiler_, mmq_, group.split_w[i], xf);
            auto ys = y.split_with_sizes(group.split_outs[i], -1);
            for (auto & p : ys) parts.push_back(p);
        }
    } else {
        check_device(group.w, xf);
        auto y = run_nint_linear(
            profiler_, mmq_,
            group.w, x.reshape({-1, x.size(-1)}));
        parts = y.split_with_sizes(group.outs, -1);
    }
    for (auto & p : parts) {
        auto s = shape;
        s.back() = p.size(-1);
        p = p.reshape(s);
    }
    return parts;
}

::tyloi::DenseTensor Projector::swiglu(
        const NintLinearGroup& group, ::tyloi::DenseTensor x) const {
    if (!group.split_w.empty() || group.outs.size() != 2 || group.outs[0] != group.outs[1]) {
        throw std::runtime_error("NINT SwiGLU fusion requires one packed [gate, up] group");
    }
    auto shape = x.sizes().vec();
    auto y = nint_matmul_swiglu(
        profiler_, group.w, x.reshape({-1, x.size(-1)}));
    shape.back() = y.size(-1);
    return y.reshape(shape);
}

::tyloi::DenseTensor Projector::geglu(
        const NintLinearGroup& group, ::tyloi::DenseTensor x) const {
    if (!group.split_w.empty() || group.outs.size() != 2 || group.outs[0] != group.outs[1]) {
        throw std::runtime_error("NINT GeGLU fusion requires one packed [gate, up] group");
    }
    auto shape = x.sizes().vec();
    auto y = nint_matmul_geglu(
        profiler_, group.w, x.reshape({-1, x.size(-1)}));
    shape.back() = y.size(-1);
    return y.reshape(shape);
}

#endif

::tyloi::DenseTensor embed(const QuantLinear& weight, ::tyloi::DenseTensor ids,
        Collectives& copies, CudaProfiler& profiler, int device) {
    ids = copies.copy(ids.to(::tyloi::kInt64), device).contiguous();
    return profiler.measure("model.embed", [&] { return quant_embedding_lookup(weight, ids); });
}

::tyloi::DenseTensor logits(const QuantLinear& weight, ::tyloi::DenseTensor hidden,
        const Projector& project, CudaProfiler& profiler, double softcap) {
    auto output = profiler.measure("model.lm_head", [&] { return project.run(weight, hidden); });
    return softcap > 0.0 ? ::tyloi::cuda::tanh(output / softcap) * softcap : output;
}

} // namespace tyloi::cuda
