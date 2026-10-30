#include "projection.h"
#include "engine/executor/options.h"
#include "engine/backends/cuda/native/branches.h"
#include "engine/backends/cuda/native/collectives.h"
#include "engine/executor/tq.h"
#include "engine/backends/cuda/kernels/mfq_cuda_activation_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_quant_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_nint_route_hint.h"

#include <algorithm>
#include <utility>

mfq_tensor_backend::Tensor run_quant_linear_shard(
        CudaProfiler& profiler,
        KlMmqState& kl_mmq,
        TqExec& tq,
        const QuantLinearShard & shard,
        mfq_tensor_backend::Tensor x,
        MfqOptional<mfq_tensor_backend::Tensor> gate,
        InputGateMode gate_mode) {
    MfqCudaGuard guard(shard.device);
    if (shard.kind == QuantLinearKind::Tq) {
        if (gate.has_value()) {
            TYLOINFER_RUNTIME_CHECK(gate_mode == InputGateMode::Sigmoid || gate_mode == InputGateMode::Silu,
                "TQ input gate mode must be sigmoid or SiLU");
            auto local_gate = gate.value().to(x.scalar_type());
            x = x * (gate_mode == InputGateMode::Sigmoid
                ? mfq_tensor_backend::sigmoid(local_gate)
                : mfq_tensor_backend::silu(local_gate));
        }
        return profiler.measure("tq.tensor_parallel", [&] {
            const auto resources=tq.get(shard.device);
            return shard.tq.forward(&resources->blas(),std::move(x));
        });
    }
    if ((mfq::mfq_compat_enabled && shard.kind == QuantLinearKind::Mxfp4Sq) ||
            (mfq::mfq_compat_enabled && shard.kind == QuantLinearKind::Fp8Sq)) {
        if (gate.has_value()) {
            TYLOINFER_RUNTIME_CHECK(
                gate_mode == InputGateMode::Sigmoid || gate_mode == InputGateMode::Silu,
                "SQ input gate mode must be sigmoid or SiLU");
            x = x * (gate_mode == InputGateMode::Sigmoid
                ? mfq_tensor_backend::sigmoid(gate.value())
                : mfq_tensor_backend::silu(gate.value()));
        }
        return (mfq::mfq_compat_enabled && shard.kind == QuantLinearKind::Mxfp4Sq)
            ? shard.mxfp4_sq.forward(x)
            : shard.fp8_sq.forward(x);
    }
    if ((mfq::mfq_compat_enabled && shard.kind == QuantLinearKind::Nint)) {
        return run_nint_linear(profiler, kl_mmq, shard.nint, x, gate, gate_mode);
    }
    if ((mfq::mfq_compat_enabled && shard.kind == QuantLinearKind::Nvq)) {
        return run_nvq_linear(profiler, kl_mmq, shard.nvq, x, gate, gate_mode);
    }
    if (shard.kind == QuantLinearKind::Mxfp4) {
        TYLOINFER_RUNTIME_CHECK(
            !gate.has_value(),
            "MXFP4 tensor-parallel linear does not support input gating");
        return mxfp4_matmul(shard.mxfp4, x);
    }
    if (shard.kind == QuantLinearKind::Dense) {
        auto local = x.to(shard.dense.scalar_type());
        if (gate.has_value()) {
            TYLOINFER_RUNTIME_CHECK(
                gate_mode == InputGateMode::Sigmoid || gate_mode == InputGateMode::Silu,
                "dense input gate mode must be sigmoid or SiLU");
            auto local_gate = gate.value().to(local.scalar_type());
            local = gate_mode == InputGateMode::Sigmoid
                ? local * mfq_tensor_backend::sigmoid(local_gate)
                : local * mfq_tensor_backend::silu(local_gate);
        }
        return dense_projection(std::move(local), shard.dense);
    }
    TYLOINFER_RUNTIME_CHECK(
        !gate.has_value(),
        "MXFP8 tensor-parallel linear does not support input gating");
    return mxfp8_matmul(profiler, shard.mxfp8, x);
}

namespace mfq::cuda {

std::vector<mfq_tensor_backend::Tensor>
Projector::grouped(
        const QuantLinearProjectionRefs& projections,
        mfq_tensor_backend::Tensor x) const {
    TYLOINFER_RUNTIME_CHECK(
        tensor_parallel_output_projections_compatible(projections),
        "tensor-parallel output projections are incompatible");
    auto output_shape = x.sizes().vec();
    auto flat = x.reshape({-1, x.size(-1)});
    const size_t shard_count =
        projections.front()->tensor_parallel_shards.size();
    if (projection_major_) {
        std::vector<mfq_tensor_backend::Tensor> local_inputs(shard_count);
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
        std::vector<mfq_tensor_backend::Tensor> result;
        result.reserve(projections.size());
        for (const auto * projection : projections) {
            std::vector<mfq_tensor_backend::Tensor> local_outputs(shard_count);
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
            std::vector<mfq_tensor_backend::Tensor> gathered;
            gathered.reserve(shard_count);
            for (auto & output : local_outputs) {
                gathered.push_back(collectives_.copy(output, primary));
            }
            auto combined = mfq_tensor_backend::cat(
                gathered, -1).contiguous();
            auto shape = output_shape;
            shape.back() = combined.size(-1);
            result.push_back(combined.reshape(shape));
        }
        return result;
    }
    std::vector<std::vector<mfq_tensor_backend::Tensor>>
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
    std::vector<mfq_tensor_backend::Tensor> result;
    result.reserve(projections.size());
    for (auto & projection_outputs : local_outputs) {
        std::vector<mfq_tensor_backend::Tensor> gathered;
        gathered.reserve(shard_count);
        for (auto & output : projection_outputs) {
            gathered.push_back(collectives_.copy(output, primary));
        }
        auto combined = mfq_tensor_backend::cat(
            gathered, -1).contiguous();
        auto shape = output_shape;
        shape.back() = combined.size(-1);
        result.push_back(combined.reshape(shape));
    }
    return result;
}

mfq_tensor_backend::Tensor Projector::sharded(
        const QuantLinear& weight,
        mfq_tensor_backend::Tensor x,
        MfqOptional<mfq_tensor_backend::Tensor> gate,
        InputGateMode gate_mode) const {
    TYLOINFER_RUNTIME_CHECK(
        weight.tensor_parallel(),
        "tensor-parallel linear has no shards");
    TYLOINFER_RUNTIME_CHECK(
        weight.tensor_parallel_axis == TensorParallelAxis::Output ||
        weight.tensor_parallel_axis == TensorParallelAxis::Input,
        "tensor-parallel linear has an invalid axis");
    std::vector<mfq_tensor_backend::Tensor> local_outputs(
        weight.tensor_parallel_shards.size());
    for (size_t launch_position = 0;
         launch_position < weight.tensor_parallel_shards.size();
         ++launch_position) {
        const size_t index = parallel_index(
            options_.peer_first, launch_position, weight.tensor_parallel_shards.size());
        const auto & shard = weight.tensor_parallel_shards[index];
        MfqCudaGuard guard(shard.device);
        mfq_tensor_backend::Tensor local_x = x;
        mfq_tensor_backend::Tensor local_gate;
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
            local_outputs[index] =
                mxfp8_matmul_f32(
                    profiler_, shard.mxfp8, local_x);
        } else {
            local_outputs[index] =
                run_quant_linear_shard(
                    profiler_, mmq_, tq_,
                    shard, local_x,
                    gate.has_value()
                        ? MfqOptional<mfq_tensor_backend::Tensor>(
                            local_gate)
                        : mfq_nullopt,
                    gate_mode);
        }
    }

    const int primary = primary_;
    MfqCudaGuard primary_guard(primary);
    if (weight.tensor_parallel_axis == TensorParallelAxis::Output) {
        std::vector<mfq_tensor_backend::Tensor> gathered;
        gathered.reserve(local_outputs.size());
        for (auto & output : local_outputs) {
            gathered.push_back(
                collectives_.copy(output, primary));
        }
        return mfq_tensor_backend::cat(gathered, -1).contiguous();
    }

    auto reduced = collectives_.sum(std::move(local_outputs), primary_,
            options_.reduce_to_primary, options_.fp16_reduce);
    return weight.is_mxfp8()
        ? reduced.to(x.scalar_type()).contiguous()
        : reduced;
}

mfq_tensor_backend::Tensor Projector::run(
        const QuantLinear& weight,
        mfq_tensor_backend::Tensor x) const {
    if(weight.is_tq() && !weight.tensor_parallel())return profiler_.measure("tq.linear",[&] {
        const auto resources=tq_.get(x.get_device());
        return weight.tq.forward(&resources->blas(),std::move(x));
    });
    if (weight.tensor_parallel() || (mfq::mfq_compat_enabled && weight.is_nint()) || (mfq::mfq_compat_enabled && weight.is_nvq())) {
        auto shape = x.sizes().vec();
        auto flat = x.reshape({-1, x.size(-1)});
#if TYLOINFER_ENABLE_MFQ_COMPAT
        auto y = weight.tensor_parallel()
            ? sharded(weight, flat, mfq_nullopt, InputGateMode::None)
            : (mfq::mfq_compat_enabled && weight.is_nint()) ? run_nint_linear(
                  profiler_, mmq_, weight.nint, flat)
                        : run_nvq_linear(
                  profiler_, mmq_, weight.nvq, flat);
#else
        auto y = sharded(weight, flat, mfq_nullopt, InputGateMode::None);
#endif
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
    if (weight.is_mxfp4()) return weight.mxfp4.forward(x);
    if ((mfq::mfq_compat_enabled && weight.is_mxfp4_sq())) return weight.mxfp4_sq.forward(x);
    if ((mfq::mfq_compat_enabled && weight.is_fp8_sq())) return weight.fp8_sq.forward(x);
    if (weight.is_dense()) return dense_projection(x, weight.dense);
    return weight.mxfp8.forward(profiler_, x);
}

std::vector<mfq_tensor_backend::Tensor> Projector::run(
        const Projections& projections,
        mfq_tensor_backend::Tensor x) const {
    const auto join = [&projections](std::vector<mfq_tensor_backend::Tensor> outputs) {
        if (projections.prefix <= 1 || outputs.empty()) return outputs;
        std::vector<mfq_tensor_backend::Tensor> prefix(
            outputs.begin(), outputs.begin() + projections.prefix);
        auto combined = mfq_tensor_backend::cat(prefix, -1);
        outputs.erase(outputs.begin(), outputs.begin() + projections.prefix);
        outputs.insert(outputs.begin(), std::move(combined));
        return outputs;
    };
    const auto run_separate = [&] {
        std::vector<mfq_tensor_backend::Tensor> outputs;
        outputs.reserve(projections.weights.size());
        for (const auto & weight : projections.weights) {
            outputs.push_back(run(weight, x));
        }
        return join(std::move(outputs));
    };

    if (projections.group.tq.defined() && x.is_cuda()) {
        auto grouped = run(projections.group, x);
        std::vector<mfq_tensor_backend::Tensor> outputs(
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

mfq_tensor_backend::Tensor Projector::groupwise(
        const QuantLinear& weight,
        mfq_tensor_backend::Tensor grouped,
        int64_t groups) const {
    TYLOINFER_RUNTIME_CHECK(
        weight.is_mxfp8(),
        "groupwise MXFP8 projection requires an MXFP8 tensor");
    if (!weight.tensor_parallel()) {
        return mxfp8_groupwise_matmul(
            profiler_, weight.mxfp8.weight, grouped, groups);
    }
    TYLOINFER_RUNTIME_CHECK(
        weight.tensor_parallel_axis == TensorParallelAxis::Input,
        "groupwise MXFP8 tensor parallelism requires input-axis shards");
    std::vector<mfq_tensor_backend::Tensor> partials;
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

mfq_tensor_backend::Tensor Projector::gated(
        const QuantLinear& weight,
        mfq_tensor_backend::Tensor x,
        mfq_tensor_backend::Tensor gate,
        InputGateMode mode) const {
    if (weight.tensor_parallel() || (mfq::mfq_compat_enabled && weight.is_nint()) || (mfq::mfq_compat_enabled && weight.is_nvq())) {
        auto shape = x.sizes().vec();
        auto flat = x.reshape({-1, x.size(-1)});
        auto flat_gate = gate.reshape({-1, gate.size(-1)});
#if TYLOINFER_ENABLE_MFQ_COMPAT
        auto y = weight.tensor_parallel()
            ? sharded(weight, flat, flat_gate, mode)
            : (mfq::mfq_compat_enabled && weight.is_nint()) ? run_nint_linear(
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
            ? local * mfq_tensor_backend::sigmoid(local_gate)
            : local * mfq_tensor_backend::silu(local_gate);
        return dense_projection(gated, weight.dense);
    }
    TYLOINFER_RUNTIME_CHECK(mode == InputGateMode::Sigmoid || mode == InputGateMode::Silu,
        "input gate mode must be sigmoid or SiLU");
    auto local_gate = gate.to(x.scalar_type());
    auto gated = mode == InputGateMode::Sigmoid
        ? x * mfq_tensor_backend::sigmoid(local_gate)
        : x * mfq_tensor_backend::silu(local_gate);
    return run(weight, gated);
}

std::vector<mfq_tensor_backend::Tensor> Projector::run(
        const QuantLinearGroup& group,
        mfq_tensor_backend::Tensor x) const {
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
        std::vector<mfq_tensor_backend::Tensor> result;
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
    if (mfq::mfq_compat_enabled && group.nint_grouped) {
        return run(group.nint, x);
    }
    if (mfq::mfq_compat_enabled && default_mmq && group.nvq_prefix2 && options_.nvq_fusion) {
        auto shape = x.sizes().vec();
        const auto flat =
            x.reshape({-1, x.size(-1)});
        std::vector<mfq_tensor_backend::Tensor> branch_outputs;
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
        std::vector<mfq_tensor_backend::Tensor> result;
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
    std::vector<mfq_tensor_backend::Tensor> result;
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

mfq_tensor_backend::Tensor Projector::swiglu(
        const QuantLinearGroup& group,
        mfq_tensor_backend::Tensor x) const {
    if(group.tq.defined() && group.outs.size()==2 && group.outs[0]==group.outs[1])
        return profiler_.measure("tq.swiglu",[&] {
            const auto resources=tq_.get(x.get_device());
            return group.tq.swiglu(&resources->blas(),std::move(x));
        });
    const bool default_mmq =
        mmq_.mode == KlMmqMode::Default;
    if (mfq::mfq_compat_enabled && default_mmq && group.nint_grouped && group.nint.split_w.empty() &&
            x.numel() / x.size(-1) >= 1 && x.numel() / x.size(-1) <= 6) {
        return swiglu(group.nint, x);
    }
    if (mfq::mfq_compat_enabled && default_mmq && group.nvq_prefix2 && group.layers.size() == 2 &&
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
    return mfq_tensor_backend::silu(parts[0]) * parts[1];
}

mfq_tensor_backend::Tensor Projector::geglu(
        const QuantLinearGroup& group,
        mfq_tensor_backend::Tensor x) const {
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
std::vector<mfq_tensor_backend::Tensor> Projector::run(
        const NintLinearGroup& group, mfq_tensor_backend::Tensor x) const {
    auto shape = x.sizes().vec();
    std::vector<mfq_tensor_backend::Tensor> parts;
    auto xf = x.reshape({-1, x.size(-1)});
    if (group.w.q8_zero && xf.size(0) > 64 &&
            group.projection_w.size() == group.outs.size()) {
        parts.reserve(group.projection_w.size());
        for (const auto & projection : group.projection_w) {
            parts.push_back(run_nint_linear(
                profiler_, mmq_, projection, xf));
        }
    } else if (!group.split_w.empty()) {
        parts.reserve(group.outs.size());
        std::vector<int> shared_groups;
        if(serial_ && runtime_options::nint_group_quantize().value_or(true) &&
                mmq_.mode==KlMmqMode::Default && xf.is_contiguous() &&
                xf.size(0)>=1 && xf.size(0)<=6 &&
                (xf.scalar_type()==mfq_tensor_backend::kFloat16 || xf.scalar_type()==mfq_tensor_backend::kFloat32)) {
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
        std::vector<mfq_tensor_backend::Tensor> grouped_outputs;
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

mfq_tensor_backend::Tensor Projector::swiglu(
        const NintLinearGroup& group, mfq_tensor_backend::Tensor x) const {
    if (!group.split_w.empty() || group.outs.size() != 2 || group.outs[0] != group.outs[1]) {
        throw std::runtime_error("NINT SwiGLU fusion requires one packed [gate, up] group");
    }
    auto shape = x.sizes().vec();
    auto y = nint_matmul_swiglu(
        profiler_, group.w, x.reshape({-1, x.size(-1)}));
    shape.back() = y.size(-1);
    return y.reshape(shape);
}

mfq_tensor_backend::Tensor Projector::geglu(
        const NintLinearGroup& group, mfq_tensor_backend::Tensor x) const {
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

mfq_tensor_backend::Tensor embed(const QuantLinear& weight, mfq_tensor_backend::Tensor ids,
        Collectives& copies, CudaProfiler& profiler, int device) {
    ids = copies.copy(ids.to(mfq_tensor_backend::kInt64), device).contiguous();
    return profiler.measure("model.embed", [&] { return quant_embedding_lookup(weight, ids); });
}

mfq_tensor_backend::Tensor logits(const QuantLinear& weight, mfq_tensor_backend::Tensor hidden,
        const Projector& project, CudaProfiler& profiler, double softcap) {
    auto output = profiler.measure("model.lm_head", [&] { return project.run(weight, hidden); });
    return softcap > 0.0 ? mfq_tensor_backend::tanh(output / softcap) * softcap : output;
}

} // namespace mfq::cuda
