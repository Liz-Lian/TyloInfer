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
#include <numeric>
#include <utility>

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/ops/projection/bias.h"
#include "engine/ops.h"
#endif

namespace {
#if TYLOINFER_ENABLE_MFQ_COMPAT
bool extend_rows(tyloi::DenseTensor& first, const tyloi::DenseTensor& next) {
    if (!first.defined() || !next.defined() || !first.is_contiguous() || !next.is_contiguous() ||
        first.dim() != next.dim() || first.dim() == 0 || first.scalar_type() != next.scalar_type() ||
        first.device() != next.device() || first.storage().get() != next.storage().get() ||
        first.storage().owner_before(next.storage()) || next.storage().owner_before(first.storage()) ||
        first.byte_offset() > first.storage()->bytes || first.nbytes() > first.storage()->bytes - first.byte_offset() ||
        next.byte_offset() > next.storage()->bytes ||
        next.byte_offset() != first.byte_offset() + first.nbytes() ||
        next.nbytes() > next.storage()->bytes - next.byte_offset()) return false;
    for (int64_t i = 0; i < first.dim(); ++i)
        if ((i && first.size(i) != next.size(i)) || first.stride(i) != next.stride(i)) return false;
    auto view = first.view_descriptor();
    view.sizes[0] += next.size(0);
    first = first.alias(view, first.byte_offset());
    return true;
}

bool extend_nint(tyloi::NintTensor& first, const tyloi::NintTensor& next) {
    if (first.ng != next.ng || first.gs != next.gs || first.neuron_len != next.neuron_len ||
        first.q8_zero != next.q8_zero) return false;
    auto merged = first;
    if (first.q8_zero) {
        if (!extend_rows(merged.q_packed, next.q_packed) ||
            !extend_rows(merged.q8_zero_scale, next.q8_zero_scale)) return false;
    } else {
        const auto& a = first.q_packed;
        const auto& b = next.q_packed;
        if (!a.defined() || !b.defined() || !a.is_contiguous() || !b.is_contiguous() ||
            a.storage().get() != b.storage().get() || a.storage().owner_before(b.storage()) ||
            b.storage().owner_before(a.storage()) || a.byte_offset() != b.byte_offset() ||
            a.data_ptr() != b.data_ptr() || a.nbytes() != b.nbytes() ||
            a.byte_offset() > a.storage()->bytes || a.nbytes() > a.storage()->bytes - a.byte_offset() ||
            !extend_rows(merged.row_q_bits, next.row_q_bits) ||
            !extend_rows(merged.row_q_bit_offsets, next.row_q_bit_offsets) ||
            !extend_rows(merged.sub_scale, next.sub_scale) || !extend_rows(merged.sub_min, next.sub_min) ||
            !extend_rows(merged.neuron_scale, next.neuron_scale) ||
            !extend_rows(merged.neuron_min, next.neuron_min)) return false;
    }
    merged.out += next.out;
    merged.shape[0] = merged.out;
    merged.aligned_q8 = merged.aligned_q8 && next.aligned_q8;
    if (merged.uniform_qbits != next.uniform_qbits) merged.uniform_qbits = 0;
    first = std::move(merged);
    return true;
}
#endif
}

QuantLinearGroup tyloi::cuda::Projector::group(
        std::span<const std::shared_ptr<const ::tyloi::Tensor>> weights) const {
    TYLOINFER_RUNTIME_CHECK(!weights.empty(), "empty quantized linear group");
    QuantLinearGroup result;
    result.decode_branch_parallel = !independent_;
    result.layers.assign(weights.begin(), weights.end());
    std::vector<::tyloi::encoding::TqTensor> tq_weights;
    bool all_tq = true;
    for (const auto& weight : weights) {
        TYLOINFER_RUNTIME_CHECK(weight != nullptr, "missing projection weight");
        result.outs.push_back(weight->size(0));
        result.cols.push_back(weight->size(1));
        result.placements.push_back(projection_placement(*weight));
        const auto* tq = dynamic_cast<const ::tyloi::encoding::TqTensor*>(weight.get());
        if (!tq || !tq->is_cuda()) all_tq = false;
        else tq_weights.push_back(*tq);
    }
    if (all_tq) result.tq = *tq_.get(tq_weights.front().get_device())->group(std::move(tq_weights));
    const auto* nvq0 = weights.size() >= 2 ? dynamic_cast<const tyloi::NvqTensor*>(weights[0].get()) : nullptr;
    const auto* nvq1 = weights.size() >= 2 ? dynamic_cast<const tyloi::NvqTensor*>(weights[1].get()) : nullptr;
    result.nvq_prefix2 = tyloi::mfq_compat_enabled && nvq0 && nvq1 &&
        nvq0->is_cuda() && nvq_pair_compatible(*nvq0, *nvq1);
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (options_.nint_grouped && !independent_ && tyloi::mfq_compat_enabled) {
        std::vector<tyloi::NintTensor> segments;
        std::vector<std::vector<int64_t>> sizes;
        bool all_nint = true;
        for (const auto& weight : weights) {
            const auto* nint = dynamic_cast<const tyloi::NintTensor*>(weight.get());
            if (!nint || !nint->q_packed.is_cuda()) { all_nint = false; break; }
            if (!segments.empty() && extend_nint(segments.back(), *nint)) {
                sizes.back().push_back(nint->out);
            } else {
                segments.push_back(*nint);
                sizes.push_back({nint->out});
            }
        }
        if (all_nint) {
            result.nint_grouped = true;
            result.nint.outs = result.outs;
            if (segments.size() == 1) {
                result.nint.w = std::move(segments.front());
                if (result.nint.w.q8_zero)
                    for (const auto& weight : weights)
                        result.nint.projection_w.push_back(static_cast<const tyloi::NintTensor&>(*weight));
            } else {
                result.nint.split_w = std::move(segments);
                result.nint.split_outs = std::move(sizes);
            }
        }
    }
#endif
    return result;
}

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
tyloi::DenseTensor tyloi::cuda::ops::embedding(tyloi::DenseTensor ids, const ::tyloi::Tensor& weight) {
    std::optional<MfqCudaGuard> guard;
    if (ids.is_cuda()) guard.emplace(ids.device().index);
    return quant_embedding_lookup(weight, ids);
}

tyloi::DenseTensor tyloi::cuda::ops::project(tyloi::DenseTensor input, const ::tyloi::Tensor& weight) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    return tyloi::cuda::Executor::current().projector(tyloi::cuda::Executor::current_ffn()).run(
        weight, input);
}

tyloi::DenseTensor tyloi::cuda::ops::linear(tyloi::DenseTensor input, const ::tyloi::Tensor& weight,
                                  const tyloi::DenseTensor* bias) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    return biased(Executor::current().projector(Executor::current_ffn()), weight,
                  bias, input);
}

tyloi::DenseTensor tyloi::cuda::ops::affine(tyloi::DenseTensor input, const ::tyloi::Tensor& weight,
                                    const tyloi::DenseTensor* bias) {
    auto output = project(std::move(input), weight);
    if (!bias || !bias->defined()) return output;
    if (output.device() != bias->device())
        throw std::invalid_argument("tensor devices do not match");
    std::optional<MfqCudaGuard> guard;
    if (output.is_cuda()) guard.emplace(output.device().index);
    auto native = output;
    return native + bias->to(native.scalar_type());
}

std::vector<tyloi::DenseTensor> tyloi::cuda::ops::project_group(tyloi::DenseTensor input,
    std::span<const std::shared_ptr<const tyloi::Tensor>> weights) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    const auto project = tyloi::cuda::Executor::current().projector(tyloi::cuda::Executor::current_ffn());
    const auto group = project.group(weights);
    auto parts = project.run(group, input);
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
    const ::tyloi::Tensor& weight) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    return tyloi::cuda::Executor::current().projector(tyloi::cuda::Executor::current_ffn()).gated(
        weight, input,
        gate, InputGateMode::Sigmoid);
}
#endif

::tyloi::DenseTensor run_quant_linear_shard(
        CudaProfiler& profiler,
        KlMmqState& kl_mmq,
        TqExec& tq,
        const tyloi::ShardedTensor::Shard & shard,
        ::tyloi::DenseTensor x,
        MfqOptional<::tyloi::DenseTensor> gate,
        InputGateMode gate_mode) {
    MfqCudaGuard guard(shard.device);
    const auto& weight = *shard.weight;
    const auto gated = [&](InputGateMode first, InputGateMode second) {
        TYLOINFER_RUNTIME_CHECK(gate_mode == first || gate_mode == second,
            "input gate mode is not supported by this projection");
    };
    if (const auto* tq_weight = dynamic_cast<const tyloi::encoding::TqTensor*>(&weight)) {
        if (gate.has_value()) {
            gated(InputGateMode::Sigmoid, InputGateMode::Silu);
            auto local_gate = gate.value().to(x.scalar_type());
            x = x * (gate_mode == InputGateMode::Sigmoid
                ? ::tyloi::cuda::sigmoid(local_gate)
                : ::tyloi::cuda::silu(local_gate));
        }
        return profiler.measure("tq.tensor_parallel", [&] {
            const auto resources = tq.get(shard.device);
            return tyloi::cuda::tq::project(*tq_weight, &resources->blas(), std::move(x));
        });
    }
    if (const auto* sq4 = dynamic_cast<const tyloi::Mxfp4SqTensor*>(&weight)) {
        TYLOINFER_RUNTIME_CHECK(tyloi::mfq_compat_enabled, "MXFP4-SQ linear requires a compatibility build");
        if (gate.has_value()) gated(InputGateMode::Sigmoid, InputGateMode::Silu);
        check_device(*sq4, x);
        if (gate.has_value())
            x = x * (gate_mode == InputGateMode::Sigmoid
                ? ::tyloi::cuda::sigmoid(gate.value()) : ::tyloi::cuda::silu(gate.value()));
        return tyloi::cuda::sq_matmul(*sq4, x);
    }
    if (const auto* sq8 = dynamic_cast<const tyloi::Fp8SqTensor*>(&weight)) {
        TYLOINFER_RUNTIME_CHECK(tyloi::mfq_compat_enabled, "FP8-SQ linear requires a compatibility build");
        if (gate.has_value()) gated(InputGateMode::Sigmoid, InputGateMode::Silu);
        check_device(*sq8, x);
        if (gate.has_value())
            x = x * (gate_mode == InputGateMode::Sigmoid
                ? ::tyloi::cuda::sigmoid(gate.value()) : ::tyloi::cuda::silu(gate.value()));
        return tyloi::cuda::sq_matmul(*sq8, x);
    }
    if (const auto* nint = dynamic_cast<const tyloi::NintTensor*>(&weight)) {
        TYLOINFER_RUNTIME_CHECK(tyloi::mfq_compat_enabled, "NINT linear requires a compatibility build");
        if (!gate.has_value() || gate_mode == InputGateMode::Sigmoid || gate_mode == InputGateMode::Silu)
            check_device(*nint, x);
        return run_nint_linear(profiler, kl_mmq, *nint, x, gate, gate_mode);
    }
    if (const auto* nvq = dynamic_cast<const tyloi::NvqTensor*>(&weight)) {
        TYLOINFER_RUNTIME_CHECK(tyloi::mfq_compat_enabled, "NVQ linear requires a compatibility build");
        if (!gate.has_value() || gate_mode == InputGateMode::Sigmoid || gate_mode == InputGateMode::Silu)
            check_device(*nvq, x);
        return run_nvq_linear(profiler, kl_mmq, *nvq, x, gate, gate_mode);
    }
    if (const auto* mx4 = dynamic_cast<const tyloi::Mxfp4Tensor*>(&weight)) {
        TYLOINFER_RUNTIME_CHECK(!gate.has_value(),
            "MXFP4 tensor-parallel linear does not support input gating");
        check_device(*mx4, x);
        return mxfp4_matmul(*mx4, x);
    }
    if (const auto* dense = dynamic_cast<const ::tyloi::DenseTensor*>(&weight)) {
        if (gate.has_value()) gated(InputGateMode::Sigmoid, InputGateMode::Silu);
        check_device(*dense, x);
        auto local = x.to(dense->scalar_type());
        if (gate.has_value()) {
            auto local_gate = gate.value().to(local.scalar_type());
            local = gate_mode == InputGateMode::Sigmoid
                ? local * ::tyloi::cuda::sigmoid(local_gate)
                : local * ::tyloi::cuda::silu(local_gate);
        }
        return dense_projection(std::move(local), *dense);
    }
    const auto& mx8 = payload_as<tyloi::Mxfp8Tensor>(weight);
    TYLOINFER_RUNTIME_CHECK(
        !gate.has_value(),
        "MXFP8 tensor-parallel linear does not support input gating");
    check_device(mx8, x);
    return mxfp8_matmul(profiler, mx8, x);
}

namespace {
template <class Format>
const Format* local_format(const ::tyloi::Tensor& weight) {
    return dynamic_cast<const Format*>(&weight);
}

template <class Format>
bool holds(const ::tyloi::Tensor& weight) {
    const auto* sharded = dynamic_cast<const tyloi::ShardedTensor*>(&weight);
    if (!sharded) return local_format<Format>(weight) != nullptr;
    return std::all_of(sharded->shards.begin(), sharded->shards.end(),
        [](const tyloi::ShardedTensor::Shard& shard) {
            return dynamic_cast<const Format*>(shard.weight.get()) != nullptr;
        });
}
}  // namespace

namespace tyloi::cuda {

using ::tyloi::indexing::Slice;

::tyloi::DenseTensor Projector::low_rank(::tyloi::DenseTensor attention,
                                               const ::tyloi::Tensor &output_a,
                                               const ::tyloi::Tensor &output_b, int64_t groups,
                                               bool groupwise_output, bool profile) const {
    const auto batch = attention.size(0);
    const auto tokens = attention.size(1);
    const auto rows = batch * tokens;
    const auto rank = output_a.size(0) / groups;
    auto grouped = attention.contiguous()
                       .reshape({rows, groups, output_a.size(1)})
                       .to(::tyloi::kFloat16);
    const auto measure = [&](const char *name, auto &&operation) {
        return profile ? profiler_.measure(name, operation) : operation();
    };
    const auto* tq_weight = dynamic_cast<const tyloi::encoding::TqTensor*>(&output_a);
    const auto* nint = dynamic_cast<const tyloi::NintTensor*>(&output_a);
    if (tq_weight != nullptr) {
        auto low_rank = measure("dsv4.output_a", [&] {
            const auto resources = tq_.get(grouped.get_device());
            return tyloi::cuda::tq::groupwise(*tq_weight,&resources->blas(), grouped, groups);
        });
        return measure("dsv4.output_b", [&] {
            return run(output_b, low_rank).reshape({batch, tokens, output_b.size(0)});
        });
    }
    if (groupwise_output && tyloi::mfq_compat_enabled && nint != nullptr &&
        nint->bits == 8 && nint->gs == 48 && nint->out == groups * rank) {
        auto low_rank = measure("dsv4.output_a", [&]() {
            return nint_matmul_groupwise_u8(profiler_, *nint, grouped, groups);
        });
        return measure("dsv4.output_b", [&]() {
            return run(output_b, low_rank).reshape({batch, tokens, output_b.size(0)});
        });
    }
    if (groupwise_output &&
        holds<tyloi::Mxfp8Tensor>(output_a) &&
        output_a.size(0) == groups * rank) {
        auto low_rank = measure("dsv4.output_a", [&]() {
            return groupwise(output_a, grouped, groups);
        });
        return measure("dsv4.output_b", [&]() {
            return run(output_b, low_rank).reshape({batch, tokens, output_b.size(0)});
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
                        .reshape({rows, output_a.size(0)})
                        .to(::tyloi::kFloat16)
                        .contiguous();
    return measure("dsv4.output_b", [&]() {
        return run(output_b, low_rank).reshape({batch, tokens, output_b.size(0)});
    });
}

std::vector<::tyloi::DenseTensor>
Projector::grouped(
        const std::vector<const ::tyloi::Tensor*>& projections,
        ::tyloi::DenseTensor x) const {
    TYLOINFER_RUNTIME_CHECK(
        tensor_parallel_output_projections_compatible(projections),
        "tensor-parallel output projections are incompatible");
    for (const auto* projection : projections) check_projection(*projection, x);
    auto output_shape = x.sizes().vec();
    auto flat = x.reshape({-1, x.size(-1)});
    const size_t shard_count =
        static_cast<const tyloi::ShardedTensor*>(projections.front())->shards.size();
    if (projection_major_) {
        std::vector<::tyloi::DenseTensor> local_inputs(shard_count);
        for (size_t launch_position = 0;
             launch_position < shard_count; ++launch_position) {
            const size_t shard = parallel_index(
                options_.peer_first, launch_position, shard_count);
            const int device =
                static_cast<const tyloi::ShardedTensor*>(projections.front())->shards[shard].device;
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
                    static_cast<const tyloi::ShardedTensor*>(projection)->shards[shard];
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
            static_cast<const tyloi::ShardedTensor*>(projections.front())->shards[shard].device;
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
                    static_cast<const tyloi::ShardedTensor*>(projections[projection])->shards[shard], local_x);
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
        const ::tyloi::ShardedTensor& weight,
        ::tyloi::DenseTensor x,
        MfqOptional<::tyloi::DenseTensor> gate,
        InputGateMode gate_mode) const {
    TYLOINFER_RUNTIME_CHECK(
        weight.axis == TensorParallelAxis::Output ||
        weight.axis == TensorParallelAxis::Input,
        "tensor-parallel linear has an invalid axis");
    std::vector<::tyloi::DenseTensor> local_outputs(
        weight.shards.size());
    for (size_t launch_position = 0;
         launch_position < weight.shards.size();
         ++launch_position) {
        const size_t index = parallel_index(
            options_.peer_first, launch_position, weight.shards.size());
        const auto & shard = weight.shards[index];
        MfqCudaGuard guard(shard.device);
        ::tyloi::DenseTensor local_x = x;
        ::tyloi::DenseTensor local_gate;
        if (weight.axis == TensorParallelAxis::Input) {
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
        if (const auto* mx8 = dynamic_cast<const tyloi::Mxfp8Tensor*>(shard.weight.get());
            mx8 != nullptr && weight.axis == TensorParallelAxis::Input) {
            TYLOINFER_RUNTIME_CHECK(
                !gate.has_value(),
                "MXFP8 input-axis tensor parallelism does not support gating");
            check_device(*mx8, local_x);
            local_outputs[index] = mxfp8_matmul_f32(profiler_, *mx8, local_x);
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
    if (weight.axis == TensorParallelAxis::Output) {
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
    return holds<tyloi::Mxfp8Tensor>(weight)
        ? reduced.to(x.scalar_type()).contiguous()
        : reduced;
}

::tyloi::DenseTensor Projector::run(
        const ::tyloi::Tensor& weight,
        ::tyloi::DenseTensor x) const {
    check_projection(weight, x);
    if (const auto* tq = local_format<tyloi::encoding::TqTensor>(weight); tq != nullptr)
        return profiler_.measure("tq.linear", [&] {
            const auto resources = tq_.get(x.get_device());
            return tyloi::cuda::tq::project(*tq, &resources->blas(), std::move(x));
        });
    if (const auto* nint = local_format<tyloi::NintTensor>(weight); nint != nullptr) {
        TYLOINFER_RUNTIME_CHECK(tyloi::mfq_compat_enabled, "NINT linear requires a compatibility build");
        auto shape = x.sizes().vec();
        auto flat = x.reshape({-1, x.size(-1)});
        auto y = run_nint_linear(profiler_, mmq_, *nint, flat);
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
    if (const auto* nvq = local_format<tyloi::NvqTensor>(weight); nvq != nullptr) {
        TYLOINFER_RUNTIME_CHECK(tyloi::mfq_compat_enabled, "NVQ linear requires a compatibility build");
        auto shape = x.sizes().vec();
        auto flat = x.reshape({-1, x.size(-1)});
        auto y = run_nvq_linear(profiler_, mmq_, *nvq, flat);
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
    if (const auto* sharded_weight = dynamic_cast<const tyloi::ShardedTensor*>(&weight)) {
        auto shape = x.sizes().vec();
        auto y = sharded(*sharded_weight, x.reshape({-1, x.size(-1)}), mfq_nullopt, InputGateMode::None);
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
    if (const auto* mx4 = local_format<tyloi::Mxfp4Tensor>(weight); mx4 != nullptr) {
        auto shape = x.sizes().vec();
        auto y = mxfp4_matmul(*mx4, x.reshape({-1, x.size(-1)}));
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
    if (const auto* sq4 = local_format<tyloi::Mxfp4SqTensor>(weight); sq4 != nullptr) {
        TYLOINFER_RUNTIME_CHECK(tyloi::mfq_compat_enabled, "MXFP4-SQ linear requires a compatibility build");
        return sq_matmul(*sq4, x);
    }
    if (const auto* sq8 = local_format<tyloi::Fp8SqTensor>(weight); sq8 != nullptr) {
        TYLOINFER_RUNTIME_CHECK(tyloi::mfq_compat_enabled, "FP8-SQ linear requires a compatibility build");
        return sq_matmul(*sq8, x);
    }
    if (const auto* dense = local_format<::tyloi::DenseTensor>(weight); dense != nullptr)
        return dense_projection(x, *dense);
    const auto& mx8 = payload_as<tyloi::Mxfp8Tensor>(weight);
    auto shape = x.sizes().vec();
    auto y = mxfp8_matmul(profiler_, mx8, x.reshape({-1, x.size(-1)}));
    shape.back() = y.size(-1);
    return y.reshape(shape);
}

bool Projector::grouped(std::span<const std::shared_ptr<const ::tyloi::Tensor>> weights) const {
    if (weights.empty()) return false;
    const ::tyloi::encoding::TqTensor* first = nullptr;
    size_t tq_count = 0;
    bool all_nint = options_.nint_grouped && !independent_ && tyloi::mfq_compat_enabled;
    for (const auto& weight : weights) {
        const auto* tq = dynamic_cast<const ::tyloi::encoding::TqTensor*>(weight.get());
        if (tq && tq->is_cuda()) {
            if (first && (weight->size(1) != first->size(1) || tq->device() != first->device())) return false;
            if (!first) first = tq;
            ++tq_count;
        }
#if TYLOINFER_ENABLE_MFQ_COMPAT
        const auto* nint = dynamic_cast<const tyloi::NintTensor*>(weight.get());
        all_nint = all_nint && nint && nint->q_packed.is_cuda() && weight->size(1) == weights.front()->size(1);
#else
        all_nint = false;
#endif
    }
    return tq_count >= 2 || all_nint;
}

std::vector<::tyloi::DenseTensor> Projector::run(
        std::span<const std::shared_ptr<const ::tyloi::Tensor>> weights,
        ::tyloi::DenseTensor x, size_t prefix) const {
    TYLOINFER_RUNTIME_CHECK(prefix >= 1 && prefix <= weights.size(),
        "invalid grouped projection prefix");
    struct {
        std::span<const std::shared_ptr<const ::tyloi::Tensor>> weights;
        QuantLinearGroup group;
        std::vector<size_t> indices;
        size_t prefix;
        bool joined = false;
    } projections{weights, {}, {}, prefix};
    for (const auto& weight : weights) check_projection(*weight, x);
    std::vector<std::shared_ptr<const ::tyloi::Tensor>> tq_weights;
    const ::tyloi::encoding::TqTensor* first = nullptr;
    bool compatible = true;
    for (size_t i = 0; i < weights.size(); ++i) {
        const auto* tq = dynamic_cast<const ::tyloi::encoding::TqTensor*>(weights[i].get());
        if (!tq || !tq->is_cuda()) continue;
        if (first && (weights[i]->size(1) != first->size(1) || tq->device() != first->device())) {
            compatible = false;
            break;
        }
        if (!first) first = tq;
        tq_weights.push_back(weights[i]);
        projections.indices.push_back(i);
    }
    if (compatible && tq_weights.size() >= 2) {
        projections.group = group(tq_weights);
        projections.joined = prefix > 1 && projections.indices.size() >= prefix;
        for (size_t i = 0; projections.joined && i < prefix; ++i)
            projections.joined = projections.indices[i] == i;
        if (projections.joined) projections.group.tq.join_prefix(prefix);
    } else if (compatible && grouped(weights)) {
        projections.group = group(weights);
        auto& nint = projections.group.nint;
        projections.joined = prefix > 1 && (nint.split_w.empty() || nint.split_outs.front().size() >= prefix);
        if (projections.joined) {
            auto& sizes = nint.split_w.empty() ? nint.outs : nint.split_outs.front();
            const auto combined = std::accumulate(sizes.begin(), sizes.begin() + prefix, int64_t(0));
            sizes.erase(sizes.begin(), sizes.begin() + prefix);
            sizes.insert(sizes.begin(), combined);
        }
    }
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
            outputs.push_back(run(*weight, x));
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
            if (!outputs[i].defined()) outputs[i] = run(*projections.weights[i], x);
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
        const ::tyloi::Tensor& weight,
        ::tyloi::DenseTensor grouped,
        int64_t groups) const {
    check_projection(weight, grouped);
    TYLOINFER_RUNTIME_CHECK(
        holds<tyloi::Mxfp8Tensor>(weight),
        "groupwise MXFP8 projection requires an MXFP8 tensor");
    const auto* sharded_weight = dynamic_cast<const tyloi::ShardedTensor*>(&weight);
    if (!sharded_weight) {
        return mxfp8_groupwise_matmul(
            profiler_, payload_as<tyloi::Mxfp8Tensor>(weight), grouped, groups);
    }
    TYLOINFER_RUNTIME_CHECK(
        sharded_weight->axis == TensorParallelAxis::Input,
        "groupwise MXFP8 tensor parallelism requires input-axis shards");
    std::vector<::tyloi::DenseTensor> partials;
    partials.reserve(sharded_weight->shards.size());
    for (const auto & shard : sharded_weight->shards) {
        const auto* shard_mx8 = dynamic_cast<const tyloi::Mxfp8Tensor*>(shard.weight.get());
        TYLOINFER_RUNTIME_CHECK(
            shard_mx8 != nullptr,
            "groupwise MXFP8 tensor-parallel shard representation mismatch");
        MfqCudaGuard guard(shard.device);
        auto local = grouped.narrow(
            -1, shard.input_begin,
            shard.input_end - shard.input_begin);
        local = collectives_.copy(local, shard.device);
        partials.push_back(mxfp8_groupwise_matmul_f32(
            profiler_, *shard_mx8, local, groups));
    }
    return collectives_.sum(std::move(partials), primary_,
            options_.reduce_to_primary, options_.fp16_reduce)
        .to(grouped.scalar_type()).contiguous();
}

::tyloi::DenseTensor Projector::gated(
        const ::tyloi::Tensor& weight,
        ::tyloi::DenseTensor x,
        ::tyloi::DenseTensor gate,
        InputGateMode mode) const {
    if (!x.defined() || x.dim() < 1)
        throw std::invalid_argument("projection input must have a last dimension");
    check_projection_device(weight, x.device());
    auto width = x.size(-1);
    const auto* shards = dynamic_cast<const tyloi::ShardedTensor*>(&weight);
    if ((!shards || shards->axis == TensorParallelAxis::Output) &&
        !holds<tyloi::NintTensor>(weight) && !holds<tyloi::NvqTensor>(weight) &&
        gate.defined() && gate.dim() > 0 && width == 1)
        width = gate.size(-1);
    check_projection_width(weight, width);
    if (const auto* nint = local_format<tyloi::NintTensor>(weight); nint != nullptr) {
        TYLOINFER_RUNTIME_CHECK(tyloi::mfq_compat_enabled, "NINT linear requires a compatibility build");
        auto shape = x.sizes().vec();
        auto y = run_nint_linear(profiler_, mmq_, *nint, x.reshape({-1, x.size(-1)}),
                                 gate.reshape({-1, gate.size(-1)}), mode);
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
    if (const auto* nvq = local_format<tyloi::NvqTensor>(weight); nvq != nullptr) {
        TYLOINFER_RUNTIME_CHECK(tyloi::mfq_compat_enabled, "NVQ linear requires a compatibility build");
        auto shape = x.sizes().vec();
        auto y = run_nvq_linear(profiler_, mmq_, *nvq, x.reshape({-1, x.size(-1)}),
                                gate.reshape({-1, gate.size(-1)}), mode);
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
    if (const auto* sharded_weight = dynamic_cast<const tyloi::ShardedTensor*>(&weight)) {
        auto shape = x.sizes().vec();
        auto y = sharded(*sharded_weight, x.reshape({-1, x.size(-1)}),
                         gate.reshape({-1, gate.size(-1)}), mode);
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
    if (const auto* dense = local_format<::tyloi::DenseTensor>(weight); dense != nullptr) {
        TYLOINFER_RUNTIME_CHECK(mode == InputGateMode::Sigmoid || mode == InputGateMode::Silu,
            "dense input gate mode must be sigmoid or SiLU");
        // Match the existing dense shard path, including dtype rounding.
        auto local = x.to(dense->scalar_type());
        auto local_gate = gate.to(dense->scalar_type());
        auto gated = mode == InputGateMode::Sigmoid
            ? local * ::tyloi::cuda::sigmoid(local_gate)
            : local * ::tyloi::cuda::silu(local_gate);
        return dense_projection(gated, *dense);
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
            result.push_back(run(*layer, x));
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
        const auto* nvq0 = local_format<tyloi::NvqTensor>(*group.layers[0]);
        const auto* nvq1 = local_format<tyloi::NvqTensor>(*group.layers[1]);
        TYLOINFER_RUNTIME_CHECK(nvq0 != nullptr && nvq1 != nullptr,
            "NVQ group fusion requires two local NVQ projections");
        if (nvq_pair_compatible(*nvq0, *nvq1)) {
            check_device(*nvq0, x);
            check_device(*nvq1, x);
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
                        return nvq_matmul_multi2(profiler_, *nvq0, *nvq1, flat);
                    }
                    return run(*group.layers[branch + 1], x);
                },
                branch_outputs);
        auto combined = parallel
            ? branch_outputs[0]
            : nvq_matmul_multi2(profiler_, *nvq0, *nvq1, flat);
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
                    : run(*group.layers[i], x));
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
                    return run(*group.layers[index], x);
                },
                result)) {
        return result;
    }
    result.reserve(group.layers.size());
    for (const auto & layer : group.layers) {
        result.push_back(run(*layer, x));
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
        const auto& nvq0 = payload_as<tyloi::NvqTensor>(*group.layers[0]);
        const auto& nvq1 = payload_as<tyloi::NvqTensor>(*group.layers[1]);
        auto shape = x.sizes().vec();
        auto y = nvq_matmul_swiglu(profiler_, nvq0, nvq1, x.reshape({-1, x.size(-1)}));
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

::tyloi::DenseTensor embed(const ::tyloi::Tensor& weight, ::tyloi::DenseTensor ids,
        Collectives& copies, CudaProfiler& profiler, int device) {
    ids = copies.copy(ids.to(::tyloi::kInt64), device).contiguous();
    return profiler.measure("model.embed", [&] { return quant_embedding_lookup(weight, ids); });
}

::tyloi::DenseTensor logits(const ::tyloi::Tensor& weight, ::tyloi::DenseTensor hidden,
        const Projector& project, CudaProfiler& profiler, double softcap) {
    auto output = profiler.measure("model.lm_head", [&] { return project.run(weight, hidden); });
    return softcap > 0.0 ? ::tyloi::cuda::tanh(output / softcap) * softcap : output;
}

} // namespace tyloi::cuda
