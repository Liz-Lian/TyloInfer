#include "../../runtime/execution_options.h"
#include "mfq_cuda_activation_ops.h"
#include "mfq_cuda_quant_ops.h"
#include "mfq_cuda_nint_route_hint.h"
#include "quant_linear.h"
#include "tyloinfer/format_features.h"
#include "weight_memory.h"

#include "format.h"
#include "fp8_sq.h"
#include "mfe_weight.h"
#include "mx.h"
#include "mxfp4_sq.h"
#include "nint.h"
#include "vq.h"
#include "nvq_codebooks.generated.h"
#include "tyloinfer/mxfp4_sq_decode.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
#define TYLOINFER_CPU_X86_GNU 1
#endif

using mfq_tensor_backend::indexing::Slice;
using namespace mfq::cuda::quant_format;

mfq_tensor_backend::Tensor reduce_model_parallel_outputs(
    CudaExecutionContext& execution,
    std::vector<mfq_tensor_backend::Tensor> outputs);

#if TYLOINFER_ENABLE_MFQ_COMPAT
static NintLinearGroup make_linear_group(const std::vector<NintWeight>& weights);
static mfq_tensor_backend::Tensor dequant_mxfp4_sq_cpu(const Mxfp4SqWeight& weight);

#endif

#if TYLOINFER_ENABLE_MFQ_COMPAT
#include "../mfq/linear_dispatch.inc"
#endif

mfq_tensor_backend::Tensor run_quant_linear_shard(
        CudaProfiler& profiler,
        KlMmqState& kl_mmq,
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
        return profiler.measure("tq.tensor_parallel", [&] { return shard.tq.forward(std::move(x)); });
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

mfq_tensor_backend::Tensor tensor_to_cuda_device(
        ModelParallelCollectiveRuntime& collectives,
        mfq_tensor_backend::Tensor value,
        int device,
        mfq_tensor_backend::Tensor reusable) {
    if (value.is_cuda() && value.get_device() == device) {
        return value.contiguous();
    }
    const auto reusable_matches = [&]() {
        return reusable.defined() && reusable.is_cuda() &&
            reusable.get_device() == device &&
            reusable.sizes() == value.sizes() &&
            reusable.scalar_type() == value.scalar_type() &&
            reusable.is_contiguous();
    };
    const auto destination_tensor = [&]() {
        if (reusable_matches()) return reusable;
        MfqCudaGuard destination_guard(device);
        return mfq_tensor_backend::empty(
            value.sizes(),
            value.options().device(
                mfq_tensor_backend::Device(
                    mfq_tensor_backend::kCUDA, device)));
    };
    if (value.numel() == 0) {
        return destination_tensor();
    }
#if defined(TYLOINFER_NATIVE_CUDA_RUNTIME) && defined(TYLOINFER_HAVE_NCCL)
    auto& runtime = collectives;
    if (value.is_cuda() && runtime.collectives_enabled) {
        const int source_device = value.get_device();
        const auto source_rank_it = std::find(
            runtime.devices.begin(), runtime.devices.end(), source_device);
        const auto destination_rank_it = std::find(
            runtime.devices.begin(), runtime.devices.end(), device);
        if (source_rank_it != runtime.devices.end() &&
                destination_rank_it != runtime.devices.end()) {
            const auto source_stream =
                mfq_get_current_cuda_stream(source_device);
            cudaStreamCaptureStatus capture_status =
                cudaStreamCaptureStatusNone;
            {
                MfqCudaGuard source_guard(source_device);
                TYLOINFER_CUDA_CHECK(cudaStreamIsCapturing(
                    source_stream.stream(), &capture_status));
            }
            if (capture_status != cudaStreamCaptureStatusNone) {
                auto source = value.contiguous();
                auto destination = destination_tensor();
                const auto source_rank = static_cast<size_t>(
                    source_rank_it - runtime.devices.begin());
                const auto destination_rank = static_cast<size_t>(
                    destination_rank_it - runtime.devices.begin());
                {
                    MfqCudaGuard source_guard(source_device);
                    TYLOINFER_CUDA_CHECK(cudaEventRecord(
                        runtime.ready[source_rank],
                        source_stream.stream()));
                    TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(
                        runtime.streams[source_rank].stream(),
                        runtime.ready[source_rank], 0));
                }
                {
                    MfqCudaGuard destination_guard(device);
                    // Cross-device event waits make the receiving NCCL stream
                    // participate in the same capture before ncclRecv runs.
                    TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(
                        runtime.streams[destination_rank].stream(),
                        runtime.ready[source_rank], 0));
                }
                TYLOINFER_NCCL_CHECK(ncclGroupStart());
                TYLOINFER_NCCL_CHECK(ncclSend(
                    source.data_ptr(), source.nbytes(), ncclUint8,
                    static_cast<int>(destination_rank),
                    runtime.communicators[source_rank],
                    runtime.streams[source_rank].stream()));
                TYLOINFER_NCCL_CHECK(ncclRecv(
                    destination.data_ptr(), destination.nbytes(), ncclUint8,
                    static_cast<int>(source_rank),
                    runtime.communicators[destination_rank],
                    runtime.streams[destination_rank].stream()));
                TYLOINFER_NCCL_CHECK(ncclGroupEnd());
                {
                    MfqCudaGuard source_guard(source_device);
                    TYLOINFER_CUDA_CHECK(cudaEventRecord(
                        runtime.completed[source_rank],
                        runtime.streams[source_rank].stream()));
                    TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(
                        source_stream.stream(),
                        runtime.completed[source_rank], 0));
                }
                {
                    MfqCudaGuard destination_guard(device);
                    const auto destination_stream =
                        mfq_get_current_cuda_stream(device);
                    TYLOINFER_CUDA_CHECK(cudaEventRecord(
                        runtime.completed[destination_rank],
                        runtime.streams[destination_rank].stream()));
                    TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(
                        destination_stream.stream(),
                        runtime.completed[destination_rank], 0));
                }
                return destination;
            }
        }
    }
#endif
    MfqCudaGuard guard(device);
    if (reusable_matches()) {
        reusable.copy_(value, true);
        return reusable;
    }
    return value.to(
        value.options().device(mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, device)),
        true, false).contiguous();
}

mfq_tensor_backend::Tensor reduce_model_parallel_outputs(
        CudaExecutionContext& execution,
        std::vector<mfq_tensor_backend::Tensor> outputs) {
    if (outputs.empty()) {
        throw std::runtime_error(
            "cannot reduce an empty model-parallel output");
    }
#ifdef TYLOINFER_HAVE_NCCL
    auto& runtime = execution.model_parallel_collectives;
    if (runtime.collectives_enabled &&
            outputs.size() == runtime.devices.size()) {
        const auto shape = outputs.front().sizes().vec();
        const auto output_dtype = outputs.front().scalar_type();
        const int64_t elements = outputs.front().numel();
        const bool reduce_to_primary =
            execution.config.model_parallel_reduce_to_primary;
        // A two-input FP16 sum has the same final FP16 rounding as the former
        // FP32 reduction, while avoiding both conversion passes.
        const bool fp16_reduce =
            execution.config.model_parallel_fp16_reduce &&
            reduce_to_primary &&
            outputs.size() == 2 &&
            output_dtype == mfq_tensor_backend::kFloat16;
        const int primary = model_parallel_primary_device(execution);
        const auto primary_rank_it = std::find(
            runtime.devices.begin(), runtime.devices.end(), primary);
        if (primary_rank_it == runtime.devices.end()) {
            throw std::runtime_error(
                "model-parallel primary device is absent from NCCL ranks");
        }
        const auto primary_rank = static_cast<size_t>(
            primary_rank_it - runtime.devices.begin());
        for (size_t index = 0; index < outputs.size(); ++index) {
            const int device = runtime.devices[index];
            if (!outputs[index].defined() || !outputs[index].is_cuda() ||
                    outputs[index].get_device() != device ||
                    outputs[index].sizes().vec() != shape ||
                    outputs[index].scalar_type() != output_dtype) {
                throw std::runtime_error(
                    "NCCL model-parallel reduction received mismatched shards");
            }
            MfqCudaGuard guard(device);
            if (fp16_reduce) {
                outputs[index] = outputs[index].contiguous();
            }
            const auto producer =
                mfq_get_current_cuda_stream(device);
            TYLOINFER_CUDA_CHECK(cudaEventRecord(
                runtime.ready[index], producer.stream()));
            const auto communication = runtime.streams[index];
            TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(
                communication.stream(), runtime.ready[index], 0));
            if (fp16_reduce) {
                mfq_cuda_record_stream(outputs[index], communication);
            } else {
                MfqCudaStreamGuard stream_guard(communication);
                auto & buffer = runtime.reduction_buffers[index];
                if (!buffer.defined() || buffer.sizes().vec() != shape ||
                        buffer.get_device() != device ||
                        buffer.scalar_type() != mfq_tensor_backend::kFloat32) {
                    buffer = mfq_tensor_backend::empty(
                        shape,
                        outputs[index].options().dtype(mfq_tensor_backend::kFloat32));
                }
                buffer.copy_(outputs[index], true);
                mfq_cuda_record_stream(outputs[index], communication);
            }
        }

        TYLOINFER_NCCL_CHECK(ncclGroupStart());
        for (size_t index = 0; index < outputs.size(); ++index) {
            auto & buffer = runtime.reduction_buffers[index];
            if (reduce_to_primary) {
                void * reduction_data = fp16_reduce
                    ? outputs[index].data_ptr()
                    : buffer.data_ptr<float>();
                TYLOINFER_NCCL_CHECK(ncclReduce(
                    reduction_data,
                    reduction_data,
                    static_cast<size_t>(elements),
                    fp16_reduce ? ncclFloat16 : ncclFloat32,
                    ncclSum,
                    static_cast<int>(primary_rank),
                    runtime.communicators[index],
                    runtime.streams[index].stream()));
            } else {
                TYLOINFER_NCCL_CHECK(ncclAllReduce(
                    buffer.data_ptr<float>(),
                    buffer.data_ptr<float>(),
                    static_cast<size_t>(elements),
                    ncclFloat32,
                    ncclSum,
                    runtime.communicators[index],
                    runtime.streams[index].stream()));
            }
        }
        TYLOINFER_NCCL_CHECK(ncclGroupEnd());

        mfq_tensor_backend::Tensor result;
        for (size_t index = 0; index < outputs.size(); ++index) {
            MfqCudaGuard guard(runtime.devices[index]);
            const auto communication = runtime.streams[index];
            if (runtime.devices[index] == primary) {
                if (fp16_reduce) {
                    result = outputs[index];
                } else {
                    MfqCudaStreamGuard stream_guard(communication);
                    result = runtime.reduction_buffers[index]
                        .to(output_dtype).contiguous();
                }
            }
            TYLOINFER_CUDA_CHECK(cudaEventRecord(
                runtime.completed[index], communication.stream()));
        }
        MfqCudaGuard primary_guard(primary);
        const auto parent = mfq_get_current_cuda_stream(primary);
        TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(
            parent.stream(), runtime.completed[primary_rank], 0));
        mfq_cuda_record_stream(result, parent);
        return result;
    }
#endif
    const int primary = model_parallel_primary_device(execution);
    MfqCudaGuard primary_guard(primary);
    const auto output_dtype =
        outputs.front().scalar_type();
    mfq_tensor_backend::Tensor reduced;
    for (auto & output : outputs) {
        auto partial =
            tensor_to_cuda_device(
                execution.model_parallel_collectives, output, primary)
                .to(mfq_tensor_backend::kFloat32);
        if (!reduced.defined()) {
            reduced = std::move(partial);
        } else {
            reduced.add_(partial);
        }
    }
    return reduced.to(output_dtype).contiguous();
}

mfq_tensor_backend::Tensor quant_embedding_lookup(
        const QuantLinear & embedding,
        mfq_tensor_backend::Tensor token_ids) {
    TYLOINFER_RUNTIME_CHECK(
        !embedding.tensor_parallel(),
        "quantized embedding does not support tensor-parallel shards");
    if (embedding.is_dense()) {
        auto output_shape = token_ids.sizes().vec();
        output_shape.push_back(embedding.dense.size(1));
        return embedding.dense.index_select(
            0, token_ids.reshape({-1})).reshape(output_shape);
    }
    if (embedding.is_tq()) return embedding.tq.embedding(token_ids);
    if ((mfq::mfq_compat_enabled && embedding.is_nvq())) {
        return nvq_embedding(embedding.nvq, token_ids);
    }
    if ((mfq::mfq_compat_enabled && embedding.is_nint())) {
        if (embedding.nint.q8_zero) {
            return nint8_zero_embedding_lookup_cuda(
                embedding.nint.q_packed,
                embedding.nint.q8_zero_scale,
                token_ids, embedding.nint.neuron_len);
        }
        return nint_embedding_cuda(
            embedding.nint.q_packed,
            embedding.nint.row_q_bits,
            embedding.nint.row_q_bit_offsets,
            embedding.nint.sub_scale,
            embedding.nint.sub_min,
            embedding.nint.neuron_scale,
            embedding.nint.neuron_min,
            token_ids, embedding.nint.neuron_len,
            embedding.nint.gs);
    }
    if (embedding.is_mxfp4()) {
        return mxfp4_embedding_lookup_cuda(
            embedding.mxfp4.weight.values,
            embedding.mxfp4.weight.scales, token_ids);
    }
    if (embedding.is_mxfp8()) {
        return mxfp8_embedding_lookup_cuda(
            embedding.mxfp8.weight.values,
            embedding.mxfp8.weight.scales, token_ids);
    }
    TYLOINFER_RUNTIME_CHECK(
        !(mfq::mfq_compat_enabled && embedding.is_mxfp4_sq()) && !(mfq::mfq_compat_enabled && embedding.is_fp8_sq()),
        "SQ tensors do not support embedding lookup");
    throw std::runtime_error("unsupported quantized embedding kind");
}

bool tensor_parallel_output_projections_compatible(
        const QuantLinearProjectionRefs & projections) {
    if (projections.size() < 2 ||
            !std::all_of(
                projections.begin(), projections.end(),
                [](const QuantLinear * layer) {
                    return layer != nullptr &&
                        layer->tensor_parallel() &&
                        layer->tensor_parallel_axis ==
                            TensorParallelAxis::Output;
                })) {
        return false;
    }
    const size_t shard_count =
        projections.front()->tensor_parallel_shards.size();
    if (shard_count < 2) return false;
    for (const auto * layer : projections) {
        if (layer->tensor_parallel_shards.size() != shard_count) {
            return false;
        }
        for (size_t shard = 0; shard < shard_count; ++shard) {
            const auto & candidate =
                layer->tensor_parallel_shards[shard];
            const auto & reference =
                projections.front()->tensor_parallel_shards[shard];
            if (candidate.device != reference.device ||
                    candidate.input_begin != reference.input_begin ||
                    candidate.input_end != reference.input_end) {
                return false;
            }
        }
    }
    return true;
}

std::vector<mfq_tensor_backend::Tensor>
forward_tensor_parallel_output_projections(
        CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor x,
        const QuantLinearProjectionRefs & projections) {
    TYLOINFER_RUNTIME_CHECK(
        tensor_parallel_output_projections_compatible(projections),
        "tensor-parallel output projections are incompatible");
    auto output_shape = x.sizes().vec();
    auto flat = x.reshape({-1, x.size(-1)});
    const size_t shard_count =
        projections.front()->tensor_parallel_shards.size();
    if (execution.decode_graph_tp_projection_major) {
        std::vector<mfq_tensor_backend::Tensor> local_inputs(shard_count);
        for (size_t launch_position = 0;
             launch_position < shard_count; ++launch_position) {
            const size_t shard = model_parallel_launch_index(
                execution.config, launch_position, shard_count);
            const int device =
                projections.front()->tensor_parallel_shards[shard].device;
            MfqCudaGuard guard(device);
            local_inputs[shard] = tensor_to_cuda_device(
                execution.model_parallel_collectives, flat, device);
        }

        const int primary = model_parallel_primary_device(execution);
        std::vector<mfq_tensor_backend::Tensor> result;
        result.reserve(projections.size());
        for (const auto * projection : projections) {
            std::vector<mfq_tensor_backend::Tensor> local_outputs(shard_count);
            for (size_t launch_position = 0;
                 launch_position < shard_count; ++launch_position) {
                const size_t shard = model_parallel_launch_index(
                execution.config, launch_position, shard_count);
                const auto & weight =
                    projection->tensor_parallel_shards[shard];
                MfqCudaGuard guard(weight.device);
                local_outputs[shard] = run_quant_linear_shard(
                    execution.profiler, execution.kl_mmq,
                    weight, local_inputs[shard]);
            }
            MfqCudaGuard primary_guard(primary);
            std::vector<mfq_tensor_backend::Tensor> gathered;
            gathered.reserve(shard_count);
            for (auto & output : local_outputs) {
                gathered.push_back(tensor_to_cuda_device(
                    execution.model_parallel_collectives,
                    output, primary));
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
        const size_t shard = model_parallel_launch_index(
                execution.config, launch_position, shard_count);
        const int device =
            projections.front()->tensor_parallel_shards[shard].device;
        MfqCudaGuard guard(device);
        // All projections consume the same immutable activation. Transfer it
        // once per rank, then keep the independent local projection work on
        // that rank before gathering each output.
        auto local_x = tensor_to_cuda_device(
            execution.model_parallel_collectives, flat, device);
        for (size_t projection = 0;
                projection < projections.size(); ++projection) {
            local_outputs[projection][shard] =
                run_quant_linear_shard(
                    execution.profiler, execution.kl_mmq,
                    projections[projection]
                        ->tensor_parallel_shards[shard], local_x);
        }
    }

    const int primary = model_parallel_primary_device(execution);
    MfqCudaGuard primary_guard(primary);
    std::vector<mfq_tensor_backend::Tensor> result;
    result.reserve(projections.size());
    for (auto & projection_outputs : local_outputs) {
        std::vector<mfq_tensor_backend::Tensor> gathered;
        gathered.reserve(shard_count);
        for (auto & output : projection_outputs) {
            gathered.push_back(tensor_to_cuda_device(
                execution.model_parallel_collectives,
                output, primary));
        }
        auto combined = mfq_tensor_backend::cat(
            gathered, -1).contiguous();
        auto shape = output_shape;
        shape.back() = combined.size(-1);
        result.push_back(combined.reshape(shape));
    }
    return result;
}

QuantLinearGroup make_quant_group(
        CudaExecutionContext& execution,
        std::vector<QuantLinear> layers,
        bool preserve_projection_boundaries) {
    const mfq::cuda::WeightAllocationScope weight_memory;
    if (layers.empty()) throw std::runtime_error("empty quantized linear group");
    QuantLinearGroup result;
    result.decode_branch_parallel = !preserve_projection_boundaries;
    result.outs.reserve(layers.size());
    bool all_nint = true;
    std::vector<NintWeight> nint_weights;
    nint_weights.reserve(layers.size());
    for (const auto & layer : layers) {
        result.outs.push_back(layer.out());
        all_nint = all_nint && (mfq::mfq_compat_enabled && layer.is_nint());
        if ((mfq::mfq_compat_enabled && layer.is_nint()) && !layer.tensor_parallel()) {
            nint_weights.push_back(layer.nint);
        }
    }
    const bool diagnostic_keep_nint_separate =
        !execution.config.diagnostic_nint_group;
    const bool cpu_layer = execution.loading_cpu_layer;
    const bool any_tensor_parallel = std::any_of(
        layers.begin(), layers.end(),
        [](const QuantLinear & layer) {
            return layer.tensor_parallel();
        });
    const bool all_tq = std::all_of(layers.begin(),layers.end(),
        [](const QuantLinear& layer){return layer.is_tq() && layer.tq.image.is_cuda();});
    if(all_tq && !any_tensor_parallel) {
        std::vector<TqWeight> weights;weights.reserve(layers.size());
        for(const auto& layer:layers)weights.push_back(layer.tq);
        result.tq=TqProjectionGroup(std::move(weights));
    }
    if (mfq::mfq_compat_enabled && all_nint && !preserve_projection_boundaries &&
        !diagnostic_keep_nint_separate && !cpu_layer &&
        !any_tensor_parallel) {
        result.nint_grouped = true;
#if TYLOINFER_ENABLE_MFQ_COMPAT
        result.nint = make_linear_group(nint_weights);
#endif
    } else {
        result.layers = std::move(layers);
        result.nvq_prefix2 = !cpu_layer && result.layers.size() >= 2 &&
            !any_tensor_parallel &&
            (mfq::mfq_compat_enabled && result.layers[0].is_nvq()) && (mfq::mfq_compat_enabled && result.layers[1].is_nvq()) &&
            nvq_pair_compatible(result.layers[0].nvq, result.layers[1].nvq);
    }
    return result;
}

static bool quant_linear_pair_compatible(const QuantLinear & a, const QuantLinear & b) {
    if (a.tensor_parallel() || b.tensor_parallel()) {
        return a.tensor_parallel() && b.tensor_parallel() &&
            a.kind == b.kind &&
            a.tensor_parallel_axis == b.tensor_parallel_axis &&
            a.tensor_parallel_shards.size() ==
                b.tensor_parallel_shards.size();
    }
    if (a.kind != b.kind) return false;
    if (a.is_tq()) return a.tq.neuron_len == b.tq.neuron_len;
    if ((mfq::mfq_compat_enabled && a.is_nvq())) return nvq_pair_compatible(a.nvq, b.nvq);
    if (a.is_mxfp8()) {
        return a.mxfp8.weight.neuron_len == b.mxfp8.weight.neuron_len;
    }
    if (a.is_mxfp4()) {
        return a.mxfp4.weight.neuron_len == b.mxfp4.weight.neuron_len;
    }
    if ((mfq::mfq_compat_enabled && a.is_mxfp4_sq())) {
        return a.mxfp4_sq.weight.neuron_len ==
            b.mxfp4_sq.weight.neuron_len;
    }
    if ((mfq::mfq_compat_enabled && a.is_fp8_sq())) {
        return a.fp8_sq.weight.dtype == b.fp8_sq.weight.dtype &&
            a.fp8_sq.weight.neuron_len == b.fp8_sq.weight.neuron_len &&
            a.fp8_sq.weight.block_rows == b.fp8_sq.weight.block_rows &&
            a.fp8_sq.weight.block_columns == b.fp8_sq.weight.block_columns &&
            a.fp8_sq.weight.scale_kind == b.fp8_sq.weight.scale_kind;
    }
    if (a.is_dense()) {
        return a.dense.size(1) == b.dense.size(1);
    }
    const auto & x = a.nint;
    const auto & y = b.nint;
    return x.ng == y.ng && x.gs == y.gs &&
        x.neuron_len == y.neuron_len && x.q8_zero == y.q8_zero;
}

DenseLinearGroup make_dense_group(const std::vector<mfq_tensor_backend::Tensor> & ws) {
    if (ws.empty()) throw std::runtime_error("empty dense group");
    DenseLinearGroup g;
    std::vector<mfq_tensor_backend::Tensor> parts;
    parts.reserve(ws.size());
    for (const auto & w : ws) {
        if (w.dim() != 2) throw std::runtime_error("dense linear group expects 2D weights");
        if (!parts.empty() && w.size(1) != parts[0].size(1)) {
            throw std::runtime_error("cannot group dense tensors with different input width");
        }
        g.outs.push_back(w.size(0));
        parts.push_back(w.to(mfq_tensor_backend::kFloat32));
    }
    g.w = mfq_tensor_backend::cat(parts, 0).contiguous();
    return g;
}

#if TYLOINFER_ENABLE_MFQ_COMPAT
#include "../mfq/linear_reference.inc"
#endif

static mfq_tensor_backend::Tensor dequant_quant_linear_f32(
        CudaExecutionContext& execution,
        const QuantLinear& linear) {
    if (!linear.tensor_parallel()) {
        if (linear.is_tq()) {
            return quant_linear_reference_weight(linear).to(mfq_tensor_backend::kFloat32).contiguous();
        }
#if TYLOINFER_ENABLE_MFQ_COMPAT
        if ((mfq::mfq_compat_enabled && linear.is_nint())) {
            return dequant_nint_dense_f32(linear.nint);
        }
#endif
        if ((mfq::mfq_compat_enabled && linear.is_nvq())) {
            return nvq_dequant(linear.nvq)
                .to(mfq_tensor_backend::kFloat32).contiguous();
        }
        if (linear.is_mxfp8()) {
            return mxfp8_dequant_cuda(
                linear.mxfp8.weight.values,
                linear.mxfp8.weight.scales)
                .to(mfq_tensor_backend::kFloat32).contiguous();
        }
        if (linear.is_mxfp4()) {
            return mxfp4_dequant_cuda(
                linear.mxfp4.weight.values,
                linear.mxfp4.weight.scales)
                .to(mfq_tensor_backend::kFloat32).contiguous();
        }
#if TYLOINFER_ENABLE_MFQ_COMPAT
        if ((mfq::mfq_compat_enabled && linear.is_mxfp4_sq())) {
            const auto & weight = linear.mxfp4_sq.weight;
            if (!weight.blob.is_cuda()) return dequant_mxfp4_sq_cpu(weight);
            return mxfp4_sq_dequant_cuda(
                weight.blob,
                weight.row_q,
                weight.row_symbol_byte_offsets,
                weight.row_auxiliary,
                weight.bits,
                weight.out,
                weight.neuron_len,
                weight.matrix_scale_base,
                weight.q_sum,
                weight.sq4_rows,
                true).contiguous();
        }
#endif
        if ((mfq::mfq_compat_enabled && linear.is_fp8_sq())) {
            return dequant_fp8_sq(linear.fp8_sq.weight, true).contiguous();
        }
        if (linear.is_dense()) {
            return linear.dense.to(mfq_tensor_backend::kFloat32).contiguous();
        }
        throw std::runtime_error(
            "unsupported linear kind for FP32 reconstruction");
    }
    if (linear.tensor_parallel_axis != TensorParallelAxis::Output &&
        linear.tensor_parallel_axis != TensorParallelAxis::Input) {
        throw std::runtime_error(
            "cannot reconstruct a mirrored tensor-parallel linear");
    }
    const int primary = model_parallel_primary_device(execution);
    std::vector<mfq_tensor_backend::Tensor> parts;
    parts.reserve(linear.tensor_parallel_shards.size());
    for (const auto & shard : linear.tensor_parallel_shards) {
        MfqCudaGuard shard_guard(shard.device);
        mfq_tensor_backend::Tensor part;
        if (shard.kind == QuantLinearKind::Tq) {
            QuantLinear local;
            local.kind=QuantLinearKind::Tq;local.tq=shard.tq;
            part=quant_linear_reference_weight(local).to(mfq_tensor_backend::kFloat32).contiguous();
        } else
#if TYLOINFER_ENABLE_MFQ_COMPAT
        if ((mfq::mfq_compat_enabled && shard.kind == QuantLinearKind::Nint)) {
            part = dequant_nint_dense_f32(shard.nint);
        } else if ((mfq::mfq_compat_enabled && shard.kind == QuantLinearKind::Nvq)) {
            part = nvq_dequant(shard.nvq)
                .to(mfq_tensor_backend::kFloat32).contiguous();
        } else
#endif
        if (shard.kind == QuantLinearKind::Mxfp8) {
            part = mxfp8_dequant_cuda(
                shard.mxfp8.values,
                shard.mxfp8.scales)
                .to(mfq_tensor_backend::kFloat32).contiguous();
        } else if (shard.kind == QuantLinearKind::Mxfp4) {
            part = mxfp4_dequant_cuda(
                shard.mxfp4.values, shard.mxfp4.scales)
                .to(mfq_tensor_backend::kFloat32).contiguous();
        } else if ((mfq::mfq_compat_enabled && shard.kind == QuantLinearKind::Mxfp4Sq) ||
                (mfq::mfq_compat_enabled && shard.kind == QuantLinearKind::Fp8Sq)) {
            QuantLinear local;
            local.kind = shard.kind;
            local.mxfp4_sq = shard.mxfp4_sq;
            local.fp8_sq = shard.fp8_sq;
            part = dequant_quant_linear_f32(execution, local);
        } else if (shard.kind == QuantLinearKind::Dense) {
            part = shard.dense.to(mfq_tensor_backend::kFloat32).contiguous();
        } else {
            throw std::runtime_error(
                "unsupported tensor-parallel shard kind for reconstruction");
        }
        parts.push_back(
            tensor_to_cuda_device(
                execution.model_parallel_collectives, part, primary)
                .to(mfq_tensor_backend::kFloat32).contiguous());
    }
    MfqCudaGuard primary_guard(primary);
    auto dense = mfq_tensor_backend::cat(
        parts,
        linear.tensor_parallel_axis == TensorParallelAxis::Output
            ? 0 : 1).contiguous();
    if (dense.size(0) != linear.out() ||
        dense.size(1) != linear.neuron_len()) {
        throw std::runtime_error(
            "reconstructed tensor-parallel linear shape mismatch");
    }
    return dense;
}

DenseLinearGroup make_fp32_quant_group(
        CudaExecutionContext& execution,
        QuantLinearGroup group) {
    DenseLinearGroup dense;
    dense.outs = group.outs;
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (group.nint_grouped) {
        if (group.nint.split_w.empty()) {
            dense.w = dequant_nint_dense_f32(group.nint.w);
        } else {
            std::vector<mfq_tensor_backend::Tensor> parts;
            parts.reserve(group.nint.split_w.size());
            for (const auto & weight : group.nint.split_w) {
                parts.push_back(dequant_nint_dense_f32(weight));
            }
            dense.w = mfq_tensor_backend::cat(parts, 0).contiguous();
        }
    } else
#endif
    {
        std::vector<mfq_tensor_backend::Tensor> parts;
        parts.reserve(group.layers.size());
        for (const auto & linear : group.layers) {
            parts.push_back(dequant_quant_linear_f32(
                execution, linear));
        }
        dense.w = mfq_tensor_backend::cat(parts, 0).contiguous();
    }
    TYLOINFER_RUNTIME_CHECK(
        dense.w.dim() == 2 &&
            dense.w.size(0) ==
                std::accumulate(
                    dense.outs.begin(), dense.outs.end(), int64_t{0}),
        "FP32 compressor projection shape mismatch");
    return dense;
}

mfq_tensor_backend::Tensor quant_linear_reference_weight(
        const QuantLinear& linear) {
    TYLOINFER_RUNTIME_CHECK(!linear.tensor_parallel(),"linear reference requires local weights");
    if (linear.is_tq()) {
        const auto ids=mfq_tensor_backend::arange(linear.tq.out,
            linear.tq.image.options().dtype(mfq_tensor_backend::kInt64));
        return linear.tq.embedding(ids);
    }
    if ((mfq::mfq_compat_enabled && linear.is_nint())) {
        const auto& weight = linear.nint;
        return weight.q8_zero
            ? nint8_zero_dequant_cuda(
                  weight.q_packed, weight.q8_zero_scale,
                  weight.neuron_len)
            : nint_decode_cuda(
                  weight.q_packed, weight.row_q_bits,
                  weight.row_q_bit_offsets, weight.sub_scale,
                  weight.sub_min, weight.neuron_scale,
                  weight.neuron_min, weight.neuron_len, weight.gs);
    }
    if ((mfq::mfq_compat_enabled && linear.is_nvq())) return nvq_dequant(linear.nvq);
    if (linear.is_mxfp4()) {
        return mxfp4_dequant_cuda(
            linear.mxfp4.weight.values, linear.mxfp4.weight.scales);
    }
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if ((mfq::mfq_compat_enabled && linear.is_mxfp4_sq())) {
        const auto& weight = linear.mxfp4_sq.weight;
        if (!weight.blob.is_cuda()) {
            return dequant_mxfp4_sq_cpu(weight)
                .to(mfq_tensor_backend::kFloat16);
        }
        return mxfp4_sq_dequant_cuda(
            weight.blob, weight.row_q, weight.row_symbol_byte_offsets,
            weight.row_auxiliary, weight.bits, weight.out,
            weight.neuron_len, weight.matrix_scale_base,
            weight.q_sum, weight.sq4_rows, false);
    }
#endif
    if ((mfq::mfq_compat_enabled && linear.is_fp8_sq())) {
        return dequant_fp8_sq(linear.fp8_sq.weight, false);
    }
    if (linear.is_mxfp8()) {
        return mxfp8_cpu_reference(linear.mxfp8.weight);
    }
    if (linear.is_dense()) return linear.dense;
    throw std::runtime_error("unsupported linear reference format");
}

mfq_tensor_backend::Tensor QuantLinear::forward_tensor_parallel_flat(
        CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor x,
        MfqOptional<mfq_tensor_backend::Tensor> gate,
        InputGateMode gate_mode) const {
    TYLOINFER_RUNTIME_CHECK(
        tensor_parallel(),
        "tensor-parallel linear has no shards");
    TYLOINFER_RUNTIME_CHECK(
        tensor_parallel_axis == TensorParallelAxis::Output ||
        tensor_parallel_axis == TensorParallelAxis::Input,
        "tensor-parallel linear has an invalid axis");
    std::vector<mfq_tensor_backend::Tensor> local_outputs(
        tensor_parallel_shards.size());
    for (size_t launch_position = 0;
         launch_position < tensor_parallel_shards.size();
         ++launch_position) {
        const size_t index = model_parallel_launch_index(
            execution.config, launch_position, tensor_parallel_shards.size());
        const auto & shard = tensor_parallel_shards[index];
        MfqCudaGuard guard(shard.device);
        mfq_tensor_backend::Tensor local_x = x;
        mfq_tensor_backend::Tensor local_gate;
        if (tensor_parallel_axis == TensorParallelAxis::Input) {
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
        local_x = tensor_to_cuda_device(
            execution.model_parallel_collectives,
            local_x, shard.device);
        if (gate.has_value()) {
            local_gate =
                tensor_to_cuda_device(
                    execution.model_parallel_collectives,
                    local_gate, shard.device);
        }
        if (is_mxfp8() &&
                tensor_parallel_axis == TensorParallelAxis::Input) {
            TYLOINFER_RUNTIME_CHECK(
                !gate.has_value(),
                "MXFP8 input-axis tensor parallelism does not support gating");
            local_outputs[index] =
                mxfp8_matmul_f32(
                    execution.profiler, shard.mxfp8, local_x);
        } else {
            local_outputs[index] =
                run_quant_linear_shard(
                    execution.profiler, execution.kl_mmq,
                    shard, local_x,
                    gate.has_value()
                        ? MfqOptional<mfq_tensor_backend::Tensor>(
                            local_gate)
                        : mfq_nullopt,
                    gate_mode);
        }
    }

    const int primary = model_parallel_primary_device(execution);
    MfqCudaGuard primary_guard(primary);
    if (tensor_parallel_axis == TensorParallelAxis::Output) {
        std::vector<mfq_tensor_backend::Tensor> gathered;
        gathered.reserve(local_outputs.size());
        for (auto & output : local_outputs) {
            gathered.push_back(
                tensor_to_cuda_device(
                    execution.model_parallel_collectives,
                    output, primary));
        }
        return mfq_tensor_backend::cat(gathered, -1).contiguous();
    }

    auto reduced = reduce_model_parallel_outputs(
        execution, std::move(local_outputs));
    return is_mxfp8()
        ? reduced.to(x.scalar_type()).contiguous()
        : reduced;
}

mfq_tensor_backend::Tensor dense_projection(
        mfq_tensor_backend::Tensor input, const mfq_tensor_backend::Tensor& weight) {
    input = input.to(weight.scalar_type());
    const int64_t rows = input.numel() / input.size(-1);
    if (rows > 1 && rows <= 6) {
        auto shape = input.sizes().vec();
        shape.back() = weight.size(0);
        auto flat = input.reshape({rows, input.size(-1)});
        std::vector<mfq_tensor_backend::Tensor> outputs;
        outputs.reserve(rows);
        // A strided-batched GEMM may select different accumulation from M=1.
        // Keep short target/predictor windows on the decode projection path.
        // ponytail: at most six launches; fuse only with decode-equivalent accumulation.
        for (int64_t row = 0; row < rows; ++row)
            outputs.push_back(mfq_tensor_backend::matmul(flat.narrow(0, row, 1), weight.transpose(0, 1)));
        return mfq_tensor_backend::cat(outputs, 0).reshape(shape);
    }
    return mfq_tensor_backend::matmul(input, weight.transpose(0, 1));
}

mfq_tensor_backend::Tensor QuantLinear::forward_dense(mfq_tensor_backend::Tensor x) const {
    return dense_projection(std::move(x), dense);
}

mfq_tensor_backend::Tensor QuantLinear::forward(
        CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor x) const {
    if(is_tq() && !tensor_parallel())return execution.profiler.measure("tq.linear",[&] {return tq.forward(std::move(x));});
    if (tensor_parallel() || (mfq::mfq_compat_enabled && is_nint()) || (mfq::mfq_compat_enabled && is_nvq())) {
        auto shape = x.sizes().vec();
        auto flat = x.reshape({-1, x.size(-1)});
#if TYLOINFER_ENABLE_MFQ_COMPAT
        auto y = tensor_parallel()
            ? forward_tensor_parallel_flat(execution, flat, mfq_nullopt, InputGateMode::None)
            : (mfq::mfq_compat_enabled && is_nint()) ? run_nint_linear(
                  execution.profiler, execution.kl_mmq, nint, flat)
                        : run_nvq_linear(
                  execution.profiler, execution.kl_mmq, nvq, flat);
#else
        auto y = forward_tensor_parallel_flat(execution, flat, mfq_nullopt, InputGateMode::None);
#endif
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
    if (is_mxfp4()) return mxfp4.forward(x);
    if ((mfq::mfq_compat_enabled && is_mxfp4_sq())) return mxfp4_sq.forward(x);
    if ((mfq::mfq_compat_enabled && is_fp8_sq())) return fp8_sq.forward(x);
    if (is_dense()) return forward_dense(x);
    return mxfp8.forward(execution.profiler, x);
}

mfq_tensor_backend::Tensor QuantLinear::forward_bf16_output(
        CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor x) const {
    return forward(execution, x)
        .to(mfq_tensor_backend::kBFloat16).contiguous();
}

mfq_tensor_backend::Tensor QuantLinear::forward_mxfp8_groupwise(
        CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor grouped,
        int64_t groups) const {
    TYLOINFER_RUNTIME_CHECK(
        is_mxfp8(),
        "groupwise MXFP8 projection requires an MXFP8 tensor");
    if (!tensor_parallel()) {
        return mxfp8_groupwise_matmul(
            execution.profiler, mxfp8.weight, grouped, groups);
    }
    TYLOINFER_RUNTIME_CHECK(
        tensor_parallel_axis == TensorParallelAxis::Input,
        "groupwise MXFP8 tensor parallelism requires input-axis shards");
    std::vector<mfq_tensor_backend::Tensor> partials;
    partials.reserve(tensor_parallel_shards.size());
    for (const auto & shard : tensor_parallel_shards) {
        TYLOINFER_RUNTIME_CHECK(
            shard.kind == QuantLinearKind::Mxfp8,
            "groupwise MXFP8 tensor-parallel shard kind mismatch");
        MfqCudaGuard guard(shard.device);
        auto local = grouped.narrow(
            -1, shard.input_begin,
            shard.input_end - shard.input_begin);
        local = tensor_to_cuda_device(
            execution.model_parallel_collectives,
            local, shard.device);
        partials.push_back(mxfp8_groupwise_matmul_f32(
            execution.profiler, shard.mxfp8, local, groups));
    }
    return reduce_model_parallel_outputs(execution, std::move(partials))
        .to(grouped.scalar_type()).contiguous();
}

mfq_tensor_backend::Tensor QuantLinear::forward_input_mul(
        CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor x,
        mfq_tensor_backend::Tensor gate,
        InputGateMode mode) const {
    if (tensor_parallel() || (mfq::mfq_compat_enabled && is_nint()) || (mfq::mfq_compat_enabled && is_nvq())) {
        auto shape = x.sizes().vec();
        auto flat = x.reshape({-1, x.size(-1)});
        auto flat_gate = gate.reshape({-1, gate.size(-1)});
#if TYLOINFER_ENABLE_MFQ_COMPAT
        auto y = tensor_parallel()
            ? forward_tensor_parallel_flat(execution, flat, flat_gate, mode)
            : (mfq::mfq_compat_enabled && is_nint()) ? run_nint_linear(
                  execution.profiler, execution.kl_mmq,
                  nint, flat, flat_gate, mode)
                        : run_nvq_linear(
                  execution.profiler, execution.kl_mmq,
                  nvq, flat, flat_gate, mode);
#else
        auto y = forward_tensor_parallel_flat(execution, flat, flat_gate, mode);
#endif
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
    if (is_dense()) {
        TYLOINFER_RUNTIME_CHECK(mode == InputGateMode::Sigmoid || mode == InputGateMode::Silu,
            "dense input gate mode must be sigmoid or SiLU");
        // Match the existing dense shard path, including dtype rounding.
        auto local = x.to(dense.scalar_type());
        auto local_gate = gate.to(dense.scalar_type());
        auto gated = mode == InputGateMode::Sigmoid
            ? local * mfq_tensor_backend::sigmoid(local_gate)
            : local * mfq_tensor_backend::silu(local_gate);
        return forward_dense(gated);
    }
    TYLOINFER_RUNTIME_CHECK(mode == InputGateMode::Sigmoid || mode == InputGateMode::Silu,
        "input gate mode must be sigmoid or SiLU");
    auto local_gate = gate.to(x.scalar_type());
    auto gated = mode == InputGateMode::Sigmoid
        ? x * mfq_tensor_backend::sigmoid(local_gate)
        : x * mfq_tensor_backend::silu(local_gate);
    return forward(execution, gated);
}

mfq_tensor_backend::Tensor QuantLinear::forward_input_mul_f32_kld(
        CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor x,
        mfq_tensor_backend::Tensor gate,
        InputGateMode mode) const {
    TYLOINFER_RUNTIME_CHECK(
        !tensor_parallel() && (mfq::mfq_compat_enabled && is_nint()),
        "FP32-output KLD down projection requires a local NINT tensor");
    TYLOINFER_RUNTIME_CHECK(
        execution.kl_mmq.mode == KlMmqMode::Fp16,
        "FP32-output NINT MMQ is restricted to the FP16 KLD path");
    auto shape = x.sizes().vec();
    auto y = nint_matmul_input_mul_f32(
        execution.profiler, nint, x.reshape({-1, x.size(-1)}),
        gate.reshape({-1, gate.size(-1)}), static_cast<int>(mode));
    if (nint.q8_zero) ++execution.kl_mmq.dense_calls;
    shape.back() = y.size(-1);
    return y.reshape(shape);
}

int64_t QuantLinear::out() const {
    if (tensor_parallel()) return logical_out;
    if(is_tq())return tq.out;
    if ((mfq::mfq_compat_enabled && is_nint())) return nint.out;
    if ((mfq::mfq_compat_enabled && is_nvq())) return nvq.out;
    if (is_mxfp4()) return mxfp4.weight.out;
    if ((mfq::mfq_compat_enabled && is_mxfp4_sq())) return mxfp4_sq.weight.out;
    if ((mfq::mfq_compat_enabled && is_fp8_sq())) return fp8_sq.weight.out;
    if (is_dense()) return dense.size(0);
    return mxfp8.weight.out;
}

int64_t QuantLinear::neuron_len() const {
    if (tensor_parallel()) return logical_neuron_len;
    if(is_tq())return tq.neuron_len;
    if ((mfq::mfq_compat_enabled && is_nint())) return nint.neuron_len;
    if ((mfq::mfq_compat_enabled && is_nvq())) return nvq.neuron_len;
    if (is_mxfp4()) return mxfp4.weight.neuron_len;
    if ((mfq::mfq_compat_enabled && is_mxfp4_sq())) return mxfp4_sq.weight.neuron_len;
    if ((mfq::mfq_compat_enabled && is_fp8_sq())) return fp8_sq.weight.neuron_len;
    if (is_dense()) return dense.size(1);
    return mxfp8.weight.neuron_len;
}

QuantLinearProjectionRefs QuantLinearGroup::tensor_parallel_output_projections() const {
    QuantLinearProjectionRefs projections;
    projections.reserve(layers.size());
    for (const auto & layer : layers) {
        projections.push_back(&layer);
    }
    return projections;
}

bool QuantLinearGroup::tensor_parallel_output_compatible() const {
    return tensor_parallel_output_projections_compatible(
        tensor_parallel_output_projections());
}

std::vector<mfq_tensor_backend::Tensor>
QuantLinearGroup::forward_tensor_parallel_output_group(
        CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor x) const {
    return forward_tensor_parallel_output_projections(
        execution, x, tensor_parallel_output_projections());
}

std::vector<mfq_tensor_backend::Tensor> QuantLinearGroup::forward(
        CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor x) const {
    if(tq.defined())return execution.profiler.measure("tq.parallel",[&] {return tq.forward(std::move(x));});
    const bool default_mmq =
        execution.kl_mmq.mode == KlMmqMode::Default;
    if (!x.is_cuda()) {
        TYLOINFER_RUNTIME_CHECK(
            !nint_grouped,
            "CPU dense offload requires separate compact linear weights");
        std::vector<mfq_tensor_backend::Tensor> result;
        result.reserve(layers.size());
        for (const auto & layer : layers) {
            result.push_back(layer.forward(execution, x));
        }
        return result;
    }
    if (execution.config.tensor_parallel_grouped_projections &&
            tensor_parallel_output_compatible()) {
        return forward_tensor_parallel_output_group(execution, x);
    }
    if (mfq::mfq_compat_enabled && nint_grouped) {
        return nint.forward(
            execution.profiler, execution.kl_mmq,
            execution.config, execution.decode_graph_serial_branches, x);
    }
    if (mfq::mfq_compat_enabled && default_mmq && nvq_prefix2 && nvq_fusion_enabled(execution.config)) {
        auto shape = x.sizes().vec();
        const auto flat =
            x.reshape({-1, x.size(-1)});
        std::vector<mfq_tensor_backend::Tensor> branch_outputs;
        const bool parallel =
            decode_branch_parallel &&
            decode_branch_parallel_enabled(
                execution.config, execution.decode_graph_serial_branches, flat.size(0)) &&
            layers.size() > 2 &&
            branch_executor->run(
                layers.size() - 1,
                [&](size_t branch) {
                    if (branch == 0) {
                        return nvq_matmul_multi2(
                            execution.profiler, layers[0].nvq,
                            layers[1].nvq, flat);
                    }
                    return layers[branch + 1].forward(execution, x);
                },
                branch_outputs);
        auto combined = parallel
            ? branch_outputs[0]
            : nvq_matmul_multi2(
                execution.profiler, layers[0].nvq,
                layers[1].nvq, flat);
        auto pair = combined.split_with_sizes({outs[0], outs[1]}, -1);
        std::vector<mfq_tensor_backend::Tensor> result;
        result.reserve(layers.size());
        for (size_t i = 0; i < 2; ++i) {
            auto part_shape = shape;
            part_shape.back() = outs[i];
            result.push_back(pair[i].reshape(part_shape));
        }
        for (size_t i = 2; i < layers.size(); ++i) {
            result.push_back(
                parallel
                    ? branch_outputs[i - 1]
                    : layers[i].forward(execution, x));
        }
        return result;
    }
    std::vector<mfq_tensor_backend::Tensor> result;
    if (default_mmq && decode_branch_parallel &&
            decode_branch_parallel_enabled(
                execution.config, execution.decode_graph_serial_branches, x.numel() / x.size(-1)) &&
            branch_executor->run(
                layers.size(),
                [&](size_t index) {
                    return layers[index].forward(execution, x);
                },
                result)) {
        return result;
    }
    result.reserve(layers.size());
    for (const auto & layer : layers) {
        result.push_back(layer.forward(execution, x));
    }
    return result;
}

mfq_tensor_backend::Tensor QuantLinearGroup::forward_swiglu(
        CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor x) const {
    if(tq.defined() && outs.size()==2 && outs[0]==outs[1])
        return execution.profiler.measure("tq.swiglu",[&] {return tq.swiglu(std::move(x));});
    const bool default_mmq =
        execution.kl_mmq.mode == KlMmqMode::Default;
    if (mfq::mfq_compat_enabled && default_mmq && nint_grouped && nint.split_w.empty() &&
            x.numel() / x.size(-1) >= 1 && x.numel() / x.size(-1) <= 6) {
        return nint.forward_swiglu(execution.profiler, x);
    }
    if (mfq::mfq_compat_enabled && default_mmq && nvq_prefix2 && layers.size() == 2 &&
            nvq_fusion_enabled(execution.config)) {
        auto shape = x.sizes().vec();
        auto y = nvq_matmul_swiglu(
            execution.profiler, layers[0].nvq, layers[1].nvq,
            x.reshape({-1, x.size(-1)}));
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
    if (outs.size() != 2 || outs[0] != outs[1]) {
        throw std::runtime_error("SwiGLU requires equal gate/up output widths");
    }
    auto parts = forward(execution, x);
    return mfq_tensor_backend::silu(parts[0]) * parts[1];
}

mfq_tensor_backend::Tensor QuantLinearGroup::forward_geglu(
        CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor x) const {
#if TYLOINFER_ENABLE_MFQ_COMPAT
        if (execution.kl_mmq.mode == KlMmqMode::Default &&
            nint_grouped && nint.split_w.empty() &&
            x.numel() / x.size(-1) == 1) {
        return nint.forward_geglu(execution.profiler, x);
    }
#endif

    if (outs.size() != 2 || outs[0] != outs[1]) {
        throw std::runtime_error("GeGLU requires equal gate/up output widths");
    }
    auto parts = forward(execution, x);
    return gelu_mul_cuda(parts[0].contiguous(), parts[1].contiguous());
}

std::vector<mfq_tensor_backend::Tensor> DenseLinearGroup::forward(mfq_tensor_backend::Tensor x) const {
    auto shape = x.sizes().vec();
    auto y = dense_projection(x.reshape({-1, x.size(-1)}), w);
    auto parts = y.split_with_sizes(outs, -1);
    for (auto & p : parts) {
        auto s = shape;
        s.back() = p.size(-1);
        p = p.reshape(s);
    }
    return parts;
}

#if TYLOINFER_ENABLE_MFQ_COMPAT
#include "../mfq/linear_groups.inc"
#endif
