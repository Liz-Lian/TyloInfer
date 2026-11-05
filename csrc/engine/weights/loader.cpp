#include "engine/weights/mapped_embedding.h"
#include "engine/backends/cuda/native/device.h"
#include "engine/weights/moe/experts.h"
#include <iostream>
#include "engine/weights/moe/load.h"
#include "env.h"
#include "engine/weights/loader.h"
#include "engine/backends/cuda/ops/projection/projection.h"
#include "engine/weights/group.h"
#include "engine/weights/quant.h"
#include "engine/weights/encoding/dense_blob.h"
#include "engine/weights/memory.h"
#include "engine/backends/cuda/kernels/mfq_cuda_quant_ops.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>
#include "engine/backends/cuda/ops/ffn/moe.h"
#include "engine/weights/moe/expert_cache.h"
#include "engine/weights/moe/expert_store.h"
#include "engine/weights/moe/quant_range_source.h"
#include "engine/weights/tq/load.h"
#include "engine/weights/tq/image.h"
#include "engine/weights/tq/source.h"
#include "engine/weights/encoding/quant/packed_nint_expert_store.h"
#include <cstdlib>
#include <cstdio>
#include <atomic>

void LayerPlacementConfig::prepare(int64_t layers) {
    layer_devices.assign(static_cast<size_t>(layers), primary_device());
    if (!enabled()) return;
    const auto slices = tyloi::plan_tensor_parallel_slices(
        layers, 1, devices, split);
    for (const auto& slice : slices) {
        for (int64_t layer = slice.begin; layer < slice.end; ++layer) {
            layer_devices.at(static_cast<size_t>(layer)) = slice.device;
        }
    }
}

int LayerPlacementConfig::device_for_layer(int64_t layer) const {
    if (layer < 0 || layer >= static_cast<int64_t>(layer_devices.size())) {
        throw std::runtime_error("layer-placement index is outside the model");
    }
    return layer_devices.at(static_cast<size_t>(layer));
}

static std::vector<std::uint8_t> read_weight(
        bool drop_file_cache,
        const tyloi::source::Source& source,
        std::string_view name) {
    auto result = tyloi::read_tensor(source, name);
    if (drop_file_cache) source.drop_file_cache();
    return result;
}

::tyloi::DenseTensor tyloi::cuda::weight_loader::Loader::tensor(const std::string& name) const {
    const auto& mfq = source;
    const tyloi::cuda::AllocationScope weight_memory;
    const bool cpu_layer = placement.cpu;
    MfqCudaGuard guard(placement.device);
    const auto & rec = tyloi::require_tensor(mfq, name);
    if(rec.dtype=="TQ") {
        // This entry is for parameters explicitly consumed as dense tensors
        // (norm/function tables), not ordinary quantized projections.
        const auto weight=load_tq_weight(mfq,name,placement.device);
        auto ids=::tyloi::cuda::arange(weight.size(0),::tyloi::TensorOptions()
            .device(weight.device()).dtype(::tyloi::kInt64));
        auto value=tyloi::cuda::tq::embedding(weight,ids).to(::tyloi::kFloat32).contiguous();
        return cpu_layer?value.cpu().contiguous():value;
    }
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (rec.dtype == "MFE") {
        auto dense = materialize_mfe_dense(mfq, name)
            .to(::tyloi::kFloat32)
            .contiguous();
        return cpu_layer ? dense.cpu().contiguous() : dense;
    }
    if (rec.dtype != "NINT8-0" && rec.dtype != "NINT") {
        auto value = host(name);
        if (rec.dtype == "BF16" || rec.dtype == "F16")
            value = value.to(::tyloi::kFloat32);
        return cpu_layer ? value : value.to(::tyloi::kCUDA).contiguous();
    }
    auto blob = read_weight(placement.drop_cache, mfq, name);
    if (rec.dtype == "NINT8-0") {
        const auto source = tyloi::nint8::unpack(blob);
        if (cpu_layer) {
            return dequant_nint8_zero_cpu(source);
        }
        const auto packed = to_device_nint8_zero(source, true);
        auto dense = nint8_zero_dequant_f32_cuda(
            packed.q_packed,
            packed.q8_zero_scale,
            packed.neuron_len)
            .contiguous();
        if (packed.shape.size() != 2 ||
            dense.size(0) != packed.shape[0] ||
            dense.size(1) != packed.shape[1]) {
            throw std::runtime_error(
                "NINT8-0 dense tensor shape mismatch: " + name);
        }
        return dense;
    }
    auto dense = dequant_nint_dense_f32(
        to_device_nint(tyloi::nint::unpack(blob), true));
    return cpu_layer
        ? dense.cpu().contiguous()
        : dense;
#else
    auto value = host(name);
    if (rec.dtype == "BF16" || rec.dtype == "F16") value = value.to(::tyloi::kFloat32);
    return cpu_layer ? value : value.to(::tyloi::kCUDA).contiguous();
#endif
}

static bool is_nint_linear_dtype(const std::string & dtype) {
    return dtype == "NINT" || dtype == "NINT8-0";
}

static bool is_nvq_linear_dtype(const std::string & dtype) {
    return dtype == "NVQ" || dtype == "NPQ";
}

static TensorParallelAxis infer_tensor_parallel_axis(
        const std::string & name) {
    const auto ends_with = [&name](std::string_view suffix) {
        return name.size() >= suffix.size() &&
            std::string_view(name).substr(name.size() - suffix.size()) == suffix;
    };
    if (name == "model.token_embedding.weight" ||
        ends_with(".token_embedding.weight") ||
        ends_with(".text_embedding.weight") ||
        (name.find(".code_embedding.") != std::string::npos &&
         ends_with(".weight"))) {
        return TensorParallelAxis::Mirrored;
    }
    if (name == "model.output.weight" ||
        name.find(".code_output.") != std::string::npos) {
        return TensorParallelAxis::Output;
    }
    if (ends_with(".mlp.down.weight") ||
        ends_with(".mlp.shared_expert.down.weight") ||
        ends_with(".attention.output.weight") ||
        ends_with(".linear_attention.output.weight") ||
        ends_with(".attention.output_a.weight") ||
        ends_with(".attention.output_b.weight") ||
        ends_with(".projector.output.weight")) {
        return TensorParallelAxis::Input;
    }
    return TensorParallelAxis::Output;
}

static std::vector<tyloi::TensorParallelSlice>
plan_parallel_slices(
        int64_t extent,
        int64_t preferred_granularity,
        const ParallelConfig & config,
        const std::string & name) {
    (void)name;
    if (!config.enabled()) {
        throw std::runtime_error(
            "parallel slice planning requires at least two ranks");
    }
    int64_t granularity = std::max<int64_t>(
        1, std::min<int64_t>(
            preferred_granularity,
            extent / static_cast<int64_t>(config.devices.size())));
    while (granularity > 1 &&
           extent < static_cast<int64_t>(config.devices.size()) *
               granularity) {
        granularity /= 2;
    }
    auto slices = tyloi::plan_tensor_parallel_slices(
        extent, granularity, config.devices, config.split);
    tyloi::validate_tensor_parallel_slices(
        slices, extent, granularity);
    return slices;
}

std::vector<tyloi::TensorParallelSlice>
plan_moe_expert_parallel_slices(
        const ParallelConfig& parallel,
        int64_t extent,
        const std::string & name) {
    return plan_parallel_slices(extent, 1, parallel, name);
}

std::shared_ptr<const ::tyloi::Tensor> tyloi::cuda::weight_loader::Loader::weight(const std::string & name,
        std::optional<TensorParallelAxis> axis_override,
        const std::vector<tyloi::TensorParallelSlice> *
            slices_override) const {
    const auto& mfq = source;
    const tyloi::cuda::AllocationScope weight_memory;
    const bool cpu_layer = placement.cpu;
    const auto & dtype = tyloi::require_tensor(mfq, name).dtype;
    std::shared_ptr<const ::tyloi::Tensor> weight;
    std::vector<tyloi::ShardedTensor::Shard> shards;
    int64_t logical_out = 0, logical_neuron_len = 0;
    const TensorParallelAxis axis =
        axis_override.value_or(
            infer_tensor_parallel_axis(name));
    if (slices_override != nullptr &&
        axis != TensorParallelAxis::Output) {
        throw std::runtime_error(
            "explicit tensor-parallel slices require an output-axis weight");
    }
    auto select_slices = [&](
            int64_t extent,
            int64_t preferred) {
        if (slices_override == nullptr) {
            return plan_parallel_slices(
                extent, preferred, placement.tensor, name);
        }
        auto slices = *slices_override;
        tyloi::validate_tensor_parallel_slices(
            slices, extent, 1);
        if (slices.size() !=
            placement.tensor.devices.size()) {
            throw std::runtime_error(
                "explicit tensor-parallel slice count mismatch");
        }
        for (size_t index = 0;
             index < slices.size(); ++index) {
            if (slices[index].device !=
                placement.tensor.devices[index]) {
                throw std::runtime_error(
                    "explicit tensor-parallel device order mismatch");
            }
        }
        return slices;
    };
    if(dtype=="TQ") {
        const auto shape=tyloi::tq_projection_shape(mfq,name);
        logical_out=shape[0];logical_neuron_len=shape[1];
        if(placement.tensor.enabled() && axis!=TensorParallelAxis::Mirrored) {
            const auto blob=read_weight(placement.drop_cache,mfq,name);
            const tyloi::TqRows rows(blob.size(),[&](std::uint64_t offset,std::byte* target,std::size_t bytes) {
                if(offset>blob.size() || bytes>blob.size()-offset)throw std::out_of_range("TQ shard read");
                std::memcpy(target,blob.data()+offset,bytes);
            });
            const auto extent=axis==TensorParallelAxis::Output?shape[0]:(shape[1]+255)/256;
            for(const auto& slice:select_slices(extent,axis==TensorParallelAxis::Output?128:1)) {
                tyloi::ShardedTensor::Shard shard;
                shard.device=slice.device;                shard.output_begin=axis==TensorParallelAxis::Output?slice.begin:0;
                shard.output_end=axis==TensorParallelAxis::Output?slice.end:shape[0];
                shard.input_begin=axis==TensorParallelAxis::Input?slice.begin*256:0;
                shard.input_end=axis==TensorParallelAxis::Input?std::min(slice.end*256,shape[1]):shape[1];
                const auto image=axis==TensorParallelAxis::Output
                    ?rows.slice_rows(slice.begin,slice.end):rows.slice_columns(slice.begin*256,slice.end*256);
                shard.weight = std::make_shared<tyloi::encoding::TqTensor>(to_device_tq(image,shard.input_end-shard.input_begin,slice.device));
                shards.push_back(std::move(shard));
            }
        } else weight = std::make_shared<tyloi::encoding::TqTensor>(load_tq_weight(mfq,name,placement.device,!cpu_layer));
    } else
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (is_nint_linear_dtype(dtype)) {
        const auto blob = read_weight(placement.drop_cache, mfq, name);
        if (dtype == "NINT8-0") {
            const auto cpu = tyloi::nint8::unpack(blob);
            logical_out = cpu.out;
            logical_neuron_len = cpu.neuron_len;
            if (placement.tensor.enabled() &&
                axis != TensorParallelAxis::Mirrored) {
                const int64_t extent =
                    axis == TensorParallelAxis::Output
                    ? cpu.out : cpu.ng;
                const int64_t preferred =
                    axis == TensorParallelAxis::Output
                    ? 128
                    : 4;
                for (const auto & slice :
                     select_slices(
                         extent, preferred)) {
                    auto shard_cpu =
                        axis == TensorParallelAxis::Output
                        ? slice_nint8_zero_cpu_output(
                            cpu, slice.begin, slice.end)
                        : slice_nint8_zero_cpu_input_groups(
                            cpu, slice.begin, slice.end);
                    tyloi::ShardedTensor::Shard shard;
                    shard.device = slice.device;
                    shard.output_begin =
                        axis == TensorParallelAxis::Output
                        ? slice.begin : 0;
                    shard.output_end =
                        axis == TensorParallelAxis::Output
                        ? slice.end : cpu.out;
                    shard.input_begin =
                        axis == TensorParallelAxis::Input
                        ? slice.begin * 32 : 0;
                    shard.input_end =
                        axis == TensorParallelAxis::Input
                        ? std::min<int64_t>(
                            slice.end * 32,
                            cpu.neuron_len)
                        : cpu.neuron_len;
                    shard.weight = std::make_shared<tyloi::NintTensor>(to_device_nint8_zero(
                        shard_cpu, true, slice.device));
                    shards.push_back(
                        std::move(shard));
                }
            } else {
                weight = std::make_shared<tyloi::NintTensor>(to_device_nint8_zero(
                    cpu, !cpu_layer,
                    cpu_layer ? -1 : placement.device));
            }
        } else {
            const auto cpu = tyloi::nint::unpack(blob);
            logical_out = cpu.out;
            logical_neuron_len = cpu.neuron_len;
            if (placement.tensor.enabled() &&
                axis != TensorParallelAxis::Mirrored) {
                const int64_t extent =
                    axis == TensorParallelAxis::Output
                    ? cpu.out : cpu.ng;
                const int64_t preferred =
                    axis == TensorParallelAxis::Output
                    ? 128
                    : std::lcm<int64_t>(cpu.gs, 128) / cpu.gs;
                for (const auto & slice :
                     select_slices(
                         extent, preferred)) {
                    auto shard_cpu =
                        axis == TensorParallelAxis::Output
                        ? slice_nint_cpu_output(
                            cpu, slice.begin, slice.end)
                        : slice_nint_cpu_input_groups(
                            cpu, slice.begin, slice.end);
                    tyloi::ShardedTensor::Shard shard;
                    shard.device = slice.device;
                    shard.output_begin =
                        axis == TensorParallelAxis::Output
                        ? slice.begin : 0;
                    shard.output_end =
                        axis == TensorParallelAxis::Output
                        ? slice.end : cpu.out;
                    shard.input_begin =
                        axis == TensorParallelAxis::Input
                        ? slice.begin * cpu.gs : 0;
                    shard.input_end =
                        axis == TensorParallelAxis::Input
                        ? std::min<int64_t>(
                            slice.end * cpu.gs,
                            cpu.neuron_len)
                        : cpu.neuron_len;
                    shard.weight = std::make_shared<tyloi::NintTensor>(to_device_nint(
                        shard_cpu, true, slice.device));
                    shards.push_back(
                        std::move(shard));
                }
            } else {
                weight = std::make_shared<tyloi::NintTensor>(to_device_nint(
                    cpu, !cpu_layer,
                    cpu_layer ? -1 : placement.device));
            }
        }
    } else if (is_nvq_linear_dtype(dtype)) {
        const auto cpu = tyloi::nvq::unpack(
            read_weight(placement.drop_cache, mfq, name), dtype);
        logical_out = cpu.out;
        logical_neuron_len = cpu.neuron_len;
        if (placement.tensor.enabled() &&
            axis != TensorParallelAxis::Mirrored) {
            const int64_t extent =
                axis == TensorParallelAxis::Output
                ? cpu.out : cpu.ng;
            const int64_t preferred =
                axis == TensorParallelAxis::Output
                ? 128
                : std::lcm<int64_t>(cpu.gs, 128) / cpu.gs;
            for (const auto & slice :
                 select_slices(
                     extent, preferred)) {
                auto shard_cpu = slice_nvq_cpu(
                    cpu, axis, slice.begin, slice.end);
                tyloi::ShardedTensor::Shard shard;
                shard.device = slice.device;
                shard.output_begin =
                    axis == TensorParallelAxis::Output
                    ? slice.begin : 0;
                shard.output_end =
                    axis == TensorParallelAxis::Output
                    ? slice.end : cpu.out;
                shard.input_begin =
                    axis == TensorParallelAxis::Input
                    ? slice.begin * cpu.gs : 0;
                shard.input_end =
                    axis == TensorParallelAxis::Input
                    ? std::min<int64_t>(
                        slice.end * cpu.gs,
                        cpu.neuron_len)
                    : cpu.neuron_len;
                shard.weight = std::make_shared<tyloi::NvqTensor>(to_device_nvq(
                    shard_cpu, true, false, false, slice.device));
                shards.push_back(
                    std::move(shard));
            }
        } else {
            weight = std::make_shared<tyloi::NvqTensor>(to_device_nvq(
                cpu, !cpu_layer, options.quant.nvq2,
                options.quant.nvq_groups,
                cpu_layer ? -1 : placement.device));
        }
    } else if (dtype == "MXFP4-SQ") {
        const auto payload = read_weight(placement.drop_cache, mfq, name);
        const auto layout = tyloi::sq::parse(payload.data(), payload.size());
        logical_out = layout.outputs;
        logical_neuron_len = layout.width;
        if (!cpu_layer && placement.tensor.enabled() &&
                axis != TensorParallelAxis::Mirrored) {
            const bool output_axis = axis == TensorParallelAxis::Output;
            for (const auto& slice : select_slices(
                    output_axis ? layout.outputs : layout.width,
                    output_axis ? 128 : 32)) {
                tyloi::ShardedTensor::Shard shard;
                shard.device = slice.device;
                shard.output_begin = output_axis ? slice.begin : 0;
                shard.output_end = output_axis ? slice.end : layout.outputs;
                shard.input_begin = output_axis ? 0 : slice.begin;
                shard.input_end = output_axis ? layout.width : slice.end;
                std::vector<int64_t> rows(
                    shard.output_end - shard.output_begin);
                std::iota(rows.begin(), rows.end(), shard.output_begin);
                shard.weight = std::make_shared<tyloi::Mxfp4SqTensor>(to_device_mxfp4_sq(
                    tyloi::sq::select_rows(
                        payload, rows, shard.input_begin, shard.input_end),
                    true, slice.device));
                shards.push_back(std::move(shard));
            }
        } else {
            weight = std::make_shared<tyloi::Mxfp4SqTensor>(to_device_mxfp4_sq(
                payload, !cpu_layer,
                cpu_layer ? -1 : placement.device));
        }
    } else if (tyloi::fp8sq::is_dtype(dtype)) {
        const auto payload = read_weight(placement.drop_cache, mfq, name);
        const auto layout = tyloi::fp8sq::parse(
            dtype, payload.data(), payload.size());
        logical_out = layout.outputs;
        logical_neuron_len = layout.width;
        if (!cpu_layer && placement.tensor.enabled() &&
                axis != TensorParallelAxis::Mirrored) {
            const bool output_axis = axis == TensorParallelAxis::Output;
            const auto alignment = output_axis
                ? layout.block_rows : layout.block_columns;
            for (const auto& slice : select_slices(
                    output_axis ? layout.outputs : layout.width, alignment)) {
                tyloi::ShardedTensor::Shard shard;
                shard.device = slice.device;
                shard.output_begin = output_axis ? slice.begin : 0;
                shard.output_end = output_axis ? slice.end : layout.outputs;
                shard.input_begin = output_axis ? 0 : slice.begin;
                shard.input_end = output_axis ? layout.width : slice.end;
                shard.weight = std::make_shared<tyloi::Fp8SqTensor>(to_device_fp8_sq(
                    dtype, tyloi::fp8sq::slice(
                        dtype, payload,
                        shard.output_begin, shard.output_end,
                        shard.input_begin, shard.input_end),
                    true, slice.device));
                shards.push_back(std::move(shard));
            }
        } else {
            weight = std::make_shared<tyloi::Fp8SqTensor>(to_device_fp8_sq(
                dtype, payload, !cpu_layer,
                cpu_layer ? -1 : placement.device));
        }
    } else
#endif
    if (dtype == "MXFP4") {
        const auto cpu = tyloi::mxfp4::unpack(
            read_weight(placement.drop_cache, mfq, name));
        logical_out = cpu.out;
        logical_neuron_len = cpu.neuron_len;
        if (placement.tensor.enabled() &&
                axis != TensorParallelAxis::Mirrored) {
            const int64_t extent = axis == TensorParallelAxis::Output
                ? cpu.out : cpu.neuron_len;
            const int64_t preferred = axis == TensorParallelAxis::Output
                ? 128 : 32;
            for (const auto & slice : select_slices(extent, preferred)) {
                auto shard_cpu = slice_mxfp4_cpu(
                    cpu, axis, slice.begin, slice.end);
                tyloi::ShardedTensor::Shard shard;
                shard.device = slice.device;
                shard.output_begin = axis == TensorParallelAxis::Output
                    ? slice.begin : 0;
                shard.output_end = axis == TensorParallelAxis::Output
                    ? slice.end : cpu.out;
                shard.input_begin = axis == TensorParallelAxis::Input
                    ? slice.begin : 0;
                shard.input_end = axis == TensorParallelAxis::Input
                    ? slice.end : cpu.neuron_len;
                shard.weight = std::make_shared<tyloi::Mxfp4Tensor>(to_device_mxfp4(
                    shard_cpu, true, slice.device));
                shards.push_back(std::move(shard));
            }
        } else {
            weight = std::make_shared<tyloi::Mxfp4Tensor>(to_device_mxfp4(
                cpu, !cpu_layer,
                cpu_layer ? -1 : placement.device));
        }
    } else if (dtype == "MXFP8") {
        const auto cpu = tyloi::mxfp8::unpack(
            read_weight(placement.drop_cache, mfq, name));
        logical_out = cpu.out;
        logical_neuron_len = cpu.neuron_len;
        if (placement.tensor.enabled() &&
                axis != TensorParallelAxis::Mirrored) {
            const int64_t extent = axis == TensorParallelAxis::Output
                ? cpu.out : cpu.neuron_len;
            for (const auto & slice : select_slices(extent, 128)) {
                auto shard_cpu = slice_mxfp8_cpu(
                    cpu, axis, slice.begin, slice.end);
                tyloi::ShardedTensor::Shard shard;
                shard.device = slice.device;
                shard.output_begin = axis == TensorParallelAxis::Output
                    ? slice.begin : 0;
                shard.output_end = axis == TensorParallelAxis::Output
                    ? slice.end : cpu.out;
                shard.input_begin = axis == TensorParallelAxis::Input
                    ? slice.begin : 0;
                shard.input_end = axis == TensorParallelAxis::Input
                    ? slice.end : cpu.neuron_len;
                shard.weight = std::make_shared<tyloi::Mxfp8Tensor>(to_device_mxfp8(
                    shard_cpu, true, slice.device));
                shards.push_back(
                    std::move(shard));
            }
        } else if (cpu_layer) {
            weight = std::make_shared<tyloi::Mxfp8Tensor>(to_device_mxfp8(cpu, false));
        } else {
            weight = std::make_shared<tyloi::Mxfp8Tensor>(to_device_mxfp8(
                cpu, true, placement.device));
        }
    } else if (dtype == "BF16" || dtype == "F16" || dtype == "F32") {
        auto cpu = host(name);
        TYLOINFER_RUNTIME_CHECK(cpu.dim() == 2, "dense linear tensor must be rank 2: ", name);
        logical_out = cpu.size(0);
        logical_neuron_len = cpu.size(1);
        // Keep native floating-point linears whole on the primary TP rank.
        // Splitting these matrices changes the cuBLAS GEMM geometry and causes
        // materially larger drift than the weight formats under test. They
        // are a small fraction of the model; routed experts and MXFP8/NINT/NVQ
        // weights remain sharded.
        const bool shard_native_float =
            options.shard_dense;
        if (shard_native_float && placement.tensor.enabled() &&
                axis != TensorParallelAxis::Mirrored) {
            const int64_t extent = axis == TensorParallelAxis::Output
                ? cpu.size(0) : cpu.size(1);
            for (const auto & slice : select_slices(extent, 128)) {
                tyloi::ShardedTensor::Shard shard;
                shard.device = slice.device;
                shard.output_begin = axis == TensorParallelAxis::Output
                    ? slice.begin : 0;
                shard.output_end = axis == TensorParallelAxis::Output
                    ? slice.end : cpu.size(0);
                shard.input_begin = axis == TensorParallelAxis::Input
                    ? slice.begin : 0;
                shard.input_end = axis == TensorParallelAxis::Input
                    ? slice.end : cpu.size(1);
                const int64_t dimension =
                    axis == TensorParallelAxis::Output ? 0 : 1;
                MfqCudaGuard guard(slice.device);
                shard.weight = std::make_shared<::tyloi::DenseTensor>(cpu.narrow(
                        dimension, slice.begin,
                        slice.end - slice.begin)
                    .to(::tyloi::Device(::tyloi::kCUDA, slice.device))
                    .contiguous());
                shards.push_back(
                    std::move(shard));
            }
        } else if (cpu_layer) {
            weight = std::make_shared<::tyloi::DenseTensor>(cpu.contiguous());
        } else {
            MfqCudaGuard guard(placement.device);
            weight = std::make_shared<::tyloi::DenseTensor>(cpu.to(::tyloi::kCUDA).contiguous());
        }
    } else {
        throw std::runtime_error(
            "linear tensor must be NINT/NVQ/MXFP4-SQ/MXFP8-SQ/FP8-128SQ/MXFP4/MXFP8/BF16/F16/F32: " +
            name + " dtype=" + dtype);
    }
    if (shards.empty()) return weight;
    const auto shard_dtype = shards.front().weight->scalar_type();
    return std::make_shared<tyloi::ShardedTensor>(
        logical_out, logical_neuron_len, shard_dtype, axis, std::move(shards));
}

bool is_quant_dtype(const std::string & dtype) {
    return dtype == "TQ" || is_nint_linear_dtype(dtype) ||
        is_nvq_linear_dtype(dtype) || dtype == "MXFP4-SQ" ||
        tyloi::fp8sq::is_dtype(dtype) ||
        dtype == "MXFP4" || dtype == "MXFP8";
}

QuantLinearGroup tyloi::cuda::weight_loader::Loader::build_group(const std::vector<std::string> & names,
    size_t required_compatible_prefix,
    const std::vector<tyloi::TensorParallelSlice> *
        slices_override,
    bool preserve_projection_boundaries) const {
    std::vector<std::shared_ptr<const ::tyloi::Tensor>> layers;
    layers.reserve(names.size());
    for (const auto & name : names) {
        layers.push_back(weight(name,
            slices_override != nullptr
                ? std::optional<TensorParallelAxis>(
                    TensorParallelAxis::Output)
                : std::nullopt,
            slices_override));
    }
    if (required_compatible_prefix > layers.size()) {
        throw std::runtime_error("invalid required quantized-group prefix length");
    }
    // A compatible prefix is fused opportunistically by make_quant_group().
    // Mixed-precision Q/K and gate/up pairs remain separate
    // branches and preserve the recipe-selected layouts.
    return make_quant_group(
        std::move(layers), options.grouped,
        placement.cpu, preserve_projection_boundaries);
}

static std::vector<tyloi::TensorParallelSlice>
tensor_parallel_output_slices_for_input(
        const ::tyloi::Tensor & input_parallel) {
    const auto* sharded = dynamic_cast<const tyloi::ShardedTensor*>(&input_parallel);
    if (!sharded || sharded->axis != TensorParallelAxis::Input) {
        return {};
    }
    std::vector<tyloi::TensorParallelSlice> result;
    result.reserve(
        sharded->shards.size());
    for (const auto & shard :
         sharded->shards) {
        result.push_back({
            shard.device,
            shard.input_begin,
            shard.input_end,
        });
    }
    tyloi::validate_tensor_parallel_slices(
        result,
        input_parallel.size(1),
        1);
    return result;
}

QuantLinearGroup tyloi::cuda::weight_loader::Loader::build_gate_up(const std::vector<std::string> & names,
        const ::tyloi::Tensor & down,
        size_t required_compatible_prefix,
        bool preserve_projection_boundaries) const {
    // An ungated MLP has one Up projection; only the available prefix may fuse.
    required_compatible_prefix = std::min(required_compatible_prefix, names.size());
    auto slices =
        tensor_parallel_output_slices_for_input(
            down);
    return slices.empty()
        ? build_group(names,
            required_compatible_prefix,
            nullptr,
            preserve_projection_boundaries)
        : build_group(names,
            required_compatible_prefix,
            &slices,
            preserve_projection_boundaries);
}

std::shared_ptr<tyloi::NintRows> load_nint_row_table(
        const tyloi::source::Source& source, const std::string& name, bool prefer_mapping) {
#if TYLOINFER_ENABLE_MFQ_COMPAT

    const auto& metadata = tyloi::require_tensor(source, name);
    TYLOINFER_RUNTIME_CHECK(metadata.dtype == "NINT", "NINT row table requires packed NINT");
    TYLOINFER_RUNTIME_CHECK(metadata.nbytes <= std::numeric_limits<std::size_t>::max(),
        "NINT row table size overflow");
    auto read = source.tensor_reader(name);
    bool parallel=source.supports_parallel_tensor_reads();
    const auto* batched=tyloi::cuda::env_lookup("TYLOINFER_PLE_BATCH_READ");
    auto batch=(!batched || batched[0]!='0') ? source.tensor_batch_reader(name) : tyloi::source::Source::TensorBatchReader{};
    bool mapped_file=false;
    const auto* mapped=tyloi::cuda::env_lookup("TYLOINFER_PLE_FILE_MAP");
    if(mapped ? mapped[0]=='1' : prefer_mapping && source.file_read_stats().mode!=tyloi::FileReadMode::Direct) {
        if(auto mapping=source.mapped_tensor_batch_reader(name)) {
            batch=std::move(mapping);
            read=[batch](uint64_t offset,std::byte* out,std::size_t count){batch({{offset,out,count}});};
            parallel=true;mapped_file=true;
        } else if(auto mapping=source.mapped_tensor_reader(name)) {
            read=std::move(mapping);parallel=true;batch={};mapped_file=true;
        }
    }
    if(const auto* trace=tyloi::cuda::env_lookup("TYLOINFER_TRACE_PLE_TIMINGS");trace && trace[0]=='1')
        std::fprintf(stderr,"nint_row_source mapped_file=%d nbytes=%llu name=%s\n",int(mapped_file),
            static_cast<unsigned long long>(metadata.nbytes),name.c_str());
    auto table=std::make_shared<tyloi::NintRows>(static_cast<std::size_t>(metadata.nbytes),
        [read = std::move(read)](std::size_t offset, std::uint8_t* out, std::size_t count) {
            read(offset, reinterpret_cast<std::byte*>(out), count);
        }, parallel,std::move(batch));
    tyloi::cuda::observe_streamed_ple(table,static_cast<std::size_t>(metadata.nbytes));
    return table;

#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}

tyloi::cuda::Experts tyloi::cuda::weight_loader::Loader::build_experts(const std::string & name,
        bool cacheable,
        int layer_id,
        const std::string & projection_role) const {
    const auto& mfq = source;
#if TYLOINFER_ENABLE_MFQ_COMPAT

    const auto& quant = options.quant;
    const tyloi::cuda::AllocationScope weight_memory;
    auto& cache = this->cache.moe;
    if (cache && cacheable &&
            !placement.experts.enabled() &&
            options.ranges) {
        const auto & record = tyloi::require_tensor(mfq, name);
        tyloi::source::Source::TensorReader reader;
        tyloi::source::Source::TensorBatchReader batch_reader;
        try {
            // Streaming workers read independent experts concurrently. A
            // shared buffered FILE serializes their seeks across all fields.
            const bool direct=mfq.file_read_stats().mode==tyloi::FileReadMode::Direct;
            bool mapped=false;
            if(!options.preload && !direct) {
                reader=mfq.mapped_tensor_reader(name);
                batch_reader=mfq.mapped_tensor_batch_reader(name);
                mapped=bool(reader);
            }
            // Final GPU allocation can leave too little RAM for complete
            // residency. Keep batched I/O ready even when that decision is
            // made after these canonical sources have been registered.
            if(!batch_reader)batch_reader=mfq.tensor_batch_reader(name);
            if(direct && batch_reader)reader=[batch=batch_reader](uint64_t offset,std::byte* out,std::size_t count) {
                batch({{offset,out,count}});
            };
            if(!reader)reader=mfq.tensor_reader(name);
            static std::atomic_bool reported{false};
            if(reader && !reported.exchange(true))
                std::fprintf(stderr,"moe_ssd_source_dispatch mapped=%d parallel_reads=%d batch_fields=%d direct=%d\n",
                    int(mapped),int(mapped || mfq.supports_parallel_tensor_reads()),int(bool(batch_reader)),int(direct));
        } catch (const tyloi::TensorReaderUnsupported&) {
            // Sources without retained readers keep the existing CPU loader.
        }
        if (reader) {
            if (record.dtype == "NINT") {
                if(batch_reader)batch_reader=[batch=std::move(batch_reader)](const std::vector<tyloi::ReadSpan>& spans) {
                    static std::atomic_bool reported{false};
                    if(!reported.exchange(true))
                        std::fprintf(stderr,"moe_nint_expert_batch_dispatch fields=%zu\n",spans.size());
                    batch(spans);
                };
                auto store = tyloi::make_packed_nint_expert_store(record.nbytes,
                    [reader](std::size_t offset, std::uint8_t* output, std::size_t bytes) {
                        reader(offset, reinterpret_cast<std::byte*>(output), bytes);
                    },std::move(batch_reader));
                const int minimum = std::min(options.min_slots, store->num_experts());
                return cache_quant_moe_weight(cache, name,
                    std::make_shared<MoeQuantRangeSource>(std::move(store)), minimum, layer_id, projection_role);
            }
            try {
                auto store=std::make_shared<tyloi::MfeQuantExpertStore>(record.nbytes,
                    [reader](std::size_t offset,std::uint8_t* output,std::size_t bytes) {
                        reader(offset,reinterpret_cast<std::byte*>(output),bytes);
                    },batch_reader);
                const int minimum=std::min(options.min_slots,store->num_experts());
                return cache_quant_moe_weight(cache,name,std::make_shared<MoeQuantRangeSource>(std::move(store)),
                    minimum,layer_id,projection_role);
            } catch (const tyloi::MfeQuantRangeUnsupported&) {
                // Other canonical pool families retain the existing CPU path.
            }
        }
    }
    auto cpu = load_mfe_cpu(mfq, name);
    const bool has_matrix_local_sq = std::any_of(
        cpu.pools.begin(), cpu.pools.end(), [](const tyloi::mfe::Pool & pool) {
            return pool.dtype == "MXFP4-SQ" ||
                tyloi::fp8sq::is_dtype(pool.dtype);
        });
    if (placement.experts.enabled()) {
        auto slices = plan_moe_expert_parallel_slices(
            placement.experts, cpu.n_experts, name);
        tyloi::cuda::Experts result;
        result.n_experts = cpu.n_experts;
        result.out_per_expert =
            cpu.out_per_expert;
        result.neuron_len =
            cpu.neuron_len;
        for (const auto & slice : slices) {
            auto shard =
                std::make_shared<tyloi::cuda::Experts>(
                    to_cuda_device_moe_expert_slice(
                        cpu, slice.begin,
                        slice.end,
                        slice.device,
                        quant));
            result.expert_parallel_shards.push_back({
                slice.device,
                slice.begin,
                slice.end,
                std::move(shard),
            });
        }
        return result;
    }
    if (cache && cacheable && !has_matrix_local_sq) {
        auto runtime =
            make_mixed_moe_runtime(cpu, false);
        return cache_moe_weight(
            cache,
            name, runtime,
            std::min(
                options.min_slots,
                runtime->n_experts),
            layer_id,
            projection_role);
    }
    const bool all_nint = std::all_of(
        cpu.pools.begin(), cpu.pools.end(), [](const tyloi::mfe::Pool & pool) {
            return pool.dtype == "NINT";
        });
    return all_nint
        ? to_gpu_mfe(cpu)
        : to_gpu_mixed_moe(cpu, quant);

#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}

std::shared_ptr<MixedMoeRuntime> load_mfe_cpu_offloaded(
        const tyloi::source::Source & mfq, const std::string & name) {
#if TYLOINFER_ENABLE_MFQ_COMPAT

    return make_mixed_moe_runtime(load_mfe_cpu(mfq, name), false);

#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}

namespace tyloi::cuda::weight_loader {
std::vector<std::shared_ptr<const ::tyloi::Tensor>> Loader::group(const std::vector<std::string>& names,
    size_t prefix, const std::vector<tyloi::TensorParallelSlice>* slices, bool preserve_boundaries) const {
    std::vector<std::shared_ptr<const ::tyloi::Tensor>> layers;
    layers.reserve(names.size());
    for (const auto& name : names)
        layers.push_back(weight(name, slices ? std::optional<TensorParallelAxis>(TensorParallelAxis::Output)
                                            : std::nullopt, slices));
    if (prefix > layers.size()) throw std::runtime_error("invalid required quantized-group prefix length");
    if (layers.empty()) throw std::runtime_error("empty quantized linear group");
    std::vector<::tyloi::encoding::TqTensor> tq_weights;
    for (const auto& layer : layers) {
        const auto* tq = dynamic_cast<const ::tyloi::encoding::TqTensor*>(layer.get());
        if (!tq || !tq->is_cuda()) { tq_weights.clear(); break; }
        tq_weights.push_back(*tq);
    }
    if (!tq_weights.empty()) tyloi::cuda::tq::validate_group(tq_weights);
#if TYLOINFER_ENABLE_MFQ_COMPAT
    const bool all_nint = std::all_of(layers.begin(), layers.end(), [](const auto& layer) {
        const auto* nint = dynamic_cast<const tyloi::NintTensor*>(layer.get());
        return nint && nint->q_packed.is_cuda();
    });
    if (all_nint && options.grouped && !placement.cpu && !preserve_boundaries) {
        auto group = make_quant_group(layers, true, false);
        return nint_aliases(layers, group.nint);
    }
#endif
    return layers;
}
std::vector<std::shared_ptr<const ::tyloi::Tensor>> Loader::gate_up(const std::vector<std::string>& names,
    const ::tyloi::Tensor& down, size_t prefix, bool preserve_boundaries) const {
    auto slices = tensor_parallel_output_slices_for_input(down);
    return group(names, std::min(prefix, names.size()), slices.empty() ? nullptr : &slices,
                 preserve_boundaries);
}
std::shared_ptr<const tyloi::cuda::Experts> Loader::experts(const std::string& name, bool cacheable,
    int layer, const std::string& role) const {
    return std::make_shared<Experts>(build_experts(name, cacheable, layer, role));
}
DenseTensor Loader::dense(const std::string& name) const {
    const auto& file = source;
    const tyloi::cuda::AllocationScope weight_memory;
    const auto& dtype=tyloi::require_tensor(file, name).dtype;
    const bool dense_storage = dtype=="F32" || dtype=="F16" || dtype=="BF16";
    const bool quantized_storage = tyloi::mfq_compat_enabled &&
        (dtype=="NINT" || dtype=="NINT8-0" || dtype=="MFE");
    TYLOINFER_RUNTIME_CHECK(dense_storage || quantized_storage,
        "expected a dense parameter: ",name);
    auto value=tensor(name);
    if (!dense_storage) return value;
    return value.to(dtype=="F16" ? ::tyloi::kFloat16 : dtype=="BF16" ? ::tyloi::kBFloat16 : ::tyloi::kFloat32);
}

Routed Loader::routed(const std::string& name, int layer,
    int64_t experts, int64_t output, int64_t input, const std::string& role) const {
    const auto& file = source;
    const auto& dtype=tyloi::require_tensor(file, name).dtype;
    if(dtype=="TQ") {
        auto weight=load_tq_routed_weight(cache,tq,file,name,experts,output,input);
        Routed result=[weight](const Projector&,const DenseTensor& source,const DenseTensor& ids) {
            return weight->forward(source,ids);
        };
        result.tq_projections.push_back(std::move(weight));
        return result;
    }
    if (dtype=="F32" || dtype=="F16" || dtype=="BF16") {
        TYLOINFER_RUNTIME_CHECK(!placement.experts.enabled(),
            "expert parallelism requires packed routed tensors: ",name);
        auto w=dense(name);
        TYLOINFER_RUNTIME_CHECK(w.sizes().vec()==std::vector<int64_t>({experts,output,input}),
            "dense expert tensor shape mismatch: ",name);
        return [w,output,input](const Projector&, const DenseTensor& x,const DenseTensor& ids) {
            const auto rows=ids.size(0), routes=ids.size(1);
            auto selected=w.index_select(0,ids.reshape({-1}).to(::tyloi::kInt64)).reshape({rows,routes,output,input});
            auto source=x.dim()==2 ? x.unsqueeze(1).expand({rows,routes,input}) : x;
            return ::tyloi::cuda::matmul(selected,source.to(w.scalar_type()).unsqueeze(-1)).squeeze(-1);
        };
    }

#if TYLOINFER_ENABLE_MFQ_COMPAT
    // Predictor layers have their own expert images and must not share the
    // backbone's layer-zero transfer range or prefill lookahead.
    if(name.starts_with("predictor."))layer=-1-layer;
    auto w=std::make_shared<tyloi::cuda::Experts>(this->build_experts(name,true,layer,role.empty() ? this->role : role));
    TYLOINFER_RUNTIME_CHECK(w->n_experts==experts && w->out_per_expert==output && w->neuron_len==input,
        "routed tensor shape mismatch: ",name);
    Routed result=[w,experts](const Projector& project, const DenseTensor& x,const DenseTensor& ids) {
        auto route=build_moe_route_plan(ids.to(::tyloi::kInt32).contiguous(),int(experts));
        return project.run(*w, x.contiguous(), route);
    };
    result.projections.push_back(w);
    return result;

#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}

Routed Loader::routed_gate_up(const std::string& mlp_prefix,
    int layer, int64_t experts, int64_t width, int64_t input, const std::string& role) const {
    const auto& file = source;
    const auto base=mlp_prefix+".experts";
    const auto gate_name=base+".gate.weight",up_name=base+".up.weight";
    const bool has_gate=tyloi::has_tensor(file, gate_name),has_up=tyloi::has_tensor(file, up_name);
    TYLOINFER_RUNTIME_CHECK(has_gate==has_up,"incomplete routed Gate/Up pair under ",base);
    if (!has_gate) return routed(base+".gate_up.weight",layer,experts,2*width,input,role);
    auto gate=routed(gate_name,layer,experts,width,input,"gate");
    auto up=routed(up_name,layer,experts,width,input,"up");
    auto projections=gate.projections;
    projections.insert(projections.end(),up.projections.begin(),up.projections.end());
    auto tq_projections=gate.tq_projections;
    tq_projections.insert(tq_projections.end(),up.tq_projections.begin(),up.tq_projections.end());
    Routed result=[gate=std::move(gate),up=std::move(up)](const Projector& project, const DenseTensor& x,const DenseTensor& ids) {
        return ::tyloi::cuda::cat({gate(project,x,ids),up(project,x,ids)},-1);
    };
    result.projections=std::move(projections);
    result.tq_projections=std::move(tq_projections);
    return result;
}



} // namespace tyloi::cuda::weight_loader

::tyloi::DenseTensor tyloi::cuda::weight_loader::Loader::host(const std::string &name) const {
    const auto &record = tyloi::require_tensor(source, name);
    ::tyloi::ScalarType dtype;
    size_t item_size;
    if (record.dtype == "BF16") { dtype = ::tyloi::kBFloat16; item_size = 2; }
    else if (record.dtype == "F16") { dtype = ::tyloi::kFloat16; item_size = 2; }
    else if (record.dtype == "F32") { dtype = ::tyloi::kFloat32; item_size = 4; }
    else if (record.dtype == "I64") { dtype = ::tyloi::kInt64; item_size = 8; }
    else if (record.dtype == "I32") { dtype = ::tyloi::kInt32; item_size = 4; }
    else throw std::runtime_error("unsupported dense dtype: " + record.dtype + " tensor " + name);

    auto blob = tyloi::read_tensor(source, name);
    if (placement.drop_cache) source.drop_file_cache();
    tyloi::dense::Layout layout;
    try {
        layout = tyloi::dense::parse(blob, item_size, record.shape);
    } catch (const std::runtime_error& error) {
        throw std::runtime_error(std::string(error.what()) + ": " + name);
    }
    return ::tyloi::cuda::from_blob(blob.data() + layout.offset, layout.shape,
                         ::tyloi::TensorOptions().dtype(dtype)).clone();
}

::tyloi::DenseTensor tyloi::cuda::weight_loader::Loader::native(const std::string &name) const {
    const tyloi::cuda::AllocationScope weight_memory;
    MfqCudaGuard guard(placement.device);
    return host(name).to(::tyloi::kCUDA).contiguous();
}

std::shared_ptr<const ::tyloi::Tensor> tyloi::cuda::weight_loader::Loader::embedding(const std::string& name, bool independent_output) const {
    const auto* mapped=tyloi::cuda::env_lookup("TYLOINFER_EMBEDDING_MAPPED");
    const bool mapped_control=mapped && mapped[0]=='2';
    const bool use_mapped=mapped ? mapped[0]=='1' || mapped_control :
        options.mapped_embedding;
    const auto& dtype=tyloi::require_tensor(source,name).dtype;
    const bool nint=tyloi::mfq_compat_enabled && dtype=="NINT";
    if(use_mapped && independent_output &&
            !placement.tensor.enabled() && !placement.cpu &&
            (dtype=="BF16" || dtype=="F16" || dtype=="F32" || nint)) {
        try {
            const tyloi::cuda::AllocationScope weight_memory;
            if(nint) {
                auto cpu = for_layer(placement.device, true).weight(name);
                auto result=map_nint_embedding(cpu,placement.device,mapped_control);
                std::cerr<<"mapped_token_embedding rows="<<result->size(0)<<" width="<<result->size(1)
                         <<" dtype="<<dtype<<" read="<<(mapped_control?"device_control":"mapped_host")<<std::endl;
                return result;
            }
            auto cpu=host(name);
            auto result=map_dense_embedding(cpu,placement.device,mapped_control);
            std::cerr<<"mapped_token_embedding bytes="<<cpu.numel()*cpu.element_size()
                     <<" rows="<<cpu.size(0)<<" width="<<cpu.size(1)<<" dtype="<<dtype
                     <<" read="<<(mapped_control?"device_control":"mapped_host")<<std::endl;
            return result;
        } catch(const tyloi::cuda::Error& error) {
            // Strata keeps the original GPU table when mapped pinning is
            // unavailable. Explicit diagnostic controls must report failure.
            if(mapped)throw;
            (void)cudaGetLastError();
            std::cerr<<"mapped_token_embedding unavailable; using device table: "<<error.what()<<std::endl;
        }
    }
    return weight(name);

}

namespace tyloi::weights {

BiasWeight bias_rows(const BiasWeight& value, int64_t begin, int64_t count) {
    const auto& weight = *value.weight;
    const auto& bias = value.bias;

    const auto* dense = dynamic_cast<const ::tyloi::DenseTensor*>(&weight);
    if (dense == nullptr) {
        throw std::runtime_error("fused vision projection must stay dense to be split");
    }
    if (begin < 0 || count <= 0 || begin + count > dense->size(0)) {
        throw std::runtime_error("fused vision projection row range is invalid");
    }
    BiasWeight result;
    result.weight = std::make_shared<::tyloi::DenseTensor>(
        dense->narrow(0, begin, count).contiguous());
    if (bias) {
        const auto& dense_bias = dynamic_cast<const ::tyloi::DenseTensor&>(*bias);
        if (dense_bias.defined()) result.bias = std::make_shared<::tyloi::DenseTensor>(
            dense_bias.narrow(0, begin, count).contiguous());
    }
    return result;
}

} // namespace tyloi::weights

namespace tyloi::cuda::weight_loader {

weights::BiasWeight Loader::biased(const std::string& weight_name,
                          const std::string& bias_name) const {
    weights::BiasWeight result;
    result.weight = weight(weight_name);
    if (!bias_name.empty() && has(bias_name)) result.bias = std::make_shared<DenseTensor>(tensor(bias_name));
    return result;
}

weights::BiasWeight Loader::affine(const std::string& weight_name,
                          const std::string& bias_name) const {
    weights::BiasWeight result{weight(weight_name), {}};
    if (has(bias_name)) {
        auto bias = tensor(bias_name);
        if (bias.dim() != 1 || bias.size(0) != result.weight->size(0)) {
            throw std::runtime_error("vision affine bias shape mismatch: " + bias_name);
        }
        result.bias = std::make_shared<DenseTensor>(std::move(bias));
    }
    return result;
}

Loader::DenseTensor Loader::norm(const std::string& name, std::int64_t width) const {
    auto weight = tensor(name).to(::tyloi::kFloat32).contiguous();
    if (weight.dim() != 1 || weight.size(0) != width) {
        throw std::runtime_error("vision RMSNorm width mismatch: " + name);
    }
    return weight;
}

} // namespace tyloi::cuda::weight_loader
