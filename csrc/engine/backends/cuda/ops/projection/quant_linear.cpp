#include "quant_linear.h"
#include "engine/backends/cuda/kernels/mfq_cuda_embedding_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_quant_ops.h"
#include "engine/weights/encoding/mxfp4_sq_decode.h"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

const QuantLinear& as_quant_linear(const tyloi::weights::Weights& weight) {
    const auto* result = dynamic_cast<const QuantLinear*>(&weight);
    if (!result || weight.backend() != tyloi::DeviceType::cuda ||
        weight.layout() != tyloi::weights::Layout::matrix)
        throw std::invalid_argument("matrix was not prepared by the CUDA backend");
    return *result;
}

const QuantLinearGroup& as_quant_linear_group(const tyloi::weights::Weights& group) {
    const auto* result = dynamic_cast<const QuantLinearGroup*>(&group);
    if (!result || group.backend() != tyloi::DeviceType::cuda ||
        group.layout() != tyloi::weights::Layout::group)
        throw std::invalid_argument("matrix group was not prepared by the CUDA backend");
    return *result;
}

namespace {
void check_width(int64_t width, int64_t expected, bool padded) {
    if (width <= 0 || expected <= 0 || (padded ? width > expected : width != expected))
        throw std::invalid_argument("projection input width does not match weight");
}

void check_placement(const tyloi::weights::Placement& placement, ::tyloi::Device device) {
    if (placement.caller && *placement.caller == (device.is_cuda() ? tyloi::DeviceType::cuda : tyloi::DeviceType::cpu))
        return;
    for (const auto& place : placement.devices)
        if (device == (place.is_cuda() ? ::tyloi::Device(::tyloi::kCUDA, place.index)
                                      : ::tyloi::Device(::tyloi::kCPU))) return;
    throw std::invalid_argument("projection weight and input devices do not match");
}

void check_fields(std::initializer_list<const ::tyloi::Tensor*> fields, ::tyloi::Device device) {
    for (const auto* field : fields) {
        if (!field->defined() || field->numel() == 0) continue;
        if (field->device() != device)
            throw std::invalid_argument("projection weight fields and input devices do not match");
    }
}

void check_nint_fields(const NintWeight& w, ::tyloi::Device device) {
    check_fields({&w.q_packed, &w.row_q_bits, &w.row_q_bit_offsets, &w.sub_scale,
                  &w.sub_min, &w.neuron_scale, &w.neuron_min}, device);
}

void check_fields(const NintWeight& w, ::tyloi::Device device) {
    if (w.q8_zero) check_fields({&w.q_packed, &w.q8_zero_scale}, device);
    else check_nint_fields(w, device);
}

void check_fields(const NvqWeight& w, ::tyloi::Device device) {
    check_fields({&w.indices_packed, &w.aux_packed, &w.sub_scale_packed,
                  &w.neuron_scale, &w.codebook}, device);
}

void check_fields(const Mxfp4Weight& w, ::tyloi::Device device) {
    check_fields({&w.values, &w.scales}, device);
}

void check_fields(const Mxfp8Weight& w, ::tyloi::Device device) {
    check_fields({&w.values, &w.scales}, device);
}

void check_fields(const Mxfp4SqWeight& w, ::tyloi::Device device) {
    if (!w.blob.is_cuda()) return;
    check_fields({&w.blob, &w.row_q, &w.row_symbol_byte_offsets, &w.row_auxiliary}, device);
}

void check_fields(const Fp8SqWeight& w, ::tyloi::Device device) {
    if (!w.blob.is_cuda() || (w.dtype != "MXFP8-SQ" && w.dtype != "FP8-128SQ")) return;
    check_fields({&w.blob, &w.row_q, &w.row_symbol_byte_offsets}, device);
}

const ::tyloi::Tensor& shard_payload(const QuantLinearShard& shard) {
    switch (shard.kind) {
    case QuantLinearKind::Nint: return shard.nint.q_packed;
    case QuantLinearKind::Nvq: return shard.nvq.indices_packed;
    case QuantLinearKind::Mxfp4: return shard.mxfp4.values;
    case QuantLinearKind::Mxfp4Sq: return shard.mxfp4_sq.blob;
    case QuantLinearKind::Fp8Sq: return shard.fp8_sq.blob;
    case QuantLinearKind::Mxfp8: return shard.mxfp8.values;
    case QuantLinearKind::Dense: return shard.dense;
    case QuantLinearKind::Tq: return shard.tq.image;
    }
    throw std::invalid_argument("unsupported projection weight");
}

int64_t shard_width(const QuantLinearShard& shard) {
    switch (shard.kind) {
    case QuantLinearKind::Nint: return shard.nint.neuron_len;
    case QuantLinearKind::Nvq: return shard.nvq.neuron_len;
    case QuantLinearKind::Mxfp4: return shard.mxfp4.neuron_len;
    case QuantLinearKind::Mxfp4Sq: return shard.mxfp4_sq.neuron_len;
    case QuantLinearKind::Fp8Sq: return shard.fp8_sq.neuron_len;
    case QuantLinearKind::Mxfp8: return shard.mxfp8.neuron_len;
    case QuantLinearKind::Dense: return shard.dense.size(1);
    case QuantLinearKind::Tq: return shard.tq.neuron_len;
    }
    throw std::invalid_argument("unsupported projection weight");
}
}

void check_projection_width(const QuantLinear& weight, int64_t width) {
    if (!weight.tensor_parallel()) {
        check_width(width, weight.neuron_len(), weight.is_nint() || weight.is_nvq());
        return;
    }
    if (weight.tensor_parallel_axis == TensorParallelAxis::Input) {
        check_width(width, weight.neuron_len(), false);
        for (const auto& shard : weight.tensor_parallel_shards) {
            if (shard.input_begin < 0 || shard.input_end <= shard.input_begin || shard.input_end > width)
                throw std::invalid_argument("projection input does not cover tensor-parallel shard");
            check_width(shard.input_end - shard.input_begin, shard_width(shard),
                        shard.kind == QuantLinearKind::Nint || shard.kind == QuantLinearKind::Nvq);
        }
    } else if (weight.tensor_parallel_axis == TensorParallelAxis::Output) {
        for (const auto& shard : weight.tensor_parallel_shards)
            check_width(width, shard_width(shard),
                        shard.kind == QuantLinearKind::Nint || shard.kind == QuantLinearKind::Nvq);
    } else {
        throw std::invalid_argument("tensor-parallel linear has an invalid axis");
    }
}

void check_projection_device(const QuantLinear& weight, ::tyloi::Device device) {
    if (!weight.tensor_parallel()) {
        check_placement(weight.placement(), device);
        if (weight.is_nint()) check_fields(weight.nint, device);
        else if (weight.is_nvq()) check_fields(weight.nvq, device);
        else if (weight.is_mxfp4()) check_fields(weight.mxfp4, device);
        else if (weight.is_mxfp8()) check_fields(weight.mxfp8, device);
        else if (weight.is_mxfp4_sq()) check_fields(weight.mxfp4_sq, device);
        else if (weight.is_fp8_sq()) check_fields(weight.fp8_sq, device);
        return;
    }
    for (const auto& shard : weight.tensor_parallel_shards) {
        const auto& payload = shard_payload(shard);
        if (!payload.defined() || !payload.is_cuda() || payload.get_device() != shard.device)
            throw std::invalid_argument("tensor-parallel weight is not prepared on its device");
        const ::tyloi::Device local(::tyloi::kCUDA, shard.device);
        switch (shard.kind) {
        case QuantLinearKind::Nint: check_fields(shard.nint, local); break;
        case QuantLinearKind::Nvq: check_fields(shard.nvq, local); break;
        case QuantLinearKind::Mxfp4: check_fields(shard.mxfp4, local); break;
        case QuantLinearKind::Mxfp4Sq: check_fields(shard.mxfp4_sq, local); break;
        case QuantLinearKind::Fp8Sq: check_fields(shard.fp8_sq, local); break;
        case QuantLinearKind::Mxfp8: check_fields(shard.mxfp8, local); break;
        case QuantLinearKind::Dense:
        case QuantLinearKind::Tq: break;
        }
    }
}

void check_projection(const QuantLinear& weight, const ::tyloi::Tensor& input) {
    if (!input.defined() || input.dim() < 1)
        throw std::invalid_argument("projection input must have a last dimension");
    check_projection_device(weight, input.device());
    check_projection_width(weight, input.size(-1));
}

void check_projection(const QuantLinearGroup& group, const ::tyloi::Tensor& input) {
    if (!input.defined() || input.dim() < 1)
        throw std::invalid_argument("projection input must have a last dimension");
    if (group.outs.empty() || group.cols.size() != group.outs.size() ||
        group.placements.size() != group.outs.size())
        throw std::invalid_argument("projection group is not prepared");
    for (const auto out : group.outs)
        if (out <= 0) throw std::invalid_argument("invalid projection group output width");
    if (!group.layers.empty()) {
        if (group.layers.size() != group.outs.size())
            throw std::invalid_argument("projection group has invalid layers");
        for (const auto& layer : group.layers) check_projection(layer, input);
        return;
    }
    if (!group.nint_grouped || !input.is_cuda())
        throw std::invalid_argument("projection group is not prepared for input");
    for (size_t i = 0; i < group.outs.size(); ++i) {
        check_width(input.size(-1), group.cols[i], true);
        check_placement(group.placements[i], input.device());
    }
    const auto check_nint_weight = [&](const NintWeight& weight) {
        if (!weight.q_packed.defined())
            throw std::invalid_argument("projection weight is not prepared");
        check_width(input.size(-1), weight.neuron_len, true);
        check_device(weight, input);
    };
    if (group.nint.split_w.empty()) check_nint_weight(group.nint.w);
    for (const auto& weight : group.nint.projection_w) check_nint_weight(weight);
    for (const auto& weight : group.nint.split_w) check_nint_weight(weight);
}

void check_device(const ::tyloi::Tensor& weight, const ::tyloi::Tensor& input) {
    if (input.is_cuda() && weight.defined() && weight.numel() != 0 && weight.device() != input.device())
        throw std::invalid_argument("projection weight and input devices do not match");
}

void check_device(const NintWeight& weight, const ::tyloi::Tensor& input) {
    if (input.is_cuda()) check_fields(weight, input.device());
}

void check_nint(const NintWeight& weight, const ::tyloi::Tensor& input) {
    if (input.is_cuda()) check_nint_fields(weight, input.device());
}

void check_device(const NvqWeight& weight, const ::tyloi::Tensor& input) {
    if (input.is_cuda()) check_fields(weight, input.device());
}

void check_device(const Mxfp4Weight& weight, const ::tyloi::Tensor& input) {
    if (input.is_cuda()) check_fields(weight, input.device());
}

void check_device(const Mxfp8Weight& weight, const ::tyloi::Tensor& input) {
    if (input.is_cuda()) check_fields(weight, input.device());
}

void check_device(const Mxfp4SqWeight& weight, const ::tyloi::Tensor& input) {
    if (input.is_cuda()) check_fields(weight, input.device());
}

void check_device(const Fp8SqWeight& weight, const ::tyloi::Tensor& input) {
    if (input.is_cuda()) check_fields(weight, input.device());
}

void check_device(const QuantLinear& weight, const ::tyloi::Tensor& input) {
    if (weight.tensor_parallel() || weight.is_tq()) return;
    if (weight.is_nint()) check_device(weight.nint, input);
    else if (weight.is_nvq()) check_device(weight.nvq, input);
    else if (weight.is_mxfp4()) check_device(weight.mxfp4, input);
    else if (weight.is_mxfp4_sq()) check_device(weight.mxfp4_sq, input);
    else if (weight.is_fp8_sq()) check_device(weight.fp8_sq, input);
    else if (weight.is_dense()) check_device(weight.dense, input);
    else check_device(weight.mxfp8, input);
}

#if TYLOINFER_ENABLE_MFQ_COMPAT
#include "linear_dispatch.inc"

::tyloi::Tensor dequant_nint_dense_f32(const NintWeight & w) {
    if (w.q8_zero) {
        return nint8_zero_dequant_f32_cuda(
            w.q_packed, w.q8_zero_scale, w.neuron_len).contiguous();
    }
    return nint_decode_f32_cuda(
        w.q_packed, w.row_q_bits, w.row_q_bit_offsets,
        w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min,
        w.neuron_len, w.gs).contiguous();
}
#endif

::tyloi::Tensor quant_embedding_lookup(
        const QuantLinear & embedding,
        ::tyloi::Tensor token_ids) {
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
    if ((tyloi::mfq_compat_enabled && embedding.is_nvq())) {
        check_device(embedding.nvq, token_ids);
        return nvq_embedding(embedding.nvq, token_ids);
    }
    if ((tyloi::mfq_compat_enabled && embedding.is_nint())) {
        check_device(embedding.nint, token_ids);
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
        check_device(embedding.mxfp4, token_ids);
        return mxfp4_embedding_lookup_cuda(
            embedding.mxfp4.values,
            embedding.mxfp4.scales, token_ids);
    }
    if (embedding.is_mxfp8()) {
        check_device(embedding.mxfp8, token_ids);
        return mxfp8_embedding_lookup_cuda(
            embedding.mxfp8.values,
            embedding.mxfp8.scales, token_ids);
    }
    TYLOINFER_RUNTIME_CHECK(
        !(tyloi::mfq_compat_enabled && embedding.is_mxfp4_sq()) && !(tyloi::mfq_compat_enabled && embedding.is_fp8_sq()),
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

::tyloi::Tensor quant_linear_reference_weight(
        const QuantLinear& linear) {
    TYLOINFER_RUNTIME_CHECK(!linear.tensor_parallel(),"linear reference requires local weights");
    if (linear.is_tq()) {
        const auto ids=::tyloi::cuda::arange(linear.tq.out,
            linear.tq.image.options().dtype(::tyloi::kInt64));
        return linear.tq.embedding(ids);
    }
    if ((tyloi::mfq_compat_enabled && linear.is_nint())) {
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
    if ((tyloi::mfq_compat_enabled && linear.is_nvq())) return nvq_dequant(linear.nvq);
    if (linear.is_mxfp4()) {
        return mxfp4_dequant_cuda(
            linear.mxfp4.values, linear.mxfp4.scales);
    }
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if ((tyloi::mfq_compat_enabled && linear.is_mxfp4_sq())) {
        const auto& weight = linear.mxfp4_sq;
        if (!weight.blob.is_cuda()) {
            return dequant_mxfp4_sq_cpu(weight)
                .to(::tyloi::kFloat16);
        }
        return mxfp4_sq_dequant_cuda(
            weight.blob, weight.row_q, weight.row_symbol_byte_offsets,
            weight.row_auxiliary, weight.bits, weight.out,
            weight.neuron_len, weight.matrix_scale_base,
            weight.q_sum, weight.sq4_rows, false);
    }
#endif
    if ((tyloi::mfq_compat_enabled && linear.is_fp8_sq())) {
        return dequant_fp8_sq(linear.fp8_sq, false);
    }
    if (linear.is_mxfp8()) {
        return mxfp8_cpu_reference(linear.mxfp8);
    }
    if (linear.is_dense()) return linear.dense;
    throw std::runtime_error("unsupported linear reference format");
}

::tyloi::Tensor dense_projection(
        ::tyloi::Tensor input, const ::tyloi::Tensor& weight) {
    input = input.to(weight.scalar_type());
    const int64_t rows = input.numel() / input.size(-1);
    if (rows > 1 && rows <= 6) {
        auto shape = input.sizes().vec();
        shape.back() = weight.size(0);
        auto flat = input.reshape({rows, input.size(-1)});
        std::vector<::tyloi::Tensor> outputs;
        outputs.reserve(rows);
        // A strided-batched GEMM may select different accumulation from M=1.
        // Keep short target/predictor windows on the decode projection path.
        // ponytail: at most six launches; fuse only with decode-equivalent accumulation.
        for (int64_t row = 0; row < rows; ++row)
            outputs.push_back(::tyloi::cuda::matmul(flat.narrow(0, row, 1), weight.transpose(0, 1)));
        return ::tyloi::cuda::cat(outputs, 0).reshape(shape);
    }
    return ::tyloi::cuda::matmul(input, weight.transpose(0, 1));
}

tyloi::weights::Placement QuantLinear::placement(std::size_t index) const {
    if (index != 0) throw std::out_of_range("matrix index out of range");
    tyloi::weights::Placement result;
    if (tensor_parallel()) {
        for (const auto& shard : tensor_parallel_shards)
            result.devices.emplace_back(tyloi::DeviceType::cuda, shard.device);
        return result;
    }
    const ::tyloi::Tensor* payload = nullptr;
    if (is_tq()) payload = &tq.image;
    else if (is_nint()) payload = &nint.q_packed;
    else if (is_nvq()) payload = &nvq.indices_packed;
    else if (is_mxfp4()) payload = &mxfp4.values;
    else if (is_mxfp8()) payload = &mxfp8.values;
    else if (is_mxfp4_sq()) payload = &mxfp4_sq.blob;
    else if (is_fp8_sq()) payload = &fp8_sq.blob;
    else if (is_dense()) payload = &dense;
    if (!payload || !payload->defined())
        throw std::invalid_argument("matrix weight is not prepared");
    if (is_tq() && payload->is_cpu()) result.caller = tyloi::DeviceType::cuda;
    else if (payload->is_cuda()) result.devices.emplace_back(tyloi::DeviceType::cuda, payload->get_device());
    else if (payload->is_cpu()) result.devices.emplace_back(tyloi::DeviceType::cpu);
    else throw std::invalid_argument("unsupported matrix weight device");
    return result;
}

int64_t QuantLinear::out() const {
    if (tensor_parallel()) return logical_out;
    if(is_tq())return tq.out;
    if ((tyloi::mfq_compat_enabled && is_nint())) return nint.out;
    if ((tyloi::mfq_compat_enabled && is_nvq())) return nvq.out;
    if (is_mxfp4()) return mxfp4.out;
    if ((tyloi::mfq_compat_enabled && is_mxfp4_sq())) return mxfp4_sq.out;
    if ((tyloi::mfq_compat_enabled && is_fp8_sq())) return fp8_sq.out;
    if (is_dense()) return dense.size(0);
    return mxfp8.out;
}

int64_t QuantLinear::neuron_len() const {
    if (tensor_parallel()) return logical_neuron_len;
    if(is_tq())return tq.neuron_len;
    if ((tyloi::mfq_compat_enabled && is_nint())) return nint.neuron_len;
    if ((tyloi::mfq_compat_enabled && is_nvq())) return nvq.neuron_len;
    if (is_mxfp4()) return mxfp4.neuron_len;
    if ((tyloi::mfq_compat_enabled && is_mxfp4_sq())) return mxfp4_sq.neuron_len;
    if ((tyloi::mfq_compat_enabled && is_fp8_sq())) return fp8_sq.neuron_len;
    if (is_dense()) return dense.size(1);
    return mxfp8.neuron_len;
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

std::vector<::tyloi::Tensor> DenseLinearGroup::forward(::tyloi::Tensor x) const {
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
