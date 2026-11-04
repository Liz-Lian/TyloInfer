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

const QuantLinear& as_quant_linear(const mfq::weights::Weights& weight) {
    const auto* result = dynamic_cast<const QuantLinear*>(&weight);
    if (!result || weight.backend() != mfq::DeviceType::cuda ||
        weight.layout() != mfq::weights::Layout::matrix)
        throw std::invalid_argument("matrix was not prepared by the CUDA backend");
    return *result;
}

const QuantLinearGroup& as_quant_linear_group(const mfq::weights::Weights& group) {
    const auto* result = dynamic_cast<const QuantLinearGroup*>(&group);
    if (!result || group.backend() != mfq::DeviceType::cuda ||
        group.layout() != mfq::weights::Layout::group)
        throw std::invalid_argument("matrix group was not prepared by the CUDA backend");
    return *result;
}

namespace {
void check_width(int64_t width, int64_t expected, bool padded) {
    if (width <= 0 || expected <= 0 || (padded ? width > expected : width != expected))
        throw std::invalid_argument("projection input width does not match weight");
}

void check_placement(const mfq::weights::Placement& placement, mfq_tensor_backend::Device device) {
    if (placement.caller && *placement.caller == (device.is_cuda() ? mfq::DeviceType::cuda : mfq::DeviceType::cpu))
        return;
    for (const auto& place : placement.devices)
        if (device == (place.is_cuda() ? mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, place.index)
                                      : mfq_tensor_backend::Device(mfq_tensor_backend::kCPU))) return;
    throw std::invalid_argument("projection weight and input devices do not match");
}

void check_fields(std::initializer_list<const mfq_tensor_backend::Tensor*> fields, mfq_tensor_backend::Device device) {
    for (const auto* field : fields) {
        if (!field->defined() || field->numel() == 0) continue;
        if (field->device() != device)
            throw std::invalid_argument("projection weight fields and input devices do not match");
    }
}

void check_nint_fields(const NintWeight& w, mfq_tensor_backend::Device device) {
    check_fields({&w.q_packed, &w.row_q_bits, &w.row_q_bit_offsets, &w.sub_scale,
                  &w.sub_min, &w.neuron_scale, &w.neuron_min}, device);
}

void check_fields(const NintWeight& w, mfq_tensor_backend::Device device) {
    if (w.q8_zero) check_fields({&w.q_packed, &w.q8_zero_scale}, device);
    else check_nint_fields(w, device);
}

void check_fields(const NvqWeight& w, mfq_tensor_backend::Device device) {
    check_fields({&w.indices_packed, &w.aux_packed, &w.sub_scale_packed,
                  &w.neuron_scale, &w.codebook}, device);
}

void check_fields(const Mxfp4Weight& w, mfq_tensor_backend::Device device) {
    check_fields({&w.values, &w.scales}, device);
}

void check_fields(const Mxfp8Weight& w, mfq_tensor_backend::Device device) {
    check_fields({&w.values, &w.scales}, device);
}

void check_fields(const Mxfp4SqWeight& w, mfq_tensor_backend::Device device) {
    if (!w.blob.is_cuda()) return;
    check_fields({&w.blob, &w.row_q, &w.row_symbol_byte_offsets, &w.row_auxiliary}, device);
}

void check_fields(const Fp8SqWeight& w, mfq_tensor_backend::Device device) {
    if (!w.blob.is_cuda() || (w.dtype != "MXFP8-SQ" && w.dtype != "FP8-128SQ")) return;
    check_fields({&w.blob, &w.row_q, &w.row_symbol_byte_offsets}, device);
}

const mfq_tensor_backend::Tensor& shard_payload(const QuantLinearShard& shard) {
    switch (shard.kind) {
    case QuantLinearKind::Nint: return shard.nint.q_packed;
    case QuantLinearKind::Nvq: return shard.nvq.indices_packed;
    case QuantLinearKind::Mxfp4: return shard.mxfp4.values;
    case QuantLinearKind::Mxfp4Sq: return shard.mxfp4_sq.weight.blob;
    case QuantLinearKind::Fp8Sq: return shard.fp8_sq.weight.blob;
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
    case QuantLinearKind::Mxfp4Sq: return shard.mxfp4_sq.weight.neuron_len;
    case QuantLinearKind::Fp8Sq: return shard.fp8_sq.weight.neuron_len;
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

void check_projection_device(const QuantLinear& weight, mfq_tensor_backend::Device device) {
    if (!weight.tensor_parallel()) {
        check_placement(weight.placement(), device);
        if (weight.is_nint()) check_fields(weight.nint, device);
        else if (weight.is_nvq()) check_fields(weight.nvq, device);
        else if (weight.is_mxfp4()) check_fields(weight.mxfp4.weight, device);
        else if (weight.is_mxfp8()) check_fields(weight.mxfp8.weight, device);
        else if (weight.is_mxfp4_sq()) check_fields(weight.mxfp4_sq.weight, device);
        else if (weight.is_fp8_sq()) check_fields(weight.fp8_sq.weight, device);
        return;
    }
    for (const auto& shard : weight.tensor_parallel_shards) {
        const auto& payload = shard_payload(shard);
        if (!payload.defined() || !payload.is_cuda() || payload.get_device() != shard.device)
            throw std::invalid_argument("tensor-parallel weight is not prepared on its device");
        const mfq_tensor_backend::Device local(mfq_tensor_backend::kCUDA, shard.device);
        switch (shard.kind) {
        case QuantLinearKind::Nint: check_fields(shard.nint, local); break;
        case QuantLinearKind::Nvq: check_fields(shard.nvq, local); break;
        case QuantLinearKind::Mxfp4: check_fields(shard.mxfp4, local); break;
        case QuantLinearKind::Mxfp4Sq: check_fields(shard.mxfp4_sq.weight, local); break;
        case QuantLinearKind::Fp8Sq: check_fields(shard.fp8_sq.weight, local); break;
        case QuantLinearKind::Mxfp8: check_fields(shard.mxfp8, local); break;
        case QuantLinearKind::Dense:
        case QuantLinearKind::Tq: break;
        }
    }
}

void check_projection(const QuantLinear& weight, const mfq_tensor_backend::Tensor& input) {
    if (!input.defined() || input.dim() < 1)
        throw std::invalid_argument("projection input must have a last dimension");
    check_projection_device(weight, input.device());
    check_projection_width(weight, input.size(-1));
}

void check_projection(const QuantLinearGroup& group, const mfq_tensor_backend::Tensor& input) {
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

void check_device(const mfq_tensor_backend::Tensor& weight, const mfq_tensor_backend::Tensor& input) {
    if (input.is_cuda() && weight.defined() && weight.numel() != 0 && weight.device() != input.device())
        throw std::invalid_argument("projection weight and input devices do not match");
}

void check_device(const NintWeight& weight, const mfq_tensor_backend::Tensor& input) {
    if (input.is_cuda()) check_fields(weight, input.device());
}

void check_nint(const NintWeight& weight, const mfq_tensor_backend::Tensor& input) {
    if (input.is_cuda()) check_nint_fields(weight, input.device());
}

void check_device(const NvqWeight& weight, const mfq_tensor_backend::Tensor& input) {
    if (input.is_cuda()) check_fields(weight, input.device());
}

void check_device(const Mxfp4Weight& weight, const mfq_tensor_backend::Tensor& input) {
    if (input.is_cuda()) check_fields(weight, input.device());
}

void check_device(const Mxfp8Weight& weight, const mfq_tensor_backend::Tensor& input) {
    if (input.is_cuda()) check_fields(weight, input.device());
}

void check_device(const Mxfp4SqWeight& weight, const mfq_tensor_backend::Tensor& input) {
    if (input.is_cuda()) check_fields(weight, input.device());
}

void check_device(const Fp8SqWeight& weight, const mfq_tensor_backend::Tensor& input) {
    if (input.is_cuda()) check_fields(weight, input.device());
}

void check_device(const QuantLinear& weight, const mfq_tensor_backend::Tensor& input) {
    if (weight.tensor_parallel() || weight.is_tq()) return;
    if (weight.is_nint()) check_device(weight.nint, input);
    else if (weight.is_nvq()) check_device(weight.nvq, input);
    else if (weight.is_mxfp4()) check_device(weight.mxfp4.weight, input);
    else if (weight.is_mxfp4_sq()) check_device(weight.mxfp4_sq.weight, input);
    else if (weight.is_fp8_sq()) check_device(weight.fp8_sq.weight, input);
    else if (weight.is_dense()) check_device(weight.dense, input);
    else check_device(weight.mxfp8.weight, input);
}

#if TYLOINFER_ENABLE_MFQ_COMPAT
#include "linear_dispatch.inc"

mfq_tensor_backend::Tensor dequant_nint_dense_f32(const NintWeight & w) {
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
        check_device(embedding.nvq, token_ids);
        return nvq_embedding(embedding.nvq, token_ids);
    }
    if ((mfq::mfq_compat_enabled && embedding.is_nint())) {
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
        check_device(embedding.mxfp4.weight, token_ids);
        return mxfp4_embedding_lookup_cuda(
            embedding.mxfp4.weight.values,
            embedding.mxfp4.weight.scales, token_ids);
    }
    if (embedding.is_mxfp8()) {
        check_device(embedding.mxfp8.weight, token_ids);
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

mfq::weights::Placement QuantLinear::placement(std::size_t index) const {
    if (index != 0) throw std::out_of_range("matrix index out of range");
    mfq::weights::Placement result;
    if (tensor_parallel()) {
        for (const auto& shard : tensor_parallel_shards)
            result.devices.emplace_back(mfq::DeviceType::cuda, shard.device);
        return result;
    }
    const mfq_tensor_backend::Tensor* payload = nullptr;
    if (is_tq()) payload = &tq.image;
    else if (is_nint()) payload = &nint.q_packed;
    else if (is_nvq()) payload = &nvq.indices_packed;
    else if (is_mxfp4()) payload = &mxfp4.weight.values;
    else if (is_mxfp8()) payload = &mxfp8.weight.values;
    else if (is_mxfp4_sq()) payload = &mxfp4_sq.weight.blob;
    else if (is_fp8_sq()) payload = &fp8_sq.weight.blob;
    else if (is_dense()) payload = &dense;
    if (!payload || !payload->defined())
        throw std::invalid_argument("matrix weight is not prepared");
    if (is_tq() && payload->is_cpu()) result.caller = mfq::DeviceType::cuda;
    else if (payload->is_cuda()) result.devices.emplace_back(mfq::DeviceType::cuda, payload->get_device());
    else if (payload->is_cpu()) result.devices.emplace_back(mfq::DeviceType::cpu);
    else throw std::invalid_argument("unsupported matrix weight device");
    return result;
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
