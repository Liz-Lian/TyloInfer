#include "quant_linear.h"
#include "engine/backends/cuda/kernels/mfq_cuda_embedding_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_quant_ops.h"
#include "formats/mxfp4_sq_decode.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#if TYLOINFER_ENABLE_MFQ_COMPAT
#include "../mfq/linear_dispatch.inc"

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
