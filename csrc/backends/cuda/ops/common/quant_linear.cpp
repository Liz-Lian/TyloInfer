#include "quant_linear.h"
#include "native/collectives.h"
#include "mfq_cuda_embedding_ops.h"
#include "mfq_cuda_quant_ops.h"
#include "formats/mxfp4_sq_decode.h"
#include "weight_memory.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace mfq::cuda::quant_format;

#if TYLOINFER_ENABLE_MFQ_COMPAT
static NintLinearGroup make_linear_group(const std::vector<NintWeight>& weights);
static mfq_tensor_backend::Tensor dequant_mxfp4_sq_cpu(const Mxfp4SqWeight& weight);

#endif

#if TYLOINFER_ENABLE_MFQ_COMPAT
#include "../mfq/linear_dispatch.inc"
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

QuantLinearGroup make_quant_group(
        std::vector<QuantLinear> layers,
        bool group_nint,
        bool cpu_layer,
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
        group_nint && !cpu_layer &&
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
        mfq::cuda::Collectives& collectives, int primary,
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
            part = dequant_quant_linear_f32(collectives, primary, local);
        } else if (shard.kind == QuantLinearKind::Dense) {
            part = shard.dense.to(mfq_tensor_backend::kFloat32).contiguous();
        } else {
            throw std::runtime_error(
                "unsupported tensor-parallel shard kind for reconstruction");
        }
        parts.push_back(
            collectives.copy(part, primary)
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
        mfq::cuda::Collectives& collectives,
        int primary,
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
                collectives, primary, linear));
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
