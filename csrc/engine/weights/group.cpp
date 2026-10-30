#include "engine/weights/group.h"

#include "native/cuda/collectives.h"
#include "engine/backends/cuda/kernels/mfq_cuda_quant_ops.h"
#include "weight_memory.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#if TYLOINFER_ENABLE_MFQ_COMPAT
static NintWeight cat_weights(const std::vector<NintWeight> & ws) {
    if (ws.empty()) throw std::runtime_error("empty NINT group");
    const auto & a = ws[0];
    std::vector<mfq_tensor_backend::Tensor> qp, rqb, rqoff, q8s, ss, sm, ns, nm;
    int64_t out = 0;
    int64_t q_bit_base = 0;
    bool aligned_q8 = !a.q8_zero;
    int uniform_qbits=a.uniform_qbits;
    for (const auto & w : ws) {
        if(w.uniform_qbits!=uniform_qbits)uniform_qbits=0;
        if (w.ng != a.ng || w.gs != a.gs ||
            w.neuron_len != a.neuron_len || w.q8_zero != a.q8_zero) {
            throw std::runtime_error("cannot group NINT tensors with different input layout");
        }
        qp.push_back(w.q_packed);
        if (!w.q8_zero) {
            aligned_q8 = aligned_q8 && w.aligned_q8 && (q_bit_base & 31) == 0;
            rqb.push_back(w.row_q_bits);
            rqoff.push_back(w.row_q_bit_offsets + q_bit_base);
            q_bit_base += w.q_packed.numel() * 8;
        }
        if (w.q8_zero) {
            q8s.push_back(w.q8_zero_scale);
            out += w.out;
            continue;
        }
        ss.push_back(w.sub_scale);
        sm.push_back(w.sub_min);
        ns.push_back(w.neuron_scale);
        nm.push_back(w.neuron_min);
        out += w.out;
    }
    NintWeight g;
    g.out = out;
    g.ng = a.ng;
    g.gs = a.gs;
    g.bits = a.bits;
    g.neuron_len = a.neuron_len;
    g.q8_zero = a.q8_zero;
    g.aligned_q8 = aligned_q8;
    g.uniform_qbits=uniform_qbits;
    g.shape = a.shape;
    g.shape[0] = out;
    g.q_packed = mfq_tensor_backend::cat(qp, 0).contiguous();
    if (!a.q8_zero) {
        g.row_q_bits = mfq_tensor_backend::cat(rqb, 0).contiguous();
        g.row_q_bit_offsets = mfq_tensor_backend::cat(rqoff, 0).contiguous();
    }
    if (a.q8_zero) {
        g.q8_zero_scale = mfq_tensor_backend::cat(q8s, 0).contiguous();
        return g;
    }
    g.sub_scale = mfq_tensor_backend::cat(ss, 0).contiguous();
    g.sub_min = mfq_tensor_backend::cat(sm, 0).contiguous();
    g.neuron_scale = mfq_tensor_backend::cat(ns, 0).contiguous();
    g.neuron_min = mfq_tensor_backend::cat(nm, 0).contiguous();
    return g;
}

static NintLinearGroup make_linear_group(const std::vector<NintWeight> & ws) {
    NintLinearGroup g;
    for (const auto & w : ws) g.outs.push_back(w.out);
    bool same = !ws.empty();
    for (size_t i = 1; i < ws.size(); ++i) {
        if (ws[i].ng != ws[0].ng || ws[i].gs != ws[0].gs ||
            ws[i].neuron_len != ws[0].neuron_len ||
            ws[i].q8_zero != ws[0].q8_zero) {
            same = false;
            break;
        }
    }
    if (same) {
        g.w = cat_weights(ws);
        if (g.w.q8_zero) {
            g.projection_w.reserve(ws.size());
            int64_t offset = 0;
            for (const auto & source : ws) {
                NintWeight projection = g.w;
                projection.out = source.out;
                projection.shape = source.shape;
                projection.q_packed =
                    g.w.q_packed.narrow(0, offset, source.out);
                projection.q8_zero_scale =
                    g.w.q8_zero_scale.narrow(0, offset, source.out);
                projection.workspaces.clear();
                g.projection_w.push_back(std::move(projection));
                offset += source.out;
            }
            TYLOINFER_RUNTIME_CHECK(offset == g.w.out,
                        "Q8 projection views do not cover grouped output");
        }
    } else {
        std::vector<NintWeight> cur;
        std::vector<int64_t> cur_outs;
        auto flush = [&]() {
            if (cur.empty()) return;
            g.split_w.push_back(cur.size() == 1 ? cur[0] : cat_weights(cur));
            g.split_outs.push_back(cur_outs);
            cur.clear();
            cur_outs.clear();
        };
        for (const auto & w : ws) {
            bool append = !cur.empty() &&
                w.ng == cur[0].ng && w.gs == cur[0].gs &&
                w.neuron_len == cur[0].neuron_len &&
                w.q8_zero == cur[0].q8_zero;
            if (!append) flush();
            cur.push_back(w);
            cur_outs.push_back(w.out);
        }
        flush();
    }
    return g;
}
#endif

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

namespace mfq::cuda {

Projections make_projections(
        std::vector<QuantLinear> weights,
        bool group_nint,
        bool cpu_layer,
        size_t prefix) {
    Projections result;
    result.weights = std::move(weights);
    result.prefix = prefix;
    if (result.weights.empty()) return result;
    TYLOINFER_RUNTIME_CHECK(
        prefix >= 1 && prefix <= result.weights.size(),
        "invalid grouped projection prefix");
    if (cpu_layer) return result;

    std::vector<QuantLinear> tq_weights;
    std::vector<size_t> tq_indices;
    for (size_t i = 0; i < result.weights.size(); ++i) {
        const auto & weight = result.weights[i];
        if (!weight.is_tq() || weight.tensor_parallel() ||
                !weight.tq.image.is_cuda()) {
            continue;
        }
        if (!tq_weights.empty() &&
                (weight.neuron_len() != tq_weights.front().neuron_len() ||
                 weight.tq.image.device() !=
                     tq_weights.front().tq.image.device())) {
            return result;
        }
        tq_weights.push_back(weight);
        tq_indices.push_back(i);
    }
    if (tq_weights.size() >= 2) {
        result.group = make_quant_group(
            tq_weights, group_nint, cpu_layer);
        TYLOINFER_RUNTIME_CHECK(
            result.group.tq.defined(),
            "TQ projection group was not constructed");
        result.indices = std::move(tq_indices);
        result.joined = prefix > 1 && result.indices.size() >= prefix;
        for (size_t i = 0; result.joined && i < prefix; ++i) {
            result.joined = result.indices[i] == i;
        }
        if (result.joined) result.group.tq.join_prefix(prefix);
        return result;
    }

#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (!group_nint) return result;
    for (size_t i = 0; i < result.weights.size(); ++i) {
        const auto & weight = result.weights[i];
        if (!weight.is_nint() || weight.tensor_parallel() ||
                !weight.nint.q_packed.is_cuda()) {
            return result;
        }
        if (i > 0 && weight.neuron_len() != result.weights.front().neuron_len()) {
            return result;
        }
    }

    auto group = make_quant_group(
        result.weights, group_nint, cpu_layer);
    if (!group.nint_grouped) return result;
    result.group = std::move(group);

    std::vector<QuantLinear> aliases;
    aliases.reserve(result.weights.size());
    size_t matrix_index = 0;
    int64_t row = 0;
    int64_t packed_bytes = 0;
    for (const auto & original : result.weights) {
        const auto & matrix = result.group.nint.split_w.empty()
            ? result.group.nint.w
            : result.group.nint.split_w.at(matrix_index);
        TYLOINFER_RUNTIME_CHECK(
            row + original.out() <= matrix.out &&
                original.nint.ng == matrix.ng &&
                original.nint.gs == matrix.gs &&
                original.nint.q8_zero == matrix.q8_zero,
            "grouped projection storage geometry mismatch");
        auto alias = original;
        auto & view = alias.nint;
        view.workspaces.clear();
        view.q_packed = matrix.q_packed;
        if (view.q8_zero) {
            view.q_packed = matrix.q_packed.narrow(0, row, view.out);
            view.q8_zero_scale = matrix.q8_zero_scale.narrow(0, row, view.out);
        } else {
            view.row_q_bits = matrix.row_q_bits.narrow(0, row, view.out);
            view.row_q_bit_offsets =
                matrix.row_q_bit_offsets.narrow(0, row, view.out);
            view.sub_scale = matrix.sub_scale.narrow(0, row, view.out);
            view.sub_min = matrix.sub_min.narrow(0, row, view.out);
            view.neuron_scale = matrix.neuron_scale.narrow(0, row, view.out);
            view.neuron_min = matrix.neuron_min.narrow(0, row, view.out);
            view.aligned_q8 =
                original.nint.aligned_q8 && (packed_bytes % 4 == 0);
        }
        aliases.push_back(std::move(alias));
        row += original.out();
        packed_bytes += original.nint.q_packed.numel();
        if (row == matrix.out) {
            ++matrix_index;
            row = 0;
            packed_bytes = 0;
        }
    }
    TYLOINFER_RUNTIME_CHECK(
        row == 0 && matrix_index ==
            (result.group.nint.split_w.empty()
                ? 1 : result.group.nint.split_w.size()),
        "grouped projection storage is not fully covered");
    result.weights = std::move(aliases);

    result.joined = prefix > 1 &&
        (result.group.nint.split_w.empty() ||
         result.group.nint.split_outs.front().size() >= prefix);
    const auto join_sizes = [prefix](auto & sizes) {
        const auto combined = std::accumulate(
            sizes.begin(), sizes.begin() + prefix, int64_t(0));
        sizes.erase(sizes.begin(), sizes.begin() + prefix);
        sizes.insert(sizes.begin(), combined);
    };
    if (result.joined) {
        if (result.group.nint.split_w.empty()) {
            join_sizes(result.group.nint.outs);
        } else {
            join_sizes(result.group.nint.split_outs.front());
        }
    }
#endif
    return result;
}

} // namespace mfq::cuda

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
