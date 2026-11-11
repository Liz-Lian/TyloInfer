#include "projection.h"

#include "engine/backends/cuda/native/collectives.h"
#include "engine/backends/cuda/kernels/mfq_cuda_quant_ops.h"
#include "engine/backends/cuda/native/allocation.h"

#include <algorithm>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tyloi::cuda {

#if TYLOINFER_ENABLE_MFQ_COMPAT
static tyloi::NintTensor cat_weights(const std::vector<tyloi::NintTensor>& ws) {
    if (ws.empty()) throw std::runtime_error("empty NINT group");
    const auto& a = ws[0];
    std::vector<::tyloi::DenseTensor> qp, rqb, rqoff, q8s, ss, sm, ns, nm;
    int64_t out = 0;
    int64_t q_bit_base = 0;
    bool aligned_q8 = !a.q8_zero;
    int uniform_qbits = a.uniform_qbits;
    for (const auto& w : ws) {
        if (w.uniform_qbits != uniform_qbits) uniform_qbits = 0;
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
    tyloi::NintTensor g;
    g.out = out;
    g.ng = a.ng;
    g.gs = a.gs;
    g.bits = a.bits;
    g.neuron_len = a.neuron_len;
    g.q8_zero = a.q8_zero;
    g.aligned_q8 = aligned_q8;
    g.uniform_qbits = uniform_qbits;
    g.shape = a.shape;
    g.shape[0] = out;
    g.q_packed = ::tyloi::cuda::cat(qp, 0).contiguous();
    if (!a.q8_zero) {
        g.row_q_bits = ::tyloi::cuda::cat(rqb, 0).contiguous();
        g.row_q_bit_offsets = ::tyloi::cuda::cat(rqoff, 0).contiguous();
    }
    if (a.q8_zero) {
        g.q8_zero_scale = ::tyloi::cuda::cat(q8s, 0).contiguous();
        return g;
    }
    g.sub_scale = ::tyloi::cuda::cat(ss, 0).contiguous();
    g.sub_min = ::tyloi::cuda::cat(sm, 0).contiguous();
    g.neuron_scale = ::tyloi::cuda::cat(ns, 0).contiguous();
    g.neuron_min = ::tyloi::cuda::cat(nm, 0).contiguous();
    return g;
}

static NintLinearGroup make_linear_group(const std::vector<tyloi::NintTensor>& ws) {
    NintLinearGroup g;
    for (const auto& w : ws) g.outs.push_back(w.out);
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
            for (const auto& source : ws) {
                tyloi::NintTensor projection = g.w;
                projection.out = source.out;
                projection.shape = source.shape;
                projection.q_packed = g.w.q_packed.narrow(0, offset, source.out);
                projection.q8_zero_scale =
                    g.w.q8_zero_scale.narrow(0, offset, source.out);
                g.projection_w.push_back(std::move(projection));
                offset += source.out;
            }
            TYLOINFER_RUNTIME_CHECK(offset == g.w.out,
                "Q8 projection views do not cover grouped output");
        }
    } else {
        std::vector<tyloi::NintTensor> cur;
        std::vector<int64_t> cur_outs;
        auto flush = [&]() {
            if (cur.empty()) return;
            g.split_w.push_back(cur.size() == 1 ? cur[0] : cat_weights(cur));
            g.split_outs.push_back(cur_outs);
            cur.clear();
            cur_outs.clear();
        };
        for (const auto& w : ws) {
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
        std::vector<::tyloi::Tensor> layers,
        bool group_nint,
        bool cpu_layer,
        bool preserve_projection_boundaries) {
    const tyloi::cuda::AllocationScope weight_memory;
    if (layers.empty()) throw std::runtime_error("empty quantized linear group");
    QuantLinearGroup result;
    result.decode_branch_parallel = !preserve_projection_boundaries;
    result.outs.reserve(layers.size());
    result.cols.reserve(layers.size());
    result.placements.reserve(layers.size());
    bool all_nint = true;
    std::vector<tyloi::NintTensor> nint_weights;
    nint_weights.reserve(layers.size());
    for (const auto& layer : layers) {
        result.outs.push_back(layer.size(0));
        result.cols.push_back(layer.size(1));
        result.placements.push_back(projection_placement(layer));
        const auto* nint = layer.representation<tyloi::NintTensor>();
        all_nint = all_nint && tyloi::mfq_compat_enabled && nint != nullptr;
        if (nint != nullptr) nint_weights.push_back(*nint);
    }
    const bool any_tensor_parallel = std::any_of(
        layers.begin(), layers.end(),
        [](const ::tyloi::Tensor& layer) {
            return layer.representation<tyloi::ShardedTensor>() != nullptr;
        });
    const bool all_tq = std::all_of(layers.begin(), layers.end(),
        [](const ::tyloi::Tensor& layer) {
            const auto* tq = layer.representation<::tyloi::encoding::TqTensor>();
            return tq != nullptr && tq->is_cuda();
        });
    if (all_tq && !any_tensor_parallel) {
        std::vector<::tyloi::encoding::TqTensor> weights;
        weights.reserve(layers.size());
        for (const auto& layer : layers)
            weights.push_back(*layer.representation<::tyloi::encoding::TqTensor>());
        result.tq = TqProjectionGroup(std::move(weights));
    }
    if (tyloi::mfq_compat_enabled && all_nint && !preserve_projection_boundaries &&
        group_nint && !cpu_layer && !any_tensor_parallel) {
        result.nint_grouped = true;
#if TYLOINFER_ENABLE_MFQ_COMPAT
#ifdef TYLOINFER_NATIVE_CUDA
        std::optional<MfqCudaGuard> guard;
        if (!nint_weights.empty() && nint_weights.front().q_packed.is_cuda())
            guard.emplace(nint_weights.front().q_packed.device());
#endif
        result.nint = make_linear_group(nint_weights);
#endif
    } else {
        result.layers = std::move(layers);
        const auto* nvq0 = result.layers.size() >= 2
            ? result.layers[0].representation<tyloi::NvqTensor>() : nullptr;
        const auto* nvq1 = result.layers.size() >= 2
            ? result.layers[1].representation<tyloi::NvqTensor>() : nullptr;
        result.nvq_prefix2 = !cpu_layer && result.layers.size() >= 2 &&
            !any_tensor_parallel && tyloi::mfq_compat_enabled && nvq0 && nvq1 &&
            nvq_pair_compatible(*nvq0, *nvq1);
    }
    return result;
}

DenseLinearGroup make_dense_group(const std::vector<::tyloi::Tensor>& ws) {
    if (ws.empty()) throw std::runtime_error("empty dense group");
    DenseLinearGroup g;
    std::vector<::tyloi::DenseTensor> parts;
    parts.reserve(ws.size());
    for (const auto& weight : ws) {
        const auto& w = weight.dense();
        if (w.dim() != 2) throw std::runtime_error("dense linear group expects 2D weights");
        if (!parts.empty() && w.size(1) != parts[0].size(1))
            throw std::runtime_error("cannot group dense tensors with different input width");
        g.outs.push_back(w.size(0));
        parts.push_back(w.to(::tyloi::kFloat32));
    }
    auto joined = ::tyloi::cuda::cat(parts, 0).contiguous();
#ifdef TYLOINFER_NATIVE_CUDA
    g.w = std::make_shared<::tyloi::DenseTensor>(std::move(joined));
#else
    g.w = std::move(joined);
#endif
    return g;
}

static ::tyloi::DenseTensor dequant_quant_linear_f32(
        tyloi::cuda::Collectives& collectives, int primary,
        const ::tyloi::Tensor& linear) {
    const auto* sharded = linear.representation<tyloi::ShardedTensor>();
    if (!sharded) {
        if (linear.representation<::tyloi::encoding::TqTensor>() != nullptr)
            return quant_linear_reference_weight(linear).to(::tyloi::kFloat32).contiguous();
#if TYLOINFER_ENABLE_MFQ_COMPAT
        if (const auto* nint = linear.representation<tyloi::NintTensor>();
            nint != nullptr && tyloi::mfq_compat_enabled)
            return dequant_nint_dense_f32(*nint);
#endif
        if (const auto* nvq = linear.representation<tyloi::NvqTensor>();
            nvq != nullptr && tyloi::mfq_compat_enabled)
            return nvq_dequant(*nvq).to(::tyloi::kFloat32).contiguous();
        if (const auto* mx8 = linear.representation<tyloi::Mxfp8Tensor>();
            mx8 != nullptr)
            return mxfp8_dequant_cuda(mx8->values, mx8->scales).to(::tyloi::kFloat32).contiguous();
        if (const auto* mx4 = linear.representation<tyloi::Mxfp4Tensor>();
            mx4 != nullptr)
            return mxfp4_dequant_cuda(mx4->values, mx4->scales).to(::tyloi::kFloat32).contiguous();
#if TYLOINFER_ENABLE_MFQ_COMPAT
        if (const auto* sq4 = linear.representation<tyloi::Mxfp4SqTensor>();
            sq4 != nullptr && tyloi::mfq_compat_enabled) {
            if (!sq4->blob.is_cuda()) return dequant_mxfp4_sq_cpu(*sq4);
            return mxfp4_sq_dequant_cuda(
                sq4->blob, sq4->row_q, sq4->row_symbol_byte_offsets, sq4->row_auxiliary,
                sq4->bits, sq4->out, sq4->neuron_len, sq4->matrix_scale_base,
                sq4->q_sum, sq4->sq4_rows, true).contiguous();
        }
        if (const auto* sq8 = linear.representation<tyloi::Fp8SqTensor>();
            sq8 != nullptr && tyloi::mfq_compat_enabled)
            return dequant_fp8_sq(*sq8, true).contiguous();
#endif
        if (const auto* dense = linear.representation<::tyloi::DenseTensor>();
            dense != nullptr)
            return dense->to(::tyloi::kFloat32).contiguous();
        throw std::runtime_error("unsupported linear weight for FP32 reconstruction");
    }
    if (sharded->axis != TensorParallelAxis::Output &&
        sharded->axis != TensorParallelAxis::Input)
        throw std::runtime_error("cannot reconstruct a mirrored tensor-parallel linear");
    std::vector<::tyloi::DenseTensor> parts;
    parts.reserve(sharded->shards.size());
    for (const auto& shard : sharded->shards) {
        MfqCudaGuard shard_guard(shard.device);
        ::tyloi::DenseTensor part;
        const auto& payload = shard.weight;
        if (payload.representation<::tyloi::encoding::TqTensor>() != nullptr) {
            part = quant_linear_reference_weight(payload).to(::tyloi::kFloat32).contiguous();
        } else
#if TYLOINFER_ENABLE_MFQ_COMPAT
        if (const auto* nint = payload.representation<tyloi::NintTensor>();
            nint != nullptr && tyloi::mfq_compat_enabled) {
            part = dequant_nint_dense_f32(*nint);
        } else if (const auto* nvq = payload.representation<tyloi::NvqTensor>();
            nvq != nullptr && tyloi::mfq_compat_enabled) {
            part = nvq_dequant(*nvq).to(::tyloi::kFloat32).contiguous();
        } else
#endif
        if (const auto* mx8 = payload.representation<tyloi::Mxfp8Tensor>(); mx8 != nullptr) {
            part = mxfp8_dequant_cuda(mx8->values, mx8->scales).to(::tyloi::kFloat32).contiguous();
        } else if (const auto* mx4 = payload.representation<tyloi::Mxfp4Tensor>(); mx4 != nullptr) {
            part = mxfp4_dequant_cuda(mx4->values, mx4->scales).to(::tyloi::kFloat32).contiguous();
        } else if (payload.representation<tyloi::Mxfp4SqTensor>() != nullptr ||
                payload.representation<tyloi::Fp8SqTensor>() != nullptr) {
            part = dequant_quant_linear_f32(collectives, primary, payload);
        } else if (const auto* dense = payload.representation<::tyloi::DenseTensor>();
            dense != nullptr) {
            part = dense->to(::tyloi::kFloat32).contiguous();
        } else {
            throw std::runtime_error(
                "unsupported tensor-parallel shard representation for reconstruction");
        }
        parts.push_back(collectives.copy(part, primary).to(::tyloi::kFloat32).contiguous());
    }
    MfqCudaGuard primary_guard(primary);
    auto dense = ::tyloi::cuda::cat(
        parts, sharded->axis == TensorParallelAxis::Output ? 0 : 1).contiguous();
    if (dense.size(0) != linear.size(0) || dense.size(1) != linear.size(1))
        throw std::runtime_error("reconstructed tensor-parallel linear shape mismatch");
    return dense;
}

DenseLinearGroup make_fp32_quant_group(
        tyloi::cuda::Collectives& collectives,
        int primary,
        std::span<const ::tyloi::Tensor> weights) {
    DenseLinearGroup dense;
    std::vector<::tyloi::DenseTensor> parts;
    parts.reserve(weights.size());
    for (const auto& weight : weights) {
        dense.outs.push_back(weight.size(0));
        parts.push_back(dequant_quant_linear_f32(collectives, primary, weight));
    }
    auto joined = ::tyloi::cuda::cat(parts, 0).contiguous();
#ifdef TYLOINFER_NATIVE_CUDA
    dense.w = std::make_shared<::tyloi::DenseTensor>(std::move(joined));
    const auto& grouped = dense.w;
#else
    dense.w = std::move(joined);
    const auto& grouped = dense.w;
#endif
    TYLOINFER_RUNTIME_CHECK(
        grouped.dim() == 2 &&
            grouped.size(0) == std::accumulate(
                dense.outs.begin(), dense.outs.end(), int64_t{0}),
        "FP32 compressor projection shape mismatch");
    return dense;
}

} // namespace tyloi::cuda
