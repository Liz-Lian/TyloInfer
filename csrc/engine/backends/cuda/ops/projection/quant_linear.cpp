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

namespace {
void check_width(int64_t width, int64_t expected, bool padded) {
    if (width <= 0 || expected <= 0 || (padded ? width > expected : width != expected))
        throw std::invalid_argument("projection input width does not match weight");
}

}  // namespace

void check_fields(const ::tyloi::TensorImpl& weight, ::tyloi::Device device, bool check_device) {
    for (const auto* field : weight.buffers()) {
        field->require_dense();
        if (!check_device || !field->defined() || field->numel() == 0) continue;
        if (field->device() != device)
            throw std::invalid_argument("projection weight fields and input devices do not match");
    }
}

// NINT and NVQ pack input groups, so a narrower activation is accepted.
bool width_is_padded(const ::tyloi::Tensor& weight) {
    return weight.representation<tyloi::NintTensor>() != nullptr ||
           weight.representation<tyloi::NvqTensor>() != nullptr;
}

void check_projection_width(const ::tyloi::Tensor& weight, int64_t width) {
    const auto* sharded = weight.representation<tyloi::ShardedTensor>();
    if (!sharded) {
        check_width(width, weight.size(1), width_is_padded(weight));
        return;
    }
    if (sharded->axis == TensorParallelAxis::Input) {
        check_width(width, weight.size(1), false);
        for (const auto& shard : sharded->shards) {
            if (shard.input_begin < 0 || shard.input_end <= shard.input_begin || shard.input_end > width)
                throw std::invalid_argument("projection input does not cover tensor-parallel shard");
            check_width(shard.input_end - shard.input_begin, shard.weight.size(1),
                        width_is_padded(shard.weight));
        }
    } else if (sharded->axis == TensorParallelAxis::Output) {
        for (const auto& shard : sharded->shards)
            check_width(width, shard.weight.size(1), width_is_padded(shard.weight));
    } else {
        throw std::invalid_argument("tensor-parallel linear has an invalid axis");
    }
}

void check_projection_device(const ::tyloi::Tensor& weight, ::tyloi::Device device) {
    const auto* sharded = weight.representation<tyloi::ShardedTensor>();
    if (!sharded) {
        tyloi::weights::check_placement(projection_placement(weight), device);
        const bool cpu_tq = weight.representation<tyloi::encoding::TqTensor>() && !weight.is_cuda();
        check_fields(weight, device, !cpu_tq);
        return;
    }
    for (const auto& shard : sharded->shards) {
        if (!shard.weight || !shard.weight.defined() || !shard.weight.is_cuda() ||
            shard.weight.get_device() != shard.device)
            throw std::invalid_argument("tensor-parallel weight is not prepared on its device");
        check_fields(shard.weight, ::tyloi::Device(::tyloi::kCUDA, shard.device));
    }
}

void check_projection(const ::tyloi::Tensor& weight, const ::tyloi::DenseTensor& input) {
    input.require_dense();
    if (!input.defined() || input.dim() < 1)
        throw std::invalid_argument("projection input must have a last dimension");
    check_projection_device(weight, input.device());
    check_projection_width(weight, input.size(-1));
}

void check_projection(const QuantLinearGroup& group, const ::tyloi::DenseTensor& input) {
    input.require_dense();
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
        tyloi::weights::check_placement(group.placements[i], input.device());
    }
    const auto check_nint_weight = [&](const tyloi::NintTensor& weight) {
        if (!weight.q_packed.defined())
            throw std::invalid_argument("projection weight is not prepared");
        check_width(input.size(-1), weight.neuron_len, true);
        check_device(weight, input);
    };
    if (group.nint.split_w.empty()) check_nint_weight(group.nint.w);
    for (const auto& weight : group.nint.projection_w) check_nint_weight(weight);
    for (const auto& weight : group.nint.split_w) check_nint_weight(weight);
}

void check_device(const ::tyloi::DenseTensor& weight, const ::tyloi::DenseTensor& input) {
    weight.require_dense(); input.require_dense();
    if (input.is_cuda() && weight.defined() && weight.numel() != 0 && weight.device() != input.device())
        throw std::invalid_argument("projection weight and input devices do not match");
}

void check_device(const ::tyloi::TensorImpl& weight, const ::tyloi::DenseTensor& input) {
    check_fields(weight, input.device(), input.is_cuda());
}


#if TYLOINFER_ENABLE_MFQ_COMPAT
#include "linear_dispatch.inc"

::tyloi::DenseTensor dequant_nint_dense_f32(const tyloi::NintTensor & w) {
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

::tyloi::DenseTensor quant_embedding_lookup(
        const ::tyloi::Tensor & weight,
        ::tyloi::DenseTensor token_ids) {
    TYLOINFER_RUNTIME_CHECK(
        weight.representation<tyloi::ShardedTensor>() == nullptr,
        "quantized embedding does not support tensor-parallel shards");
    if (const auto* dense = weight.representation<::tyloi::DenseTensor>()) {
        auto output_shape = token_ids.sizes().vec();
        output_shape.push_back(dense->size(1));
        return dense->index_select(0, token_ids.reshape({-1})).reshape(output_shape);
    }
    if (const auto* tq = weight.representation<tyloi::encoding::TqTensor>())
        return tyloi::cuda::tq::embedding(*tq, token_ids);
    if (const auto* nvq = weight.representation<tyloi::NvqTensor>()) {
        TYLOINFER_RUNTIME_CHECK(tyloi::mfq_compat_enabled, "NVQ embedding weight is not enabled");
#if TYLOINFER_ENABLE_MFQ_COMPAT
        check_device(*nvq, token_ids);
        return nvq_embedding(*nvq, token_ids);
#endif
    }
    if (const auto* nint = weight.representation<tyloi::NintTensor>()) {
        TYLOINFER_RUNTIME_CHECK(tyloi::mfq_compat_enabled, "NINT embedding weight is not enabled");
#if TYLOINFER_ENABLE_MFQ_COMPAT
        check_device(*nint, token_ids);
        if (nint->q8_zero)
            return nint8_zero_embedding_lookup_cuda(
                nint->q_packed, nint->q8_zero_scale, token_ids, nint->neuron_len);
        return nint_embedding_cuda(
            nint->q_packed, nint->row_q_bits, nint->row_q_bit_offsets,
            nint->sub_scale, nint->sub_min, nint->neuron_scale, nint->neuron_min,
            token_ids, nint->neuron_len, nint->gs);
#endif
    }
    if (const auto* mx4 = weight.representation<tyloi::Mxfp4Tensor>()) {
        check_device(*mx4, token_ids);
        return mxfp4_embedding_lookup_cuda(mx4->values, mx4->scales, token_ids);
    }
    if (const auto* mx8 = weight.representation<tyloi::Mxfp8Tensor>()) {
        check_device(*mx8, token_ids);
        return mxfp8_embedding_lookup_cuda(mx8->values, mx8->scales, token_ids);
    }
    TYLOINFER_RUNTIME_CHECK(
        weight.representation<tyloi::Mxfp4SqTensor>() == nullptr &&
            weight.representation<tyloi::Fp8SqTensor>() == nullptr,
        "SQ tensors do not support embedding lookup");
    throw std::runtime_error("unsupported quantized embedding weight");
}

bool tensor_parallel_output_projections_compatible(
        const std::vector<::tyloi::Tensor>& projections) {
    if (projections.size() < 2) return false;
    const auto* first = projections.front().representation<tyloi::ShardedTensor>();
    if (!first || first->axis != TensorParallelAxis::Output || first->shards.size() < 2)
        return false;
    for (const auto& projection : projections) {
        const auto* sharded = projection.representation<tyloi::ShardedTensor>();
        if (!sharded || sharded->axis != TensorParallelAxis::Output ||
            sharded->shards.size() != first->shards.size()) return false;
        for (size_t i = 0; i < first->shards.size(); ++i) {
            const auto& candidate = sharded->shards[i];
            const auto& reference = first->shards[i];
            if (candidate.device != reference.device ||
                candidate.input_begin != reference.input_begin ||
                candidate.input_end != reference.input_end) return false;
        }
    }
    return true;
}

::tyloi::DenseTensor quant_linear_reference_weight(
        const ::tyloi::Tensor& weight) {
    TYLOINFER_RUNTIME_CHECK(weight.representation<tyloi::ShardedTensor>() == nullptr, "linear reference requires local weights");
    if (const auto* tq = weight.representation<tyloi::encoding::TqTensor>()) {
        const auto ids = ::tyloi::cuda::arange(tq->size(0),
            ::tyloi::TensorOptions{}.dtype(::tyloi::kInt64).device(tq->device()));
        return tyloi::cuda::tq::embedding(*tq, ids);
    }
    if (const auto* nint = weight.representation<tyloi::NintTensor>()) {
        TYLOINFER_RUNTIME_CHECK(tyloi::mfq_compat_enabled, "NINT linear weight is not enabled");
#if TYLOINFER_ENABLE_MFQ_COMPAT
        return nint->q8_zero
            ? nint8_zero_dequant_cuda(nint->q_packed, nint->q8_zero_scale, nint->neuron_len)
            : nint_decode_cuda(nint->q_packed, nint->row_q_bits, nint->row_q_bit_offsets,
                               nint->sub_scale, nint->sub_min, nint->neuron_scale,
                               nint->neuron_min, nint->neuron_len, nint->gs);
#endif
    }
    if (const auto* nvq = weight.representation<tyloi::NvqTensor>()) {
        TYLOINFER_RUNTIME_CHECK(tyloi::mfq_compat_enabled, "NVQ linear weight is not enabled");
#if TYLOINFER_ENABLE_MFQ_COMPAT
        return nvq_dequant(*nvq);
#endif
    }
    if (const auto* mx4 = weight.representation<tyloi::Mxfp4Tensor>())
        return mxfp4_dequant_cuda(mx4->values, mx4->scales);
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (const auto* sq4 = weight.representation<tyloi::Mxfp4SqTensor>()) {
        TYLOINFER_RUNTIME_CHECK(tyloi::mfq_compat_enabled, "MXFP4-SQ linear weight is not enabled");
        if (!sq4->blob.is_cuda()) return dequant_mxfp4_sq_cpu(*sq4).to(::tyloi::kFloat16);
        return mxfp4_sq_dequant_cuda(
            sq4->blob, sq4->row_q, sq4->row_symbol_byte_offsets, sq4->row_auxiliary,
            sq4->bits, sq4->out, sq4->neuron_len, sq4->matrix_scale_base,
            sq4->q_sum, sq4->sq4_rows, false);
    }
#endif
    if (const auto* sq8 = weight.representation<tyloi::Fp8SqTensor>()) {
        TYLOINFER_RUNTIME_CHECK(tyloi::mfq_compat_enabled, "FP8-SQ linear weight is not enabled");
#if TYLOINFER_ENABLE_MFQ_COMPAT
        return dequant_fp8_sq(*sq8, false);
#endif
    }
    if (const auto* mx8 = weight.representation<tyloi::Mxfp8Tensor>())
        return mxfp8_cpu_reference(*mx8);
    if (const auto* dense = weight.representation<::tyloi::DenseTensor>()) return *dense;
    throw std::runtime_error("unsupported linear reference format");
}

::tyloi::DenseTensor dense_projection(
        ::tyloi::DenseTensor input, const ::tyloi::DenseTensor& weight) {
    input = input.to(weight.scalar_type());
    const int64_t rows = input.numel() / input.size(-1);
    if (rows > 1 && rows <= 6) {
        auto shape = input.sizes().vec();
        shape.back() = weight.size(0);
        auto flat = input.reshape({rows, input.size(-1)});
        std::vector<::tyloi::DenseTensor> outputs;
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

tyloi::weights::Placement projection_placement(const ::tyloi::Tensor& weight, std::size_t index) {
    if (index != 0) throw std::out_of_range("matrix index out of range");
    tyloi::weights::Placement result;
    if (const auto* sharded = weight.representation<tyloi::ShardedTensor>()) {
        for (const auto& shard : sharded->shards)
            result.devices.emplace_back(tyloi::DeviceType::cuda, shard.device);
    } else {
        result.devices.push_back(weight.device());
        if (weight.representation<tyloi::encoding::TqTensor>() && !weight.is_cuda())
            result.caller = tyloi::DeviceType::cuda;
    }
    return result;
}

std::vector<::tyloi::Tensor> QuantLinearGroup::tensor_parallel_output_projections() const {
    std::vector<::tyloi::Tensor> projections;
    projections.reserve(layers.size());
    for (const auto& layer : layers) projections.push_back(layer);
    return projections;
}

bool QuantLinearGroup::tensor_parallel_output_compatible() const {
    return tensor_parallel_output_projections_compatible(
        tensor_parallel_output_projections());
}

std::vector<::tyloi::DenseTensor> DenseLinearGroup::forward(::tyloi::DenseTensor x) const {
    auto shape = x.sizes().vec();
    auto y = dense_projection(x.reshape({-1, x.size(-1)}),
        w.dense());
    auto parts = y.split_with_sizes(outs, -1);
    for (auto & p : parts) {
        auto s = shape;
        s.back() = p.size(-1);
        p = p.reshape(s);
    }
    return parts;
}
