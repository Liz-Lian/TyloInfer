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

void check_placement(const tyloi::weights::Placement& placement, ::tyloi::Device device) {
    if (placement.caller && *placement.caller == (device.is_cuda() ? tyloi::DeviceType::cuda : tyloi::DeviceType::cpu))
        return;
    for (const auto& place : placement.devices)
        if (device == (place.is_cuda() ? ::tyloi::Device(::tyloi::kCUDA, place.index)
                                      : ::tyloi::Device(::tyloi::kCPU))) return;
    throw std::invalid_argument("projection weight and input devices do not match");
}

void check_fields(const ::tyloi::Tensor& weight, ::tyloi::Device device, bool check_device = true) {
    for (const auto* field : weight.buffers()) {
        field->require_dense();
        if (!check_device || !field->defined() || field->numel() == 0) continue;
        if (field->device() != device)
            throw std::invalid_argument("projection weight fields and input devices do not match");
    }
}

}  // namespace

int64_t shard_width(const QuantLinearShard& shard) { return shard.weight->size(1); }

// NINT and NVQ pack input groups, so a narrower activation is accepted.
bool width_is_padded(const ::tyloi::Tensor& weight) {
    return dynamic_cast<const tyloi::NintTensor*>(&weight) != nullptr ||
           dynamic_cast<const tyloi::NvqTensor*>(&weight) != nullptr;
}

void check_projection_width(const QuantLinear& linear, int64_t width) {
    if (!linear.tensor_parallel()) {
        check_width(width, linear.neuron_len(), width_is_padded(linear.payload()));
        return;
    }
    if (linear.tensor_parallel_axis == TensorParallelAxis::Input) {
        check_width(width, linear.neuron_len(), false);
        for (const auto& shard : linear.tensor_parallel_shards) {
            if (shard.input_begin < 0 || shard.input_end <= shard.input_begin || shard.input_end > width)
                throw std::invalid_argument("projection input does not cover tensor-parallel shard");
            check_width(shard.input_end - shard.input_begin, shard_width(shard),
                        width_is_padded(*shard.weight));
        }
    } else if (linear.tensor_parallel_axis == TensorParallelAxis::Output) {
        for (const auto& shard : linear.tensor_parallel_shards)
            check_width(width, shard_width(shard), width_is_padded(*shard.weight));
    } else {
        throw std::invalid_argument("tensor-parallel linear has an invalid axis");
    }
}

void check_projection_device(const QuantLinear& linear, ::tyloi::Device device) {
    if (!linear.tensor_parallel()) {
        check_placement(linear.placement(), device);
        check_fields(linear.payload(), device);
        return;
    }
    for (const auto& shard : linear.tensor_parallel_shards) {
        if (!shard.weight->defined() || !shard.weight->is_cuda() ||
            shard.weight->get_device() != shard.device)
            throw std::invalid_argument("tensor-parallel weight is not prepared on its device");
        check_fields(*shard.weight, ::tyloi::Device(::tyloi::kCUDA, shard.device));
    }
}

void check_projection(const QuantLinear& weight, const ::tyloi::DenseTensor& input) {
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
        for (const auto& layer : group.layers) check_projection(*layer, input);
        return;
    }
    if (!group.nint_grouped || !input.is_cuda())
        throw std::invalid_argument("projection group is not prepared for input");
    for (size_t i = 0; i < group.outs.size(); ++i) {
        check_width(input.size(-1), group.cols[i], true);
        check_placement(group.placements[i], input.device());
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

void check_device(const ::tyloi::Tensor& weight, const ::tyloi::DenseTensor& input) {
    check_fields(weight, input.device(), input.is_cuda());
}


void check_device(const QuantLinear& weight, const ::tyloi::DenseTensor& input) {
    // TQ projections validate their own payload; tensor-parallel shards are
    // checked per shard against the shard device.
    if (weight.tensor_parallel() ||
        dynamic_cast<const tyloi::encoding::TqTensor*>(&weight.payload()) != nullptr)
        return;
    check_device(weight.payload(), input);
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
        const QuantLinear & embedding,
        ::tyloi::DenseTensor token_ids) {
    TYLOINFER_RUNTIME_CHECK(
        !embedding.tensor_parallel(),
        "quantized embedding does not support tensor-parallel shards");
    const auto& weight = embedding.payload();
    if (const auto* dense = dynamic_cast<const ::tyloi::DenseTensor*>(&weight)) {
        auto output_shape = token_ids.sizes().vec();
        output_shape.push_back(dense->size(1));
        return dense->index_select(0, token_ids.reshape({-1})).reshape(output_shape);
    }
    if (const auto* tq = dynamic_cast<const tyloi::encoding::TqTensor*>(&weight))
        return tyloi::cuda::tq::embedding(*tq, token_ids);
    if (const auto* nvq = dynamic_cast<const tyloi::NvqTensor*>(&weight)) {
        TYLOINFER_RUNTIME_CHECK(tyloi::mfq_compat_enabled, "NVQ embedding weight is not enabled");
        check_device(*nvq, token_ids);
        return nvq_embedding(*nvq, token_ids);
    }
    if (const auto* nint = dynamic_cast<const tyloi::NintTensor*>(&weight)) {
        TYLOINFER_RUNTIME_CHECK(tyloi::mfq_compat_enabled, "NINT embedding weight is not enabled");
        check_device(*nint, token_ids);
        if (nint->q8_zero)
            return nint8_zero_embedding_lookup_cuda(
                nint->q_packed, nint->q8_zero_scale, token_ids, nint->neuron_len);
        return nint_embedding_cuda(
            nint->q_packed, nint->row_q_bits, nint->row_q_bit_offsets,
            nint->sub_scale, nint->sub_min, nint->neuron_scale, nint->neuron_min,
            token_ids, nint->neuron_len, nint->gs);
    }
    if (const auto* mx4 = dynamic_cast<const tyloi::Mxfp4Tensor*>(&weight)) {
        check_device(*mx4, token_ids);
        return mxfp4_embedding_lookup_cuda(mx4->values, mx4->scales, token_ids);
    }
    if (const auto* mx8 = dynamic_cast<const tyloi::Mxfp8Tensor*>(&weight)) {
        check_device(*mx8, token_ids);
        return mxfp8_embedding_lookup_cuda(mx8->values, mx8->scales, token_ids);
    }
    TYLOINFER_RUNTIME_CHECK(
        dynamic_cast<const tyloi::Mxfp4SqTensor*>(&weight) == nullptr &&
            dynamic_cast<const tyloi::Fp8SqTensor*>(&weight) == nullptr,
        "SQ tensors do not support embedding lookup");
    throw std::runtime_error("unsupported quantized embedding weight");
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

::tyloi::DenseTensor quant_linear_reference_weight(
        const QuantLinear& linear) {
    TYLOINFER_RUNTIME_CHECK(!linear.tensor_parallel(), "linear reference requires local weights");
    const auto& weight = linear.payload();
    if (const auto* tq = dynamic_cast<const tyloi::encoding::TqTensor*>(&weight)) {
        const auto ids = ::tyloi::cuda::arange(tq->size(0),
            ::tyloi::TensorOptions{}.dtype(::tyloi::kInt64).device(tq->device()));
        return tyloi::cuda::tq::embedding(*tq, ids);
    }
    if (const auto* nint = dynamic_cast<const tyloi::NintTensor*>(&weight)) {
        TYLOINFER_RUNTIME_CHECK(tyloi::mfq_compat_enabled, "NINT linear weight is not enabled");
        return nint->q8_zero
            ? nint8_zero_dequant_cuda(nint->q_packed, nint->q8_zero_scale, nint->neuron_len)
            : nint_decode_cuda(nint->q_packed, nint->row_q_bits, nint->row_q_bit_offsets,
                               nint->sub_scale, nint->sub_min, nint->neuron_scale,
                               nint->neuron_min, nint->neuron_len, nint->gs);
    }
    if (const auto* nvq = dynamic_cast<const tyloi::NvqTensor*>(&weight)) {
        TYLOINFER_RUNTIME_CHECK(tyloi::mfq_compat_enabled, "NVQ linear weight is not enabled");
        return nvq_dequant(*nvq);
    }
    if (const auto* mx4 = dynamic_cast<const tyloi::Mxfp4Tensor*>(&weight))
        return mxfp4_dequant_cuda(mx4->values, mx4->scales);
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (const auto* sq4 = dynamic_cast<const tyloi::Mxfp4SqTensor*>(&weight)) {
        TYLOINFER_RUNTIME_CHECK(tyloi::mfq_compat_enabled, "MXFP4-SQ linear weight is not enabled");
        if (!sq4->blob.is_cuda()) return dequant_mxfp4_sq_cpu(*sq4).to(::tyloi::kFloat16);
        return mxfp4_sq_dequant_cuda(
            sq4->blob, sq4->row_q, sq4->row_symbol_byte_offsets, sq4->row_auxiliary,
            sq4->bits, sq4->out, sq4->neuron_len, sq4->matrix_scale_base,
            sq4->q_sum, sq4->sq4_rows, false);
    }
#endif
    if (const auto* sq8 = dynamic_cast<const tyloi::Fp8SqTensor*>(&weight)) {
        TYLOINFER_RUNTIME_CHECK(tyloi::mfq_compat_enabled, "FP8-SQ linear weight is not enabled");
        return dequant_fp8_sq(*sq8, false);
    }
    if (const auto* mx8 = dynamic_cast<const tyloi::Mxfp8Tensor*>(&weight))
        return mxfp8_cpu_reference(*mx8);
    if (const auto* dense = dynamic_cast<const ::tyloi::DenseTensor*>(&weight)) return *dense;
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

tyloi::weights::Placement QuantLinear::placement(std::size_t index) const {
    if (index != 0) throw std::out_of_range("matrix index out of range");
    tyloi::weights::Placement result;
    if (tensor_parallel()) {
        for (const auto& shard : tensor_parallel_shards)
            result.devices.emplace_back(tyloi::DeviceType::cuda, shard.device);
        return result;
    }
    result.devices.push_back(weight->device());
    return result;
}

int64_t QuantLinear::out() const {
    return tensor_parallel() ? logical_out : weight->size(0);
}

std::shared_ptr<const QuantLinear> QuantLinear::relocated(::tyloi::Device device) const {
    auto moved = std::make_shared<QuantLinear>(*this);
    if (moved->weight) moved->weight = weight->relocated(device);
    for (auto& shard : moved->tensor_parallel_shards)
        shard.weight = shard.weight->relocated(::tyloi::Device(::tyloi::kCUDA, shard.device));
    return moved;
}

int64_t QuantLinear::neuron_len() const {
    return tensor_parallel() ? logical_neuron_len : weight->size(1);
}


QuantLinearProjectionRefs QuantLinearGroup::tensor_parallel_output_projections() const {
    QuantLinearProjectionRefs projections;
    projections.reserve(layers.size());
    for (const auto & layer : layers) {
        projections.push_back(layer.get());
    }
    return projections;
}

bool QuantLinearGroup::tensor_parallel_output_compatible() const {
    return tensor_parallel_output_projections_compatible(
        tensor_parallel_output_projections());
}

std::vector<::tyloi::DenseTensor> DenseLinearGroup::forward(::tyloi::DenseTensor x) const {
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
