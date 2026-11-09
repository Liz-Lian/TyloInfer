#include "engine/weights/group.h"

#include "engine/backends/cuda/ops/projection/projection.h"
#include "engine/weights/encoding/tq.h"

#include <stdexcept>
#include <utility>
#include <vector>

namespace {

#if TYLOINFER_ENABLE_MFQ_COMPAT
std::vector<std::shared_ptr<const ::tyloi::Tensor>> nint_aliases(
        const std::vector<std::shared_ptr<const ::tyloi::Tensor>>& weights,
        const NintLinearGroup& group) {
    std::vector<std::shared_ptr<const ::tyloi::Tensor>> aliases;
    aliases.reserve(weights.size());
    size_t matrix_index = 0;
    int64_t row = 0;
    int64_t packed_bytes = 0;
    for (const auto & original : weights) {
        const auto & matrix = group.split_w.empty() ? group.w : group.split_w.at(matrix_index);
        const auto* original_nint = dynamic_cast<const tyloi::NintTensor*>(original.get());
        TYLOINFER_RUNTIME_CHECK(
            original_nint != nullptr && row + original->size(0) <= matrix.out &&
                original_nint->ng == matrix.ng && original_nint->gs == matrix.gs &&
                original_nint->q8_zero == matrix.q8_zero,
            "grouped projection storage geometry mismatch");
        auto view = *original_nint;
        view.q_packed = matrix.q_packed;
        if (view.q8_zero) {
            view.q_packed = matrix.q_packed.narrow(0, row, view.out);
            view.q8_zero_scale = matrix.q8_zero_scale.narrow(0, row, view.out);
        } else {
            view.row_q_bits = matrix.row_q_bits.narrow(0, row, view.out);
            view.row_q_bit_offsets = matrix.row_q_bit_offsets.narrow(0, row, view.out);
            view.sub_scale = matrix.sub_scale.narrow(0, row, view.out);
            view.sub_min = matrix.sub_min.narrow(0, row, view.out);
            view.neuron_scale = matrix.neuron_scale.narrow(0, row, view.out);
            view.neuron_min = matrix.neuron_min.narrow(0, row, view.out);
            view.aligned_q8 = original_nint->aligned_q8 && (packed_bytes % 4 == 0);
        }
        aliases.push_back(std::make_shared<tyloi::NintTensor>(std::move(view)));
        row += original->size(0);
        packed_bytes += original_nint->q_packed.numel();
        if (row == matrix.out) {
            ++matrix_index;
            row = 0;
            packed_bytes = 0;
        }
    }
    TYLOINFER_RUNTIME_CHECK(
        row == 0 && matrix_index == (group.split_w.empty() ? 1 : group.split_w.size()),
        "grouped projection storage is not fully covered");
    return aliases;
}
#endif

} // namespace

namespace tyloi::weights {

#if TYLOINFER_ENABLE_MFQ_COMPAT
std::vector<std::shared_ptr<const ::tyloi::Tensor>> pack_nint(
        std::vector<std::shared_ptr<const ::tyloi::Tensor>> weights) {
    if (weights.empty()) return weights;
    auto group = tyloi::cuda::make_quant_group(weights, true, false);
    if (group.nint_grouped) return nint_aliases(weights, group.nint);
    return weights;
}
#endif

std::vector<std::shared_ptr<const ::tyloi::Tensor>> pack_projections(
        std::vector<std::shared_ptr<const ::tyloi::Tensor>> weights,
        bool group_nint,
        bool cpu_layer) {
    if (weights.empty() || cpu_layer) return weights;
    const ::tyloi::encoding::TqTensor* first = nullptr;
    std::vector<::tyloi::encoding::TqTensor> tq_weights;
    for (const auto& weight : weights) {
        const auto* tq = dynamic_cast<const ::tyloi::encoding::TqTensor*>(weight.get());
        if (!tq || !tq->is_cuda()) continue;
        if (first && (weight->size(1) != first->size(1) || tq->device() != first->device()))
            return weights;
        if (!first) first = tq;
        tq_weights.push_back(*tq);
    }
    if (tq_weights.size() >= 2) {
        tyloi::cuda::tq::validate_group(tq_weights);
        return weights;
    }
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (!group_nint) return weights;
    for (const auto& weight : weights) {
        const auto* nint = dynamic_cast<const tyloi::NintTensor*>(weight.get());
        if (!nint || !nint->q_packed.is_cuda() || weight->size(1) != weights.front()->size(1))
            return weights;
    }
    return pack_nint(std::move(weights));
#endif
    return weights;
}

} // namespace tyloi::weights
