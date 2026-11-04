#pragma once

#include "engine/weights/encoding/encoded.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace tyloi {

// NINT: integer codes with per-row bit widths and sub-group/neuron scales.
class NintTensor final : public EncodedTensor {
    std::shared_ptr<EncodedTensor> cloned() const override {
        return std::make_shared<NintTensor>(*this);
    }

public:
    DenseTensor q_packed;
    DenseTensor row_q_bits;
    DenseTensor row_q_bit_offsets;
    DenseTensor q8_zero_scale;
    DenseTensor sub_scale;
    DenseTensor sub_min;
    DenseTensor neuron_scale;
    DenseTensor neuron_min;
    // Experimental routed row records; populated only by the comparison bench.
    DenseTensor route_metadata;
    int64_t out = 0;
    int64_t ng = 0;
    int64_t gs = 0;
    int64_t bits = 0;
    int64_t neuron_len = 0;
    int64_t q_expert_stride = 0;
    int format_version = 2;
    double aggregate_bpw = 0.0;
    double distribution_entropy = 0.0;
    bool q8_zero = false;
    // Proven once from CPU row metadata; false for unclassified/grouped weights.
    bool aligned_q8 = false;
    // Proven from immutable CPU row metadata; zero means mixed/unknown.
    int uniform_qbits = 0;
    std::vector<int64_t> shape;

    // payload and its zero-point scale are required.

    // payload and its zero-point scale are required.
    std::vector<const DenseTensor*> buffers() const override {
        if (q8_zero) return {&q_packed, &q8_zero_scale};
        return {&q_packed, &row_q_bits, &row_q_bit_offsets, &sub_scale,
                &sub_min, &neuron_scale, &neuron_min};
    }

protected:
    const DenseTensor& primary() const override { return q_packed; }
    // With q8_zero the row/sub-group streams are absent; only the packed
    std::int64_t logical_rows() const override { return out; }
    std::int64_t logical_columns() const override { return neuron_len; }
};

} // namespace tyloi
