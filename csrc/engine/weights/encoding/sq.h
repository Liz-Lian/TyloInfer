#pragma once

#include "engine/weights/encoding/encoded.h"

#include <cstdint>
#include <string>
#include <vector>

namespace tyloi {

// FP8 SQ: one packed blob with per-row quantization and symbol offsets.
class Fp8SqTensor final : public EncodedTensor {
public:
    std::string dtype;
    DenseTensor blob;
    DenseTensor row_q;
    DenseTensor row_symbol_byte_offsets;
    int64_t out = 0;
    int64_t neuron_len = 0;
    int64_t block_rows = 0;
    int64_t block_columns = 0;
    int64_t scale_rows = 0;
    int64_t scale_columns = 0;
    int64_t scale_kind = 0;
    int64_t palettes_offset = 0;
    int64_t symbols_offset = 0;
    int64_t scales_offset = 0;
    int format_version = 0;
    double aggregate_bpw = 0.0;
    double distribution_entropy = 0.0;

    std::vector<const DenseTensor*> buffers() const override {
        return {&blob, &row_q, &row_symbol_byte_offsets};
    }

protected:
    const DenseTensor& primary() const override { return blob; }
    std::int64_t logical_rows() const override { return out; }
    std::int64_t logical_columns() const override { return neuron_len; }
};

// MXFP4 SQ: packed 4-bit payload plus per-row quantization and auxiliary codes.
class Mxfp4SqTensor final : public EncodedTensor {
public:
    DenseTensor blob;
    DenseTensor row_q;
    DenseTensor row_symbol_byte_offsets;
    DenseTensor row_auxiliary;
    int64_t bits = 0;
    int64_t out = 0;
    int64_t neuron_len = 0;
    int64_t matrix_scale_base = 0;
    int64_t q_sum = 0;
    int64_t sq4_rows = 0;
    int format_version = 0;
    double aggregate_bpw = 0.0;
    double distribution_entropy = 0.0;

    std::vector<const DenseTensor*> buffers() const override {
        return {&blob, &row_q, &row_symbol_byte_offsets, &row_auxiliary};
    }

protected:
    const DenseTensor& primary() const override { return blob; }
    std::int64_t logical_rows() const override { return out; }
    std::int64_t logical_columns() const override { return neuron_len; }
};

} // namespace tyloi
