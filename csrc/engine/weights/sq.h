#pragma once

#include "backend.h"

#include <cstdint>
#include <string>

struct Fp8SqWeight {
    std::string dtype;
    ::tyloi::Tensor blob;
    ::tyloi::Tensor row_q;
    ::tyloi::Tensor row_symbol_byte_offsets;
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
};

struct Mxfp4SqWeight {
    ::tyloi::Tensor blob;
    ::tyloi::Tensor row_q;
    ::tyloi::Tensor row_symbol_byte_offsets;
    ::tyloi::Tensor row_auxiliary;
    int64_t bits = 0;
    int64_t out = 0;
    int64_t neuron_len = 0;
    int64_t matrix_scale_base = 0;
    int64_t q_sum = 0;
    int64_t sq4_rows = 0;
    int format_version = 0;
    double aggregate_bpw = 0.0;
    double distribution_entropy = 0.0;
};
