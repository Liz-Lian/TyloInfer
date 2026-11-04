#pragma once

#include "backend.h"

#include <cstdint>
#include <vector>

struct NvqWeight {
    ::tyloi::Tensor indices_packed;
    ::tyloi::Tensor aux_packed;
    ::tyloi::Tensor sub_scale_packed;
    ::tyloi::Tensor neuron_scale;
    ::tyloi::Tensor codebook;
    // Optional decode-only int8 digits: NVQ1-L 8*v+delta, S 32*v+5*delta.
    // Canonical books retain the dense/CPU and reference decode contract.
    ::tyloi::Tensor integer_codebook;
    // Complete NVQ1 GS24 records (S 4B, L 5B), aliasing indices_packed.
    ::tyloi::Tensor decode_records;
    // One byte-preserving group stream replaces indices/aux/states.
    bool dense_groups = false;
    uint32_t bank_map = 0;
    int64_t format = 0;
    int64_t kernel_format = 0;
    int64_t sign_mode = 0;
    int64_t sub_bits = 0;
    int64_t gs = 0;
    int64_t out = 0;
    int64_t ng = 0;
    int64_t neuron_len = 0;
    std::vector<int64_t> shape;
};

inline constexpr int64_t kNvq2JscXlGroupExecKernelFormat = 16;
inline constexpr int64_t kNvq3JscLGroupExecKernelFormat = 17;

struct NepqWeight {
    ::tyloi::Tensor indices_packed;
    ::tyloi::Tensor aux_packed;
    ::tyloi::Tensor state_packed;
    ::tyloi::Tensor neuron_scale;
    ::tyloi::Tensor table_pool;
    ::tyloi::Tensor grouped_table_pool;
    ::tyloi::Tensor bank_ids;
    ::tyloi::Tensor rotation_signs;
    int format = 0;
    int state_bits = 0;
    int n_experts = 0;
    int out_per_expert = 0;
    int neuron_len = 0;
    int ng = 0;
    int rotation_block = 0;
    uint64_t rotation_seed = 0;
    bool residual = false;
    int residual_position_bits = 0;
    int residual_block_vectors = 0;
    ::tyloi::Tensor residual_codebook;
    ::tyloi::Tensor residual_first;
    ::tyloi::Tensor residual_second;
};
