#pragma once

#include "engine/weights/encoding/nvq.h"

// Weight headers keep the native backend closure their consumers already use.
#include "backend.h"

#include <cstdint>
#include <vector>

struct NepqWeight {
    ::tyloi::DenseTensor indices_packed;
    ::tyloi::DenseTensor aux_packed;
    ::tyloi::DenseTensor state_packed;
    ::tyloi::DenseTensor neuron_scale;
    ::tyloi::DenseTensor table_pool;
    ::tyloi::DenseTensor grouped_table_pool;
    ::tyloi::DenseTensor bank_ids;
    ::tyloi::DenseTensor rotation_signs;
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
    ::tyloi::DenseTensor residual_codebook;
    ::tyloi::DenseTensor residual_first;
    ::tyloi::DenseTensor residual_second;

    // Every buffer an NEPQ pool keeps alive. It stays a plain aggregate: the
    // expert pool is not a projection weight, but byte accounting and residency
    // checks still need the whole list.
    std::vector<const ::tyloi::DenseTensor*> buffers() const {
        return {&indices_packed, &aux_packed, &state_packed, &neuron_scale, &table_pool,
            &grouped_table_pool, &bank_ids, &rotation_signs, &residual_codebook,
            &residual_first, &residual_second};
    }
};

inline std::vector<const ::tyloi::DenseTensor*> nepq_buffers(const NepqWeight& weight) {
    return weight.buffers();
}
