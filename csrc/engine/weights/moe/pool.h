#pragma once

#include "engine/weights/mx.h"
#include "engine/weights/sq.h"
#include "engine/weights/nint.h"
#include "engine/weights/vq.h"

#include <cstdint>

struct MfePoolWeight {
    NintWeight weight;
    mfq_tensor_backend::Tensor expert_local;
    int local_experts = 0;
};

struct MoeActivationGeometry {
    int groups = 0;
    int gs = 0;
    int transform_block = 0;
    uint64_t transform_seed = 0;

    bool operator==(const MoeActivationGeometry & other) const {
        return groups == other.groups && gs == other.gs &&
            transform_block == other.transform_block &&
            transform_seed == other.transform_seed;
    }
};

enum class MixedMoeFamily {
    Nint,
    Nint8Zero,
    Mxfp4,
    Mxfp4Sq,
    Fp8Sq,
    Nvq,
    Nepq,
};

struct MixedMoePool {
    MixedMoeFamily family = MixedMoeFamily::Nint;
    NintWeight nint;
    NintWeight q8_zero;
    Mxfp4Weight mxfp4;
    Mxfp4SqWeight mxfp4_sq;
    Fp8SqWeight fp8_sq;
    NvqWeight nvq;
    NepqWeight nepq;
    mfq_tensor_backend::Tensor expert_local;
    int local_experts = 0;
};
