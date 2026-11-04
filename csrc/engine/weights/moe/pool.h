#pragma once

#include "engine/weights/mx.h"
#include "engine/weights/sq.h"
#include "engine/weights/nint.h"
#include "engine/weights/vq.h"

#include <cstdint>

struct MfePoolWeight {
    tyloi::NintTensor weight;
    ::tyloi::DenseTensor expert_local;
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
    tyloi::NintTensor nint;
    tyloi::NintTensor q8_zero;
    tyloi::Mxfp4Tensor mxfp4;
    tyloi::Mxfp4SqTensor mxfp4_sq;
    tyloi::Fp8SqTensor fp8_sq;
    tyloi::NvqTensor nvq;
    NepqWeight nepq;
    ::tyloi::DenseTensor expert_local;
    int local_experts = 0;
};
