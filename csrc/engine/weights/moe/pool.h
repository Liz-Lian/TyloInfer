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

    template <class Visitor>
    void visit_tensors(Visitor visit) const {
        for (const auto* buffer : weight.buffers())
            if (buffer && buffer->defined()) visit(*buffer);
        if (expert_local.defined()) visit(expert_local);
    }
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

    // The encoded representation this pool uses. NEPQ keeps a plain aggregate,
    // so it has no Tensor payload and reports its buffers separately.
    const ::tyloi::Tensor* payload() const {
        switch (family) {
            case MixedMoeFamily::Nint: return &nint;
            case MixedMoeFamily::Nint8Zero: return &q8_zero;
            case MixedMoeFamily::Mxfp4: return &mxfp4;
            case MixedMoeFamily::Mxfp4Sq: return &mxfp4_sq;
            case MixedMoeFamily::Fp8Sq: return &fp8_sq;
            case MixedMoeFamily::Nvq: return &nvq;
            case MixedMoeFamily::Nepq: return nullptr;
        }
        return nullptr;
    }

    // Every buffer the pool requires: payload buffers plus the expert map.
    template <class Visitor>
    void visit_tensors(Visitor visit) const {
        if (const auto* weight = payload())
            for (const auto* buffer : weight->buffers())
                if (buffer && buffer->defined()) visit(*buffer);
        else for (const auto* buffer : nepq_buffers(nepq))
            if (buffer && buffer->defined()) visit(*buffer);
        if (expert_local.defined()) visit(expert_local);
    }
};
