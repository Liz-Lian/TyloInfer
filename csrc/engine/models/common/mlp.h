#pragma once

#include "engine/ops.h"

#include <memory>
#include <stdexcept>
#include <utility>

namespace tyloi::models {

struct Mlp {
    std::shared_ptr<const QuantLinearGroup> gate_up;
    std::shared_ptr<const ::tyloi::Tensor> down;
    ops::MlpActivation activation = ops::MlpActivation::silu_gated;
    double swiglu_limit = 0.0;

    DenseTensor forward(DenseTensor input) const {
        if (!gate_up || !down) throw std::logic_error("MLP weights are missing");
        return ops::mlp(std::move(input), *gate_up, *down, activation, swiglu_limit);
    }
};

} // namespace tyloi::models
