#pragma once

#include "engine/ops.h"

#include <memory>
#include <stdexcept>
#include <utility>

namespace tyloi::models {

struct Mlp {
    std::shared_ptr<const weights::Weights> gate_up;
    std::shared_ptr<const weights::Weights> down;
    ops::MlpActivation activation = ops::MlpActivation::silu_gated;
    double swiglu_limit = 0.0;

    Tensor forward(Tensor input) const {
        if (!gate_up || !down) throw std::logic_error("MLP weights are missing");
        return ops::mlp(std::move(input), *gate_up, *down, activation, swiglu_limit);
    }
};

} // namespace tyloi::models
