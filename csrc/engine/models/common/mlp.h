#pragma once

#include "engine/ops.h"

#include <memory>
#include <stdexcept>
#include <utility>

namespace tyloi::models {

struct Mlp {
    std::vector<std::shared_ptr<const Tensor>> gate_up;
    std::shared_ptr<const ::tyloi::Tensor> down;
    ops::MlpActivation activation = ops::MlpActivation::silu_gated;
    double swiglu_limit = 0.0;

    std::shared_ptr<Tensor> forward(const Tensor& input) const {
        if (gate_up.empty() || !down) throw std::logic_error("MLP weights are missing");
        return ops::mlp(input, gate_up, *down, activation, swiglu_limit);
    }
};

} // namespace tyloi::models
