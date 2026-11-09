#pragma once

#include "engine/ops.h"

#include <memory>
#include <vector>

namespace tyloi::models {

struct Mlp {
    std::vector<Tensor> gate_up;
    Tensor down;
    ops::MlpActivation activation = ops::MlpActivation::silu_gated;
    double swiglu_limit = 0.0;
    std::unique_ptr<Mlp> important_neurons;
};

Tensor mlp(const Mlp& weights, const Tensor& input);

} // namespace tyloi::models
