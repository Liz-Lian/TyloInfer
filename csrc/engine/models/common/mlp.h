#pragma once

#include "engine/ops.h"

#include <memory>
#include <stdexcept>
#include <utility>

namespace mfq::models {

struct Mlp {
    std::shared_ptr<const weights::MatrixGroup> gate_up;
    std::shared_ptr<const weights::Matrix> down;
    ops::MlpActivation activation = ops::MlpActivation::silu_gated;
    double swiglu_limit = 0.0;

    Tensor forward(Tensor input) const {
        if (!gate_up || !down) throw std::logic_error("MLP weights are missing");
        return ops::mlp(std::move(input), *gate_up, *down, activation, swiglu_limit);
    }
};

} // namespace mfq::models
