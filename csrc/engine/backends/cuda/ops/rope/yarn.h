#pragma once

#include "engine/ops.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tyloi::cuda {

inline double yarn_attention_factor(const tyloi::ops::YarnScaling& yarn) { return yarn.enabled() ? 1+0.1*std::log(yarn.factor) : 1; }

inline double yarn_frequency(const tyloi::ops::YarnScaling& yarn, int pair, int dimension, double theta) {
    const auto frequency=std::pow(theta,-2.0*pair/dimension);
    if(!yarn.enabled())return frequency;
    if(yarn.original_context<=0 || !std::isfinite(yarn.factor) || !std::isfinite(yarn.beta_fast) ||
        !std::isfinite(yarn.beta_slow) || yarn.beta_slow<=0 || yarn.beta_fast<=yarn.beta_slow || theta<=1)
        throw std::invalid_argument("invalid Qwen YaRN scaling");
    const auto correction=[&](double rotations) {
        return dimension*std::log(yarn.original_context/(rotations*2*std::acos(-1.0)))/(2*std::log(theta));
    };
    const auto low=std::max(0.0,std::floor(correction(yarn.beta_fast)));
    auto high=std::min(double(dimension-1),std::ceil(correction(yarn.beta_slow)));
    if(low==high)high+=0.001;
    const auto ramp=std::clamp((pair-low)/(high-low),0.0,1.0);
    return frequency*(1-ramp+ramp/yarn.factor);
}

} // namespace tyloi::cuda
