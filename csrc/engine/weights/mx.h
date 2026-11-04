#pragma once

#include "backend.h"

#include <cstdint>

struct Mxfp8Weight {
    ::tyloi::Tensor values;
    ::tyloi::Tensor scales;
    int64_t out = 0;
    int64_t neuron_len = 0;
};

struct Mxfp4Weight {
    ::tyloi::Tensor values;
    ::tyloi::Tensor scales;
    int64_t out = 0;
    int64_t neuron_len = 0;
};
