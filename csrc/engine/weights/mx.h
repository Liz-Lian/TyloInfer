#pragma once

#include "backend.h"

#include <cstdint>

struct Mxfp8Weight {
    mfq_tensor_backend::Tensor values;
    mfq_tensor_backend::Tensor scales;
    int64_t out = 0;
    int64_t neuron_len = 0;
};

struct Mxfp4Weight {
    mfq_tensor_backend::Tensor values;
    mfq_tensor_backend::Tensor scales;
    int64_t out = 0;
    int64_t neuron_len = 0;
};
