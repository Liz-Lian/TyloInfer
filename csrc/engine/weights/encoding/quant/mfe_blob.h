#pragma once

#include "nint_blob.h"
#include "../mx_blob.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mfq::mfe {

struct Pool {
    std::vector<int32_t> expert_ids;
    std::string dtype;
    std::vector<uint8_t> payload;
    std::vector<uint8_t> runtime_payload;
    mfq::nint::Data weight;
    mfq::nint8::Data q8_zero;
    mfq::mxfp4::Data mxfp4;
};

struct Data {
    int n_experts = 0;
    int out_per_expert = 0;
    int neuron_len = 0;
    std::vector<Pool> pools;
};

Data unpack(const std::vector<uint8_t>& blob);

} // namespace mfq::mfe
