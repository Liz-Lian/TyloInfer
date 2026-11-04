#pragma once

#include "nint_blob.h"
#include "../mx_blob.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace tyloi::mfe {

struct Pool {
    std::vector<int32_t> expert_ids;
    std::string dtype;
    std::vector<uint8_t> payload;
    std::vector<uint8_t> runtime_payload;
    tyloi::nint::Data weight;
    tyloi::nint8::Data q8_zero;
    tyloi::mxfp4::Data mxfp4;
};

struct Data {
    int n_experts = 0;
    int out_per_expert = 0;
    int neuron_len = 0;
    std::vector<Pool> pools;
};

std::vector<std::byte> header(
    std::uint32_t experts,
    std::uint32_t output,
    std::uint32_t input,
    std::uint32_t pools);

std::vector<std::byte> expert_prefix(
    std::int32_t expert,
    std::string_view dtype,
    std::uint64_t nbytes);

Data unpack(const std::vector<uint8_t>& blob);

} // namespace tyloi::mfe
