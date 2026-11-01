#pragma once

#include "policy.h"

#include <cstdint>

namespace mfq::engine::mtp {

struct State {
    GenerationStats stats;
    PolicyState policy;
    std::uint64_t cycles = 0;
    std::uint64_t accepted = 0;
    std::uint64_t rejected = 0;
};

} // namespace mfq::engine::mtp
