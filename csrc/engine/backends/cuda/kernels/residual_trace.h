#pragma once

#include <cuda_runtime_api.h>
#include <array>
#include <cstdint>

/** Timestamp destinations and actual phase order for a fused residual launch. */
struct ResidualTraceData {
    enum Phase { begin, norm, quantize_input, down, dense_projection, up_mix, end, count };
    struct Point { std::uint64_t* timestamp = nullptr; cudaEvent_t event = nullptr; void* owner = nullptr; };
    std::array<Point, count> points{};
    std::array<Phase, 16> sequence{};
    unsigned used = 0;
};
