#pragma once

#include "backend.h"

#include <cstdint>

#include "engine/ops.h"
namespace tyloi::cuda::ops {
tyloi::ops::TailRopeTable tail_rope_table(std::int64_t max_positions, double base,
    std::int64_t compress_ratio, Device device, double compress_base,
    std::int64_t original_positions, double factor, double beta_fast, double beta_slow);
tyloi::DenseTensor tail_rope(tyloi::DenseTensor input, tyloi::DenseTensor positions,
    const tyloi::ops::TailRopeTable& table, bool inverse);
}
