#pragma once
#include <mlx/mlx.h>
#include <vector>

namespace mfq::metal {
struct MlxGgufRotation {
    int block;
    std::vector<float> pattern;
    mlx::core::array signs;
};
}
