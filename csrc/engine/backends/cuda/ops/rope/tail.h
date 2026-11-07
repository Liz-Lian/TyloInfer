#pragma once

#include "backend.h"

#include <cstdint>

#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/ops.h"
namespace tyloi::cuda::ops {
tyloi::ops::TailRopeTable tail_rope_table(std::int64_t max_positions, double base,
    std::int64_t compress_ratio, Device device, double compress_base,
    std::int64_t original_positions, double factor, double beta_fast, double beta_slow);
tyloi::DenseTensor tail_rope(tyloi::DenseTensor input, tyloi::DenseTensor positions,
    const tyloi::ops::TailRopeTable& table, bool inverse);
}
#else
namespace tyloi::cuda {

class TailRope {
public:
    TailRope() = default;
    TailRope(int64_t max_position_embeddings, double rope_base, int64_t compress_ratio,
             double compress_rope_base = 0.0, int64_t original_positions = 0,
             double rope_factor = 1.0, double beta_fast = 32.0, double beta_slow = 1.0);

    ::tyloi::DenseTensor forward(::tyloi::DenseTensor input,
                                       ::tyloi::DenseTensor positions,
                                       bool inverse = false) const;
    const ::tyloi::DenseTensor &cos() const { return cos_; }
    const ::tyloi::DenseTensor &sin() const { return sin_; }

private:
    ::tyloi::DenseTensor cos_, sin_, negative_sin_;
};

} // namespace tyloi::cuda
#endif
