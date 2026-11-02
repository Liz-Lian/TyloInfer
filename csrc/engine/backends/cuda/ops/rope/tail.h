#pragma once

#include "backend.h"

#include <cstdint>

namespace mfq::cuda {

class TailRope {
public:
    TailRope() = default;
    TailRope(int64_t max_position_embeddings, double rope_base, int64_t compress_ratio,
             double compress_rope_base = 0.0, int64_t original_positions = 0,
             double rope_factor = 1.0, double beta_fast = 32.0, double beta_slow = 1.0);

    mfq_tensor_backend::Tensor forward(mfq_tensor_backend::Tensor input,
                                       mfq_tensor_backend::Tensor positions,
                                       bool inverse = false) const;
    const mfq_tensor_backend::Tensor &cos() const { return cos_; }
    const mfq_tensor_backend::Tensor &sin() const { return sin_; }

private:
    mfq_tensor_backend::Tensor cos_, sin_, negative_sin_;
};

} // namespace mfq::cuda
