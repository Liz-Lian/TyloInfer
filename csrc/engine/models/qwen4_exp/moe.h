#pragma once

#include "engine/models/qwen4_exp/config.h"
#include "engine/tensor.h"

#include <array>
#include <vector>

namespace tyloi::cuda::qwen4_exp {

#ifndef TYLOINFER_NATIVE_CUDA
class Moe {
  public:
    tyloi::models::qwen4_exp::Config config{};
    std::vector<::tyloi::Tensor> gate_up;
    ::tyloi::Tensor down, router, shared_gate;
    std::array<::tyloi::Tensor, 3> shared{};
    bool tq = false;

    Moe() = default;
    Moe(std::vector<::tyloi::Tensor> gate_up, ::tyloi::Tensor down,
        ::tyloi::Tensor router, ::tyloi::Tensor shared_gate, ::tyloi::Tensor gate,
        ::tyloi::Tensor up, ::tyloi::Tensor shared_down,
        const tyloi::models::qwen4_exp::Config&);
    ::tyloi::DenseTensor forward(const ::tyloi::DenseTensor&) const;
};
#endif

} // namespace tyloi::cuda::qwen4_exp
