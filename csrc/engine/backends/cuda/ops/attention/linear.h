#pragma once

#include "engine/backends/cuda/ops/projection/projection.h"

#include <array>
#include <cstdint>

namespace tyloi::cuda {

struct LinearInputs {
  ::tyloi::DenseTensor qk;
  ::tyloi::DenseTensor value;
  ::tyloi::DenseTensor qkv;
  ::tyloi::DenseTensor gate;
  ::tyloi::DenseTensor beta;
  int64_t batch = 0;
  int64_t tokens = 0;
  bool split = false;
};

struct LinearView {
    ::tyloi::DenseTensor& convolution;
    ::tyloi::DenseTensor& recurrent;
    bool transposed;
};

struct LinearAttention {
    struct Options {
        bool shared_input = true;
        bool fused_conv = true;
    };

    int64_t key_heads = 0, value_heads = 0;
    int64_t key_dim = 0, value_dim = 0;
    int64_t conv_kernel = 0;
    double rms_norm_eps = 1e-6;
    double norm_weight_offset = 1.0;
    ::tyloi::DenseTensor attn_norm, conv_weight, conv_bias,
        dt_bias, a_log, linear_norm;
    bool split_in_proj = false;
    bool split_dense_zab = false;
    bool dense_ab_tail = false;
    bool ab_is_nint = false;
    bool dense_out_proj = false;
    bool tiled_v_heads = false;
    QuantLinearGroup in_proj;
    QuantLinearGroup qkv_proj;
    QuantLinearGroup qkvz_proj;
    QuantLinear z_proj;
    QuantLinearGroup ab_nint_proj;
    DenseLinearGroup ab_proj;
    DenseLinearGroup zab_proj;
    QuantLinear out_proj;
    ::tyloi::DenseTensor out_proj_dense;

    std::array<::tyloi::DenseTensor, 2> forward(
        const Projector& project, CudaProfiler& profiler, const Options& options,
        ::tyloi::DenseTensor input,
        LinearView state, LinearInputs* captured = nullptr) const;
    void replay(LinearView state, const LinearInputs& inputs, int64_t retained_tokens) const;

  private:
    std::array<::tyloi::DenseTensor, 2> cpu(
        const Projector& project, ::tyloi::DenseTensor input,
        LinearView state) const;
    std::array<::tyloi::DenseTensor, 2> cuda(
        const Projector& project, CudaProfiler& profiler, const Options& options,
        ::tyloi::DenseTensor input,
        LinearView state, LinearInputs* captured) const;
};

} // namespace tyloi::cuda
