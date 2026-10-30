#include "residual.h"
#include "ops/norm/rms.h"
#include "engine/backends/cuda/kernels/mfq_cuda_norm_ops.h"

#include <vector>

namespace mfq::cuda {

using Tensor = mfq_tensor_backend::Tensor;

std::array<Tensor, 2> full_norm(
    CudaProfiler& profiler, const Tensor& ffn_norm, double eps, double offset,
    bool bf16, bool fused_norm, bool fp32, Tensor residual, Tensor output) {
    const int64_t B = residual.size(0), T = residual.size(1), H = residual.size(2);
    auto pair = profiler.measure("full.attn_residual_ffn_norm", [&] {
        auto rr = residual.reshape({-1, H});
        auto oo2 = output.reshape({-1, H});
        if (!rr.is_cuda()) {
            auto summed =
                (rr.to(mfq_tensor_backend::kFloat32) + oo2.to(mfq_tensor_backend::kFloat32))
                    .to(rr.scalar_type())
                    .contiguous();
            auto normalized =
                bf16 ? qwen_rms_norm_bf16(fused_norm, summed, ffn_norm,
                                                   eps, offset)
                              : qwen_rms_norm(summed.to(mfq_tensor_backend::kFloat32),
                                              ffn_norm, eps, offset);
            return std::vector<mfq_tensor_backend::Tensor>{summed, normalized};
        }
        if (oo2.scalar_type() != rr.scalar_type()) {
            oo2 = oo2.to(rr.scalar_type()).contiguous();
        }
        if (fp32) {
            return acc_rms_norm_cuda(rr.to(mfq_tensor_backend::kFloat32),
                                     oo2.to(mfq_tensor_backend::kFloat32), ffn_norm,
                                     eps, offset);
        }
        if (rr.scalar_type() == mfq_tensor_backend::kFloat16 &&
            oo2.scalar_type() == mfq_tensor_backend::kFloat16) {
            return acc_rms_norm_f16_cuda(rr, oo2, ffn_norm, eps,
                                         offset);
        }
        if (rr.scalar_type() == mfq_tensor_backend::kBFloat16 &&
            oo2.scalar_type() == mfq_tensor_backend::kBFloat16) {
            if (bf16) {
                return acc_rms_norm_bf16_cuda(rr, oo2, ffn_norm, eps,
                                              offset);
            }
            auto sum = (rr + oo2).contiguous();
            auto norm = qwen_rms_norm(sum.to(mfq_tensor_backend::kFloat32), ffn_norm,
                                      eps, offset)
                            .to(mfq_tensor_backend::kBFloat16)
                            .contiguous();
            return std::vector<mfq_tensor_backend::Tensor>{sum, norm};
        }
        return acc_rms_norm_cuda(rr, oo2, ffn_norm, eps, offset);
    });
    return {pair[0].reshape({B, T, H}), pair[1].reshape({B, T, H})};
}

std::array<Tensor, 2> linear_norm(
    CudaProfiler& profiler, const Tensor& ffn_norm, double eps, double offset,
    bool fp32, Tensor residual, Tensor output) {
    const int64_t B = residual.size(0), T = residual.size(1), H = residual.size(2);
    if (!residual.is_cuda()) {
        auto sum = (residual.to(mfq_tensor_backend::kFloat32) +
                    output.to(mfq_tensor_backend::kFloat32))
                       .to(residual.scalar_type()).contiguous();
        auto norm = qwen_rms_norm(sum.reshape({B * T, H}).to(mfq_tensor_backend::kFloat32),
                                  ffn_norm, eps, offset).reshape({B, T, H});
        return {sum, norm};
    }
    auto pair = profiler.measure("linear.attn_residual_ffn_norm", [&] {
        auto rr = residual.reshape({-1, H});
        auto oo2 = output.reshape({-1, H});
        if (oo2.scalar_type() != rr.scalar_type()) {
          oo2 = oo2.to(rr.scalar_type()).contiguous();
        }
        if (fp32) {
          return acc_rms_norm_cuda(
              rr.to(mfq_tensor_backend::kFloat32),
              oo2.to(mfq_tensor_backend::kFloat32), ffn_norm,
              eps,
              offset);
        }
        if (rr.scalar_type() == mfq_tensor_backend::kFloat16 &&
            oo2.scalar_type() == mfq_tensor_backend::kFloat16) {
          return acc_rms_norm_f16_cuda(
              rr, oo2, ffn_norm, eps,
              offset);
        }
        return acc_rms_norm_cuda(
            rr, oo2, ffn_norm, eps,
            offset);
    });
    return {pair[0].reshape({B, T, H}), pair[1].reshape({B, T, H})};
}

Tensor full_add(CudaProfiler& profiler, bool fp32, bool bf16_acc,
                Tensor residual, Tensor ff) {
    const int64_t B = residual.size(0), T = residual.size(1), H = residual.size(2);
    return profiler.measure("full.ffn_residual", [&] {
        auto rr = residual.reshape({-1, H});
        auto ff2 = ff.reshape({-1, H});
        if (!rr.is_cuda()) {
            return (rr.to(mfq_tensor_backend::kFloat32) +
                    ff2.to(mfq_tensor_backend::kFloat32))
                .to(rr.scalar_type())
                .reshape({B, T, H})
                .contiguous();
        }
        if (fp32) {
            rr = rr.to(mfq_tensor_backend::kFloat32);
            ff2 = ff2.to(mfq_tensor_backend::kFloat32);
        } else if (ff2.scalar_type() != rr.scalar_type()) {
            ff2 = ff2.to(rr.scalar_type()).contiguous();
        }
        if (rr.scalar_type() == mfq_tensor_backend::kBFloat16 &&
            ff2.scalar_type() == mfq_tensor_backend::kBFloat16) {
            if (bf16_acc) {
                return acc_cuda(rr, ff2).reshape({B, T, H});
            }
            return (rr + ff2).contiguous().reshape({B, T, H});
        }
        return acc_cuda(rr, ff2).reshape({B, T, H});
    });
}

Tensor linear_add(CudaProfiler& profiler, bool fp32, Tensor residual, Tensor ff) {
    if (!residual.is_cuda()) {
        return (residual.to(mfq_tensor_backend::kFloat32) + ff.to(mfq_tensor_backend::kFloat32))
            .to(residual.scalar_type()).contiguous();
    }
    const int64_t B = residual.size(0), T = residual.size(1), H = residual.size(2);
    return profiler.measure("linear.ffn_residual", [&] {
        auto rr = residual.reshape({-1, H});
        auto ff2 = ff.reshape({-1, H});
        if (fp32) {
          rr = rr.to(mfq_tensor_backend::kFloat32);
          ff2 = ff2.to(mfq_tensor_backend::kFloat32);
        } else if (ff2.scalar_type() != rr.scalar_type()) {
          ff2 = ff2.to(rr.scalar_type()).contiguous();
        }
        return acc_cuda(rr, ff2).reshape({B, T, H});
    });
}

} // namespace mfq::cuda
