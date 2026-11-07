#include "residual.h"
#include "engine/backends/cuda/ops/norm/rms.h"
#include "engine/backends/cuda/kernels/mfq_cuda_norm_ops.h"

#include <vector>

namespace tyloi::cuda {


std::array<DenseTensor, 2> full_norm(
    CudaProfiler& profiler,
#ifdef TYLOINFER_NATIVE_CUDA
    const ::tyloi::Tensor& norm,
#else
    const DenseTensor& ffn_norm,
#endif
    double eps, double offset,
    bool bf16, bool fused_norm, bool fp32, DenseTensor residual, DenseTensor output) {
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& ffn_norm = dynamic_cast<const DenseTensor&>(norm);
#endif
    const int64_t B = residual.size(0), T = residual.size(1), H = residual.size(2);
    auto pair = profiler.measure("full.attn_residual_ffn_norm", [&] {
        auto rr = residual.reshape({-1, H});
        auto oo2 = output.reshape({-1, H});
        if (!rr.is_cuda()) {
            auto summed =
                (rr.to(::tyloi::kFloat32) + oo2.to(::tyloi::kFloat32))
                    .to(rr.scalar_type())
                    .contiguous();
            auto normalized =
                bf16 ? qwen_rms_norm_bf16(fused_norm, summed, ffn_norm,
                                                   eps, offset)
                              : qwen_rms_norm(summed.to(::tyloi::kFloat32),
                                              ffn_norm, eps, offset);
            return std::vector<::tyloi::DenseTensor>{summed, normalized};
        }
        if (oo2.scalar_type() != rr.scalar_type()) {
            oo2 = oo2.to(rr.scalar_type()).contiguous();
        }
        if (fp32) {
            return acc_rms_norm_cuda(rr.to(::tyloi::kFloat32),
                                     oo2.to(::tyloi::kFloat32), ffn_norm,
                                     eps, offset);
        }
        if (rr.scalar_type() == ::tyloi::kFloat16 &&
            oo2.scalar_type() == ::tyloi::kFloat16) {
            return acc_rms_norm_f16_cuda(rr, oo2, ffn_norm, eps,
                                         offset);
        }
        if (rr.scalar_type() == ::tyloi::kBFloat16 &&
            oo2.scalar_type() == ::tyloi::kBFloat16) {
            if (bf16) {
                return acc_rms_norm_bf16_cuda(rr, oo2, ffn_norm, eps,
                                              offset);
            }
            auto sum = (rr + oo2).contiguous();
            auto norm = qwen_rms_norm(sum.to(::tyloi::kFloat32), ffn_norm,
                                      eps, offset)
                            .to(::tyloi::kBFloat16)
                            .contiguous();
            return std::vector<::tyloi::DenseTensor>{sum, norm};
        }
        return acc_rms_norm_cuda(rr, oo2, ffn_norm, eps, offset);
    });
    return {pair[0].reshape({B, T, H}), pair[1].reshape({B, T, H})};
}

std::array<DenseTensor, 2> linear_norm(
    CudaProfiler& profiler,
#ifdef TYLOINFER_NATIVE_CUDA
    const ::tyloi::Tensor& norm,
#else
    const DenseTensor& ffn_norm,
#endif
    double eps, double offset,
    bool fp32, DenseTensor residual, DenseTensor output) {
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& ffn_norm = dynamic_cast<const DenseTensor&>(norm);
#endif
    const int64_t B = residual.size(0), T = residual.size(1), H = residual.size(2);
    if (!residual.is_cuda()) {
        auto sum = (residual.to(::tyloi::kFloat32) +
                    output.to(::tyloi::kFloat32))
                       .to(residual.scalar_type()).contiguous();
        auto norm = qwen_rms_norm(sum.reshape({B * T, H}).to(::tyloi::kFloat32),
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
              rr.to(::tyloi::kFloat32),
              oo2.to(::tyloi::kFloat32), ffn_norm,
              eps,
              offset);
        }
        if (rr.scalar_type() == ::tyloi::kFloat16 &&
            oo2.scalar_type() == ::tyloi::kFloat16) {
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

DenseTensor full_add(CudaProfiler& profiler, bool fp32, bool bf16_acc,
                DenseTensor residual, DenseTensor ff) {
    const int64_t B = residual.size(0), T = residual.size(1), H = residual.size(2);
    return profiler.measure("full.ffn_residual", [&] {
        auto rr = residual.reshape({-1, H});
        auto ff2 = ff.reshape({-1, H});
        if (!rr.is_cuda()) {
            return (rr.to(::tyloi::kFloat32) +
                    ff2.to(::tyloi::kFloat32))
                .to(rr.scalar_type())
                .reshape({B, T, H})
                .contiguous();
        }
        if (fp32) {
            rr = rr.to(::tyloi::kFloat32);
            ff2 = ff2.to(::tyloi::kFloat32);
        } else if (ff2.scalar_type() != rr.scalar_type()) {
            ff2 = ff2.to(rr.scalar_type()).contiguous();
        }
        if (rr.scalar_type() == ::tyloi::kBFloat16 &&
            ff2.scalar_type() == ::tyloi::kBFloat16) {
            if (bf16_acc) {
                return acc_cuda(rr, ff2).reshape({B, T, H});
            }
            return (rr + ff2).contiguous().reshape({B, T, H});
        }
        return acc_cuda(rr, ff2).reshape({B, T, H});
    });
}

DenseTensor linear_add(CudaProfiler& profiler, bool fp32, DenseTensor residual, DenseTensor ff) {
    if (!residual.is_cuda()) {
        return (residual.to(::tyloi::kFloat32) + ff.to(::tyloi::kFloat32))
            .to(residual.scalar_type()).contiguous();
    }
    const int64_t B = residual.size(0), T = residual.size(1), H = residual.size(2);
    return profiler.measure("linear.ffn_residual", [&] {
        auto rr = residual.reshape({-1, H});
        auto ff2 = ff.reshape({-1, H});
        if (fp32) {
          rr = rr.to(::tyloi::kFloat32);
          ff2 = ff2.to(::tyloi::kFloat32);
        } else if (ff2.scalar_type() != rr.scalar_type()) {
          ff2 = ff2.to(rr.scalar_type()).contiguous();
        }
        return acc_cuda(rr, ff2).reshape({B, T, H});
    });
}

} // namespace tyloi::cuda
