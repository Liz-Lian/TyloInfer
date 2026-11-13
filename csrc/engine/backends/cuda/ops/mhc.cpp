#include "engine/ops/mhc.h"
#include "engine/backends/cuda/kernels/deepseek_v41.h"
#include "engine/backends/cuda/kernels/deepseek_v4_hc.h"
#include "backend.h"
#include <utility>

namespace tyloi::ops {
MhcResult mhc_collapse(const Tensor& residual, const Tensor& previous_pre, const Tensor& function,
        const Tensor& scale, const Tensor& base, const Tensor& norm, MhcConfig config) {
    MhcResult output{Tensor{DenseTensor{}}, Tensor{DenseTensor{}}, Tensor{DenseTensor{}}, Tensor{DenseTensor{}}};
    const auto& x=residual.dense();
    const auto& pre=previous_pre.dense();
    const auto& weights=function.dense();
    const auto& scales=scale.dense();
    const auto& bases=base.dense();
    const auto& normalization=norm.dense();
    const MfqCudaGuard guard(x.device());
    if(x.size(0)*x.size(1)<=8) {
        auto partials=deepseek_v41_hc_project_cuda(x,pre,weights);
        auto values=deepseek_v41_hc_finalize_cuda(x,pre,normalization,partials,scales,bases,
            config.iterations,config.rms_eps,config.coefficient_eps);
        output.branch.dense()=std::move(values[0]);
        output.next_pre.dense()=std::move(values[1]);
        output.post.dense()=std::move(values[2]);
        output.combination.dense()=std::move(values[3]);
        return output;
    }
    auto prepared=deepseek_v41_hc_prepare_cuda(x,pre,normalization,config.rms_eps);
    auto mixes=::tyloi::cuda::matmul(prepared.at(0),weights.transpose(0,1));
    auto coefficients=dsv4_hc_coefficients_cuda(mixes.contiguous(),scales,bases,
        config.iterations,config.coefficient_eps);
    output.branch.dense()=std::move(prepared.at(1));
    output.next_pre.dense()=std::move(coefficients.at(0));
    output.post.dense()=std::move(coefficients.at(1));
    output.combination.dense()=std::move(coefficients.at(2));
    return output;
}
Tensor mhc_input(const Tensor& residual, const Tensor& previous_pre, const Tensor& norm, double rms_eps) {
    Tensor result{DenseTensor{}};
    const MfqCudaGuard guard(residual.device());
    result.dense()=deepseek_v41_hc_input_cuda(residual.dense(),previous_pre.dense(),norm.dense(),rms_eps);
    return result;
}
Tensor mhc_expand(const Tensor& branch, const Tensor& residual, const MhcResult& mix) {
    Tensor result{DenseTensor{}};
    const MfqCudaGuard guard(branch.device());
    result.dense()=dsv4_hc_post_cuda(branch.dense().contiguous(),residual.dense().contiguous(),
        mix.post.dense().contiguous(),mix.combination.dense().contiguous());
    return result;
}
} // namespace tyloi::ops
