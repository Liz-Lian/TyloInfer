#include "engine/ops/engram.h"
#include "engine/ops.h"
#include "engine/backends/cuda/kernels/deepseek_v41.h"
#include "backend.h"
#include <utility>

namespace tyloi::ops {
EngramWeights engram_weights(const Tensor& projection, const Tensor& query, const Tensor& key) {
    Tensor product{DenseTensor{}};
    const MfqCudaGuard guard(query.device());
    auto q=query.dense().to(kFloat32).contiguous();
    auto k=key.dense().to(kFloat32).contiguous();
    product.dense()=(q*k).contiguous();
    return {projection,std::move(product)};
}
Tensor engram_select(const Tensor& hidden, const Tensor& text_rows) {
    Tensor result{DenseTensor{}};
    const auto& x=hidden.dense();
    const MfqCudaGuard guard(x.device());
    if (!text_rows.defined() || !text_rows.numel()) {
        result.dense()=x;
        return result;
    }
    auto flat=x.reshape({1,-1,x.size(2),x.size(3)});
    result.dense()=flat.index_select(1,text_rows.dense()).contiguous();
    return result;
}
Tensor engram(const Tensor& hidden, const Tensor& selected, const Tensor& gathered,
              const EngramWeights& weights, double eps, const Tensor& text_rows) {
    Tensor result{DenseTensor{}}, projected{DenseTensor{}}, gated{DenseTensor{}};
    const auto& x=hidden.dense();
    const MfqCudaGuard guard(x.device());
    if (text_rows.defined() && !text_rows.numel()) {
        result.dense()=x;
        return result;
    }
    projected.dense()=ops::measure("deepseek_v41.engram.projection", [&] {
        return ops::project(gathered,weights.projection).dense();
    });
    gated.dense()=ops::measure("deepseek_v41.engram.gate_add", [&] {
        return deepseek_v41_engram_gate_add_cuda(selected.dense(),projected.dense().contiguous(),
            weights.query_key.dense(),eps);
    });
    if (!text_rows.defined()) {
        result.dense()=gated.dense();
        return result;
    }
    auto flat=x.reshape({1,-1,x.size(2),x.size(3)});
    auto scattered=flat.clone();
    scattered.index_copy_(1,text_rows.dense(),gated.dense());
    result.dense()=scattered.reshape(x.sizes());
    return result;
}
} // namespace tyloi::ops
