#include "engine/backends/cuda/executor/execution.h"
#include "engine/weights/loader.h"
#include "engine/models/common/causal_model_ops.h"

namespace tyloi::cuda {

bool CausalResources::adapter_uses_common_rope() const noexcept { return false; }

void CausalResources::adapter_configure_rope(RopeCache &, ::tyloi::Device) const {}

void CausalResources::adapter_load_final_state(const tyloi::cuda::weight_loader::Loader& weights,
                                               ::tyloi::DenseTensor &output_norm) {
    output_norm = weights.tensor("model.output_norm.weight");
}

void CausalResources::adapter_prepare_blocks(const tyloi::cuda::weight_loader::Loader& weights) {}

::tyloi::DenseTensor CausalResources::adapter_embed(::tyloi::DenseTensor output) const {
    return output;
}

::tyloi::DenseTensor
CausalResources::adapter_raw_hidden(const ::tyloi::DenseTensor &hidden,
                                    const ::tyloi::DenseTensor &) const {
    return hidden;
}

::tyloi::DenseTensor
CausalResources::adapter_logits(const ::tyloi::Tensor &lm_head,
                                ::tyloi::DenseTensor hidden) const {
    auto logits = execution->executor().profiler().measure(
        "model.lm_head", [&]() { return execution->executor().projector(execution->executor().mode().ffn, model_parallel_primary_device(execution->config)).run(lm_head, hidden); });
    return logits;
}

} // namespace tyloi::cuda
