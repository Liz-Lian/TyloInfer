#include "engine/backends/cuda/executor/execution.h"
#include "engine/backends/cuda/ops/norm/rms.h"
#include "engine/backends/cuda/kernels/mfq_cuda_sampling_ops.h"
#include "engine/weights/loader.h"
#include "engine/models/common/causal_model_ops.h"

namespace tyloi::cuda {

namespace {

::tyloi::DenseTensor normalize_hidden(::tyloi::DenseTensor hidden,
                                            const ::tyloi::DenseTensor &output_norm,
                                            const tyloi::models::CausalLmMetadata &metadata,
                                            int64_t batch, int64_t tokens, CudaProfiler &profiler) {
    return profiler.measure("model.output_norm", [&]() {
        auto flat = hidden.reshape({batch * tokens, metadata.hidden_size});
        return qwen_rms_norm(flat.to(::tyloi::kFloat32), output_norm,
                             metadata.rms_norm_eps, metadata.norm_weight_offset)
            .reshape({batch, tokens, metadata.hidden_size});
    });
}

} // namespace

void CausalResources::adapter_validate_load_options() const {}

bool CausalResources::adapter_uses_common_rope() const noexcept { return false; }

void CausalResources::adapter_configure_rope(RopeCache &, ::tyloi::Device) const {}

void CausalResources::adapter_load_final_state(const tyloi::cuda::weight_loader::Loader& weights,
                                               ::tyloi::DenseTensor &output_norm) {
    output_norm = weights.tensor("model.output_norm.weight");
}

void CausalResources::adapter_prepare_blocks(const tyloi::cuda::weight_loader::Loader& weights) {}

bool CausalResources::adapter_supports_dense_cpu_offload() const noexcept { return false; }

::tyloi::DenseTensor CausalResources::adapter_embed(::tyloi::DenseTensor output) const {
    return output;
}

void CausalResources::adapter_reset(int64_t) {}

bool CausalResources::adapter_requires_batch_reset(int64_t) const noexcept { return false; }

::tyloi::DenseTensor
CausalResources::adapter_prepare_hidden(::tyloi::DenseTensor hidden, int64_t, int64_t) const {
    return hidden;
}

void CausalResources::adapter_begin_forward(bool) {}

::tyloi::DenseTensor
CausalResources::adapter_block_positions(const ::tyloi::DenseTensor &,
                                         const ::tyloi::DenseTensor &local_positions,
                                         int) const {
    return local_positions;
}

void CausalResources::adapter_finish_forward(const ::tyloi::DenseTensor &, int64_t, int64_t) {
}

::tyloi::DenseTensor
CausalResources::adapter_finalize_hidden(::tyloi::DenseTensor hidden,
                                         const ::tyloi::DenseTensor &output_norm,
                                         int64_t batch, int64_t tokens) const {
    return normalize_hidden(std::move(hidden), output_norm, metadata, batch, tokens,
                            execution->executor().profiler());
}

::tyloi::DenseTensor
CausalResources::adapter_raw_hidden(const ::tyloi::DenseTensor &hidden,
                                    const ::tyloi::DenseTensor &) const {
    return hidden;
}

::tyloi::DenseTensor
CausalResources::adapter_logits(const QuantLinear &lm_head,
                                ::tyloi::DenseTensor hidden) const {
    auto logits = execution->executor().profiler().measure(
        "model.lm_head", [&]() { return execution->executor().projector(execution->executor().mode().ffn, model_parallel_primary_device(execution->config)).run(lm_head, hidden); });
    return logits;
}

::tyloi::DenseTensor
CausalResources::adapter_last_logits(const QuantLinear &lm_head,
                                     ::tyloi::DenseTensor hidden) const {
    return execution->executor().projector(execution->executor().mode().ffn, model_parallel_primary_device(execution->config)).run(lm_head, hidden.to(::tyloi::kFloat16).contiguous());
}

::tyloi::DenseTensor
CausalResources::adapter_next_token(const QuantLinear &lm_head,
                                    ::tyloi::DenseTensor hidden) const {
    auto input = hidden.to(::tyloi::kFloat16).contiguous();
    auto logits = execution->executor().profiler().measure("model.lm_head",
                                              [&]() { return execution->executor().projector(execution->executor().mode().ffn, model_parallel_primary_device(execution->config)).run(lm_head, input); });
    return sample_greedy_cuda(logits.contiguous().view({input.size(0), -1}));
}

bool CausalResources::adapter_supports_prepared_prompt() const noexcept { return false; }

void CausalResources::adapter_begin_speculative() {}
void CausalResources::adapter_commit_speculative() {}
void CausalResources::adapter_rollback_speculative(int64_t) {}

} // namespace tyloi::cuda
