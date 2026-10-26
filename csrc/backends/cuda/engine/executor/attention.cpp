#include "engine/executor/attention.h"
#include "engine/executor/execution.h"
#include "engine/state/kv_cache.h"

#include <algorithm>
#include <utility>

namespace mfq::cuda {

std::array<mfq_tensor_backend::Tensor, 2> run_attention(
        CudaExecutionContext& execution, const FullAttention& attention,
        KVCache& cache, AttentionScratch& scratch, mfq_tensor_backend::Tensor input,
        const AttentionStep& step, const RopeCache& rope) {
    const auto kl_capacity = execution.kl_mmq.kv_cache_capacity;
    const int64_t capacity = attention.sliding ? attention.attention_window
        : (kl_capacity > 0 ? std::max<int64_t>(step.cache_position + input.size(1), kl_capacity)
                           : attention.max_position_embeddings);
    if (!cache.defined()) {
        cache = KVCache(input.size(0), attention.kv_heads, capacity,
            attention.attention_head_dim, attention.sliding, input.device(),
            attention.official_bf16 ? mfq_tensor_backend::kBFloat16 : mfq_tensor_backend::kFloat16,
            attention.kv_quantization);
    }
    cache.quantization = attention.kv_quantization;
    return attention.forward(
        execution.projector(), execution.profiler, {
            .kernels = {
                .bf16_flash128 = execution.config.tasks.attention.kernels.bf16_flash128,
                .flash128_casts = execution.config.tasks.attention.kernels.flash128_casts,
                .bf16_gqa_decode = execution.config.tasks.attention.kernels.bf16_gqa_decode,
                .decode_aten = execution.config.tasks.attention.kernels.decode_aten,
                .mma_prefill = execution.config.tasks.attention.kernels.mma_prefill,
                .mma_decode = execution.config.tasks.attention.kernels.mma_decode,
                .split_k = execution.config.tasks.attention.kernels.split_k},
            .fused_norm = execution.config.tasks.attention.fused_norm,
            .fused_rope = execution.config.tasks.attention.fused_rope,
            .fused_qk_rope_kv = execution.config.tasks.attention.fused_qk_rope_kv,
            .fused_rope_kv = execution.config.tasks.attention.fused_rope_kv,
            .write_aten = execution.config.tasks.attention.write_aten},
        std::move(input), step, cache.view(), scratch, rope,
        attention.layer == execution.gemma_trace_layer ? execution.gemma_stage_trace : nullptr);
}

} // namespace mfq::cuda
