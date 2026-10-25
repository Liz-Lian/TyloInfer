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
                .bf16_flash128 = execution.config.minicpm_bf16_flash128,
                .flash128_casts = execution.config.minicpm_flash128_specialized_casts,
                .bf16_gqa_decode = execution.config.minicpm_bf16_gqa_decode,
                .decode_aten = execution.config.attention_decode_aten,
                .mma_prefill = execution.config.mma_attention,
                .mma_decode = execution.config.mma_attention_decode,
                .split_k = execution.config.attention_decode_split_k},
            .fused_norm = execution.config.minicpm_fused_bf16_rmsnorm,
            .fused_rope = execution.config.minicpm_fused_bf16_rope,
            .fused_qk_rope_kv = execution.config.minicpm_fused_qk_norm_rope_kv,
            .fused_rope_kv = execution.config.minicpm_fused_rope_kv,
            .write_aten = execution.config.kv_cache_write_aten},
        std::move(input), step, cache.view(), scratch, rope,
        attention.layer == execution.gemma_trace_layer ? execution.gemma_stage_trace : nullptr);
}

} // namespace mfq::cuda
