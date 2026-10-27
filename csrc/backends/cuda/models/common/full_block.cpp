#include "models/common/full_block.h"
#include "engine/executor/ffn.h"
#include "engine/executor/attention.h"
#include "backends/cuda/engine/executor/execution.h"
#include "backends/cuda/engine/executor/options.h"

#include <utility>

mfq_tensor_backend::Tensor Block::forward_context(CudaExecutionContext &execution,
                                                  mfq_tensor_backend::Tensor x,
                                                  const Context &context, const RopeCache &rope) {
    TYLOINFER_RUNTIME_CHECK(context.confirmed_prefix == 0 || supports_speculation(),
                      "block does not support speculative verification");
    return forward(execution, std::move(x), context.positions, context.cache_position,
                   context.sequence_lengths, rope, context.cache_positions, context.attention_mask);
}

void FullBlock::preallocate_kv() {
    if (!mfq::cuda::runtime_options::kv_vram_preallocated() || cpu_offloaded || cache.defined()) return;
    MfqCudaGuard guard(cuda_device);
    cache = KVCache(1, kv_heads, sliding ? attention_window : max_position_embeddings,
        attention_head_dim, sliding, mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, cuda_device),
        official_bf16 ? mfq_tensor_backend::kBFloat16 : mfq_tensor_backend::kFloat16, kv_quantization);
}

void FullBlock::reset(int64_t B) {
    if (cache.defined() && cache.batch_size() == B)
        return;
    cache = KVCache();
}

mfq_tensor_backend::Tensor
FullBlock::forward(CudaExecutionContext &execution, mfq_tensor_backend::Tensor x,
                   mfq_tensor_backend::Tensor pos, int64_t cache_pos,
                   const MfqOptional<mfq_tensor_backend::Tensor> &seq_len, const RopeCache &rope,
                   const MfqOptional<mfq_tensor_backend::Tensor> &cache_positions,
                   const MfqOptional<mfq_tensor_backend::Tensor> &attention_mask) {
    return forward_impl(execution, std::move(x), std::move(pos), cache_pos, seq_len, rope,
                        cache_positions, attention_mask, 0, 0);
}

mfq_tensor_backend::Tensor FullBlock::forward_context(CudaExecutionContext &execution,
                                                      mfq_tensor_backend::Tensor x,
                                                      const Context &context,
                                                      const RopeCache &rope) {
    TYLOINFER_RUNTIME_CHECK(context.confirmed_prefix == 0 || supports_speculation(),
                      "block does not support speculative verification");
    return forward_impl(execution, std::move(x), context.positions, context.cache_position,
                        context.sequence_lengths, rope, context.cache_positions,
                        context.attention_mask, context.planned_kv_length,
                        context.decode_attention_parts);
}

mfq_tensor_backend::Tensor FullBlock::forward_impl(
    CudaExecutionContext &execution, mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor pos,
    int64_t cache_pos, const MfqOptional<mfq_tensor_backend::Tensor> &seq_len,
    const RopeCache &rope, const MfqOptional<mfq_tensor_backend::Tensor> &cache_positions,
    const MfqOptional<mfq_tensor_backend::Tensor> &attention_mask, int64_t planned_kv_length,
    int64_t decode_attention_parts) {
    const int64_t B = x.size(0), T = x.size(1), H = x.size(2);
    const mfq::cuda::AttentionStep step{
        std::move(pos), seq_len, cache_positions, attention_mask,
        cache_pos, planned_kv_length, decode_attention_parts};
    auto attention = mfq::cuda::run_attention(
        execution, *this, cache, execution.attention_scratch[this], std::move(x), step, rope);
    return forward_ffn(execution, std::move(attention[0]), std::move(attention[1]), B, T, H);
}

mfq_tensor_backend::Tensor FullBlock::forward_ffn(
    CudaExecutionContext& execution, mfq_tensor_backend::Tensor residual,
    mfq_tensor_backend::Tensor output, int64_t, int64_t, int64_t) {
    return execution.ffn().full(
        ffn, ffn_norm, *this, std::move(residual), std::move(output),
        layer == execution.gemma_trace_layer ? execution.gemma_stage_trace : nullptr);
}
