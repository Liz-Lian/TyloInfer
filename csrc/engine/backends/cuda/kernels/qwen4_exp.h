#pragma once
#include "mfq_cuda_kv_hot_cache.h"

#include "backend.h"

#include <optional>
#include <vector>

namespace mfq_qwen4_exp {
using ::tyloi::DenseTensor;
void fill_decode_positions(const DenseTensor& cache_positions, const DenseTensor& positions,
    int64_t start, int64_t delta=0);

DenseTensor grouped_rms_norm(const DenseTensor&, const DenseTensor&, int64_t group_size, double eps);
DenseTensor gated_residual_bottleneck(const DenseTensor&, int64_t streams);
DenseTensor gated_residual_injection(const DenseTensor&, int64_t streams);
DenseTensor gated_residual_mix(const DenseTensor& projection, const DenseTensor& normalized, int64_t streams);
std::vector<DenseTensor> gated_residual_pre(
    const DenseTensor&, const DenseTensor&, const DenseTensor&, const DenseTensor&,
    const std::optional<DenseTensor>& injection, int64_t hidden_size,
    int64_t hc_count, double eps);
DenseTensor gated_residual_post(
    const DenseTensor&, const DenseTensor&, const DenseTensor&, int64_t hc_count);
std::vector<DenseTensor> gated_residual_post_norm(
    const DenseTensor& branch, const DenseTensor& residual, const DenseTensor& injection,
    const DenseTensor& norm, int64_t hc_count, double eps);
DenseTensor block_scores(const DenseTensor&, const DenseTensor&);
DenseTensor live_block_scores(const DenseTensor& query, const DenseTensor& pooled,
    const DenseTensor& cache_positions, int64_t pool);
std::vector<DenseTensor> live_index_pool(const DenseTensor& index, const DenseTensor& positions,
    const DenseTensor& cache_positions, int64_t pool, int64_t maximum);
DenseTensor live_topk_blocks(const DenseTensor& scores, const DenseTensor& cache_positions,
    int64_t pool, int64_t count);
std::vector<DenseTensor> ple_dilated_conv_silu(
    const DenseTensor&, const DenseTensor&, const std::optional<DenseTensor>& state,
    int64_t dilation);
DenseTensor dense_gqa_attention(
    const DenseTensor&, const DenseTensor&, const DenseTensor&, int64_t offset, tyloi::cuda::HotKvView hot = {});
DenseTensor sparse_gqa_attention(
    const DenseTensor&, const DenseTensor&, const DenseTensor&, const DenseTensor& indices, tyloi::cuda::HotKvView hot = {});
DenseTensor sparse_gqa_attention_gate(const DenseTensor&, const DenseTensor&, const DenseTensor&,
    const DenseTensor& indices, const DenseTensor& gate, bool half_output, tyloi::cuda::HotKvView hot = {});
DenseTensor qsa_selected_tokens(const DenseTensor& blocks, const DenseTensor& positions,
    int64_t pool, int64_t budget);
DenseTensor attention_gate(const DenseTensor& attended, const DenseTensor& gate, bool half_output);
DenseTensor causal_gqa_attention_gate(const DenseTensor&, const DenseTensor&, const DenseTensor&,
    const DenseTensor& positions, const DenseTensor& gate, int64_t columns, bool half_output, tyloi::cuda::HotKvView hot = {});

} // namespace mfq_qwen4_exp
