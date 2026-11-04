#pragma once

#include "backend.h"
#include "mfq_cuda_kv_hot_cache.h"

#include <initializer_list>

namespace mfq_selected_attention {
using ::tyloi::DenseTensor;

void values(std::initializer_list<const DenseTensor*> tensors);
DenseTensor promoted_matmul(const DenseTensor& left, const DenseTensor& right);
DenseTensor dots(const DenseTensor& query, const DenseTensor& keys);
DenseTensor dense(const DenseTensor& query, const DenseTensor& key, const DenseTensor& value,
    int64_t offset, double scale, tyloi::cuda::HotKvView hot = {});
DenseTensor sparse(const DenseTensor& query, const DenseTensor& key, const DenseTensor& value,
    const DenseTensor& indices, double scale, bool key_is_cache, tyloi::cuda::HotKvView hot = {});
DenseTensor sparse_gated(const DenseTensor& query, const DenseTensor& key, const DenseTensor& value,
    const DenseTensor& indices, const DenseTensor& gate, double scale, bool half_output, tyloi::cuda::HotKvView hot = {});
DenseTensor block_indices_to_tokens(const DenseTensor& blocks, const DenseTensor& positions,
    int64_t pool, int64_t budget);
DenseTensor gated(const DenseTensor& attended, const DenseTensor& gate, bool half_output);
DenseTensor causal_gated(const DenseTensor& query, const DenseTensor& key, const DenseTensor& value,
    const DenseTensor& positions, const DenseTensor& gate, int64_t columns, double scale,
    bool half_output, bool fused_mma_gate = true, bool fused_mma_prepare = true, tyloi::cuda::HotKvView hot = {});
DenseTensor mma_reduce_gated(const DenseTensor& partial_output, const DenseTensor& metadata,
    const DenseTensor& gate, bool half_output, int kv_heads, int total_blocks,
    int blocks_per_tile, int query_columns, int head_columns);

} // namespace mfq_selected_attention
