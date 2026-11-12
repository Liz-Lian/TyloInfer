#pragma once
#include <torch/extension.h>
#include "sparse_mma_raw.h"
#include <initializer_list>

namespace mfq_selected_jit {
using Tensor = torch::Tensor;
using Hot = tyloi::cuda::sparse_mma_raw::Hot;
/** Validate colocated CUDA floating-point tensors. */
void values(std::initializer_list<const Tensor*> tensors);
/** Multiply tensors with the original mixed-dtype promotion. */
Tensor promoted_matmul(const Tensor& left, const Tensor& right);
/** Compute rectified selected-attention index scores. */
Tensor dots(const Tensor& query, const Tensor& keys);
/** Compute causal dense grouped attention. */
Tensor dense(const Tensor& query, const Tensor& key, const Tensor& value,
    int64_t offset, double scale, Hot hot = {});
/** Compute selected attention, optionally treating values as a key view. */
Tensor sparse(const Tensor& query, const Tensor& key, const Tensor& value,
    const Tensor& indices, double scale, bool key_is_cache, Hot hot = {});
/** Compute selected attention and apply its gate. */
Tensor sparse_gated(const Tensor& query, const Tensor& key, const Tensor& value,
    const Tensor& indices, const Tensor& gate, double scale, bool half_output, Hot hot = {});
/** Expand selected blocks and their live tail into token indices. */
Tensor block_indices_to_tokens(const Tensor& blocks, const Tensor& positions,
    int64_t pool, int64_t budget);
/** Apply an attention gate using the Torch sigmoid policy. */
Tensor gated(const Tensor& attended, const Tensor& gate, bool half_output);
/** Compute causal attention with its optional fused gated MMA path. */
Tensor causal_gated(const Tensor& query, const Tensor& key, const Tensor& value,
    const Tensor& positions, const Tensor& gate, int64_t columns, double scale,
    bool half_output, bool fused_mma_gate = true, bool fused_mma_prepare = true, Hot hot = {});
}
