#pragma once
#include "engine/tensor.h"

namespace tyloi::ops {
/** Immutable embedding projection and prepared gating weights. */
struct EngramWeights { Tensor projection, query_key; };
/** Prepare the FP32 query/key product used by Engram gating. */
EngramWeights engram_weights(const Tensor& projection, const Tensor& query, const Tensor& key);
/** Select text rows before the corresponding embedding lookup. */
Tensor engram_select(const Tensor& hidden, const Tensor& text_rows = {});
/** Project gathered rows, apply gating and scatter selected results into the full hidden state. */
Tensor engram(const Tensor& hidden, const Tensor& selected, const Tensor& gathered,
              const EngramWeights& weights, double eps, const Tensor& text_rows = {});
} // namespace tyloi::ops
