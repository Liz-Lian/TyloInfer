#pragma once

#include "engine/tensor.h"
#include <cstdint>

namespace tyloi::cuda { class Program; }
namespace tyloi::cuda::diagnostics {
/** Compare ordinary and split-K attention on a prompt and one decode row. */
int compare_decode_splitk(Program& program, Tensor ids);
/** Compare teacher-forced decode logits with MMA disabled and enabled. */
int compare_mma_decode(Program& program, Tensor ids, int steps, std::int64_t planned_length);
/** Report repeat prefill timing and optional block and stage differences. */
int repeat_prefill(Program& program, Tensor ids, int count, bool trace);
/** Compare prefill logits with attention MMA and BF16 flash disabled and enabled. */
int compare_mma_attention(Program& program, Tensor ids);
}
