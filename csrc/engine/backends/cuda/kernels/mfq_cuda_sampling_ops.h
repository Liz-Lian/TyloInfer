#pragma once

#include "backend.h"

#include <cstdint>
#include <vector>

void decode_graph_commit_cuda(::tyloi::DenseTensor next, ::tyloi::DenseTensor generated, ::tyloi::DenseTensor step,
                              ::tyloi::DenseTensor input, ::tyloi::DenseTensor pos, ::tyloi::DenseTensor len);
::tyloi::DenseTensor sample_greedy_cuda(::tyloi::DenseTensor logits);
::tyloi::DenseTensor sample_softmax_cuda(::tyloi::DenseTensor logits, ::tyloi::DenseTensor random, double temperature);
::tyloi::DenseTensor sample_top_k_top_p_cuda(
    ::tyloi::DenseTensor logits, ::tyloi::DenseTensor random, double temperature, int64_t top_k, double top_p);
// [B,top_k] int64 packets: high 32 bits original f32 logit, low 32 bits token ID.
::tyloi::DenseTensor sample_top_k_compact_cuda(::tyloi::DenseTensor logits,int64_t top_k);
// A stable device flag selects work at replay time without another graph
// shape. The owner must publish it before launch and retain it through replay.
::tyloi::DenseTensor sample_top_k_compact_if_cuda(
    ::tyloi::DenseTensor logits,int64_t top_k,const volatile int32_t* enabled);
void sample_token_counts_add_cuda(::tyloi::DenseTensor counts, ::tyloi::DenseTensor tokens);
::tyloi::DenseTensor sample_apply_penalties_cuda(
    ::tyloi::DenseTensor logits, ::tyloi::DenseTensor counts,
    double presence_penalty, double frequency_penalty, double repetition_penalty);
