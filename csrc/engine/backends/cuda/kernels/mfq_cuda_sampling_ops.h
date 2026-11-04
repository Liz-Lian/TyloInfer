#pragma once

#include "backend.h"

#include <cstdint>
#include <vector>

void decode_graph_commit_cuda(::tyloi::Tensor next, ::tyloi::Tensor generated, ::tyloi::Tensor step,
                              ::tyloi::Tensor input, ::tyloi::Tensor pos, ::tyloi::Tensor len);
::tyloi::Tensor sample_greedy_cuda(::tyloi::Tensor logits);
::tyloi::Tensor sample_softmax_cuda(::tyloi::Tensor logits, ::tyloi::Tensor random, double temperature);
::tyloi::Tensor sample_top_k_top_p_cuda(
    ::tyloi::Tensor logits, ::tyloi::Tensor random, double temperature, int64_t top_k, double top_p);
// [B,top_k] int64 packets: high 32 bits original f32 logit, low 32 bits token ID.
::tyloi::Tensor sample_top_k_compact_cuda(::tyloi::Tensor logits,int64_t top_k);
// A stable device flag selects work at replay time without another graph
// shape. The owner must publish it before launch and retain it through replay.
::tyloi::Tensor sample_top_k_compact_if_cuda(
    ::tyloi::Tensor logits,int64_t top_k,const volatile int32_t* enabled);
void sample_token_counts_add_cuda(::tyloi::Tensor counts, ::tyloi::Tensor tokens);
::tyloi::Tensor sample_apply_penalties_cuda(
    ::tyloi::Tensor logits, ::tyloi::Tensor counts,
    double presence_penalty, double frequency_penalty, double repetition_penalty);
