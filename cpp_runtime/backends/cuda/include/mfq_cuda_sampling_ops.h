#pragma once

#include "mfq_tensor_backend.h"

#include <cstdint>
#include <vector>

void decode_graph_commit_cuda(mfq_tensor_backend::Tensor next, mfq_tensor_backend::Tensor generated, mfq_tensor_backend::Tensor step,
                              mfq_tensor_backend::Tensor input, mfq_tensor_backend::Tensor pos, mfq_tensor_backend::Tensor len);
mfq_tensor_backend::Tensor sample_greedy_cuda(mfq_tensor_backend::Tensor logits);
// [B] int64 packets: low32 vocabulary ID, high32 full-softmax f32 probability.
// An optional stable device flag disables unused probability work on replay;
// disabled packets preserve the token and carry -1.f in the probability field.
mfq_tensor_backend::Tensor sample_greedy_probability_cuda(mfq_tensor_backend::Tensor logits,
    mfq_tensor_backend::Tensor output = {},const volatile int32_t* probability_enabled=nullptr);
// Feed valid low32 vocabulary IDs to another GPU step and optionally publish
// them to aligned mapped int32 host slots. No probability conversion or wait.
void sample_export_greedy_ids_cuda(mfq_tensor_backend::Tensor packets,mfq_tensor_backend::Tensor ids,
    int32_t* mapped_ids=nullptr);
// Publish raw packets to a stable, aligned mapped host alias. The caller owns
// that storage until stream completion; every packet bit is preserved.
void sample_publish_packets_cuda(mfq_tensor_backend::Tensor packets,int64_t* mapped_packets);
mfq_tensor_backend::Tensor sample_softmax_cuda(mfq_tensor_backend::Tensor logits, mfq_tensor_backend::Tensor random, double temperature);
mfq_tensor_backend::Tensor sample_top_k_top_p_cuda(
    mfq_tensor_backend::Tensor logits, mfq_tensor_backend::Tensor random, double temperature, int64_t top_k, double top_p);
// [B,top_k] int64 packets: high 32 bits original f32 logit, low 32 bits token ID.
mfq_tensor_backend::Tensor sample_top_k_compact_cuda(mfq_tensor_backend::Tensor logits,int64_t top_k);
// A stable device flag selects work at replay time without another graph
// shape. The owner must publish it before launch and retain it through replay.
mfq_tensor_backend::Tensor sample_top_k_compact_if_cuda(
    mfq_tensor_backend::Tensor logits,int64_t top_k,const volatile int32_t* enabled,
    mfq_tensor_backend::Tensor output = {});
void sample_token_counts_add_cuda(mfq_tensor_backend::Tensor counts, mfq_tensor_backend::Tensor tokens);
mfq_tensor_backend::Tensor sample_apply_penalties_cuda(
    mfq_tensor_backend::Tensor logits, mfq_tensor_backend::Tensor counts,
    double presence_penalty, double frequency_penalty, double repetition_penalty);
