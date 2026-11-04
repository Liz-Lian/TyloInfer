#pragma once

#include "backend.h"

#include <cstdint>
#include <vector>

::tyloi::Tensor dsv4_compress_cuda(
    ::tyloi::Tensor kv,
    ::tyloi::Tensor gate,
    ::tyloi::Tensor ape,
    ::tyloi::Tensor norm,
    ::tyloi::Tensor prev_kv,
    ::tyloi::Tensor prev_gate,
    ::tyloi::Tensor positions,
    ::tyloi::Tensor cos,
    ::tyloi::Tensor sin,
    std::int64_t ratio,
    bool overlap,
    std::int64_t quant_mode,
    double eps);

::tyloi::Tensor dsv4_fp4_sim_cuda(
    ::tyloi::Tensor input);

::tyloi::Tensor dsv4_decode_pool_update_cuda(
    ::tyloi::Tensor kv_token,
    ::tyloi::Tensor gate_token,
    ::tyloi::Tensor ape,
    ::tyloi::Tensor norm,
    ::tyloi::Tensor state_kv,
    ::tyloi::Tensor state_gate,
    ::tyloi::Tensor prev_kv,
    ::tyloi::Tensor prev_gate,
    ::tyloi::Tensor pool,
    ::tyloi::Tensor seq_len,
    ::tyloi::Tensor cos,
    ::tyloi::Tensor sin,
    std::int64_t ratio,
    bool overlap,
    std::int64_t quant_mode,
    double eps);

::tyloi::Tensor dsv4_indexer_scores_cuda(
    ::tyloi::Tensor q,
    ::tyloi::Tensor k,
    ::tyloi::Tensor weights,
    std::int64_t query_offset,
    std::int64_t ratio);

::tyloi::Tensor dsv4_topk512_cuda(
    ::tyloi::Tensor scores);

std::vector<::tyloi::Tensor> dsv4_build_prefill_plan_cuda(
    ::tyloi::Tensor topk,
    std::int64_t query_offset,
    std::int64_t local_history,
    std::int64_t pool_len,
    std::int64_t ratio,
    std::int64_t window);

std::vector<::tyloi::Tensor> dsv4_build_decode_plan_cuda(
    ::tyloi::Tensor topk,
    ::tyloi::Tensor seq_len,
    std::int64_t pool_len,
    std::int64_t ratio,
    std::int64_t window);

::tyloi::Tensor attention_dsv4_sparse_cuda(
    ::tyloi::Tensor q,
    ::tyloi::Tensor kv,
    ::tyloi::Tensor indices,
    ::tyloi::Tensor mask,
    ::tyloi::Tensor sinks,
    ::tyloi::Tensor meta,
    double scale);

::tyloi::Tensor dsv4_indexer_candidate_scores_cuda(
    ::tyloi::Tensor q, ::tyloi::Tensor k,
    ::tyloi::Tensor weights, ::tyloi::Tensor candidates,
    int64_t query_offset, int64_t ratio);
