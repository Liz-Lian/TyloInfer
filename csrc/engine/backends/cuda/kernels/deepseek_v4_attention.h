#pragma once

#include "backend.h"

#include <cstdint>
#include <vector>

::tyloi::DenseTensor dsv4_compress_cuda(
    ::tyloi::DenseTensor kv,
    ::tyloi::DenseTensor gate,
    ::tyloi::DenseTensor ape,
    ::tyloi::DenseTensor norm,
    ::tyloi::DenseTensor prev_kv,
    ::tyloi::DenseTensor prev_gate,
    ::tyloi::DenseTensor positions,
    ::tyloi::DenseTensor cos,
    ::tyloi::DenseTensor sin,
    std::int64_t ratio,
    bool overlap,
    std::int64_t quant_mode,
    double eps);

::tyloi::DenseTensor dsv4_fp4_sim_cuda(
    ::tyloi::DenseTensor input);

::tyloi::DenseTensor dsv4_decode_pool_update_cuda(
    ::tyloi::DenseTensor kv_token,
    ::tyloi::DenseTensor gate_token,
    ::tyloi::DenseTensor ape,
    ::tyloi::DenseTensor norm,
    ::tyloi::DenseTensor state_kv,
    ::tyloi::DenseTensor state_gate,
    ::tyloi::DenseTensor prev_kv,
    ::tyloi::DenseTensor prev_gate,
    ::tyloi::DenseTensor pool,
    ::tyloi::DenseTensor seq_len,
    ::tyloi::DenseTensor cos,
    ::tyloi::DenseTensor sin,
    std::int64_t ratio,
    bool overlap,
    std::int64_t quant_mode,
    double eps);

::tyloi::DenseTensor dsv4_indexer_scores_cuda(
    ::tyloi::DenseTensor q,
    ::tyloi::DenseTensor k,
    ::tyloi::DenseTensor weights,
    std::int64_t query_offset,
    std::int64_t ratio);

::tyloi::DenseTensor dsv4_topk512_cuda(
    ::tyloi::DenseTensor scores);

std::vector<::tyloi::DenseTensor> dsv4_build_prefill_plan_cuda(
    ::tyloi::DenseTensor topk,
    std::int64_t query_offset,
    std::int64_t local_history,
    std::int64_t pool_len,
    std::int64_t ratio,
    std::int64_t window);

std::vector<::tyloi::DenseTensor> dsv4_build_decode_plan_cuda(
    ::tyloi::DenseTensor topk,
    ::tyloi::DenseTensor seq_len,
    std::int64_t pool_len,
    std::int64_t ratio,
    std::int64_t window);

::tyloi::DenseTensor attention_dsv4_sparse_cuda(
    ::tyloi::DenseTensor q,
    ::tyloi::DenseTensor kv,
    ::tyloi::DenseTensor indices,
    ::tyloi::DenseTensor mask,
    ::tyloi::DenseTensor sinks,
    ::tyloi::DenseTensor meta,
    double scale);

::tyloi::DenseTensor dsv4_indexer_candidate_scores_cuda(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor k,
    ::tyloi::DenseTensor weights, ::tyloi::DenseTensor candidates,
    int64_t query_offset, int64_t ratio);
