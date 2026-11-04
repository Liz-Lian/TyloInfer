#pragma once

#include "backend.h"

#include <cstdint>
#include <vector>

void qsa_pool_rotary_cache_cuda(
    const ::tyloi::Tensor& index, const ::tyloi::Tensor& weight,
    const ::tyloi::Tensor& positions, const ::tyloi::Tensor& cache_positions,
    const ::tyloi::Tensor& frequencies, const ::tyloi::Tensor& axes,
    const ::tyloi::Tensor& output, int64_t pool, int64_t maximum, int64_t rotary_maximum, double eps);

::tyloi::Tensor rotary_embedding_cached_cuda(
    ::tyloi::Tensor input, ::tyloi::Tensor positions,
    ::tyloi::Tensor frequencies, ::tyloi::Tensor axes,
    int64_t maximum);
::tyloi::Tensor rotary_normalized_cached_cuda(
    ::tyloi::Tensor input, ::tyloi::Tensor weight,
    ::tyloi::Tensor positions, ::tyloi::Tensor frequencies,
    ::tyloi::Tensor axes, int64_t maximum, double eps,
    ::tyloi::Tensor key_cache, ::tyloi::Tensor cache_positions,
    ::tyloi::Tensor projected_value, ::tyloi::Tensor value_cache);

std::vector<::tyloi::Tensor> rotary_normalized_grouped_cuda(
    const std::vector<::tyloi::Tensor>& inputs,
    const std::vector<::tyloi::Tensor>& weights,
    ::tyloi::Tensor positions,::tyloi::Tensor frequencies,
    ::tyloi::Tensor axes,int64_t maximum,double eps,int cache_entry,
    ::tyloi::Tensor key_cache,::tyloi::Tensor cache_positions,
    ::tyloi::Tensor projected_value,::tyloi::Tensor value_cache);

::tyloi::Tensor minicpm_qk_norm_rope_cache_write_bf16_cuda(
    ::tyloi::Tensor q, ::tyloi::Tensor k, ::tyloi::Tensor v,
    ::tyloi::Tensor q_weight, ::tyloi::Tensor k_weight,
    ::tyloi::Tensor rope_pos, ::tyloi::Tensor write_pos,
    ::tyloi::Tensor cos, ::tyloi::Tensor sin,
    ::tyloi::Tensor k_cache, ::tyloi::Tensor v_cache,
    double eps, double weight_offset);
::tyloi::Tensor rope_table_cuda(::tyloi::Tensor x, ::tyloi::Tensor pos, ::tyloi::Tensor cos, ::tyloi::Tensor sin,
                              int64_t rotary_dim, ::tyloi::Tensor sections);
::tyloi::Tensor rope_table_bf16_cuda(::tyloi::Tensor x, ::tyloi::Tensor pos, ::tyloi::Tensor cos, ::tyloi::Tensor sin,
                                   int64_t rotary_dim);
::tyloi::Tensor minicpm_bf16_rope_cache_write_cuda(
    ::tyloi::Tensor q, ::tyloi::Tensor k, ::tyloi::Tensor v,
    ::tyloi::Tensor rope_pos, ::tyloi::Tensor write_pos,
    ::tyloi::Tensor cos, ::tyloi::Tensor sin,
    ::tyloi::Tensor k_cache, ::tyloi::Tensor v_cache,
    int64_t rotary_dim);
::tyloi::Tensor attention_cuda(::tyloi::Tensor q, ::tyloi::Tensor k, ::tyloi::Tensor v, double scale, bool causal);
::tyloi::Tensor attention_swa_cuda(::tyloi::Tensor q, ::tyloi::Tensor k, ::tyloi::Tensor v,
                                 double scale, int64_t window);
::tyloi::Tensor attention_cache_swa_cuda(
    ::tyloi::Tensor q, ::tyloi::Tensor k_cache, ::tyloi::Tensor v_cache,
    ::tyloi::Tensor seq_len, double scale, int64_t window);
::tyloi::Tensor attention_cache_swa_planned_cuda(
    ::tyloi::Tensor q, ::tyloi::Tensor k_cache, ::tyloi::Tensor v_cache,
    ::tyloi::Tensor seq_len, double scale, int64_t window, int64_t planned_length);
::tyloi::Tensor mfq_attention_mma256_cuda(::tyloi::Tensor q, ::tyloi::Tensor k, ::tyloi::Tensor v, double scale);
::tyloi::Tensor mfq_attention_mma128_cuda(::tyloi::Tensor q, ::tyloi::Tensor k, ::tyloi::Tensor v, double scale);
::tyloi::Tensor minicpm_flash128_q_cast_cuda(
    ::tyloi::Tensor q);
std::vector<::tyloi::Tensor> minicpm_flash128_kv_cast_cuda(
    ::tyloi::Tensor k, ::tyloi::Tensor v);
::tyloi::Tensor minicpm_flash128_output_cast_cuda(
    ::tyloi::Tensor output);
::tyloi::Tensor mfq_attention_mma512_cuda(::tyloi::Tensor q, ::tyloi::Tensor k, ::tyloi::Tensor v, double scale);
::tyloi::Tensor mfq_attention_mma256_swa_cuda(
    ::tyloi::Tensor q, ::tyloi::Tensor k, ::tyloi::Tensor v, double scale, int64_t window);
::tyloi::Tensor mfq_attention_mma256_decode_cuda(
    ::tyloi::Tensor q, ::tyloi::Tensor k_cache, ::tyloi::Tensor v_cache,
    ::tyloi::Tensor seq_len, double scale, int64_t planned_len,
    ::tyloi::Tensor mask, ::tyloi::Tensor kv_max, ::tyloi::Tensor meta);
::tyloi::Tensor mfq_attention_mma512_decode_cuda(
    ::tyloi::Tensor q, ::tyloi::Tensor k_cache, ::tyloi::Tensor v_cache,
    ::tyloi::Tensor seq_len, double scale, int64_t planned_len,
    ::tyloi::Tensor mask, ::tyloi::Tensor kv_max, ::tyloi::Tensor meta);
::tyloi::Tensor attention_glm_mla576_cuda(
    ::tyloi::Tensor q, ::tyloi::Tensor kv, double scale);
::tyloi::Tensor attention_glm_mla576_cached_cuda(
    ::tyloi::Tensor q, ::tyloi::Tensor kv_cache, int64_t logical_len,
    ::tyloi::Tensor mask, ::tyloi::Tensor kv_max, ::tyloi::Tensor meta,
    double scale);
::tyloi::Tensor attention_glm_mla576_decode_cuda(
    ::tyloi::Tensor q, ::tyloi::Tensor kv_cache, ::tyloi::Tensor seq_len,
    double scale, int64_t planned_len, ::tyloi::Tensor mask,
    ::tyloi::Tensor kv_max, ::tyloi::Tensor meta);
::tyloi::Tensor glm_interleaved_rope_cuda(
    ::tyloi::Tensor x, ::tyloi::Tensor positions,
    ::tyloi::Tensor cos, ::tyloi::Tensor sin, int64_t rotary_dim);
::tyloi::Tensor dsv4_tail_rope_cuda(
    ::tyloi::Tensor x, ::tyloi::Tensor positions,
    ::tyloi::Tensor cos, ::tyloi::Tensor sin, int64_t rotary_dim);
::tyloi::Tensor glm_dsa_indexer_scores_cuda(
    ::tyloi::Tensor q, ::tyloi::Tensor k, ::tyloi::Tensor weights,
    int64_t query_offset, int64_t logical_k);
::tyloi::Tensor glm_dsa_indexer_scores_decode_cuda(
    ::tyloi::Tensor q, ::tyloi::Tensor k, ::tyloi::Tensor weights,
    ::tyloi::Tensor seq_len, int64_t planned_k);
::tyloi::Tensor attention_glm_mla_sparse_cuda(
    ::tyloi::Tensor q, ::tyloi::Tensor kv, ::tyloi::Tensor indices,
    ::tyloi::Tensor meta, double scale);
::tyloi::Tensor mfq_attention_mma256_swa_decode_cuda(
    ::tyloi::Tensor q, ::tyloi::Tensor k_cache, ::tyloi::Tensor v_cache,
    ::tyloi::Tensor seq_len, double scale, int64_t planned_len,
    ::tyloi::Tensor mask, ::tyloi::Tensor kv_max, ::tyloi::Tensor meta);
::tyloi::Tensor attention_cache_decode_cuda(::tyloi::Tensor q, ::tyloi::Tensor k_cache, ::tyloi::Tensor v_cache,
                                          ::tyloi::Tensor seq_len, double scale);
::tyloi::Tensor attention_cache_decode_split_cuda(
    ::tyloi::Tensor q, ::tyloi::Tensor k_cache, ::tyloi::Tensor v_cache,
    ::tyloi::Tensor seq_len, double scale,
    ::tyloi::Tensor partial_o, ::tyloi::Tensor partial_m, ::tyloi::Tensor partial_l,
    int64_t parts);
::tyloi::Tensor attention_cache_decode_dynamic_cuda(
    ::tyloi::Tensor q, ::tyloi::Tensor k_cache,
    ::tyloi::Tensor v_cache, ::tyloi::Tensor seq_len,
    double scale, ::tyloi::Tensor partial_o,
    ::tyloi::Tensor partial_m,
    ::tyloi::Tensor partial_l, int64_t max_parts);
