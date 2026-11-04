#pragma once

#include "backend.h"

#include <cstdint>
#include <vector>

void qsa_pool_rotary_cache_cuda(
    const ::tyloi::DenseTensor& index, const ::tyloi::DenseTensor& weight,
    const ::tyloi::DenseTensor& positions, const ::tyloi::DenseTensor& cache_positions,
    const ::tyloi::DenseTensor& frequencies, const ::tyloi::DenseTensor& axes,
    const ::tyloi::DenseTensor& output, int64_t pool, int64_t maximum, int64_t rotary_maximum, double eps);

::tyloi::DenseTensor rotary_embedding_cached_cuda(
    ::tyloi::DenseTensor input, ::tyloi::DenseTensor positions,
    ::tyloi::DenseTensor frequencies, ::tyloi::DenseTensor axes,
    int64_t maximum);
::tyloi::DenseTensor rotary_normalized_cached_cuda(
    ::tyloi::DenseTensor input, ::tyloi::DenseTensor weight,
    ::tyloi::DenseTensor positions, ::tyloi::DenseTensor frequencies,
    ::tyloi::DenseTensor axes, int64_t maximum, double eps,
    ::tyloi::DenseTensor key_cache, ::tyloi::DenseTensor cache_positions,
    ::tyloi::DenseTensor projected_value, ::tyloi::DenseTensor value_cache);

std::vector<::tyloi::DenseTensor> rotary_normalized_grouped_cuda(
    const std::vector<::tyloi::DenseTensor>& inputs,
    const std::vector<::tyloi::DenseTensor>& weights,
    ::tyloi::DenseTensor positions,::tyloi::DenseTensor frequencies,
    ::tyloi::DenseTensor axes,int64_t maximum,double eps,int cache_entry,
    ::tyloi::DenseTensor key_cache,::tyloi::DenseTensor cache_positions,
    ::tyloi::DenseTensor projected_value,::tyloi::DenseTensor value_cache);

::tyloi::DenseTensor minicpm_qk_norm_rope_cache_write_bf16_cuda(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor k, ::tyloi::DenseTensor v,
    ::tyloi::DenseTensor q_weight, ::tyloi::DenseTensor k_weight,
    ::tyloi::DenseTensor rope_pos, ::tyloi::DenseTensor write_pos,
    ::tyloi::DenseTensor cos, ::tyloi::DenseTensor sin,
    ::tyloi::DenseTensor k_cache, ::tyloi::DenseTensor v_cache,
    double eps, double weight_offset);
::tyloi::DenseTensor rope_table_cuda(::tyloi::DenseTensor x, ::tyloi::DenseTensor pos, ::tyloi::DenseTensor cos, ::tyloi::DenseTensor sin,
                              int64_t rotary_dim, ::tyloi::DenseTensor sections);
::tyloi::DenseTensor rope_table_bf16_cuda(::tyloi::DenseTensor x, ::tyloi::DenseTensor pos, ::tyloi::DenseTensor cos, ::tyloi::DenseTensor sin,
                                   int64_t rotary_dim);
::tyloi::DenseTensor minicpm_bf16_rope_cache_write_cuda(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor k, ::tyloi::DenseTensor v,
    ::tyloi::DenseTensor rope_pos, ::tyloi::DenseTensor write_pos,
    ::tyloi::DenseTensor cos, ::tyloi::DenseTensor sin,
    ::tyloi::DenseTensor k_cache, ::tyloi::DenseTensor v_cache,
    int64_t rotary_dim);
::tyloi::DenseTensor attention_cuda(::tyloi::DenseTensor q, ::tyloi::DenseTensor k, ::tyloi::DenseTensor v, double scale, bool causal);
::tyloi::DenseTensor attention_swa_cuda(::tyloi::DenseTensor q, ::tyloi::DenseTensor k, ::tyloi::DenseTensor v,
                                 double scale, int64_t window);
::tyloi::DenseTensor attention_cache_swa_cuda(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor k_cache, ::tyloi::DenseTensor v_cache,
    ::tyloi::DenseTensor seq_len, double scale, int64_t window);
::tyloi::DenseTensor attention_cache_swa_planned_cuda(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor k_cache, ::tyloi::DenseTensor v_cache,
    ::tyloi::DenseTensor seq_len, double scale, int64_t window, int64_t planned_length);
::tyloi::DenseTensor mfq_attention_mma256_cuda(::tyloi::DenseTensor q, ::tyloi::DenseTensor k, ::tyloi::DenseTensor v, double scale);
::tyloi::DenseTensor mfq_attention_mma128_cuda(::tyloi::DenseTensor q, ::tyloi::DenseTensor k, ::tyloi::DenseTensor v, double scale);
::tyloi::DenseTensor minicpm_flash128_q_cast_cuda(
    ::tyloi::DenseTensor q);
std::vector<::tyloi::DenseTensor> minicpm_flash128_kv_cast_cuda(
    ::tyloi::DenseTensor k, ::tyloi::DenseTensor v);
::tyloi::DenseTensor minicpm_flash128_output_cast_cuda(
    ::tyloi::DenseTensor output);
::tyloi::DenseTensor mfq_attention_mma512_cuda(::tyloi::DenseTensor q, ::tyloi::DenseTensor k, ::tyloi::DenseTensor v, double scale);
::tyloi::DenseTensor mfq_attention_mma256_swa_cuda(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor k, ::tyloi::DenseTensor v, double scale, int64_t window);
::tyloi::DenseTensor mfq_attention_mma256_decode_cuda(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor k_cache, ::tyloi::DenseTensor v_cache,
    ::tyloi::DenseTensor seq_len, double scale, int64_t planned_len,
    ::tyloi::DenseTensor mask, ::tyloi::DenseTensor kv_max, ::tyloi::DenseTensor meta);
::tyloi::DenseTensor mfq_attention_mma512_decode_cuda(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor k_cache, ::tyloi::DenseTensor v_cache,
    ::tyloi::DenseTensor seq_len, double scale, int64_t planned_len,
    ::tyloi::DenseTensor mask, ::tyloi::DenseTensor kv_max, ::tyloi::DenseTensor meta);
::tyloi::DenseTensor attention_glm_mla576_cuda(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor kv, double scale);
::tyloi::DenseTensor attention_glm_mla576_cached_cuda(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor kv_cache, int64_t logical_len,
    ::tyloi::DenseTensor mask, ::tyloi::DenseTensor kv_max, ::tyloi::DenseTensor meta,
    double scale);
::tyloi::DenseTensor attention_glm_mla576_decode_cuda(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor kv_cache, ::tyloi::DenseTensor seq_len,
    double scale, int64_t planned_len, ::tyloi::DenseTensor mask,
    ::tyloi::DenseTensor kv_max, ::tyloi::DenseTensor meta);
::tyloi::DenseTensor glm_interleaved_rope_cuda(
    ::tyloi::DenseTensor x, ::tyloi::DenseTensor positions,
    ::tyloi::DenseTensor cos, ::tyloi::DenseTensor sin, int64_t rotary_dim);
::tyloi::DenseTensor dsv4_tail_rope_cuda(
    ::tyloi::DenseTensor x, ::tyloi::DenseTensor positions,
    ::tyloi::DenseTensor cos, ::tyloi::DenseTensor sin, int64_t rotary_dim);
::tyloi::DenseTensor glm_dsa_indexer_scores_cuda(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor k, ::tyloi::DenseTensor weights,
    int64_t query_offset, int64_t logical_k);
::tyloi::DenseTensor glm_dsa_indexer_scores_decode_cuda(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor k, ::tyloi::DenseTensor weights,
    ::tyloi::DenseTensor seq_len, int64_t planned_k);
::tyloi::DenseTensor attention_glm_mla_sparse_cuda(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor kv, ::tyloi::DenseTensor indices,
    ::tyloi::DenseTensor meta, double scale);
::tyloi::DenseTensor mfq_attention_mma256_swa_decode_cuda(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor k_cache, ::tyloi::DenseTensor v_cache,
    ::tyloi::DenseTensor seq_len, double scale, int64_t planned_len,
    ::tyloi::DenseTensor mask, ::tyloi::DenseTensor kv_max, ::tyloi::DenseTensor meta);
::tyloi::DenseTensor attention_cache_decode_cuda(::tyloi::DenseTensor q, ::tyloi::DenseTensor k_cache, ::tyloi::DenseTensor v_cache,
                                          ::tyloi::DenseTensor seq_len, double scale);
::tyloi::DenseTensor attention_cache_decode_split_cuda(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor k_cache, ::tyloi::DenseTensor v_cache,
    ::tyloi::DenseTensor seq_len, double scale,
    ::tyloi::DenseTensor partial_o, ::tyloi::DenseTensor partial_m, ::tyloi::DenseTensor partial_l,
    int64_t parts);
::tyloi::DenseTensor attention_cache_decode_dynamic_cuda(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor k_cache,
    ::tyloi::DenseTensor v_cache, ::tyloi::DenseTensor seq_len,
    double scale, ::tyloi::DenseTensor partial_o,
    ::tyloi::DenseTensor partial_m,
    ::tyloi::DenseTensor partial_l, int64_t max_parts);
