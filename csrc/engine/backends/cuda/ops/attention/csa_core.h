#pragma once
#include "engine/ops/csa.h"
#include "engine/ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_norm_ops.h"
#include "engine/backends/cuda/kernels/deepseek_v41.h"
#include "engine/backends/cuda/kernels/deepseek_v4_attention.h"
#include "backend.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace tyloi::ops {
namespace {
using DenseTensor = ::tyloi::DenseTensor;
struct WorkShared {
    DenseTensor compressed_kv, index_k, topk, candidate_blocks;
    std::int64_t compressed_length = 0, ratio = 0;
    Tensor* scratch = nullptr;
    std::size_t scratch_count = 0;
    DenseTensor keep(DenseTensor value) {
        if (scratch && scratch_count < 32) scratch[scratch_count++].dense() = value;
        return value;
    }
};
inline DenseTensor weighted_rms(const DenseTensor &input,
                                const ::tyloi::Tensor &weight, double epsilon) {
    const auto shape = input.sizes().vec();
    TYLOINFER_RUNTIME_CHECK(!shape.empty() && weight.numel() == shape.back(),
                      "DeepSeek-V4.1 weighted RMS geometry mismatch");
    return rms_norm_cuda(input.reshape({-1, shape.back()}).to(::tyloi::kFloat32),
                         weight.dense(),
                         epsilon)
        .reshape(shape)
        .to(::tyloi::kFloat16)
        .contiguous();
}

inline DenseTensor rotate_token_major_tail(DenseTensor input, const DenseTensor &positions,
                                      const ::tyloi::ops::TailRopeTable &rope, bool inverse = false) {
    TYLOINFER_RUNTIME_CHECK(input.dim() == 3, "DeepSeek-V4.1 RoPE input must be [B,T,D]");
    auto value = input.unsqueeze(1).contiguous();
    auto rotated = ::tyloi::ops::tail_rope(value, positions, rope, inverse);
    return rotated.dense().squeeze(1).contiguous();
}

inline DenseTensor empty_topk(std::int64_t batch, std::int64_t tokens,
                         const ::tyloi::Device &device) {
    return ::tyloi::cuda::empty(
        {batch, tokens, 0},
        ::tyloi::TensorOptions().device(device).dtype(::tyloi::kInt32));
}

inline DenseTensor normalize_topk(DenseTensor selected, const DenseTensor &values, std::int64_t pool_length) {
    selected = selected.to(::tyloi::kInt64).contiguous();
    auto valid = (selected >= 0.0) & (selected < static_cast<double>(pool_length));
    valid = valid & ::tyloi::cuda::isneginf(values).logical_not();
    auto masked = ::tyloi::cuda::where(valid, selected, static_cast<double>(pool_length));
    auto ordered = std::get<0>(::tyloi::cuda::sort(masked, -1, false));
    return ::tyloi::cuda::where(ordered < static_cast<double>(pool_length), ordered, -1.0)
        .to(::tyloi::kInt32)
        .contiguous();
}

inline DenseTensor select_topk(const DenseTensor &scores, std::int64_t requested, std::int64_t pool_length,
                          const std::optional<DenseTensor> &source_indices = std::nullopt) {
    const auto width = scores.size(-1);
    const auto count = std::min(requested, width);
    if (count <= 0) {
        return empty_topk(scores.size(0), scores.size(1), scores.device());
    }
    DenseTensor relative;
    DenseTensor values;
    if (count == width) {
        relative =
            ::tyloi::cuda::arange(width, scores.options().dtype(::tyloi::kInt64))
                .reshape({1, 1, width})
                .expand({scores.size(0), scores.size(1), width})
                .contiguous();
        values = scores;
    } else if (count == 512 && scores.scalar_type() == ::tyloi::kFloat16) {
        relative =
            dsv4_topk512_cuda(scores.contiguous()).to(::tyloi::kInt64).contiguous();
        values = scores.gather(-1, relative);
    } else {
        auto result = ::tyloi::cuda::topk(scores, count, -1, true, false);
        values = std::get<0>(result);
        relative = std::get<1>(result);
    }
    auto selected = source_indices
                        ? source_indices->to(::tyloi::kInt64).gather(-1, relative)
                        : relative;
    return normalize_topk(selected, values, pool_length);
}

inline DenseTensor candidate_blocks_from_scores(const DenseTensor &scores, std::int64_t ratio,
                                           std::int64_t position, std::int64_t requested_blocks,
                                           std::int64_t block_size) {
    const auto batch = scores.size(0);
    const auto tokens = scores.size(1);
    const auto width = scores.size(2);
    const auto blocks = (width + block_size - 1) / block_size;
    const auto padded = blocks * block_size;
    DenseTensor padded_scores = scores;
    if (padded != width) {
        padded_scores = ::tyloi::cuda::cat(
            {scores,
             ::tyloi::cuda::full({batch, tokens, padded - width},
                                      -std::numeric_limits<float>::infinity(), scores.options())},
            -1);
    }
    auto block_scores = padded_scores.reshape({batch, tokens, blocks, block_size}).amax(-1);
    auto int_options = scores.options().dtype(::tyloi::kInt64);
    auto visible = (::tyloi::cuda::arange(position + 1, position + tokens + 1, int_options)
                        .reshape({1, tokens, 1}) /
                    static_cast<double>(ratio));
    auto last = ::tyloi::cuda::where(visible > 0.0,
                                          (visible - 1.0) / static_cast<double>(block_size), -1.0);
    auto block_ids = ::tyloi::cuda::arange(blocks, int_options).reshape({1, 1, blocks});
    block_scores = ::tyloi::cuda::where(block_ids == last,
                                             std::numeric_limits<float>::infinity(), block_scores);
    const auto count = std::min(requested_blocks, blocks);
    auto result = ::tyloi::cuda::topk(block_scores, count, -1, true, false);
    auto values = std::get<0>(result);
    auto selected = std::get<1>(result);
    return ::tyloi::cuda::where(::tyloi::cuda::isneginf(values).logical_not(), selected,
                                     -1.0)
        .to(::tyloi::kInt32)
        .contiguous();
}

inline DenseTensor candidate_positions(const DenseTensor &blocks, std::int64_t block_size,
                                  std::int64_t pool_length) {
    const auto count = blocks.size(-1);
    auto block64 = blocks.to(::tyloi::kInt64);
    auto within =
        ::tyloi::cuda::arange(block_size, blocks.options().dtype(::tyloi::kInt64))
            .reshape({1, 1, 1, block_size});
    auto expanded = (block64.unsqueeze(-1) * static_cast<double>(block_size) + within)
                        .reshape({blocks.size(0), blocks.size(1), count * block_size});
    auto valid = (expanded >= 0.0) & (expanded < static_cast<double>(pool_length));
    return ::tyloi::cuda::where(valid, expanded, -1.0)
        .to(::tyloi::kInt64)
        .contiguous();
}

} // namespace
struct Csa::Impl {
    using State = Csa::State;
    CsaConfig config;
    std::int64_t layer, ratio;
    Tensor query_a, query_b, key_value, output_a, output_b;
    Tensor compressor_key_value, compressor_gate, index_query, index_key, index_score;
    Tensor query_a_norm, key_value_norm, sinks, compressor_norm, index_key_norm;
    TailRopeTable rope;
    Impl(CsaWeights w, CsaConfig c)
        : config(c), layer(c.layer), ratio(c.ratio), query_a(std::move(w.query_a)), query_b(std::move(w.query_b)),
          key_value(std::move(w.key_value)), output_a(std::move(w.output_a)), output_b(std::move(w.output_b)),
          compressor_key_value(std::move(w.compressor_key_value)), compressor_gate(std::move(w.compressor_gate)),
          index_query(std::move(w.index_query)), index_key(std::move(w.index_key)), index_score(std::move(w.index_score)),
          query_a_norm(std::move(w.query_a_norm)), key_value_norm(std::move(w.key_value_norm)), sinks(std::move(w.sinks)),
          compressor_norm(std::move(w.compressor_norm)), index_key_norm(std::move(w.index_key_norm)),
          rope(tyloi::ops::tail_rope_table(c.maximum,c.rope_theta,c.ratio,Device(kCUDA,c.device),
              c.compress_rope_theta,c.rope_original,c.rope_factor,c.rope_beta_fast,c.rope_beta_slow)) {}
    bool kv_source() const noexcept { return config.kv_source; }
    bool index_source() const noexcept { return config.index_source; }
    DenseTensor output_projection(DenseTensor attention) const {
        return tyloi::ops::low_rank(attention, output_a, output_b, config.o_groups, true, false).dense();
    }

    std::optional<DenseTensor> update_compressed_source(State& layer_state, WorkShared& shared,
                                                   const DenseTensor &input, std::int64_t position) {
        const auto batch = input.size(0);
        const auto tokens = input.size(1);
        DenseTensor projected_kv, projected_score;
                const auto direct = [&] {
                return weighted_rms(tyloi::ops::project(input, compressor_key_value).dense(),
                                    compressor_norm,
                                    config.rms_eps);
            };
        const auto project = [&] {
                projected_kv = tyloi::ops::project(input, compressor_key_value).dense()
                                   .to(::tyloi::kFloat32)
                                   .contiguous();
                shared.keep(projected_kv);
                projected_score = tyloi::ops::project(input, compressor_gate).dense()
                                      .to(::tyloi::kFloat32)
                                      .contiguous();
                shared.keep(projected_score);
                if(layer_state.speculative_pending) {
                    layer_state.speculative_projected_kv.dense()=projected_kv;
                    layer_state.speculative_projected_score.dense()=projected_score;
                }
            };
        const auto pool = [&](int64_t complete, int64_t cutoff) {
                auto kv_groups = projected_kv.narrow(1, 0, cutoff)
                                     .reshape({batch, complete, ratio, config.head_dim});
                auto score_groups = projected_score.narrow(1, 0, cutoff)
                                        .reshape({batch, complete, ratio, config.head_dim});
                auto pooled = (kv_groups * ::tyloi::cuda::softmax(score_groups, 2)).sum(2);
                return weighted_rms(pooled.to(::tyloi::kFloat16),
                                    compressor_norm,
                                    config.rms_eps);
            };
        const auto save = [&](int64_t cutoff, int64_t remainder) {
                layer_state.partial_kv.dense().narrow(1, 0, remainder)
                    .copy_(projected_kv.narrow(1, cutoff, remainder));
                layer_state.partial_score.dense().narrow(1, 0, remainder)
                    .copy_(projected_score.narrow(1, cutoff, remainder));
            };
        const auto write = [&](int64_t token, int64_t slot) {
                layer_state.partial_kv.dense().narrow(1, slot, 1).copy_(projected_kv.narrow(1, token, 1));
                layer_state.partial_score.dense().narrow(1, slot, 1).copy_(projected_score.narrow(1, token, 1));
            };
        const auto partial_pool = [&] {
                auto pooled =
                    (layer_state.partial_kv.dense() * ::tyloi::cuda::softmax(layer_state.partial_score.dense(), 1))
                        .sum(1, true);
                return weighted_rms(pooled.to(::tyloi::kFloat16),
                                    compressor_norm,
                                    config.rms_eps);
            };
        const auto concat = [](const auto &emitted) { return ::tyloi::cuda::cat(emitted, 1).contiguous(); };
        TYLOINFER_RUNTIME_CHECK(ratio > 0, "invalid compression ratio");
        if (ratio == 1) return std::optional<DenseTensor>(direct());
        project();
        std::vector<DenseTensor> emitted;
        if (layer_state.partial_length == 0 && position % ratio == 0) {
            const auto complete=tokens/ratio, cutoff=complete*ratio;
            if(complete>0)emitted.push_back(pool(complete,cutoff));
            const auto remainder=tokens-cutoff;
            if(remainder>0)save(cutoff,remainder);
            layer_state.partial_length=remainder;
        } else {
            for(int64_t token=0;token<tokens;++token) {
                const auto slot=(position+token)%ratio;
                write(token,slot);
                if(slot+1==ratio)emitted.push_back(partial_pool());
            }
            layer_state.partial_length=(position+tokens)%ratio;
        }
        if(emitted.empty())return std::optional<DenseTensor>{};
        return std::optional<DenseTensor>(emitted.size()==1?std::move(emitted.front()):concat(emitted));
    }

    void publish_compressed_source(State& layer_state, WorkShared& shared, const DenseTensor &input,
                                   std::int64_t position) {
        std::optional<DenseTensor> latent;
        if (layer_state.prefilled_global_end == position + input.size(1)) layer_state.prefilled_global_end = -1;
        else latent = update_compressed_source(layer_state, shared, input, position);
        if (latent) {
            shared.keep(*latent);
            const auto count = latent->size(1);
            const auto begin = layer_state.compressed_length;
            TYLOINFER_RUNTIME_CHECK(begin + count <= layer_state.compressed_kv.size(1),
                              "DeepSeek-V4.1 compressed cache capacity exceeded");
            auto group_positions =
                ::tyloi::cuda::arange(begin * ratio, (begin + count) * ratio, ratio,
                                           latent->options().dtype(::tyloi::kInt64));
            auto index_values =
                weighted_rms(tyloi::ops::project(*latent, index_key).dense(),
                             index_key_norm, config.rms_eps);
            shared.keep(index_values);
            index_values = rotate_token_major_tail(index_values, group_positions, rope);
            shared.keep(index_values);
            index_values = dsv4_fp4_sim_cuda(index_values.contiguous());
            shared.keep(index_values);
            auto compressed = rotate_token_major_tail(*latent, group_positions, rope);
            shared.keep(compressed);
            compressed = deepseek_v41_mxfp4_e4m3_scale_sim_cuda(compressed.contiguous());
            shared.keep(compressed);
            auto rows = ::tyloi::cuda::arange(
                begin, begin + count, latent->options().dtype(::tyloi::kInt64));
            layer_state.index_k.dense().index_copy_(1, rows, index_values);
            layer_state.compressed_kv.dense().index_copy_(1, rows, compressed);
            layer_state.compressed_length += count;
        }
        shared.compressed_kv = layer_state.compressed_kv.dense();
        shared.index_k = layer_state.index_k.dense();
        shared.compressed_length = layer_state.compressed_length;
        shared.ratio = ratio;
    }

    DenseTensor compute_topk(State& layer_state, WorkShared& shared, const DenseTensor &input, const DenseTensor &q_rank,
                        const DenseTensor &positions, std::int64_t position) {
        const auto batch = input.size(0);
        const auto tokens = input.size(1);
        const auto pool_length = shared.compressed_length;
        DenseTensor candidates;
        // Restrict scoring only when the candidate budget excludes some KV.
        // An all-inclusive list adds scattered loads without reducing work.
        const bool restricted = pool_length > config.candidate_topk_blocks * config.candidate_block_size;
                const auto empty = [&] { return empty_topk(batch, tokens, input.device()); };
        const auto reuse = [&] {
                TYLOINFER_RUNTIME_CHECK(shared.topk.defined(),
                                  "DeepSeek-V4.1 CSA2 consumer has no published index plan");
                return shared.topk;
            };
        const auto score = [&] {
                auto query =
                    tyloi::ops::project(q_rank, index_query).dense()
                        .reshape({batch, tokens, config.index_n_heads, config.index_head_dim})
                        .transpose(1, 2)
                        .contiguous();
                auto rotated = ::tyloi::ops::tail_rope(query, positions, rope);
                query = rotated.dense()
                            .transpose(1, 2)
                            .contiguous();
                shared.keep(query);
                query = dsv4_fp4_sim_cuda(query.to(::tyloi::kFloat16).contiguous());
                shared.keep(query);
                auto weights = tyloi::ops::project(input, index_score).dense()
                                   .reshape({batch, tokens, config.index_n_heads})
                                   .to(::tyloi::kFloat16)
                                   .contiguous();
                shared.keep(weights);
                auto keys = shared.index_k.narrow(1, 0, pool_length).contiguous();
                shared.keep(keys);
                if (layer > config.candidate_source_layer && restricted) {
                    TYLOINFER_RUNTIME_CHECK(shared.candidate_blocks.defined(),
                                      "DeepSeek-V4.1 hierarchical indexer has no candidates");
                    candidates = candidate_positions(shared.candidate_blocks,
                                                     config.candidate_block_size, pool_length);
                    shared.keep(candidates);
                    return tyloi::ops::measure("deepseek_v41.indexer.candidate_scores", [&] {
                        return dsv4_indexer_candidate_scores_cuda(query, keys, weights, candidates,
                                                                  position, ratio);
                    });
                }
                return tyloi::ops::measure("deepseek_v41.indexer.full_scores", [&] {
                    return dsv4_indexer_scores_cuda(query, keys, weights, position, ratio);
                });
            };
        const auto candidates_stage = [&](DenseTensor scores) {
                if (restricted) shared.candidate_blocks = candidate_blocks_from_scores(
                    scores, ratio, position, config.candidate_topk_blocks,
                    config.candidate_block_size);
            };
        const auto select_candidates = [&](DenseTensor scores) {
                if (!restricted) return select_topk(scores, config.index_topk, pool_length);
                TYLOINFER_RUNTIME_CHECK(candidates.defined(), "DeepSeek-V4.1 candidate scores are unavailable");
                return select_topk(scores, config.index_topk, pool_length, candidates);
            };
        const auto select = [&](DenseTensor scores) { return select_topk(scores, config.index_topk, pool_length); };
        const auto publish = [&](DenseTensor topk) { shared.topk = std::move(topk); };
        if(ratio==0)return empty();
        if(!index_source())return reuse();
        auto selected=empty();
        if(pool_length>0) {
            auto scores=shared.keep(score());
            if(layer==config.candidate_source_layer)candidates_stage(scores);
            selected=layer>config.candidate_source_layer?select_candidates(scores):select(scores);
        }
        publish(selected);
        return selected;
    }

    DenseTensor attention_forward(State& layer_state, WorkShared& shared, const DenseTensor &input,
                             const DenseTensor &positions, std::int64_t position,
                             const MfqOptional<DenseTensor> &sequence_lengths,
                             bool truncated_local = false) {
        auto& local_kv = layer_state.local_kv.dense();
        const auto batch = input.size(0);
        const auto tokens = input.size(1);
        const auto local_history = truncated_local ? 0 : std::min(position, config.sliding_window);
                const auto rank = [&] {
                return weighted_rms(tyloi::ops::project(input, query_a).dense(),
                                    query_a_norm,
                                    config.rms_eps);
            };
        const auto query_stage = [&](DenseTensor q_rank) {
                auto query = tyloi::ops::project(q_rank, query_b).dense()
                                 .reshape({batch, tokens, config.n_heads, config.head_dim})
                                 .transpose(1, 2)
                                 .contiguous()
                                 .to(::tyloi::kFloat32);
                auto rotated = ::tyloi::ops::tail_rope(query, positions, rope);
                query = rotated.dense();

                return query;
            };
        const auto local_stage = [&] {
                auto local = weighted_rms(tyloi::ops::project(input, key_value).dense(),
                                          key_value_norm,
                                          config.rms_eps)
                                 .reshape({batch, tokens, config.head_dim});
                local = rotate_token_major_tail(local, positions, rope);
                local = deepseek_v41_mxfp8_e4m3_sim_cuda(local.contiguous());

                return local;
            };
        const auto publish = [&] { publish_compressed_source(layer_state, shared, input, position); };
        const auto validate = [&] {
                TYLOINFER_RUNTIME_CHECK(shared.compressed_kv.defined() && shared.index_k.defined() &&
                                      shared.ratio == ratio,
                                  "DeepSeek-V4.1 CSA2 consumer ran before its source layer");
            };
        const auto index_stage = [&](DenseTensor q_rank) {
                return compute_topk(layer_state, shared, input, q_rank, positions, position);
            };
        const auto cache_stage = [&](DenseTensor local) {
                DenseTensor local_for_attention;
                if (tokens == 1) {
                    auto slot = ::tyloi::cuda::full(
                        {1}, position % config.sliding_window,
                        positions.options().dtype(::tyloi::kInt64));
                    local_kv.index_copy_(1, slot, local);
                    local_for_attention = local_kv;
                } else {
                    const auto history = local_history;
                    std::vector<DenseTensor> local_parts;
                    if (history > 0) {
                        auto slots = ::tyloi::cuda::arange(
                                         position - history, position,
                                         positions.options().dtype(::tyloi::kInt64))
                                         .remainder(static_cast<double>(config.sliding_window))
                                         .to(::tyloi::kInt64)
                                         .contiguous();
                        local_parts.push_back(local_kv.index_select(1, slots));
                    }
                    local_parts.push_back(local);
                    local_for_attention =
                        local_parts.size() == 1
                            ? local_parts.front()
                            : ::tyloi::cuda::cat(local_parts, 1).contiguous();
                    const auto recent = std::min(tokens, config.sliding_window);
                    auto recent_slots = ::tyloi::cuda::arange(
                                            position + tokens - recent, position + tokens,
                                            positions.options().dtype(::tyloi::kInt64))
                                            .remainder(static_cast<double>(config.sliding_window))
                                            .to(::tyloi::kInt64)
                                            .contiguous();
                    local_kv.index_copy_(1, recent_slots,
                                               local.narrow(1, tokens - recent, recent));
                }

                const auto pool_length = ratio > 0 ? shared.compressed_length : 0;
                auto cache =
                    pool_length > 0
                        ? ::tyloi::cuda::cat({local_for_attention,
                                                   shared.compressed_kv.narrow(1, 0, pool_length)},
                                                  1)
                              .contiguous()
                        : local_for_attention.contiguous();
                return cache;
            };
        const auto plan_stage = [&](DenseTensor topk) {
                const auto pool_length = ratio > 0 ? shared.compressed_length : 0;
                std::vector<DenseTensor> plan;
                if (tokens == 1) {
                    auto lengths = sequence_lengths.has_value()
                                       ? sequence_lengths.value()
                                             .to(input.device(), ::tyloi::kInt64)
                                             .contiguous()
                                       : ::tyloi::cuda::full(
                                             {batch}, position + 1,
                                             positions.options().dtype(::tyloi::kInt64));
                    plan = dsv4_build_decode_plan_cuda(
                        topk, lengths, pool_length, ratio > 0 ? ratio : 1, config.sliding_window);
                } else {
                    TYLOINFER_RUNTIME_CHECK(
                        !sequence_lengths.has_value(),
                        "DeepSeek-V4.1 chunked prefill does not accept per-row lengths");
                    const auto history = local_history;
                    plan =
                        dsv4_build_prefill_plan_cuda(topk, position, history, pool_length,
                                                     ratio > 0 ? ratio : 1, config.sliding_window);
                }
                return plan;
            };
        const auto attend = [&](DenseTensor query, DenseTensor cache, const auto &plan) {
                return tyloi::ops::sparse_attention(
                           query, cache, plan.at(0), plan.at(1), sinks,
                           1.0 / std::sqrt(static_cast<double>(config.head_dim)))
                    .dense();
            };
        const auto inverse = [&](DenseTensor attended) {
                auto value = attended.transpose(1, 2).contiguous();
                auto rotated = ::tyloi::ops::tail_rope(value, positions, rope, true);
                return rotated.dense()
                    .transpose(1, 2)
                    .contiguous();
            };
        const auto commit = [&] { layer_state.position += tokens; };
        const auto output = [&](DenseTensor attended) { return output_projection(attended); };
        auto q_rank=shared.keep(rank());
        auto q=shared.keep(query_stage(q_rank));
        auto kv=shared.keep(local_stage());
        if(kv_source())publish();
        else if(ratio>0)validate();
        auto topk=shared.keep(index_stage(q_rank));
        auto values=shared.keep(cache_stage(std::move(kv)));
        auto indices=plan_stage(std::move(topk));
        for (const auto& value : indices) shared.keep(value);
        auto attended=shared.keep(attend(std::move(q),std::move(values),indices));
        auto hidden=shared.keep(inverse(std::move(attended)));
        commit();
        return shared.keep(output(std::move(hidden)));
    }

    /** Reset this layer's caches, reusing compatible backing storage. */
    void reset(State& layer_state, std::int64_t batch) {
        MfqCudaGuard guard(config.device);
        const ::tyloi::Device device(::tyloi::kCUDA, config.device);
        const auto reset_tensor = [&](::tyloi::Tensor& value, std::int64_t length,
                                      std::int64_t width, ::tyloi::ScalarType dtype, double fill) {
            const bool matches = value.defined() && value.dim() == 3 && value.size(0) == batch &&
                value.size(1) == length && value.size(2) == width && value.scalar_type() == dtype &&
                value.device() == device && value.representation<DenseTensor>() != nullptr && value.is_contiguous();
            if (!matches)
                value = tyloi::ops::empty({batch, length, width},
                    ::tyloi::TensorOptions().device(device).dtype(dtype));
            tyloi::ops::fill(value, fill);
        };
        reset_tensor(layer_state.local_kv, config.sliding_window, config.head_dim, ::tyloi::kFloat16, 0);
        if (kv_source()) {
            const auto capacity = std::max<std::int64_t>(1, config.maximum / ratio);
            reset_tensor(layer_state.compressed_kv, capacity, config.head_dim, ::tyloi::kFloat16, 0);
            reset_tensor(layer_state.index_k, capacity, config.index_head_dim, ::tyloi::kFloat16, 0);
            if (ratio > 1) {
                reset_tensor(layer_state.partial_kv, ratio, config.head_dim, ::tyloi::kFloat32, 0);
                reset_tensor(layer_state.partial_score, ratio, config.head_dim, ::tyloi::kFloat32,
                    -std::numeric_limits<float>::infinity());
            } else {
                layer_state.partial_kv = {}; layer_state.partial_score = {};
            }
        } else {
            layer_state.compressed_kv = {}; layer_state.index_k = {};
            layer_state.partial_kv = {}; layer_state.partial_score = {};
        }
        layer_state.position = 0;
        layer_state.compressed_length = 0;
        layer_state.partial_length = 0;
        layer_state.speculative_local_slots = {}; layer_state.speculative_slot_ids = {};
        layer_state.speculative_partial_kv = {}; layer_state.speculative_partial_score = {};
        layer_state.speculative_projected_kv = {}; layer_state.speculative_projected_score = {};
        layer_state.speculative_position = -1;
        layer_state.speculative_pending = false;
        layer_state.prefilled_global_end = -1;
        layer_state.replay_local_window = false;
        layer_state.failed.clear();
    }


    /** Save the mutable cache state before a speculative window. */
    void begin_speculative(State& layer_state, std::int64_t draft_tokens) {
        TYLOINFER_RUNTIME_CHECK(!layer_state.speculative_pending && draft_tokens > 0 &&
                              draft_tokens <= config.sliding_window && layer_state.local_kv.defined(),
                          "invalid DeepSeek-V4.1 speculative checkpoint");
        layer_state.speculative_slot_ids = Tensor{DenseTensor{}};
        layer_state.speculative_local_slots = Tensor{DenseTensor{}};
        layer_state.speculative_partial_kv = Tensor{DenseTensor{}};
        layer_state.speculative_partial_score = Tensor{DenseTensor{}};
        layer_state.speculative_position = layer_state.position;
        layer_state.speculative_compressed_length = layer_state.compressed_length;
        layer_state.speculative_partial_length = layer_state.partial_length;
        layer_state.speculative_slot_ids.dense() =
            ::tyloi::cuda::arange(layer_state.position, layer_state.position + draft_tokens,
                                       layer_state.local_kv.options().dtype(::tyloi::kInt64))
                .remainder(static_cast<double>(config.sliding_window))
                .to(::tyloi::kInt64)
                .contiguous();
        layer_state.speculative_local_slots.dense() = layer_state.local_kv.dense().index_select(1, layer_state.speculative_slot_ids.dense()).clone();
        if (layer_state.partial_kv.defined()) {
            layer_state.speculative_partial_kv.dense() = layer_state.partial_kv.dense().clone();
            layer_state.speculative_partial_score.dense() = layer_state.partial_score.dense().clone();
        }
        layer_state.speculative_pending = true;
    }

    /** Release the speculative checkpoint after a successful commit. */
    void commit_speculative(State& layer_state) {
        layer_state.speculative_pending = false;
        layer_state.speculative_local_slots = {}; layer_state.speculative_slot_ids = {};
        layer_state.speculative_partial_kv = {}; layer_state.speculative_partial_score = {};
        layer_state.speculative_projected_kv = {}; layer_state.speculative_projected_score = {};
        layer_state.speculative_position = -1;
    }

    /** Restore rejected cache writes while retaining the accepted prefix. */
    void rollback_speculative(State& layer_state, std::int64_t position) {
        TYLOINFER_RUNTIME_CHECK(layer_state.speculative_pending && position >= layer_state.speculative_position &&
                              position <= layer_state.position &&
                              layer_state.speculative_local_slots.defined() && layer_state.speculative_slot_ids.defined(),
                          "DeepSeek-V4.1 speculative checkpoint is unavailable");
        const auto kept=position-layer_state.speculative_position;
        TYLOINFER_RUNTIME_CHECK(kept<=layer_state.speculative_slot_ids.numel(),"DeepSeek-V4.1 retained prefix is out of range");
        const auto rejected=layer_state.speculative_slot_ids.numel()-kept;
        if(rejected)layer_state.local_kv.dense().index_copy_(1,layer_state.speculative_slot_ids.dense().narrow(0,kept,rejected),
            layer_state.speculative_local_slots.dense().narrow(1,kept,rejected));
        if (layer_state.speculative_partial_kv.defined()) {
            layer_state.partial_kv.dense().copy_(layer_state.speculative_partial_kv.dense());
            layer_state.partial_score.dense().copy_(layer_state.speculative_partial_score.dense());
        }
        layer_state.position = position;
        layer_state.compressed_length = layer_state.speculative_compressed_length;
        layer_state.partial_length = layer_state.speculative_partial_length;
        if(kept && kv_source()) {
            const auto total=layer_state.speculative_partial_length+kept;
            layer_state.compressed_length+=total/ratio;
            layer_state.partial_length=total%ratio;
            // Complete compressed rows already contain the accepted prefix.
            // Rebuild only its unfinished group from the saved projections;
            // this needs no attention, projection or expert recomputation.
            for(int64_t slot=0;slot<layer_state.partial_length;++slot) {
                const auto source=total-layer_state.partial_length+slot-layer_state.speculative_partial_length;
                if(source<0)continue; // The original partial group was restored above.
                TYLOINFER_RUNTIME_CHECK(layer_state.speculative_projected_kv.defined() && layer_state.speculative_projected_score.defined(),
                                  "DeepSeek-V4.1 retained compressor projections are unavailable");
                layer_state.partial_kv.dense().narrow(1,slot,1).copy_(layer_state.speculative_projected_kv.dense().narrow(1,source,1));
                layer_state.partial_score.dense().narrow(1,slot,1).copy_(layer_state.speculative_projected_score.dense().narrow(1,source,1));
            }
        }
        commit_speculative(layer_state);
    }


};
} // namespace tyloi::ops
