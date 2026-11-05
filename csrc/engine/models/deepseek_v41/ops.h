#pragma once
#include "engine/backends/cuda/executor/execution.h"
#include "engine/backends/cuda/kernels/mfq_cuda_norm_ops.h"
#include "engine/backends/cuda/ops/norm/rms.h"
#include "engine/backends/cuda/ops/ffn/ffn.h"
#include "engine/backends/cuda/ops/rope/tail.h"
#include "engine/weights/moe/expert_cache.h"
#include "engine/models/common/transformer_layer.h"
#include "engine/models/deepseek_v41/causal_lm.h"

#include "engine/models/common/causal_model_ops.h"
#include "engine/models/causal_model.h"
#include "engine/models/common/prompt.h"
#include "engine/models/info.h"
#include "engine/metrics.h"
#include "engine/models/common/session/size.h"
#include "engine/backends/cuda/ops/projection/projection.h"
#include "dspark.h"
#include "engram.h"
#include "engine/backends/cuda/kernels/deepseek_v41.h"
#include "engine/backends/cuda/kernels/deepseek_v4_attention.h"
#include "engine/backends/cuda/kernels/deepseek_v4_hc.h"
#include "engine/models/deepseek_v41/config.h"
#include "engine/models/deepseek_v41/state.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace tyloi::cuda { class Pipelines; }

namespace tyloi::cuda::deepseek_v41_runtime {

class Vision;

using DenseTensor = ::tyloi::DenseTensor;
using CommonConfig = tyloi::models::deepseek_v41::Config;

// Per-request, per-layer runtime: the KV ring plus the speculative checkpoint
// and layer flags. Blocks reference their entry through SharedState, so the
// weight records stay state-free once the model/state split completes.
struct LayerState {
    DenseTensor local_kv;
    DenseTensor compressed_kv;
    DenseTensor index_k;
    DenseTensor partial_kv;
    DenseTensor partial_score;
    std::int64_t position = 0;
    std::int64_t compressed_length = 0;
    std::int64_t partial_length = 0;

    DenseTensor speculative_local_slots;
    DenseTensor speculative_slot_ids;
    DenseTensor speculative_partial_kv;
    DenseTensor speculative_partial_score;
    DenseTensor speculative_projected_kv;
    DenseTensor speculative_projected_score;
    DenseTensor current_ids;
    std::int64_t speculative_position = -1;
    std::int64_t speculative_compressed_length = 0;
    std::int64_t speculative_partial_length = 0;
    std::int64_t prefilled_global_end = -1;
    bool speculative_pending = false;
    bool replay_local_window = false;
};

struct SharedState {
    CommonConfig config;
    std::vector<LayerState> layers;
    DenseTensor compressed_kv;
    DenseTensor index_k;
    DenseTensor topk;
    DenseTensor candidate_blocks;
    DenseTensor previous_pre;
    DenseTensor attention_meta;
    std::unique_ptr<EngramHashState> engram_hash;
    EngramHashBatch engram_hashes;
    std::vector<Engram*> engram_prefetch;
    std::vector<const tyloi::cuda::Ffn*> router_prefetch;
    std::vector<DenseTensor> dspark_target_hiddens;
    bool capture_dspark_targets = false;
    DenseTensor image_mask, text_rows, routing_mask;
    std::vector<int64_t> text_row_ids;
    std::int64_t compressed_length = 0;
    std::int64_t ratio = 0;
    std::int64_t cache_position = -1;
    std::int64_t batch = 0;
    std::int64_t tokens = 0;
    std::int64_t next_layer = 0;

    void set_image_mask(DenseTensor mask) {
        if(!mask.defined()) {image_mask={};text_rows={};routing_mask={};text_row_ids.clear();return;}
        TYLOINFER_RUNTIME_CHECK(mask.is_cpu() && mask.dim()==2 && mask.is_contiguous() &&
                          mask.scalar_type()==::tyloi::kUInt8,
                          "DeepSeek-V4.1 image mask must be contiguous CPU uint8 [B,T]");
        std::vector<int64_t> images;text_row_ids.clear();
        const auto* flags=mask.data_ptr<uint8_t>();
        for(int64_t i=0;i<mask.numel();++i)(flags[i]?images:text_row_ids).push_back(i);
        if(images.empty()){set_image_mask({});return;}
        auto options=::tyloi::TensorOptions().device(::tyloi::kCUDA)
            .dtype(::tyloi::kInt64);
        auto text=::tyloi::cuda::tensor(text_row_ids,options);
        auto routing=mask.reshape({-1,1}).to(::tyloi::kCUDA).to(::tyloi::kBool);
        image_mask=std::move(mask);text_rows=std::move(text);routing_mask=std::move(routing);
    }

    void begin_forward(bool capture, std::size_t target_count) {
        capture_dspark_targets = capture;
        dspark_target_hiddens.clear();
        if (capture)
            dspark_target_hiddens.resize(target_count);
    }

    void capture_dspark_target(std::int64_t layer, const CommonConfig &config,
                               const DenseTensor &hidden) {
        if (!capture_dspark_targets)
            return;
        const auto found = std::find(config.dspark_target_layer_ids.begin(),
                                     config.dspark_target_layer_ids.end(), layer);
        if (found == config.dspark_target_layer_ids.end())
            return;
        const auto slot =
            static_cast<std::size_t>(std::distance(config.dspark_target_layer_ids.begin(), found));
        TYLOINFER_RUNTIME_CHECK(slot < dspark_target_hiddens.size() && hidden.dim() == 4 &&
                              hidden.size(2) == config.hc_mult && hidden.size(3) == config.hidden,
                          "DeepSeek-V4.1 DSpark target geometry mismatch");
        dspark_target_hiddens[slot] = hidden.mean(2).contiguous();
    }

    DenseTensor dspark_target_hidden() const {
        TYLOINFER_RUNTIME_CHECK(capture_dspark_targets && !dspark_target_hiddens.empty() &&
                              std::all_of(dspark_target_hiddens.begin(),
                                          dspark_target_hiddens.end(),
                                          [](const DenseTensor &value) { return value.defined(); }),
                          "DeepSeek-V4.1 DSpark target layers were not fully captured");
        return dspark_target_hiddens.size() == 1
                   ? dspark_target_hiddens.front()
                   : ::tyloi::cuda::cat(dspark_target_hiddens, -1).contiguous();
    }

    void begin_speculative() {
        TYLOINFER_RUNTIME_CHECK(engram_hash, "DeepSeek-V4.1 Engram state is unavailable");
        engram_hash->begin_speculative();
    }

    void commit_speculative() noexcept {
        if (engram_hash)
            engram_hash->commit_speculative();
    }

    void rollback_speculative(int64_t position) {
        TYLOINFER_RUNTIME_CHECK(engram_hash, "DeepSeek-V4.1 Engram state is unavailable");
        engram_hash->rollback_speculative(position);
    }

    void reset_session(std::int64_t current_batch) {
        compressed_kv = DenseTensor();
        index_k = DenseTensor();
        topk = DenseTensor();
        candidate_blocks = DenseTensor();
        previous_pre = DenseTensor();
        engram_hashes = {};
        dspark_target_hiddens.clear();
        capture_dspark_targets = false;
        compressed_length = 0;
        ratio = 0;
        cache_position = -1;
        batch = 0;
        tokens = 0;
        next_layer = 0;
        TYLOINFER_RUNTIME_CHECK(engram_hash && current_batch > 0 &&
                              current_batch <= std::numeric_limits<int>::max(),
                          "DeepSeek-V4.1 Engram state is unavailable");
        engram_hash->reset(static_cast<int>(current_batch));
    }

    void enter_layer(std::int64_t layer, std::int64_t layer_count, std::int64_t current_batch,
                     std::int64_t current_tokens, std::int64_t current_position,
                     const ::tyloi::Device &device, const DenseTensor &token_ids) {
        if (layer == 0) {
            compressed_kv = DenseTensor();
            index_k = DenseTensor();
            topk = DenseTensor();
            candidate_blocks = DenseTensor();
            compressed_length = 0;
            ratio = 0;
            cache_position = current_position;
            batch = current_batch;
            tokens = current_tokens;
            next_layer = 0;
            previous_pre =
                ::tyloi::cuda::zeros({current_batch, current_tokens, 4},
                                          ::tyloi::TensorOptions().device(device).dtype(
                                              ::tyloi::kFloat32));
            previous_pre.narrow(2, 0, 1).fill_(1.0);
            TYLOINFER_RUNTIME_CHECK(engram_hash && token_ids.dim() == 2 &&
                                  token_ids.size(0) == current_batch &&
                                  token_ids.size(1) == current_tokens,
                              "DeepSeek-V4.1 Engram token state is unavailable");
            TYLOINFER_RUNTIME_CHECK(!image_mask.defined() || image_mask.sizes()==token_ids.sizes(),
                              "DeepSeek-V4.1 image participation shape mismatch");
            engram_hashes = engram_hash->forward(token_ids, current_position,
                image_mask.defined()?image_mask.data_ptr<uint8_t>():nullptr);
        }
        TYLOINFER_RUNTIME_CHECK(layer == next_layer && layer >= 0 && layer < layer_count &&
                              current_batch == batch && current_tokens == tokens &&
                              current_position == cache_position && previous_pre.defined(),
                              "DeepSeek-V4.1 layer execution order changed");
        // Keep current/next Engram reads ahead of attention without launching
        // every layer's SSD scan together. During CED encoder prefill, decoder
        // tables will later consume only the bounded replay's hash rows.
        const auto prefetch_end=current_tokens>config.sliding_window && layer<config.causal_encoder_layers
            ?config.causal_encoder_layers:layer_count;
        int issued=0;
        for(const auto* memory:engram_prefetch) {
            if(memory->layer_id()<layer || memory->layer_id()>=prefetch_end)continue;
            memory->prefetch(engram_hashes,image_mask.defined()?&text_row_ids:nullptr);
            if(++issued==2)break;
        }
        ++next_layer;
        if (!attention_meta.defined() || attention_meta.device() != device) {
            attention_meta = ::tyloi::cuda::empty(
                {8 * 1024 * 1024}, ::tyloi::TensorOptions().device(device).dtype(
                                       ::tyloi::kFloat32));
        }
    }

    DenseTensor final_collapse(const DenseTensor &hidden, std::int64_t layer_count) const {
        TYLOINFER_RUNTIME_CHECK(hidden.dim() == 4 && hidden.size(2) == 4 && previous_pre.defined() &&
                              previous_pre.sizes() == hidden.sizes().slice(0, 3) &&
                              next_layer == layer_count,
                          "DeepSeek-V4.1 final Mega-mHC state is incomplete");
        return (previous_pre.to(::tyloi::kFloat32).unsqueeze(-1) *
                hidden.to(::tyloi::kFloat32))
            .sum(2)
            .to(::tyloi::kFloat16)
            .contiguous();
    }
};

struct AttentionState {
    DenseTensor local_kv;
    DenseTensor compressed_kv;
    DenseTensor index_k;
    DenseTensor partial_kv;
    DenseTensor partial_score;
    std::int64_t position = 0;
    std::int64_t compressed_length = 0;
    std::int64_t partial_length = 0;
};

inline DenseTensor weighted_rms(const DenseTensor &input, const DenseTensor &weight, double epsilon) {
    const auto shape = input.sizes().vec();
    TYLOINFER_RUNTIME_CHECK(!shape.empty() && weight.numel() == shape.back(),
                      "DeepSeek-V4.1 weighted RMS geometry mismatch");
    return rms_norm_cuda(input.reshape({-1, shape.back()}).to(::tyloi::kFloat32), weight,
                         epsilon)
        .reshape(shape)
        .to(::tyloi::kFloat16)
        .contiguous();
}

inline DenseTensor rotate_token_major_tail(DenseTensor input, const DenseTensor &positions,
                                      const TailRope &rope, bool inverse = false) {
    TYLOINFER_RUNTIME_CHECK(input.dim() == 3, "DeepSeek-V4.1 RoPE input must be [B,T,D]");
    return rope.forward(input.unsqueeze(1).contiguous(), positions, inverse)
        .squeeze(1)
        .contiguous();
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

struct MhcResult {
    DenseTensor branch;
    DenseTensor next_pre;
    DenseTensor post;
    DenseTensor combination;
};

struct Block final {
    // Placement, as the layer loop sets it for CPU-offloaded blocks.
    int cuda_device = 0;
    bool cpu_offloaded = false;

    CommonConfig config;
    std::int64_t layer = -1;
    std::int64_t ratio = 0;
    std::int64_t max_context = 0;
    std::shared_ptr<SharedState> shared;

    LayerState &L() const {
        TYLOINFER_RUNTIME_CHECK(shared && layer >= 0 &&
                              layer < std::int64_t(shared->layers.size()),
                          "DeepSeek-V4.1 layer state is not allocated");
        return shared->layers[static_cast<std::size_t>(layer)];
    }

    DenseTensor attention_norm;
    DenseTensor mlp_norm;
    DenseTensor query_a_norm;
    DenseTensor key_value_norm;
    DenseTensor sinks;
    DenseTensor attention_mhc_function;
    DenseTensor attention_mhc_scale;
    DenseTensor attention_mhc_base;
    DenseTensor mlp_mhc_function;
    DenseTensor mlp_mhc_scale;
    DenseTensor mlp_mhc_base;
    DenseTensor compressor_norm;
    DenseTensor index_key_norm;

    std::shared_ptr<const ::tyloi::Tensor> query_a;
    std::shared_ptr<const ::tyloi::Tensor> query_b;
    std::shared_ptr<const ::tyloi::Tensor> key_value;
    std::shared_ptr<const ::tyloi::Tensor> output_a;
    std::shared_ptr<const ::tyloi::Tensor> output_b;
    std::shared_ptr<const ::tyloi::Tensor> compressor_key_value;
    std::shared_ptr<const ::tyloi::Tensor> compressor_gate;
    std::shared_ptr<const ::tyloi::Tensor> index_query;
    std::shared_ptr<const ::tyloi::Tensor> index_key;
    std::shared_ptr<const ::tyloi::Tensor> index_score;
    tyloi::cuda::Ffn mlp;
    DenseTensor vision_router_bias;
    TailRope rope;
    std::unique_ptr<Engram> engram;

    bool kv_source() const noexcept { return config.is_kv_source(layer); }

    bool index_source() const noexcept { return config.is_index_source(layer); }

    void reset(std::int64_t batch) {
        const auto half = ::tyloi::TensorOptions()
                              .device(::tyloi::kCUDA)
                              .dtype(::tyloi::kFloat16);
        L().local_kv =
            ::tyloi::cuda::zeros({batch, config.sliding_window, config.head_dim}, half);
        L().compressed_kv = DenseTensor();
        L().index_k = DenseTensor();
        L().partial_kv = DenseTensor();
        L().partial_score = DenseTensor();
        if (kv_source()) {
            const auto capacity = std::max<std::int64_t>(1, max_context / ratio);
            L().compressed_kv =
                ::tyloi::cuda::zeros({batch, capacity, config.head_dim}, half);
            L().index_k =
                ::tyloi::cuda::zeros({batch, capacity, config.index_head_dim}, half);
            if (ratio > 1) {
                const auto fp32 = half.dtype(::tyloi::kFloat32);
                L().partial_kv = ::tyloi::cuda::zeros({batch, ratio, config.head_dim}, fp32);
                L().partial_score = ::tyloi::cuda::full(
                    {batch, ratio, config.head_dim}, -std::numeric_limits<float>::infinity(), fp32);
            }
        }
        L().position = 0;
        L().compressed_length = 0;
        L().partial_length = 0;
        L().speculative_local_slots = DenseTensor();
        L().speculative_slot_ids = DenseTensor();
        L().speculative_partial_kv = DenseTensor();
        L().speculative_partial_score = DenseTensor();
        L().speculative_projected_kv = DenseTensor();
        L().speculative_projected_score = DenseTensor();
        L().speculative_position = -1;
        L().speculative_pending = false;
        L().prefilled_global_end = -1;
        L().replay_local_window = false;
        if (layer == 0)
            shared->reset_session(batch);
    }

    bool supports_speculation() const noexcept { return true; }

    void begin_speculative(std::int64_t draft_tokens) {
        TYLOINFER_RUNTIME_CHECK(!L().speculative_pending && draft_tokens > 0 &&
                              draft_tokens <= config.sliding_window && L().local_kv.defined(),
                          "invalid DeepSeek-V4.1 speculative checkpoint");
        L().speculative_position = L().position;
        L().speculative_compressed_length = L().compressed_length;
        L().speculative_partial_length = L().partial_length;
        L().speculative_slot_ids =
            ::tyloi::cuda::arange(L().position, L().position + draft_tokens,
                                       L().local_kv.options().dtype(::tyloi::kInt64))
                .remainder(static_cast<double>(config.sliding_window))
                .to(::tyloi::kInt64)
                .contiguous();
        L().speculative_local_slots = L().local_kv.index_select(1, L().speculative_slot_ids).clone();
        if (L().partial_kv.defined()) {
            L().speculative_partial_kv = L().partial_kv.clone();
            L().speculative_partial_score = L().partial_score.clone();
        }
        L().speculative_pending = true;
    }

    void commit_speculative() {
        L().speculative_pending = false;
        L().speculative_local_slots = DenseTensor();
        L().speculative_slot_ids = DenseTensor();
        L().speculative_partial_kv = DenseTensor();
        L().speculative_partial_score = DenseTensor();
        L().speculative_projected_kv = DenseTensor();
        L().speculative_projected_score = DenseTensor();
        L().speculative_position = -1;
    }

    void rollback_speculative(std::int64_t position) {
        TYLOINFER_RUNTIME_CHECK(L().speculative_pending && position >= L().speculative_position &&
                              position <= L().position &&
                              L().speculative_local_slots.defined() && L().speculative_slot_ids.defined(),
                          "DeepSeek-V4.1 speculative checkpoint is unavailable");
        const auto kept=position-L().speculative_position;
        TYLOINFER_RUNTIME_CHECK(kept<=L().speculative_slot_ids.numel(),"DeepSeek-V4.1 retained prefix is out of range");
        const auto rejected=L().speculative_slot_ids.numel()-kept;
        if(rejected)L().local_kv.index_copy_(1,L().speculative_slot_ids.narrow(0,kept,rejected),
            L().speculative_local_slots.narrow(1,kept,rejected));
        if (L().speculative_partial_kv.defined()) {
            L().partial_kv.copy_(L().speculative_partial_kv);
            L().partial_score.copy_(L().speculative_partial_score);
        }
        L().position = position;
        L().compressed_length = L().speculative_compressed_length;
        L().partial_length = L().speculative_partial_length;
        if(kept && kv_source()) {
            const auto total=L().speculative_partial_length+kept;
            L().compressed_length+=total/ratio;
            L().partial_length=total%ratio;
            // Complete compressed rows already contain the accepted prefix.
            // Rebuild only its unfinished group from the saved projections;
            // this needs no attention, projection or expert recomputation.
            for(int64_t slot=0;slot<L().partial_length;++slot) {
                const auto source=total-L().partial_length+slot-L().speculative_partial_length;
                if(source<0)continue; // The original partial group was restored above.
                TYLOINFER_RUNTIME_CHECK(L().speculative_projected_kv.defined() && L().speculative_projected_score.defined(),
                                  "DeepSeek-V4.1 retained compressor projections are unavailable");
                L().partial_kv.narrow(1,slot,1).copy_(L().speculative_projected_kv.narrow(1,source,1));
                L().partial_score.narrow(1,slot,1).copy_(L().speculative_projected_score.narrow(1,source,1));
            }
        }
        commit_speculative();
    }

    void set_token_ids(const DenseTensor &ids) { L().current_ids = ids; }
    // No dense warmup state: the recurrent cache lives in AttentionState.
    std::vector<DenseTensor*> graph_warmup_state() { return {}; }

    MhcResult collapse(CudaProfiler &profiler, const DenseTensor &residual, const DenseTensor &previous_pre,
                       const DenseTensor &function, const DenseTensor &scale, const DenseTensor &base,
                       const DenseTensor &norm, const char *profile) const {
        return profiler.measure(profile, [&]() {
            if(residual.size(0)*residual.size(1)<=8) {
                auto partials=deepseek_v41_hc_project_cuda(residual,previous_pre,function);
                auto result=deepseek_v41_hc_finalize_cuda(residual,previous_pre,norm,partials,scale,base,
                    config.hc_sinkhorn_iters,config.rms_eps,config.hc_eps);
                return MhcResult{std::move(result[0]),std::move(result[1]),std::move(result[2]),std::move(result[3])};
            }
            auto prepared = deepseek_v41_hc_prepare_cuda(residual, previous_pre, norm, config.rms_eps);
            auto mixes = ::tyloi::cuda::matmul(prepared.at(0), function.transpose(0, 1));
            auto coefficients = dsv4_hc_coefficients_cuda(mixes.contiguous(), scale, base,
                                                          config.hc_sinkhorn_iters, config.hc_eps);
            return MhcResult{std::move(prepared.at(1)), std::move(coefficients.at(0)),
                             std::move(coefficients.at(1)), std::move(coefficients.at(2))};
        });
    }

    DenseTensor expand(CudaProfiler &profiler, const DenseTensor &branch, const DenseTensor &residual,
                  const MhcResult &mix, const char *profile) const {
        return profiler.measure(profile, [&]() {
            return dsv4_hc_post_cuda(branch.contiguous(), residual.contiguous(),
                                     mix.post.contiguous(), mix.combination.contiguous());
        });
    }

    DenseTensor output_projection(CudaExecutionContext &execution, DenseTensor attention) const {
        return execution.executor().projector(execution.executor().mode().ffn, model_parallel_primary_device(execution.config)).low_rank(attention, *output_a, *output_b,
                                               config.o_groups, true, false);
    }

    std::optional<DenseTensor> update_compressed_source(CudaExecutionContext &execution,
                                                   const DenseTensor &input, std::int64_t position) {
        const auto batch = input.size(0);
        const auto tokens = input.size(1);
        DenseTensor projected_kv, projected_score;
        return tyloi::models::deepseek_v41::compress(
            L(), tokens, position, ratio,
            [&] {
                return weighted_rms(execution.executor().projector(execution.executor().mode().ffn, model_parallel_primary_device(execution.config)).run(*compressor_key_value, input), compressor_norm,
                                    config.rms_eps);
            },
            [&] {
                projected_kv = execution.executor().projector(execution.executor().mode().ffn, model_parallel_primary_device(execution.config)).run(*compressor_key_value, input)
                                   .to(::tyloi::kFloat32)
                                   .contiguous();
                projected_score = execution.executor().projector(execution.executor().mode().ffn, model_parallel_primary_device(execution.config)).run(*compressor_gate, input)
                                      .to(::tyloi::kFloat32)
                                      .contiguous();
                if(L().speculative_pending) {
                    L().speculative_projected_kv=projected_kv;
                    L().speculative_projected_score=projected_score;
                }
            },
            [&](int64_t complete, int64_t cutoff) {
                auto kv_groups = projected_kv.narrow(1, 0, cutoff)
                                     .reshape({batch, complete, ratio, config.head_dim});
                auto score_groups = projected_score.narrow(1, 0, cutoff)
                                        .reshape({batch, complete, ratio, config.head_dim});
                auto pooled = (kv_groups * ::tyloi::cuda::softmax(score_groups, 2)).sum(2);
                return weighted_rms(pooled.to(::tyloi::kFloat16), compressor_norm,
                                    config.rms_eps);
            },
            [&](int64_t cutoff, int64_t remainder) {
                L().partial_kv.narrow(1, 0, remainder)
                    .copy_(projected_kv.narrow(1, cutoff, remainder));
                L().partial_score.narrow(1, 0, remainder)
                    .copy_(projected_score.narrow(1, cutoff, remainder));
            },
            [&](int64_t token, int64_t slot) {
                L().partial_kv.narrow(1, slot, 1).copy_(projected_kv.narrow(1, token, 1));
                L().partial_score.narrow(1, slot, 1).copy_(projected_score.narrow(1, token, 1));
            },
            [&] {
                auto pooled =
                    (L().partial_kv * ::tyloi::cuda::softmax(L().partial_score, 1))
                        .sum(1, true);
                return weighted_rms(pooled.to(::tyloi::kFloat16), compressor_norm,
                                    config.rms_eps);
            },
            [](const auto &emitted) { return ::tyloi::cuda::cat(emitted, 1).contiguous(); });
    }

    void publish_compressed_source(CudaExecutionContext &execution, const DenseTensor &input,
                                   std::int64_t position) {
        std::optional<DenseTensor> latent;
        if (L().prefilled_global_end == position + input.size(1)) L().prefilled_global_end = -1;
        else latent = update_compressed_source(execution, input, position);
        if (latent) {
            const auto count = latent->size(1);
            const auto begin = L().compressed_length;
            TYLOINFER_RUNTIME_CHECK(begin + count <= L().compressed_kv.size(1),
                              "DeepSeek-V4.1 compressed cache capacity exceeded");
            auto group_positions =
                ::tyloi::cuda::arange(begin * ratio, (begin + count) * ratio, ratio,
                                           latent->options().dtype(::tyloi::kInt64));
            auto index_values =
                weighted_rms(execution.executor().projector(execution.executor().mode().ffn, model_parallel_primary_device(execution.config)).run(*index_key, *latent), index_key_norm, config.rms_eps);
            index_values = rotate_token_major_tail(index_values, group_positions, rope);
            index_values = dsv4_fp4_sim_cuda(index_values.contiguous());
            auto compressed = rotate_token_major_tail(*latent, group_positions, rope);
            compressed = deepseek_v41_mxfp4_e4m3_scale_sim_cuda(compressed.contiguous());
            auto rows = ::tyloi::cuda::arange(
                begin, begin + count, latent->options().dtype(::tyloi::kInt64));
            L().index_k.index_copy_(1, rows, index_values);
            L().compressed_kv.index_copy_(1, rows, compressed);
            L().compressed_length += count;
        }
        shared->compressed_kv = L().compressed_kv;
        shared->index_k = L().index_k;
        shared->compressed_length = L().compressed_length;
        shared->ratio = ratio;
    }

    DenseTensor compute_topk(CudaExecutionContext &execution, const DenseTensor &input, const DenseTensor &q_rank,
                        const DenseTensor &positions, std::int64_t position) {
        const auto batch = input.size(0);
        const auto tokens = input.size(1);
        const auto pool_length = shared->compressed_length;
        DenseTensor candidates;
        // Restrict scoring only when the candidate budget excludes some KV.
        // An all-inclusive list adds scattered loads without reducing work.
        const bool restricted = pool_length > config.candidate_topk_blocks * config.candidate_block_size;
        return tyloi::models::deepseek_v41::hierarchical_index(
            ratio, index_source(), layer, config.candidate_source_layer, pool_length,
            [&] { return empty_topk(batch, tokens, input.device()); },
            [&] {
                TYLOINFER_RUNTIME_CHECK(shared->topk.defined(),
                                  "DeepSeek-V4.1 CSA2 consumer has no published index plan");
                return shared->topk;
            },
            [&] {
                auto query =
                    execution.executor().projector(execution.executor().mode().ffn, model_parallel_primary_device(execution.config)).run(*index_query, q_rank)
                        .reshape({batch, tokens, config.index_n_heads, config.index_head_dim})
                        .transpose(1, 2)
                        .contiguous();
                query = rope.forward(query, positions, false)
                            .transpose(1, 2)
                            .contiguous();
                query = dsv4_fp4_sim_cuda(query.to(::tyloi::kFloat16).contiguous());
                auto weights = execution.executor().projector(execution.executor().mode().ffn, model_parallel_primary_device(execution.config)).run(*index_score, input)
                                   .reshape({batch, tokens, config.index_n_heads})
                                   .to(::tyloi::kFloat16)
                                   .contiguous();
                auto keys = shared->index_k.narrow(1, 0, pool_length).contiguous();
                if (layer > config.candidate_source_layer && restricted) {
                    TYLOINFER_RUNTIME_CHECK(shared->candidate_blocks.defined(),
                                      "DeepSeek-V4.1 hierarchical indexer has no candidates");
                    candidates = candidate_positions(shared->candidate_blocks,
                                                     config.candidate_block_size, pool_length);
                    return execution.executor().profiler().measure("deepseek_v41.indexer.candidate_scores", [&] {
                        return dsv4_indexer_candidate_scores_cuda(query, keys, weights, candidates,
                                                                  position, ratio);
                    });
                }
                return execution.executor().profiler().measure("deepseek_v41.indexer.full_scores", [&] {
                    return dsv4_indexer_scores_cuda(query, keys, weights, position, ratio);
                });
            },
            [&](DenseTensor scores) {
                if (restricted) shared->candidate_blocks = candidate_blocks_from_scores(
                    scores, ratio, position, config.candidate_topk_blocks,
                    config.candidate_block_size);
            },
            [&](DenseTensor scores) {
                if (!restricted) return select_topk(scores, config.index_topk, pool_length);
                TYLOINFER_RUNTIME_CHECK(candidates.defined(), "DeepSeek-V4.1 candidate scores are unavailable");
                return select_topk(scores, config.index_topk, pool_length, candidates);
            },
            [&](DenseTensor scores) { return select_topk(scores, config.index_topk, pool_length); },
            [&](DenseTensor topk) { shared->topk = std::move(topk); });
    }

    DenseTensor attention_forward(CudaExecutionContext &execution, const DenseTensor &input,
                             const DenseTensor &positions, std::int64_t position,
                             const MfqOptional<DenseTensor> &sequence_lengths,
                             bool truncated_local = false) {
        const auto batch = input.size(0);
        const auto tokens = input.size(1);
        const auto local_history = truncated_local ? 0 : std::min(position, config.sliding_window);
        return tyloi::models::deepseek_v41::attention(
            kv_source(), ratio,
            [&] {
                return weighted_rms(execution.executor().projector(execution.executor().mode().ffn, model_parallel_primary_device(execution.config)).run(*query_a, input), query_a_norm,
                                    config.rms_eps);
            },
            [&](DenseTensor q_rank) {
                auto query = execution.executor().projector(execution.executor().mode().ffn, model_parallel_primary_device(execution.config)).run(*query_b, q_rank)
                                 .reshape({batch, tokens, config.n_heads, config.head_dim})
                                 .transpose(1, 2)
                                 .contiguous()
                                 .to(::tyloi::kFloat32);
                query = rope.forward(query, positions, false);

                return query;
            },
            [&] {
                auto local = weighted_rms(execution.executor().projector(execution.executor().mode().ffn, model_parallel_primary_device(execution.config)).run(*key_value, input), key_value_norm,
                                          config.rms_eps)
                                 .reshape({batch, tokens, config.head_dim});
                local = rotate_token_major_tail(local, positions, rope);
                local = deepseek_v41_mxfp8_e4m3_sim_cuda(local.contiguous());

                return local;
            },
            [&] { publish_compressed_source(execution, input, position); },
            [&] {
                TYLOINFER_RUNTIME_CHECK(shared->compressed_kv.defined() && shared->index_k.defined() &&
                                      shared->ratio == ratio,
                                  "DeepSeek-V4.1 CSA2 consumer ran before its source layer");
            },
            [&](DenseTensor q_rank) {
                return compute_topk(execution, input, q_rank, positions, position);
            },
            [&](DenseTensor local) {
                DenseTensor local_for_attention;
                if (tokens == 1) {
                    auto slot = ::tyloi::cuda::full(
                        {1}, position % config.sliding_window,
                        positions.options().dtype(::tyloi::kInt64));
                    L().local_kv.index_copy_(1, slot, local);
                    local_for_attention = L().local_kv;
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
                        local_parts.push_back(L().local_kv.index_select(1, slots));
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
                    L().local_kv.index_copy_(1, recent_slots,
                                               local.narrow(1, tokens - recent, recent));
                }

                const auto pool_length = ratio > 0 ? shared->compressed_length : 0;
                auto cache =
                    pool_length > 0
                        ? ::tyloi::cuda::cat({local_for_attention,
                                                   shared->compressed_kv.narrow(1, 0, pool_length)},
                                                  1)
                              .contiguous()
                        : local_for_attention.contiguous();
                return cache;
            },
            [&](DenseTensor topk) {
                const auto pool_length = ratio > 0 ? shared->compressed_length : 0;
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
            },
            [&](DenseTensor query, DenseTensor cache, const auto &plan) {
                return attention_dsv4_sparse_cuda(
                    query, cache, plan.at(0), plan.at(1), sinks, shared->attention_meta,
                    1.0 / std::sqrt(static_cast<double>(config.head_dim)));
            },
            [&](DenseTensor attended) {
                return rope.forward(attended.transpose(1, 2).contiguous(), positions, true)
                    .transpose(1, 2)
                    .contiguous();
            },
            [&] { L().position += tokens; },
            [&](DenseTensor attended) { return output_projection(execution, attended); });
    }

    DenseTensor forward_context(CudaExecutionContext& execution,DenseTensor hidden,
            const BlockContext& context,const RopeCache& rope) {
        if(context.confirmed_prefix>0) {
            // A joint pass writes pending and draft rows. Protect their whole
            // physical window; rollback retains the confirmed prefix as well.
            begin_speculative(hidden.size(1));
            if(layer==0)shared->begin_speculative();
        }
        TYLOINFER_RUNTIME_CHECK(context.confirmed_prefix == 0 || supports_speculation(),
                              "block does not support speculative verification");
        return forward(execution, std::move(hidden), context.positions, context.cache_position,
                       context.sequence_lengths, rope, context.cache_positions,
                       context.attention_mask);
    }

    DenseTensor forward(CudaExecutionContext &execution, DenseTensor hidden, DenseTensor positions,
                   std::int64_t cache_position, const MfqOptional<DenseTensor> &sequence_lengths,
                   const RopeCache &, const MfqOptional<DenseTensor> &cache_positions = mfq_nullopt,
                   const MfqOptional<DenseTensor> &attention_mask = mfq_nullopt) {
        auto &profiler = execution.executor().profiler();
        TYLOINFER_RUNTIME_CHECK(L().current_ids.defined() && hidden.dim() == 4 &&
                              hidden.size(2) == config.hc_mult && hidden.size(3) == config.hidden &&
                              positions.dim() == 1 && positions.numel() == hidden.size(1) &&
                              !cache_positions.has_value() && !attention_mask.has_value() &&
                              L().position == cache_position,
                          "DeepSeek-V4.1 block input/cache mismatch");
        shared->enter_layer(layer, config.n_layers, hidden.size(0), hidden.size(1), cache_position,
                            hidden.device(), L().current_ids);

        const auto batch = hidden.size(0), tokens = hidden.size(1);
        const bool truncated_local = std::exchange(L().replay_local_window, false);
        return tyloi::models::deepseek_v41::decoder_layer(
            std::move(hidden), bool(engram), shared->previous_pre,
            [&](DenseTensor value) {
                return profiler.measure("deepseek_v41.engram", [&] {
                    return engram->forward(execution.executor().projector(execution.executor().mode().ffn, model_parallel_primary_device(execution.config)), profiler, value, shared->engram_hashes,
                                          shared->text_rows,&shared->text_row_ids);
                });
            },
            [&](const DenseTensor &value) { shared->capture_dspark_target(layer, config, value); },
            [&](DenseTensor value, const DenseTensor &previous_pre, int branch) {
                return collapse(profiler, value, previous_pre,
                                branch == 0 ? attention_mhc_function : mlp_mhc_function,
                                branch == 0 ? attention_mhc_scale : mlp_mhc_scale,
                                branch == 0 ? attention_mhc_base : mlp_mhc_base,
                                branch == 0 ? attention_norm : mlp_norm,
                                branch == 0 ? "deepseek_v41.mhc.attention.collapse"
                                            : "deepseek_v41.mhc.mlp.collapse");
            },
            [&](DenseTensor value) {
                return profiler.measure("deepseek_v41.attention", [&] {
                    return attention_forward(execution, value, positions, cache_position,
                                             sequence_lengths, truncated_local);
                });
            },
            [&](DenseTensor value, DenseTensor residual, const auto &mix, int branch) {
                return expand(profiler, value, residual, mix,
                              branch == 0 ? "deepseek_v41.mhc.attention.expand"
                                          : "deepseek_v41.mhc.mlp.expand");
            },
            [&](DenseTensor value) {
                return profiler.measure("deepseek_v41.mlp", [&] {
                    if(shared->image_mask.defined()) {
                        TYLOINFER_RUNTIME_CHECK(vision_router_bias.defined(),"DeepSeek-V4.1 visual router bias is missing");
                        auto bias=::tyloi::cuda::where(shared->routing_mask,vision_router_bias,mlp.moe_router_bias).contiguous();
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                        return mlp.forward(
                            value.reshape({-1,config.hidden}), execution.executor().forward_mode(execution.config.layer_placement.primary_device()).ffn, L().current_ids, bias)
#else
                        auto executor = execution.executor().fork(model_parallel_primary_device(execution.config));
                        return mlp.forward(
                            value.reshape({-1,config.hidden}), *executor, execution.executor().forward_mode(execution.config.layer_placement.primary_device()).ffn, L().current_ids, bias)
#endif
                            .reshape({batch,tokens,config.hidden});
                    }
                    if(layer+1<static_cast<int64_t>(shared->router_prefetch.size()))
                        if(const auto* next=shared->router_prefetch[layer+1]) {
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
                            auto executor = execution.executor().fork(model_parallel_primary_device(execution.config));
#endif
                            next->prefetch(
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                                value, execution.executor().forward_mode(execution.config.layer_placement.primary_device()).ffn,
#else
                                value, *executor, execution.executor().forward_mode(execution.config.layer_placement.primary_device()).ffn,
#endif
                                moe_expert_cache_ssd_router_prefetch(
                                    execution.weights.moe),
                                L().current_ids);
                        }
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                    return mlp.forward(
                            value.reshape({-1, config.hidden}), execution.executor().forward_mode(execution.config.layer_placement.primary_device()).ffn, L().current_ids)
#else
                    auto executor = execution.executor().fork(model_parallel_primary_device(execution.config));
                    return mlp.forward(
                            value.reshape({-1, config.hidden}), *executor, execution.executor().forward_mode(execution.config.layer_placement.primary_device()).ffn, L().current_ids)
#endif
                        .reshape({batch, tokens, config.hidden});
                });
            });
    }
};

tyloi::cuda::Ffn load_moe_at(Pipelines& pipelines, const weight_loader::Loader& weights,
                const CommonConfig &config, const std::string &prefix, std::int64_t layer,
                std::int64_t top_k);

std::unique_ptr<Block> load_block(Pipelines& pipelines, const weight_loader::Loader& weights,
                                    std::int64_t layer, const std::shared_ptr<SharedState> &shared, std::size_t engram_cache_rows);
void validate_load_options(const CudaExecutionContext &execution);
std::shared_ptr<SharedState> load_shared_state(const tyloi::source::Source &source,
                                               const CommonConfig &config);
DenseTensor finalize_hidden(const std::shared_ptr<SharedState> &state, const DenseTensor &hidden,
                       CudaProfiler &profiler);

inline int run_self_check() {
    EngramHashState::self_check();
    run_dspark_self_check();
    auto options = ::tyloi::TensorOptions()
                       .device(::tyloi::kCUDA)
                       .dtype(::tyloi::kInt32);
    auto blocks =
        ::tyloi::cuda::tensor(std::vector<std::int32_t>{0, 2, -1}, options).reshape({1, 1, 3});
    auto positions = candidate_positions(blocks, 2, 5)
                         .to(::tyloi::kCPU, ::tyloi::kInt64)
                         .contiguous();
    const std::vector<std::int64_t> expected{0, 1, 4, -1, -1, -1};
    const auto *values = positions.data_ptr<std::int64_t>();
    TYLOINFER_RUNTIME_CHECK(positions.numel() == static_cast<std::int64_t>(expected.size()) &&
                          std::equal(expected.begin(), expected.end(), values),
                      "DeepSeek-V4.1 candidate hierarchy contract failed");
    std::cout << "deepseek_v41_runtime_check=ok\n";
    return 0;
}

} // namespace tyloi::cuda::deepseek_v41_runtime

namespace tyloi::cuda {

namespace deepseek_v41 {

tyloi::models::SnapshotSize snapshot_size(const Model&, const MtpModule*);

// The model record: weights, placement and load-time state. Request data
// lives in State; blocks reference the shared scratch through `shared`.
struct Model : CausalResources, tyloi::models::CausalModel {
    using CausalConfig = tyloi::models::deepseek_v41::Config;
    static bool accepts_backbone(std::string_view backbone) { return backbone == "deepseek_v41"; }

    // ---- tyloi::models::CausalModel
    tyloi::models::Info info() const override;
    MfqModelCapabilities capabilities() const override;
    tyloi::engine::Metrics metrics() const override;
    std::unique_ptr<tyloi::models::State> make_state() const override;
    tyloi::StepSequence<tyloi::models::Prepared> prepare(const std::vector<std::int64_t>& tokens,
        const MfqMultimodalInput& media, tyloi::models::State& state) const override;
    std::shared_ptr<tyloi::Tensor> embedding(const tyloi::Tensor& ids, int primary = 0) const override;
    std::shared_ptr<tyloi::Tensor> normalize(const tyloi::Tensor& hidden, std::int64_t batch,
        std::int64_t tokens) const override;
    std::shared_ptr<tyloi::Tensor> softcap(const tyloi::Tensor& logits) const override;
    std::shared_ptr<tyloi::Tensor> logits(const tyloi::Tensor& hidden, bool capped = true) const override;
    std::shared_ptr<tyloi::Tensor> forward(tyloi::models::Input input, tyloi::models::State& state,
                                 const tyloi::models::ForwardOptions& options = {}) const override;

    // ---- speculative contract used by the MTP sequence
    bool supports_speculation() const;
    bool supports_suffix_speculation() const noexcept { return true; }
    ::tyloi::DenseTensor hidden_forward_speculative_suffix(::tyloi::DenseTensor ids,
                                                           ::tyloi::DenseTensor* raw_hidden = nullptr);
    void rollback_speculative(std::int64_t accepted);
    bool adapter_prefetch_token_rows(Model& model, const ::tyloi::DenseTensor& ids);

    // ---- weights and load-time state
    std::shared_ptr<const tyloi::ModelSource> source;
    tyloi::ModelGraph graph;
    CudaModelPlan plan;
    RopeCache rope;
    RopeCache cpu_rope;
    std::unordered_map<int, RopeCache> device_ropes;
    std::shared_ptr<const ::tyloi::Tensor> embed;
    ::tyloi::DenseTensor output_norm;
    std::shared_ptr<const ::tyloi::Tensor> lm_head;
    CausalConfig config;
    std::vector<std::unique_ptr<deepseek_v41_runtime::Block>> blocks;
    // shortcut: scratch shared with the active request State; split into
    // load-once tables plus request fields when requests stop being single-flight
    std::shared_ptr<deepseek_v41_runtime::SharedState> shared;
    State* active = nullptr;
    std::unique_ptr<deepseek_v41_runtime::Vision> vision;
    tyloi::cuda::MediaCache media;

    // ---- helpers absorbed from the retired CausalLm plumbing
    auto execution_scope() const {
        return CudaScope(execution->config.layer_placement.primary_device(), execution->executor().contexts());
    }
    ::tyloi::DenseTensor device_ids(::tyloi::DenseTensor ids) const {
        return execution->executor().collectives()
            .copy(ids.to(::tyloi::kInt64), execution->config.layer_placement.primary_device())
            .contiguous();
    }
    ::tyloi::DenseTensor embed_tokens(::tyloi::DenseTensor ids) const {
        return execution->executor().profiler().measure(
            "model.embed", [&] { return adapter_embed(quant_embedding_lookup(*embed, ids)); });
    }
    static ::tyloi::DenseTensor softcap_apply(::tyloi::DenseTensor logits, double cap) {
        return ::tyloi::cuda::tanh(logits / cap) * cap;
    }
    std::int64_t hidden_size() const noexcept { return metadata.hidden_size; }
    std::int64_t max_position_embeddings() const noexcept { return config.max_position_embeddings; }
    std::int64_t num_hidden_layers() const noexcept { return config.n_layers; }
    bool tie_word_embeddings() const noexcept { return metadata.tie_word_embeddings; }
    template <class Loader> bool has_independent_output_weights(Loader &loader) const {
        return !tie_word_embeddings() && loader.has_weight("model.output.weight");
    }
    template <class Loader> void load_weights(Loader &loader, bool with_blocks = true) {
        constexpr auto embedding = "model.token_embedding.weight";
        constexpr auto output = "model.output.weight";
        this->embed = loader.embedding(embedding);
        loader.final_state(this->output_norm);
        this->lm_head = tie_word_embeddings() || !loader.has_weight(output)
                            ? loader.tied_output(this->embed, embedding)
                            : loader.output(output);
        if (with_blocks) {
            loader.prepare_blocks();
            this->blocks.reserve(static_cast<std::size_t>(num_hidden_layers()));
            for (int layer = 0; layer < num_hidden_layers(); ++layer)
                this->blocks.push_back(loader.block(layer, std::string(metadata.layer_types.at(static_cast<std::size_t>(layer)))));
        }
    }
    void load_config(std::string_view payload);

    // ---- family adapters (implementations in ops.cpp)
    void adapter_validate_load_options() const;
    void adapter_load_final_state(const tyloi::cuda::weight_loader::Loader& weights,
                                  ::tyloi::DenseTensor& output_norm);
    void adapter_prepare_blocks(const tyloi::cuda::weight_loader::Loader& weights);
    std::unique_ptr<deepseek_v41_runtime::Block> adapter_load_block(
        const tyloi::cuda::weight_loader::Loader& weights, int layer, int device,
        const std::string& type);
    void adapter_begin_forward(bool capture_raw_hidden) const;
    bool adapter_supports_prepared_prompt() const noexcept { return true; }
    void adapter_set_image_mask(std::shared_ptr<const tyloi::Tensor> mask) {
        shared->set_image_mask(mask ? dynamic_cast<const ::tyloi::DenseTensor&>(*mask) : ::tyloi::DenseTensor{});
    }
    ::tyloi::DenseTensor collapse_hidden(::tyloi::DenseTensor hidden, std::int64_t batch,
                                         std::int64_t tokens) const;
    ::tyloi::DenseTensor normalize_hidden(::tyloi::DenseTensor hidden,
                                          const ::tyloi::DenseTensor& output_norm, std::int64_t batch,
                                          std::int64_t tokens) const;
    ::tyloi::DenseTensor adapter_prepare_hidden(::tyloi::DenseTensor hidden, std::int64_t batch,
                                                std::int64_t tokens) const;
    ::tyloi::DenseTensor adapter_finalize_hidden(::tyloi::DenseTensor hidden,
                                                 const ::tyloi::DenseTensor& output_norm,
                                                 std::int64_t batch, std::int64_t tokens) const;
    ::tyloi::DenseTensor adapter_raw_hidden(const ::tyloi::DenseTensor& hidden,
                                            const ::tyloi::DenseTensor& finalized) const;
    void adapter_validate_forward(std::int64_t, std::int64_t tokens, std::int64_t cache_position,
                                  bool has_position_override, bool has_cache_position_override,
                                  bool has_attention_mask) const;
    void adapter_begin_speculative();
    void adapter_commit_speculative();
    void adapter_rollback_speculative(std::int64_t keep);
};

void read(Model& model, std::shared_ptr<const tyloi::ModelSource> source,
         std::string_view definition, std::int64_t context_size);
void load(Model& model, CudaExecutionContext& execution, int64_t context_size,
          bool defer_moe_cache_finalize, bool load_blocks, bool load_optional_components);

::tyloi::DenseTensor deepseek_v41_prefill_hidden(
    Model& model, State& state, ::tyloi::DenseTensor ids, std::int64_t compute_chunk,
    const std::function<bool()>& stopped = {}, ::tyloi::DenseTensor* raw_hidden = nullptr);
std::int64_t prefill_span(const Model&, std::size_t tokens,
                          std::int64_t chunk, std::size_t, std::size_t,
                          bool transformed, bool retained);

} // namespace deepseek_v41
} // namespace tyloi::cuda
