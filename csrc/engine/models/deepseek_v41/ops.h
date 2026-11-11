#pragma once
#ifndef TYLOINFER_NATIVE_CUDA
#include "engine/weights/cache.h"
#include "engine/state/media.h"
#include "engine/backends/cuda/executor/execution.h"
#endif
#include "engine/backends/cuda/kernels/mfq_cuda_norm_ops.h"
#include "engine/backends/cuda/ops/norm/rms.h"
#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/ops.h"
#include "engine/ops/csa.h"
#include "engine/ops/ffn.h"
#include "engine/ops/mhc.h"
#else
#include "engine/backends/cuda/ops/ffn/ffn.h"
#include "engine/backends/cuda/ops/rope/tail.h"
#endif
#include "engine/models/deepseek_v41/causal_lm.h"

#include "engine/models/common/capabilities.h"
#include "engine/models/causal_model.h"
#include "engine/models/common/mtp.h"
#include "engine/models/common/prompt.h"
#include "engine/models/metadata.h"
#include "engine/metrics.h"
#include "engine/models/session/size.h"
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

#ifndef TYLOINFER_NATIVE_CUDA
struct CudaExecutionContext;
#endif

namespace tyloi::cuda::deepseek_v41_runtime {

class Vision;

using DenseTensor = ::tyloi::DenseTensor;
using CommonConfig = tyloi::models::deepseek_v41::Config;
#ifdef TYLOINFER_NATIVE_CUDA
using Parameter = ::tyloi::Tensor;
#else
using Parameter = DenseTensor;
#endif

// Per-request, per-layer runtime: the KV ring, checkpoints and layer flags.
struct LayerState {
#ifdef TYLOINFER_NATIVE_CUDA
    tyloi::ops::Csa::State attention;
    ::tyloi::Tensor current_ids;
#else
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
#endif
};

struct ModelTables {
    std::shared_ptr<const EngramHashState::Parameters> engram_parameters;
    std::vector<const Engram*> engram_prefetch;
};

struct SharedState {
    const CommonConfig& config;
    const ModelTables& tables;
    SharedState(const CommonConfig& config, const ModelTables& tables);
    std::vector<LayerState> layers;
#ifdef TYLOINFER_NATIVE_CUDA
    tyloi::ops::Csa::Shared attention;
    /** Retain an in-flight causal encoder source until publication completes. */
    tyloi::Tensor ced_source{DenseTensor{}};
    /** Retain attention inputs until their numerical operation completes. */
    tyloi::Tensor attention_input{DenseTensor{}}, attention_positions{DenseTensor{}}, attention_lengths{DenseTensor{}};
#else
    DenseTensor compressed_kv;
    DenseTensor index_k;
    DenseTensor topk;
    DenseTensor candidate_blocks;
#endif
    DenseTensor previous_pre;
#ifndef TYLOINFER_NATIVE_CUDA
    DenseTensor attention_meta;
#endif
    std::unique_ptr<EngramHashState> engram_hash;
    std::vector<std::unique_ptr<Engram::State>> engram_states;
    EngramHashBatch engram_hashes;
    std::vector<DenseTensor> dspark_target_hiddens;
    bool capture_dspark_targets = false;
    DenseTensor image_mask, text_rows, routing_mask;
    std::vector<int64_t> text_row_ids;
#ifndef TYLOINFER_NATIVE_CUDA
    std::int64_t compressed_length = 0;
    std::int64_t ratio = 0;
#endif
    std::int64_t cache_position = -1;
    std::int64_t batch = 0;
    std::int64_t tokens = 0;
    std::int64_t next_layer = 0;

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
#ifdef TYLOINFER_NATIVE_CUDA
        dspark_target_hiddens[slot] = tyloi::ops::contiguous(tyloi::ops::mean(hidden,2)).dense();
#else
        dspark_target_hiddens[slot] = hidden.mean(2).contiguous();
#endif
    }

    /** Combine captured target layers on the primary device. */
    DenseTensor dspark_target_hidden(int primary = 0) const {
        TYLOINFER_RUNTIME_CHECK(capture_dspark_targets && !dspark_target_hiddens.empty() &&
                              std::all_of(dspark_target_hiddens.begin(),
                                          dspark_target_hiddens.end(),
                                          [](const DenseTensor &value) { return value.defined(); }),
                          "DeepSeek-V4.1 DSpark target layers were not fully captured");
#ifdef TYLOINFER_NATIVE_CUDA
        std::vector<tyloi::Tensor> values;
        values.reserve(dspark_target_hiddens.size());
        for (const auto& hidden : dspark_target_hiddens)
            values.push_back(tyloi::ops::copy(hidden, ::tyloi::Device(::tyloi::kCUDA, primary)));
        return values.size() == 1 ? values.front().dense() : tyloi::ops::cat(values, -1).dense();
#else
        (void)primary;
        return dspark_target_hiddens.size() == 1
                   ? dspark_target_hiddens.front()
                   : ::tyloi::cuda::cat(dspark_target_hiddens, -1).contiguous();
#endif
    }

    void reset_session(std::int64_t current_batch) {
#ifdef TYLOINFER_NATIVE_CUDA
        attention = {};
        ced_source.dense() = {};
        attention_input.dense() = {}; attention_positions.dense() = {}; attention_lengths.dense() = {};
        for(auto& engram:engram_states)if(engram)engram->clear();
#else
        compressed_kv = DenseTensor();
        index_k = DenseTensor();
        topk = DenseTensor();
        candidate_blocks = DenseTensor();
#endif
        previous_pre = DenseTensor();
        engram_hashes = {};
        dspark_target_hiddens.clear();
        capture_dspark_targets = false;
#ifndef TYLOINFER_NATIVE_CUDA
        compressed_length = 0;
        ratio = 0;
#endif
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
#ifdef TYLOINFER_NATIVE_CUDA
            attention = {};
#else
            compressed_kv = DenseTensor();
            index_k = DenseTensor();
            topk = DenseTensor();
            candidate_blocks = DenseTensor();
            compressed_length = 0;
            ratio = 0;
#endif
            cache_position = current_position;
            batch = current_batch;
            tokens = current_tokens;
            next_layer = 0;
#ifdef TYLOINFER_NATIVE_CUDA
            auto pre=tyloi::ops::zeros({current_batch,current_tokens,4},
                ::tyloi::TensorOptions().device(device).dtype(::tyloi::kFloat32));
            auto first=tyloi::ops::slice(pre,2,0,1);
            tyloi::ops::fill(first,1.0); previous_pre=pre.dense();
#else
            previous_pre =
                ::tyloi::cuda::zeros({current_batch, current_tokens, 4},
                                          ::tyloi::TensorOptions().device(device).dtype(
                                              ::tyloi::kFloat32));
            previous_pre.narrow(2, 0, 1).fill_(1.0);
#endif
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
        for(const auto* memory:tables.engram_prefetch) {
            if(memory->layer_id()<layer || memory->layer_id()>=prefetch_end)continue;
            memory->prefetch(*engram_states.at(static_cast<std::size_t>(memory->layer_id())),
                             engram_hashes,image_mask.defined()?&text_row_ids:nullptr);
            if(++issued==2)break;
        }
        ++next_layer;
#ifndef TYLOINFER_NATIVE_CUDA
        if (!attention_meta.defined() || attention_meta.device() != device) {
            attention_meta = ::tyloi::cuda::empty(
                {8 * 1024 * 1024}, ::tyloi::TensorOptions().device(device).dtype(
                                       ::tyloi::kFloat32));
        }
#endif
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

#ifndef TYLOINFER_NATIVE_CUDA
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
#endif

inline DenseTensor weighted_rms(const DenseTensor &input,
#ifdef TYLOINFER_NATIVE_CUDA
                                const ::tyloi::Tensor &weight, double epsilon) {
#else
                                const DenseTensor &weight, double epsilon) {
#endif
    const auto shape = input.sizes().vec();
    TYLOINFER_RUNTIME_CHECK(!shape.empty() && weight.numel() == shape.back(),
                      "DeepSeek-V4.1 weighted RMS geometry mismatch");
    return rms_norm_cuda(input.reshape({-1, shape.back()}).to(::tyloi::kFloat32),
#ifdef TYLOINFER_NATIVE_CUDA
                         weight.dense(),
#else
                         weight,
#endif
                         epsilon)
        .reshape(shape)
        .to(::tyloi::kFloat16)
        .contiguous();
}

inline DenseTensor rotate_token_major_tail(DenseTensor input, const DenseTensor &positions,
#ifdef TYLOINFER_NATIVE_CUDA
                                      const ::tyloi::ops::TailRopeTable &rope, bool inverse = false) {
#else
                                      const TailRope &rope, bool inverse = false) {
#endif
    TYLOINFER_RUNTIME_CHECK(input.dim() == 3, "DeepSeek-V4.1 RoPE input must be [B,T,D]");
#ifdef TYLOINFER_NATIVE_CUDA
    auto value = input.unsqueeze(1).contiguous();
    auto rotated = ::tyloi::ops::tail_rope(value, positions, rope, inverse);
    return rotated.dense().squeeze(1).contiguous();
#else
    return rope.forward(input.unsqueeze(1).contiguous(), positions, inverse)
        .squeeze(1)
        .contiguous();
#endif
}

#ifndef TYLOINFER_NATIVE_CUDA
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

#endif
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

#ifndef TYLOINFER_NATIVE_CUDA
struct MhcResult {
    DenseTensor branch;
    DenseTensor next_pre;
    DenseTensor post;
    DenseTensor combination;
};
#endif

struct Block final {
    // Placement, as the layer loop sets it for CPU-offloaded blocks.
    int cuda_device = 0;
    bool cpu_offloaded = false;

    CommonConfig config;
    std::int64_t layer = -1;
    std::int64_t ratio = 0;
    std::int64_t max_context = 0;
    Parameter attention_norm;
    Parameter mlp_norm;
#ifndef TYLOINFER_NATIVE_CUDA
    Parameter query_a_norm;
    Parameter key_value_norm;
    Parameter sinks;
#endif
    Parameter attention_mhc_function;
    Parameter attention_mhc_scale;
    Parameter attention_mhc_base;
    Parameter mlp_mhc_function;
    Parameter mlp_mhc_scale;
    Parameter mlp_mhc_base;
#ifndef TYLOINFER_NATIVE_CUDA
    Parameter compressor_norm;
    Parameter index_key_norm;

    ::tyloi::Tensor query_a;
    ::tyloi::Tensor query_b;
    ::tyloi::Tensor key_value;
    ::tyloi::Tensor output_a;
    ::tyloi::Tensor output_b;
    ::tyloi::Tensor compressor_key_value;
    ::tyloi::Tensor compressor_gate;
    ::tyloi::Tensor index_query;
    ::tyloi::Tensor index_key;
    ::tyloi::Tensor index_score;
#else
    std::unique_ptr<tyloi::ops::Csa> attention;
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    tyloi::ops::FfnWeights mlp;
#else
    tyloi::cuda::Ffn mlp;
#endif
    Parameter vision_router_bias;
#ifndef TYLOINFER_NATIVE_CUDA
    TailRope rope;
#endif
    std::unique_ptr<Engram> engram;

    bool kv_source() const noexcept { return config.is_kv_source(layer); }

    bool index_source() const noexcept { return config.is_index_source(layer); }

#ifdef TYLOINFER_NATIVE_CUDA
    bool supports_speculation() const noexcept { return true; }
#else
    /** Reset this layer's caches, reusing compatible backing storage. */
    void reset(SharedState& shared, std::int64_t batch) {
        auto& layer_state = shared.layers.at(static_cast<std::size_t>(layer));
#ifdef TYLOINFER_NATIVE_CUDA
        MfqCudaGuard guard(cuda_device);
        const ::tyloi::Device device(::tyloi::kCUDA, cuda_device);
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
            const auto capacity = std::max<std::int64_t>(1, max_context / ratio);
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
#else
        const auto half = ::tyloi::TensorOptions()
                              .device(::tyloi::kCUDA)
                              .dtype(::tyloi::kFloat16);
        layer_state.local_kv =
            ::tyloi::cuda::zeros({batch, config.sliding_window, config.head_dim}, half);
        layer_state.compressed_kv = DenseTensor();
        layer_state.index_k = DenseTensor();
        layer_state.partial_kv = DenseTensor();
        layer_state.partial_score = DenseTensor();
        if (kv_source()) {
            const auto capacity = std::max<std::int64_t>(1, max_context / ratio);
            layer_state.compressed_kv =
                ::tyloi::cuda::zeros({batch, capacity, config.head_dim}, half);
            layer_state.index_k =
                ::tyloi::cuda::zeros({batch, capacity, config.index_head_dim}, half);
            if (ratio > 1) {
                const auto fp32 = half.dtype(::tyloi::kFloat32);
                layer_state.partial_kv = ::tyloi::cuda::zeros({batch, ratio, config.head_dim}, fp32);
                layer_state.partial_score = ::tyloi::cuda::full(
                    {batch, ratio, config.head_dim}, -std::numeric_limits<float>::infinity(), fp32);
            }
        }
#endif
        layer_state.position = 0;
        layer_state.compressed_length = 0;
        layer_state.partial_length = 0;
#ifdef TYLOINFER_NATIVE_CUDA
        layer_state.speculative_local_slots = {}; layer_state.speculative_slot_ids = {};
        layer_state.speculative_partial_kv = {}; layer_state.speculative_partial_score = {};
        layer_state.speculative_projected_kv = {}; layer_state.speculative_projected_score = {};
#else
        layer_state.speculative_local_slots = DenseTensor();
        layer_state.speculative_slot_ids = DenseTensor();
        layer_state.speculative_partial_kv = DenseTensor();
        layer_state.speculative_partial_score = DenseTensor();
        layer_state.speculative_projected_kv = DenseTensor();
        layer_state.speculative_projected_score = DenseTensor();
#endif
        layer_state.speculative_position = -1;
        layer_state.speculative_pending = false;
        layer_state.prefilled_global_end = -1;
        layer_state.replay_local_window = false;
        if (layer == 0)
            shared.reset_session(batch);
    }

    bool supports_speculation() const noexcept { return true; }

    /** Save the mutable cache state before a speculative window. */
    void begin_speculative(SharedState& shared, std::int64_t draft_tokens) {
        auto& layer_state = shared.layers.at(static_cast<std::size_t>(layer));
        TYLOINFER_RUNTIME_CHECK(!layer_state.speculative_pending && draft_tokens > 0 &&
                              draft_tokens <= config.sliding_window && layer_state.local_kv.defined(),
                          "invalid DeepSeek-V4.1 speculative checkpoint");
        layer_state.speculative_position = layer_state.position;
        layer_state.speculative_compressed_length = layer_state.compressed_length;
        layer_state.speculative_partial_length = layer_state.partial_length;
        layer_state.speculative_slot_ids =
            ::tyloi::cuda::arange(layer_state.position, layer_state.position + draft_tokens,
                                       layer_state.local_kv.options().dtype(::tyloi::kInt64))
                .remainder(static_cast<double>(config.sliding_window))
                .to(::tyloi::kInt64)
                .contiguous();
#ifdef TYLOINFER_NATIVE_CUDA
        layer_state.speculative_local_slots = layer_state.local_kv.dense().index_select(1, layer_state.speculative_slot_ids.dense()).clone();
        if (layer_state.partial_kv.defined()) {
            layer_state.speculative_partial_kv = layer_state.partial_kv.dense().clone();
            layer_state.speculative_partial_score = layer_state.partial_score.dense().clone();
        }
#else
        layer_state.speculative_local_slots = layer_state.local_kv.index_select(1, layer_state.speculative_slot_ids).clone();
        if (layer_state.partial_kv.defined()) {
            layer_state.speculative_partial_kv = layer_state.partial_kv.clone();
            layer_state.speculative_partial_score = layer_state.partial_score.clone();
        }
#endif
        layer_state.speculative_pending = true;
    }

    /** Release the speculative checkpoint after a successful commit. */
    void commit_speculative(SharedState& shared) {
        auto& layer_state = shared.layers.at(static_cast<std::size_t>(layer));
        layer_state.speculative_pending = false;
#ifdef TYLOINFER_NATIVE_CUDA
        layer_state.speculative_local_slots = {}; layer_state.speculative_slot_ids = {};
        layer_state.speculative_partial_kv = {}; layer_state.speculative_partial_score = {};
        layer_state.speculative_projected_kv = {}; layer_state.speculative_projected_score = {};
#else
        layer_state.speculative_local_slots = DenseTensor();
        layer_state.speculative_slot_ids = DenseTensor();
        layer_state.speculative_partial_kv = DenseTensor();
        layer_state.speculative_partial_score = DenseTensor();
        layer_state.speculative_projected_kv = DenseTensor();
        layer_state.speculative_projected_score = DenseTensor();
#endif
        layer_state.speculative_position = -1;
    }

    /** Restore rejected cache writes while retaining the accepted prefix. */
    void rollback_speculative(SharedState& shared, std::int64_t position) {
        auto& layer_state = shared.layers.at(static_cast<std::size_t>(layer));
        TYLOINFER_RUNTIME_CHECK(layer_state.speculative_pending && position >= layer_state.speculative_position &&
                              position <= layer_state.position &&
                              layer_state.speculative_local_slots.defined() && layer_state.speculative_slot_ids.defined(),
                          "DeepSeek-V4.1 speculative checkpoint is unavailable");
        const auto kept=position-layer_state.speculative_position;
        TYLOINFER_RUNTIME_CHECK(kept<=layer_state.speculative_slot_ids.numel(),"DeepSeek-V4.1 retained prefix is out of range");
        const auto rejected=layer_state.speculative_slot_ids.numel()-kept;
#ifdef TYLOINFER_NATIVE_CUDA
        if(rejected)layer_state.local_kv.dense().index_copy_(1,layer_state.speculative_slot_ids.dense().narrow(0,kept,rejected),
            layer_state.speculative_local_slots.dense().narrow(1,kept,rejected));
        if (layer_state.speculative_partial_kv.defined()) {
            layer_state.partial_kv.dense().copy_(layer_state.speculative_partial_kv.dense());
            layer_state.partial_score.dense().copy_(layer_state.speculative_partial_score.dense());
        }
#else
        if(rejected)layer_state.local_kv.index_copy_(1,layer_state.speculative_slot_ids.narrow(0,kept,rejected),
            layer_state.speculative_local_slots.narrow(1,kept,rejected));
        if (layer_state.speculative_partial_kv.defined()) {
            layer_state.partial_kv.copy_(layer_state.speculative_partial_kv);
            layer_state.partial_score.copy_(layer_state.speculative_partial_score);
        }
#endif
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
#ifdef TYLOINFER_NATIVE_CUDA
                layer_state.partial_kv.dense().narrow(1,slot,1).copy_(layer_state.speculative_projected_kv.dense().narrow(1,source,1));
                layer_state.partial_score.dense().narrow(1,slot,1).copy_(layer_state.speculative_projected_score.dense().narrow(1,source,1));
#else
                layer_state.partial_kv.narrow(1,slot,1).copy_(layer_state.speculative_projected_kv.narrow(1,source,1));
                layer_state.partial_score.narrow(1,slot,1).copy_(layer_state.speculative_projected_score.narrow(1,source,1));
#endif
            }
        }
        commit_speculative(shared);
    }

#endif
    /** Retain token IDs for the next layer invocation. */
    void set_token_ids(SharedState& shared, const DenseTensor &ids) { shared.layers.at(static_cast<std::size_t>(layer)).current_ids = ids; }
    // No dense warmup state: the recurrent cache lives in AttentionState.
    std::vector<DenseTensor*> graph_warmup_state() { return {}; }

#ifndef TYLOINFER_NATIVE_CUDA
    MhcResult collapse(const DenseTensor &residual, const DenseTensor &previous_pre,
#ifdef TYLOINFER_NATIVE_CUDA
                       const ::tyloi::Tensor &function_tensor, const ::tyloi::Tensor &scale_tensor,
                       const ::tyloi::Tensor &base_tensor, const ::tyloi::Tensor &norm_tensor, const char *profile) const {
#else
                       const DenseTensor &function, const DenseTensor &scale, const DenseTensor &base,
                       const DenseTensor &norm, const char *profile) const {
#endif
        return tyloi::ops::measure(profile, [&]() {
#ifdef TYLOINFER_NATIVE_CUDA
            const auto& function = function_tensor.dense();
            const auto& scale = scale_tensor.dense();
            const auto& base = base_tensor.dense();
            const auto& norm = norm_tensor.dense();
#endif
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

    DenseTensor expand(const DenseTensor &branch, const DenseTensor &residual,
                  const MhcResult &mix, const char *profile) const {
        return tyloi::ops::measure(profile, [&]() {
            return dsv4_hc_post_cuda(branch.contiguous(), residual.contiguous(),
                                     mix.post.contiguous(), mix.combination.contiguous());
        });
    }

    DenseTensor output_projection(DenseTensor attention) const {
        return tyloi::ops::low_rank(attention, output_a, output_b, config.o_groups, true, false).dense();
    }

    std::optional<DenseTensor> update_compressed_source(SharedState& shared,
                                                   const DenseTensor &input, std::int64_t position) {
        auto& layer_state = shared.layers.at(static_cast<std::size_t>(layer));
        const auto batch = input.size(0);
        const auto tokens = input.size(1);
        DenseTensor projected_kv, projected_score;
        return tyloi::models::deepseek_v41::compress(
            layer_state, tokens, position, ratio,
            [&] {
                return weighted_rms(tyloi::ops::project(input, compressor_key_value).dense(),
#ifdef TYLOINFER_NATIVE_CUDA
                                    compressor_norm,
#else
                                    compressor_norm,
#endif
                                    config.rms_eps);
            },
            [&] {
                projected_kv = tyloi::ops::project(input, compressor_key_value).dense()
                                   .to(::tyloi::kFloat32)
                                   .contiguous();
                projected_score = tyloi::ops::project(input, compressor_gate).dense()
                                      .to(::tyloi::kFloat32)
                                      .contiguous();
                if(layer_state.speculative_pending) {
                    layer_state.speculative_projected_kv=projected_kv;
                    layer_state.speculative_projected_score=projected_score;
                }
            },
            [&](int64_t complete, int64_t cutoff) {
                auto kv_groups = projected_kv.narrow(1, 0, cutoff)
                                     .reshape({batch, complete, ratio, config.head_dim});
                auto score_groups = projected_score.narrow(1, 0, cutoff)
                                        .reshape({batch, complete, ratio, config.head_dim});
                auto pooled = (kv_groups * ::tyloi::cuda::softmax(score_groups, 2)).sum(2);
                return weighted_rms(pooled.to(::tyloi::kFloat16),
#ifdef TYLOINFER_NATIVE_CUDA
                                    compressor_norm,
#else
                                    compressor_norm,
#endif
                                    config.rms_eps);
            },
            [&](int64_t cutoff, int64_t remainder) {
#ifdef TYLOINFER_NATIVE_CUDA
                layer_state.partial_kv.dense().narrow(1, 0, remainder)
                    .copy_(projected_kv.narrow(1, cutoff, remainder));
                layer_state.partial_score.dense().narrow(1, 0, remainder)
                    .copy_(projected_score.narrow(1, cutoff, remainder));
#else
                layer_state.partial_kv.narrow(1, 0, remainder)
                    .copy_(projected_kv.narrow(1, cutoff, remainder));
                layer_state.partial_score.narrow(1, 0, remainder)
                    .copy_(projected_score.narrow(1, cutoff, remainder));
#endif
            },
            [&](int64_t token, int64_t slot) {
#ifdef TYLOINFER_NATIVE_CUDA
                layer_state.partial_kv.dense().narrow(1, slot, 1).copy_(projected_kv.narrow(1, token, 1));
                layer_state.partial_score.dense().narrow(1, slot, 1).copy_(projected_score.narrow(1, token, 1));
#else
                layer_state.partial_kv.narrow(1, slot, 1).copy_(projected_kv.narrow(1, token, 1));
                layer_state.partial_score.narrow(1, slot, 1).copy_(projected_score.narrow(1, token, 1));
#endif
            },
            [&] {
#ifdef TYLOINFER_NATIVE_CUDA
                auto pooled =
                    (layer_state.partial_kv.dense() * ::tyloi::cuda::softmax(layer_state.partial_score.dense(), 1))
                        .sum(1, true);
#else
                auto pooled =
                    (layer_state.partial_kv * ::tyloi::cuda::softmax(layer_state.partial_score, 1))
                        .sum(1, true);
#endif
                return weighted_rms(pooled.to(::tyloi::kFloat16),
#ifdef TYLOINFER_NATIVE_CUDA
                                    compressor_norm,
#else
                                    compressor_norm,
#endif
                                    config.rms_eps);
            },
            [](const auto &emitted) { return ::tyloi::cuda::cat(emitted, 1).contiguous(); });
    }

    void publish_compressed_source(SharedState& shared, const DenseTensor &input,
                                   std::int64_t position) {
        auto& layer_state = shared.layers.at(static_cast<std::size_t>(layer));
        std::optional<DenseTensor> latent;
        if (layer_state.prefilled_global_end == position + input.size(1)) layer_state.prefilled_global_end = -1;
        else latent = update_compressed_source(shared, input, position);
        if (latent) {
            const auto count = latent->size(1);
            const auto begin = layer_state.compressed_length;
            TYLOINFER_RUNTIME_CHECK(begin + count <= layer_state.compressed_kv.size(1),
                              "DeepSeek-V4.1 compressed cache capacity exceeded");
            auto group_positions =
                ::tyloi::cuda::arange(begin * ratio, (begin + count) * ratio, ratio,
                                           latent->options().dtype(::tyloi::kInt64));
            auto index_values =
                weighted_rms(tyloi::ops::project(*latent, index_key).dense(),
#ifdef TYLOINFER_NATIVE_CUDA
                             index_key_norm, config.rms_eps);
#else
                             index_key_norm, config.rms_eps);
#endif
            index_values = rotate_token_major_tail(index_values, group_positions, rope);
            index_values = dsv4_fp4_sim_cuda(index_values.contiguous());
            auto compressed = rotate_token_major_tail(*latent, group_positions, rope);
            compressed = deepseek_v41_mxfp4_e4m3_scale_sim_cuda(compressed.contiguous());
            auto rows = ::tyloi::cuda::arange(
                begin, begin + count, latent->options().dtype(::tyloi::kInt64));
#ifdef TYLOINFER_NATIVE_CUDA
            layer_state.index_k.dense().index_copy_(1, rows, index_values);
            layer_state.compressed_kv.dense().index_copy_(1, rows, compressed);
#else
            layer_state.index_k.index_copy_(1, rows, index_values);
            layer_state.compressed_kv.index_copy_(1, rows, compressed);
#endif
            layer_state.compressed_length += count;
        }
#ifdef TYLOINFER_NATIVE_CUDA
        shared.compressed_kv = layer_state.compressed_kv.dense();
        shared.index_k = layer_state.index_k.dense();
#else
        shared.compressed_kv = layer_state.compressed_kv;
        shared.index_k = layer_state.index_k;
#endif
        shared.compressed_length = layer_state.compressed_length;
        shared.ratio = ratio;
    }

    DenseTensor compute_topk(SharedState& shared, const DenseTensor &input, const DenseTensor &q_rank,
                        const DenseTensor &positions, std::int64_t position) {
        const auto batch = input.size(0);
        const auto tokens = input.size(1);
        const auto pool_length = shared.compressed_length;
        DenseTensor candidates;
        // Restrict scoring only when the candidate budget excludes some KV.
        // An all-inclusive list adds scattered loads without reducing work.
        const bool restricted = pool_length > config.candidate_topk_blocks * config.candidate_block_size;
        return tyloi::models::deepseek_v41::hierarchical_index(
            ratio, index_source(), layer, config.candidate_source_layer, pool_length,
            [&] { return empty_topk(batch, tokens, input.device()); },
            [&] {
                TYLOINFER_RUNTIME_CHECK(shared.topk.defined(),
                                  "DeepSeek-V4.1 CSA2 consumer has no published index plan");
                return shared.topk;
            },
            [&] {
                auto query =
                    tyloi::ops::project(q_rank, index_query).dense()
                        .reshape({batch, tokens, config.index_n_heads, config.index_head_dim})
                        .transpose(1, 2)
                        .contiguous();
#ifdef TYLOINFER_NATIVE_CUDA
                auto rotated = ::tyloi::ops::tail_rope(query, positions, rope);
                query = rotated.dense()
#else
                query = rope.forward(query, positions, false)
#endif
                            .transpose(1, 2)
                            .contiguous();
                query = dsv4_fp4_sim_cuda(query.to(::tyloi::kFloat16).contiguous());
                auto weights = tyloi::ops::project(input, index_score).dense()
                                   .reshape({batch, tokens, config.index_n_heads})
                                   .to(::tyloi::kFloat16)
                                   .contiguous();
                auto keys = shared.index_k.narrow(1, 0, pool_length).contiguous();
                if (layer > config.candidate_source_layer && restricted) {
                    TYLOINFER_RUNTIME_CHECK(shared.candidate_blocks.defined(),
                                      "DeepSeek-V4.1 hierarchical indexer has no candidates");
                    candidates = candidate_positions(shared.candidate_blocks,
                                                     config.candidate_block_size, pool_length);
                    return tyloi::ops::measure("deepseek_v41.indexer.candidate_scores", [&] {
                        return dsv4_indexer_candidate_scores_cuda(query, keys, weights, candidates,
                                                                  position, ratio);
                    });
                }
                return tyloi::ops::measure("deepseek_v41.indexer.full_scores", [&] {
                    return dsv4_indexer_scores_cuda(query, keys, weights, position, ratio);
                });
            },
            [&](DenseTensor scores) {
                if (restricted) shared.candidate_blocks = candidate_blocks_from_scores(
                    scores, ratio, position, config.candidate_topk_blocks,
                    config.candidate_block_size);
            },
            [&](DenseTensor scores) {
                if (!restricted) return select_topk(scores, config.index_topk, pool_length);
                TYLOINFER_RUNTIME_CHECK(candidates.defined(), "DeepSeek-V4.1 candidate scores are unavailable");
                return select_topk(scores, config.index_topk, pool_length, candidates);
            },
            [&](DenseTensor scores) { return select_topk(scores, config.index_topk, pool_length); },
            [&](DenseTensor topk) { shared.topk = std::move(topk); });
    }

    DenseTensor attention_forward(SharedState& shared, const DenseTensor &input,
                             const DenseTensor &positions, std::int64_t position,
                             const MfqOptional<DenseTensor> &sequence_lengths,
                             bool truncated_local = false) {
        auto& layer_state = shared.layers.at(static_cast<std::size_t>(layer));
#ifdef TYLOINFER_NATIVE_CUDA
        auto& local_kv = layer_state.local_kv.dense();
#else
        auto& local_kv = layer_state.local_kv;
#endif
        const auto batch = input.size(0);
        const auto tokens = input.size(1);
        const auto local_history = truncated_local ? 0 : std::min(position, config.sliding_window);
        return tyloi::models::deepseek_v41::attention(
            kv_source(), ratio,
            [&] {
                return weighted_rms(tyloi::ops::project(input, query_a).dense(),
#ifdef TYLOINFER_NATIVE_CUDA
                                    query_a_norm,
#else
                                    query_a_norm,
#endif
                                    config.rms_eps);
            },
            [&](DenseTensor q_rank) {
                auto query = tyloi::ops::project(q_rank, query_b).dense()
                                 .reshape({batch, tokens, config.n_heads, config.head_dim})
                                 .transpose(1, 2)
                                 .contiguous()
                                 .to(::tyloi::kFloat32);
#ifdef TYLOINFER_NATIVE_CUDA
                auto rotated = ::tyloi::ops::tail_rope(query, positions, rope);
                query = rotated.dense();
#else
                query = rope.forward(query, positions, false);
#endif

                return query;
            },
            [&] {
                auto local = weighted_rms(tyloi::ops::project(input, key_value).dense(),
#ifdef TYLOINFER_NATIVE_CUDA
                                          key_value_norm,
#else
                                          key_value_norm,
#endif
                                          config.rms_eps)
                                 .reshape({batch, tokens, config.head_dim});
                local = rotate_token_major_tail(local, positions, rope);
                local = deepseek_v41_mxfp8_e4m3_sim_cuda(local.contiguous());

                return local;
            },
            [&] { publish_compressed_source(shared, input, position); },
            [&] {
                TYLOINFER_RUNTIME_CHECK(shared.compressed_kv.defined() && shared.index_k.defined() &&
                                      shared.ratio == ratio,
                                  "DeepSeek-V4.1 CSA2 consumer ran before its source layer");
            },
            [&](DenseTensor q_rank) {
                return compute_topk(shared, input, q_rank, positions, position);
            },
            [&](DenseTensor local) {
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
            },
            [&](DenseTensor topk) {
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
            },
            [&](DenseTensor query, DenseTensor cache, const auto &plan) {
#ifdef TYLOINFER_NATIVE_CUDA
                return tyloi::ops::sparse_attention(
                           query, cache, plan.at(0), plan.at(1), sinks,
                           1.0 / std::sqrt(static_cast<double>(config.head_dim)))
                    .dense();
#else
                return attention_dsv4_sparse_cuda(
                    query, cache, plan.at(0), plan.at(1),
                    sinks, shared.attention_meta,
                    1.0 / std::sqrt(static_cast<double>(config.head_dim)));
#endif
            },
            [&](DenseTensor attended) {
#ifdef TYLOINFER_NATIVE_CUDA
                auto value = attended.transpose(1, 2).contiguous();
                auto rotated = ::tyloi::ops::tail_rope(value, positions, rope, true);
                return rotated.dense()
#else
                return rope.forward(attended.transpose(1, 2).contiguous(), positions, true)
#endif
                    .transpose(1, 2)
                    .contiguous();
            },
            [&] { layer_state.position += tokens; },
            [&](DenseTensor attended) { return output_projection(attended); });
    }

#endif
    DenseTensor forward(SharedState& shared,
#ifndef TYLOINFER_NATIVE_CUDA
                   CudaExecutionContext &execution,
#endif
                   DenseTensor hidden, DenseTensor positions,
                   std::int64_t cache_position, const MfqOptional<DenseTensor> &sequence_lengths,
                   const MfqOptional<DenseTensor> &cache_positions = mfq_nullopt,
                   std::int64_t confirmed_prefix = 0) {
        if(confirmed_prefix>0) {
            // A joint pass writes pending and draft rows. Protect their whole
            // physical window; rollback retains the confirmed prefix as well.
#ifdef TYLOINFER_NATIVE_CUDA
            attention->begin_speculative(shared.layers.at(static_cast<std::size_t>(layer)).attention,hidden.size(1));
#else
            begin_speculative(shared, hidden.size(1));
#endif
            if(layer==0) {
                TYLOINFER_RUNTIME_CHECK(shared.engram_hash, "DeepSeek-V4.1 Engram state is unavailable");
                shared.engram_hash->begin_speculative();
            }
        }
        TYLOINFER_RUNTIME_CHECK(confirmed_prefix == 0 || supports_speculation(),
                              "block does not support speculative verification");
        auto& layer_state = shared.layers.at(static_cast<std::size_t>(layer));
        TYLOINFER_RUNTIME_CHECK(layer_state.current_ids.defined() && hidden.dim() == 4 &&
                              hidden.size(2) == config.hc_mult && hidden.size(3) == config.hidden &&
                              positions.dim() == 1 && positions.numel() == hidden.size(1) &&
                              !cache_positions.has_value() &&
#ifdef TYLOINFER_NATIVE_CUDA
                              layer_state.attention.position == cache_position,
#else
                              layer_state.position == cache_position,
#endif
                          "DeepSeek-V4.1 block input/cache mismatch");
#ifdef TYLOINFER_NATIVE_CUDA
        const auto& ids = layer_state.current_ids.dense();
#else
        const auto& ids = layer_state.current_ids;
#endif
        shared.enter_layer(layer, config.n_layers, hidden.size(0), hidden.size(1), cache_position,
                            hidden.device(), ids);

        const auto batch = hidden.size(0), tokens = hidden.size(1);
        const auto measure=[&](const char* name,auto&& operation) {
#ifdef TYLOINFER_NATIVE_CUDA
            return tyloi::ops::measure(name,tyloi::Device(tyloi::kCUDA,cuda_device),std::forward<decltype(operation)>(operation));
#else
            return tyloi::ops::measure(name,std::forward<decltype(operation)>(operation));
#endif
        };
#ifdef TYLOINFER_NATIVE_CUDA
        const bool truncated_local = std::exchange(layer_state.attention.replay_local_window, false);
#else
        const bool truncated_local = std::exchange(layer_state.replay_local_window, false);
#endif
        if (engram) {
            hidden = measure("deepseek_v41.engram", [&] {
                return engram->forward(*shared.engram_states.at(static_cast<std::size_t>(layer)),
                                       hidden, shared.engram_hashes, shared.text_rows, &shared.text_row_ids);
            });
        }
        shared.capture_dspark_target(layer, config, hidden);
#ifdef TYLOINFER_NATIVE_CUDA
        const tyloi::ops::MhcConfig mhc{config.hc_sinkhorn_iters,config.rms_eps,config.hc_eps};
        auto attention_mix=measure("deepseek_v41.mhc.attention.collapse",[&] {
            return tyloi::ops::mhc_collapse(hidden,shared.previous_pre,attention_mhc_function,
                attention_mhc_scale,attention_mhc_base,attention_norm,mhc);
        });
#else
        auto attention_mix = collapse(hidden, shared.previous_pre,
            attention_mhc_function, attention_mhc_scale, attention_mhc_base, attention_norm,
            "deepseek_v41.mhc.attention.collapse");
#endif
#ifdef TYLOINFER_NATIVE_CUDA
        shared.attention_input.dense()=attention_mix.branch.dense();
        shared.attention_positions.dense()=positions;
        shared.attention_lengths.dense()=sequence_lengths.has_value()?sequence_lengths.value():DenseTensor{};
#endif
        auto attended = measure("deepseek_v41.attention", [&] {
#ifdef TYLOINFER_NATIVE_CUDA
            return attention->forward(layer_state.attention,shared.attention,shared.attention_input,shared.attention_positions,
                cache_position,shared.attention_lengths,truncated_local).dense();
#else
            return attention_forward(shared, attention_mix.branch, positions, cache_position,
                                     sequence_lengths, truncated_local);
#endif
        });
#ifdef TYLOINFER_NATIVE_CUDA
        shared.attention_input.dense()={}; shared.attention_positions.dense()={}; shared.attention_lengths.dense()={};
#endif
#ifdef TYLOINFER_NATIVE_CUDA
        hidden=measure("deepseek_v41.mhc.attention.expand",[&] {
            return tyloi::ops::mhc_expand(attended,hidden,attention_mix).dense();
        });
        auto ffn_mix=measure("deepseek_v41.mhc.mlp.collapse",[&] {
            return tyloi::ops::mhc_collapse(hidden,attention_mix.next_pre,mlp_mhc_function,
                mlp_mhc_scale,mlp_mhc_base,mlp_norm,mhc);
        });
#else
        hidden = expand(attended, hidden, attention_mix, "deepseek_v41.mhc.attention.expand");
        auto ffn_mix = collapse(hidden, attention_mix.next_pre,
            mlp_mhc_function, mlp_mhc_scale, mlp_mhc_base, mlp_norm,
            "deepseek_v41.mhc.mlp.collapse");
#endif
        auto output = measure("deepseek_v41.mlp", [&] {
#ifdef TYLOINFER_NATIVE_CUDA
            const auto& value = ffn_mix.branch.dense();
#else
            const auto& value = ffn_mix.branch;
#endif
            if(shared.image_mask.defined()) {
#ifdef TYLOINFER_NATIVE_CUDA
                TYLOINFER_RUNTIME_CHECK(vision_router_bias && vision_router_bias.defined(),"DeepSeek-V4.1 visual router bias is missing");
#else
                TYLOINFER_RUNTIME_CHECK(vision_router_bias.defined(),"DeepSeek-V4.1 visual router bias is missing");
#endif
                TYLOINFER_RUNTIME_CHECK(mlp.moe_router_bias, "DeepSeek-V4.1 router bias is missing");
#ifdef TYLOINFER_NATIVE_CUDA
                auto bias=tyloi::ops::contiguous(tyloi::ops::where(shared.routing_mask,vision_router_bias,mlp.moe_router_bias)).dense();
#else
                auto bias=::tyloi::cuda::where(shared.routing_mask,vision_router_bias,mlp.moe_router_bias.dense()).contiguous();
#endif
#ifdef TYLOINFER_NATIVE_CUDA
                return tyloi::ops::ffn(mlp,
                    value.reshape({-1,config.hidden}), ids, bias).dense()
#else
                auto executor = execution.executor().fork(model_parallel_primary_device(execution.config));
                return mlp.forward(
                    value.reshape({-1,config.hidden}), *executor, execution.executor().forward_mode(execution.config.layer_placement.primary_device()).ffn, ids, bias)
#endif
                    .reshape({batch,tokens,config.hidden});
            }
#ifdef TYLOINFER_NATIVE_CUDA
            return tyloi::ops::ffn(mlp,
                    value.reshape({-1, config.hidden}), ids).dense()
#else
            auto executor = execution.executor().fork(model_parallel_primary_device(execution.config));
            return mlp.forward(
                    value.reshape({-1, config.hidden}), *executor, execution.executor().forward_mode(execution.config.layer_placement.primary_device()).ffn, ids)
#endif
                .reshape({batch, tokens, config.hidden});
        });
#ifdef TYLOINFER_NATIVE_CUDA
        hidden=measure("deepseek_v41.mhc.mlp.expand",[&] {
            return tyloi::ops::mhc_expand(output,hidden,ffn_mix).dense();
        });
        shared.previous_pre = ffn_mix.next_pre.dense();
#else
        hidden = expand(output, hidden, ffn_mix, "deepseek_v41.mhc.mlp.expand");
        shared.previous_pre = std::move(ffn_mix.next_pre);
#endif
        return hidden;
    }
};

#ifdef TYLOINFER_NATIVE_CUDA
/** Load one DeepSeek routed FFN with its numerical policy. */
tyloi::ops::FfnWeights load_moe_at(const tyloi::weights::Loader& weights,
#else
tyloi::cuda::Ffn load_moe_at(const tyloi::weights::Loader& weights,
#endif
                const CommonConfig &config, const std::string &prefix, std::int64_t layer,
                std::int64_t top_k);

std::unique_ptr<Block> load_block(const tyloi::weights::Loader& weights,
                                    std::int64_t layer, const CommonConfig& config,
                                    ModelTables& tables, std::size_t engram_cache_rows);
#ifndef TYLOINFER_NATIVE_CUDA
void validate_load_options(const CudaExecutionContext &execution);
#endif
DenseTensor finalize_hidden(const SharedState &state, const DenseTensor &hidden);

inline int run_self_check() {
    EngramHashState::self_check();
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

// The model record: weights, placement and load-time state. Request data
// lives in State and is passed explicitly to each block.
struct Model : tyloi::models::CausalModel {
    using CausalConfig = tyloi::models::deepseek_v41::Config;
    static bool accepts_backbone(std::string_view backbone) { return backbone == "deepseek_v41"; }

    // ---- tyloi::models::CausalModel
    tyloi::models::Info info() const override;
    MfqModelCapabilities capabilities() const override;
    tyloi::engine::Metrics metrics() const override;
#ifdef TYLOINFER_NATIVE_CUDA
    /** Borrow this model's assembled DSpark predictor. */
    tyloi::models::Mtp* mtp() const noexcept override { return predictor.get(); }
#endif
    std::unique_ptr<tyloi::models::State> make_state() const override;
    tyloi::StepSequence<tyloi::models::Prepared> prepare(const std::vector<std::int64_t>& tokens,
        const MfqMultimodalInput& media, tyloi::models::State& state) const override;
    ::tyloi::Tensor embedding(const tyloi::Tensor& ids, int primary = 0) const override;
    ::tyloi::Tensor normalize(const tyloi::Tensor& hidden, std::int64_t batch,
        std::int64_t tokens) const override;
    ::tyloi::Tensor softcap(const tyloi::Tensor& logits) const override;
    ::tyloi::Tensor logits(const tyloi::Tensor& hidden, bool capped = true) const override;
    ::tyloi::Tensor forward(tyloi::models::Input input, tyloi::models::State& state,
                                 const tyloi::models::ForwardOptions& options = {}) const override;
#ifdef TYLOINFER_NATIVE_CUDA
    /** Run the staged causal encoder and decoder prefill. */
    tyloi::StepSequence<tyloi::models::Prefill> prefill(tyloi::models::Input input,
        tyloi::models::State& state, const tyloi::models::ForwardOptions& options,
        tyloi::models::PrefillOptions prefill) const override;
#endif

    // ---- speculative contract used by the MTP sequence
    bool supports_speculation() const;
    bool supports_suffix_speculation() const noexcept { return true; }
    ::tyloi::Tensor hidden_forward_speculative_suffix(State& state, ::tyloi::Tensor ids,
                                                      ::tyloi::Tensor* raw_hidden = nullptr);
#ifdef TYLOINFER_NATIVE_CUDA
    /** Prefetch Engram rows for host token IDs using this model's request state. */
    bool prefetch(tyloi::models::State& state, const ::tyloi::Tensor& ids) const override;
#else
    /** Prefetch Engram rows for host token IDs using this model's request state. */
    bool prefetch_token_rows(State& state, const ::tyloi::Tensor& ids) const;
#endif

    // ---- weights and load-time state
    tyloi::models::CausalLmMetadata metadata;
    int primary_device = 0;
    bool needs_host_tokens = false;
#ifndef TYLOINFER_NATIVE_CUDA
    CudaExecutionContext* execution = nullptr;
#endif
    std::shared_ptr<const tyloi::ModelSource> source;
    tyloi::ModelGraph graph;
    CudaModelPlan plan;
    ::tyloi::Tensor embed;
#ifdef TYLOINFER_NATIVE_CUDA
    ::tyloi::Tensor output_norm;
#else
    ::tyloi::DenseTensor output_norm;
#endif
    ::tyloi::Tensor lm_head;
    CausalConfig config;
    std::vector<std::unique_ptr<deepseek_v41_runtime::Block>> blocks;
    deepseek_v41_runtime::ModelTables tables;
    std::unique_ptr<deepseek_v41_runtime::Vision> vision;
#ifdef TYLOINFER_NATIVE_CUDA
    std::unique_ptr<tyloi::models::Mtp> predictor;
#else
    tyloi::cache::Media media;
#endif

    // ---- helpers absorbed from the retired CausalLm plumbing
#ifndef TYLOINFER_NATIVE_CUDA
    auto execution_scope() const {
        return CudaScope(execution->config.layer_placement.primary_device(), execution->executor().contexts());
    }
#endif
    ::tyloi::DenseTensor device_ids(::tyloi::DenseTensor ids) const {
#ifdef TYLOINFER_NATIVE_CUDA
        auto converted = tyloi::ops::cast(ids, ::tyloi::kInt64);
        return tyloi::ops::contiguous(tyloi::ops::copy(converted, ::tyloi::Device(::tyloi::kCUDA, primary_device))).dense();
#else
        return execution->executor().collectives()
            .copy(ids.to(::tyloi::kInt64), execution->config.layer_placement.primary_device())
            .contiguous();
#endif
    }
    ::tyloi::DenseTensor embed_tokens(::tyloi::DenseTensor ids) const {
        return tyloi::ops::measure("model.embed", [&] { return tyloi::ops::embedding(ids, embed).dense(); });
    }
    static ::tyloi::DenseTensor softcap_apply(::tyloi::DenseTensor logits, double cap) {
        return ::tyloi::cuda::tanh(logits / cap) * cap;
    }
    std::int64_t hidden_size() const noexcept { return metadata.hidden_size; }
    std::int64_t max_position_embeddings() const noexcept { return config.max_position_embeddings; }
    std::int64_t num_hidden_layers() const noexcept { return config.n_layers; }
    bool tie_word_embeddings() const noexcept { return metadata.tie_word_embeddings; }
    void load_config(std::string_view payload);

    // ---- family adapters (implementations in ops.cpp)
    ::tyloi::DenseTensor normalize_hidden(::tyloi::DenseTensor hidden,
#ifdef TYLOINFER_NATIVE_CUDA
                                          const ::tyloi::Tensor& output_norm, std::int64_t batch,
#else
                                          const ::tyloi::DenseTensor& output_norm, std::int64_t batch,
#endif
                                          std::int64_t tokens) const;
    ::tyloi::DenseTensor prepare_hidden(::tyloi::DenseTensor hidden, std::int64_t batch,
                                                std::int64_t tokens) const;
    ::tyloi::DenseTensor finalize_hidden(State& state, ::tyloi::DenseTensor hidden,
                                                 std::int64_t batch, std::int64_t tokens) const;
    void validate_forward(std::int64_t, std::int64_t tokens, std::int64_t cache_position,
                                  bool has_position_override, bool has_cache_position_override) const;
};

void read(Model& model, std::shared_ptr<const tyloi::ModelSource> source,
         std::string_view definition, std::int64_t context_size);
#ifndef TYLOINFER_NATIVE_CUDA
void load(Model& model, CudaExecutionContext& execution, int64_t context_size,
          bool defer_moe_cache_finalize, bool load_blocks, bool load_optional_components);
#endif

#ifndef TYLOINFER_NATIVE_CUDA
struct CedEncoding {
    ::tyloi::DenseTensor ids, tail_hidden, tail_pre, tail_ids;
    tyloi::models::deepseek_v41::EngramHashBatch tail_hashes;
    std::int64_t batch, begin, end, tokens, compute_chunk;
};
::tyloi::Tensor deepseek_v41_prefill_input(
    Model& model, State& state, const ::tyloi::Tensor& ids, std::int64_t compute_chunk);
std::optional<CedEncoding> deepseek_v41_prefill_encode(
    Model& model, State& state, const ::tyloi::Tensor& ids, std::int64_t compute_chunk,
    const std::function<bool()>& stopped, bool capture_raw_hidden);
::tyloi::Tensor deepseek_v41_prefill_decode(
    Model& model, State& state, CedEncoding encoded, const std::function<bool()>& stopped,
    ::tyloi::Tensor* raw_hidden = nullptr);
std::int64_t prefill_span(const Model&, const State&, std::size_t tokens,
                          std::int64_t chunk, std::size_t, std::size_t,
                          bool transformed, bool retained);
#endif

} // namespace deepseek_v41
} // namespace tyloi::cuda
