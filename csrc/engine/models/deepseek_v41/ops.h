#pragma once
#include "engine/backends/cuda/kernels/mfq_cuda_norm_ops.h"
#include "engine/backends/cuda/ops/norm/rms.h"
#include "engine/ops.h"
#include "engine/ops/csa.h"
#include "engine/ops/ffn.h"
#include "engine/ops/mhc.h"
#include "engine/models/deepseek_v41/causal_lm.h"

#include "engine/models/metadata.h"
#include "engine/models/source.h"
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


namespace tyloi::cuda::deepseek_v41_runtime {

class Vision;

using DenseTensor = ::tyloi::DenseTensor;
using CommonConfig = tyloi::models::deepseek_v41::Config;
using Parameter = ::tyloi::Tensor;

// Per-request, per-layer runtime: the KV ring, checkpoints and layer flags.
struct LayerState {
    tyloi::ops::Csa::State attention;
    ::tyloi::Tensor current_ids;
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
    tyloi::ops::Csa::Shared attention;
    /** Retain an in-flight causal encoder source until publication completes. */
    tyloi::Tensor ced_source{DenseTensor{}};
    /** Retain attention inputs until their numerical operation completes. */
    tyloi::Tensor attention_input{DenseTensor{}}, attention_positions{DenseTensor{}}, attention_lengths{DenseTensor{}};
    DenseTensor previous_pre;
    std::unique_ptr<EngramHashState> engram_hash;
    std::vector<std::unique_ptr<Engram::State>> engram_states;
    EngramHashBatch engram_hashes;
    std::vector<DenseTensor> dspark_target_hiddens;
    bool capture_dspark_targets = false;
    DenseTensor image_mask, text_rows, routing_mask;
    std::vector<int64_t> text_row_ids;
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
        dspark_target_hiddens[slot] = tyloi::ops::contiguous(tyloi::ops::mean(hidden,2)).dense();
    }

    /** Combine captured target layers on the primary device. */
    DenseTensor dspark_target_hidden(int primary = 0) const {
        TYLOINFER_RUNTIME_CHECK(capture_dspark_targets && !dspark_target_hiddens.empty() &&
                              std::all_of(dspark_target_hiddens.begin(),
                                          dspark_target_hiddens.end(),
                                          [](const DenseTensor &value) { return value.defined(); }),
                          "DeepSeek-V4.1 DSpark target layers were not fully captured");
        std::vector<tyloi::Tensor> values;
        values.reserve(dspark_target_hiddens.size());
        for (const auto& hidden : dspark_target_hiddens)
            values.push_back(tyloi::ops::copy(hidden, ::tyloi::Device(::tyloi::kCUDA, primary)));
        return values.size() == 1 ? values.front().dense() : tyloi::ops::cat(values, -1).dense();
    }

    void reset_session(std::int64_t current_batch) {
        attention = {};
        ced_source.dense() = {};
        attention_input.dense() = {}; attention_positions.dense() = {}; attention_lengths.dense() = {};
        for(auto& engram:engram_states)if(engram)engram->clear();
        previous_pre = DenseTensor();
        engram_hashes = {};
        dspark_target_hiddens.clear();
        capture_dspark_targets = false;
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
            attention = {};
            cache_position = current_position;
            batch = current_batch;
            tokens = current_tokens;
            next_layer = 0;
            auto pre=tyloi::ops::zeros({current_batch,current_tokens,4},
                ::tyloi::TensorOptions().device(device).dtype(::tyloi::kFloat32));
            auto first=tyloi::ops::slice(pre,2,0,1);
            tyloi::ops::fill(first,1.0); previous_pre=pre.dense();
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
    Parameter attention_mhc_function;
    Parameter attention_mhc_scale;
    Parameter attention_mhc_base;
    Parameter mlp_mhc_function;
    Parameter mlp_mhc_scale;
    Parameter mlp_mhc_base;
    std::unique_ptr<tyloi::ops::Csa> attention;
    tyloi::ops::FfnWeights mlp;
    Parameter vision_router_bias;
    std::unique_ptr<Engram> engram;

    bool kv_source() const noexcept { return config.is_kv_source(layer); }

    bool index_source() const noexcept { return config.is_index_source(layer); }

    bool supports_speculation() const noexcept { return true; }
    /** Retain token IDs for the next layer invocation. */
    void set_token_ids(SharedState& shared, const DenseTensor &ids) { shared.layers.at(static_cast<std::size_t>(layer)).current_ids = ids; }
    // No dense warmup state: the recurrent cache lives in AttentionState.
    std::vector<DenseTensor*> graph_warmup_state() { return {}; }

    DenseTensor forward(SharedState& shared,
                   DenseTensor hidden, DenseTensor positions,
                   std::int64_t cache_position, const MfqOptional<DenseTensor> &sequence_lengths,
                   const MfqOptional<DenseTensor> &cache_positions = mfq_nullopt,
                   std::int64_t confirmed_prefix = 0) {
        if(confirmed_prefix>0) {
            // A joint pass writes pending and draft rows. Protect their whole
            // physical window; rollback retains the confirmed prefix as well.
            attention->begin_speculative(shared.layers.at(static_cast<std::size_t>(layer)).attention,hidden.size(1));
            if(layer==0) {
                TYLOINFER_RUNTIME_CHECK(shared.engram_hash, "DeepSeek-V4.1 Engram state is unavailable");
                shared.engram_hash->begin_speculative();
            }
        }
        TYLOINFER_RUNTIME_CHECK(confirmed_prefix == 0 || supports_speculation(),
                              "block does not support speculative verification");
        auto& layer_state = shared.layers.at(static_cast<std::size_t>(layer));
        const auto& layer_position = layer_state.attention.position;
        TYLOINFER_RUNTIME_CHECK(layer_state.current_ids.defined() && hidden.dim() == 4 &&
                              hidden.size(2) == config.hc_mult && hidden.size(3) == config.hidden &&
                              positions.dim() == 1 && positions.numel() == hidden.size(1) &&
                              !cache_positions.has_value() &&
                              layer_position == cache_position,
                          "DeepSeek-V4.1 block input/cache mismatch");
        const auto& ids = layer_state.current_ids.dense();
        shared.enter_layer(layer, config.n_layers, hidden.size(0), hidden.size(1), cache_position,
                            hidden.device(), ids);

        const auto batch = hidden.size(0), tokens = hidden.size(1);
        const auto measure=[&](const char* name,auto&& operation) {
            return tyloi::ops::measure(name,tyloi::Device(tyloi::kCUDA,cuda_device),std::forward<decltype(operation)>(operation));
        };
        const bool truncated_local = std::exchange(layer_state.attention.replay_local_window, false);
        if (engram) {
            hidden = measure("deepseek_v41.engram", [&] {
                return engram->forward(*shared.engram_states.at(static_cast<std::size_t>(layer)),
                                       hidden, shared.engram_hashes, shared.text_rows, &shared.text_row_ids);
            });
        }
        shared.capture_dspark_target(layer, config, hidden);
        const tyloi::ops::MhcConfig mhc{config.hc_sinkhorn_iters,config.rms_eps,config.hc_eps};
        auto attention_mix=measure("deepseek_v41.mhc.attention.collapse",[&] {
            return tyloi::ops::mhc_collapse(hidden,shared.previous_pre,attention_mhc_function,
                attention_mhc_scale,attention_mhc_base,attention_norm,mhc);
        });
        shared.attention_input.dense()=attention_mix.branch.dense();
        shared.attention_positions.dense()=positions;
        shared.attention_lengths.dense()=sequence_lengths.has_value()?sequence_lengths.value():DenseTensor{};
        auto attended = measure("deepseek_v41.attention", [&] {
            return attention->forward(layer_state.attention,shared.attention,shared.attention_input,shared.attention_positions,
                cache_position,shared.attention_lengths,truncated_local).dense();
        });
        shared.attention_input.dense()={}; shared.attention_positions.dense()={}; shared.attention_lengths.dense()={};
        hidden=measure("deepseek_v41.mhc.attention.expand",[&] {
            return tyloi::ops::mhc_expand(attended,hidden,attention_mix).dense();
        });
        auto ffn_mix=measure("deepseek_v41.mhc.mlp.collapse",[&] {
            return tyloi::ops::mhc_collapse(hidden,attention_mix.next_pre,mlp_mhc_function,
                mlp_mhc_scale,mlp_mhc_base,mlp_norm,mhc);
        });
        auto output = measure("deepseek_v41.mlp", [&] {
            const auto& value = ffn_mix.branch.dense();
            if(shared.image_mask.defined()) {
                TYLOINFER_RUNTIME_CHECK(vision_router_bias && vision_router_bias.defined(),"DeepSeek-V4.1 visual router bias is missing");
                TYLOINFER_RUNTIME_CHECK(mlp.moe_router_bias, "DeepSeek-V4.1 router bias is missing");
                auto bias=tyloi::ops::contiguous(tyloi::ops::where(shared.routing_mask,vision_router_bias,mlp.moe_router_bias)).dense();
                return tyloi::ops::ffn(mlp,
                    value.reshape({-1,config.hidden}), ids, bias).dense()
                    .reshape({batch,tokens,config.hidden});
            }
            return tyloi::ops::ffn(mlp,
                    value.reshape({-1, config.hidden}), ids).dense()
                .reshape({batch, tokens, config.hidden});
        });
        hidden=measure("deepseek_v41.mhc.mlp.expand",[&] {
            return tyloi::ops::mhc_expand(output,hidden,ffn_mix).dense();
        });
        shared.previous_pre = ffn_mix.next_pre.dense();
        return hidden;
    }
};

/** Fill one routed FFN while retaining partial weights on failure. */
void load_moe_at_into(tyloi::ops::FfnWeights&, const tyloi::weights::Loader&,
    const CommonConfig&, const std::string&, std::int64_t layer, std::int64_t top_k);
/** Load one DeepSeek routed FFN with its numerical policy. */
tyloi::ops::FfnWeights load_moe_at(const tyloi::weights::Loader& weights,
                const CommonConfig &config, const std::string &prefix, std::int64_t layer,
                std::int64_t top_k);

std::unique_ptr<Block> load_block(const tyloi::weights::Loader& weights,
                                    std::int64_t layer, const CommonConfig& config,
                                    ModelTables& tables, std::size_t engram_cache_rows);
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
    /** Borrow this model's assembled DSpark predictor. */
    tyloi::models::Mtp* mtp() const noexcept override { return predictor.get(); }
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
    /** Run the staged causal encoder and decoder prefill. */
    tyloi::StepSequence<tyloi::models::Prefill> prefill(tyloi::models::Input input,
        tyloi::models::State& state, const tyloi::models::ForwardOptions& options,
        tyloi::models::PrefillOptions prefill) const override;

    // ---- speculative contract used by the MTP sequence
    bool supports_speculation() const;
    bool supports_suffix_speculation() const noexcept { return true; }
    ::tyloi::Tensor hidden_forward_speculative_suffix(State& state, ::tyloi::Tensor ids,
                                                      ::tyloi::Tensor* raw_hidden = nullptr);
    /** Prefetch Engram rows for host token IDs using this model's request state. */
    bool prefetch(tyloi::models::State& state, const ::tyloi::Tensor& ids) const override;

    // ---- weights and load-time state
    tyloi::models::CausalLmMetadata metadata;
    int primary_device = 0;
    bool needs_host_tokens = false;
    std::shared_ptr<const tyloi::ModelSource> source;
    tyloi::ModelGraph graph;
    ::tyloi::models::Plan plan;
    ::tyloi::Tensor embed;
    ::tyloi::Tensor output_norm;
    ::tyloi::Tensor lm_head;
    CausalConfig config;
    std::vector<std::unique_ptr<deepseek_v41_runtime::Block>> blocks;
    deepseek_v41_runtime::ModelTables tables;
    std::unique_ptr<deepseek_v41_runtime::Vision> vision;
    std::unique_ptr<tyloi::models::Mtp> predictor;

    // ---- helpers absorbed from the retired CausalLm plumbing
    ::tyloi::DenseTensor device_ids(::tyloi::DenseTensor ids) const {
        auto converted = tyloi::ops::cast(ids, ::tyloi::kInt64);
        return tyloi::ops::contiguous(tyloi::ops::copy(converted, ::tyloi::Device(::tyloi::kCUDA, primary_device))).dense();
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
                                          const ::tyloi::Tensor& output_norm, std::int64_t batch,
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


} // namespace deepseek_v41
} // namespace tyloi::cuda
