#pragma once

#include "engine/weights/loader.h"
#include "engine/models/session.h"

#include "engine/backends/cuda/executor/execution.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/backends/cuda/ops/norm/rms.h"
#include "engine/models/common/mtp.h"
#include "engine/models/nemotron_h/forward.h"
#include "engine/models/nemotron_h/load.h"
#include "engine/models/nemotron_h/predictor.h"

#include <algorithm>
#include <optional>
#include <string>
#include <utility>

namespace tyloi::cuda::nemotron_h {

// Nemotron-H's predictor is one multi-token-prediction head whose sub-blocks
// are listed in mtp_layers_block_type; both blocks run inside the single stage
// of the shared projected_predictor graph. The attention sub-block owns the
// predictor history; the MoE sub-block is stateless.
// The semantics follow llama.cpp (src/models/nemotron-h-moe.cpp), vLLM and
// SGLang rather than DeepSeek-V3: the backbone hands over its post-output_norm
// hidden and each further draft step
// reuses this module's final_layernorm output. The final norm stays RMSNorm as
// in vLLM/SGLang (llama.cpp uses LayerNorm; the real model will A/B the two).
struct Mtp final : MtpModule {
    using DenseTensor = ::tyloi::DenseTensor;

    tyloi::cuda::nemotron_h::Config config;
    std::shared_ptr<const ::tyloi::Tensor> embedding;
    int primary = 0;
    std::shared_ptr<const ::tyloi::Tensor> fusion;
    DenseTensor embedding_norm, hidden_norm, output_norm;
    std::vector<Layer> layers;
    std::vector<LayerState> states;
    std::vector<int64_t> lengths;
    CudaExecutionContext *execution = nullptr;
    std::unique_ptr<Executor> executor;
    int64_t batch = 0;

    // The predictor history shares the backbone context limit, which the
    // loader may have overridden with --ctx-size.
    int64_t predictor_capacity() const noexcept { return config.max_position_embeddings; }

    static std::optional<Mtp> load_if_present(CudaExecutionContext &execution,
                                              const weight_loader::Loader& weights,
                                              const tyloi::models::nemotron_h::Config &main,
                                              std::shared_ptr<const ::tyloi::Tensor> embedding) {
        const bool fusion_present = weights.has("predictor.fusion.weight");
        if (!fusion_present) {
            TYLOINFER_RUNTIME_CHECK(!weights.has_prefix("predictor."),
                              "Nemotron-H source contains an incomplete MTP head");
            return std::nullopt;
        }
        TYLOINFER_RUNTIME_CHECK(main.num_nextn_predict_layers == 1 && !main.mtp_mixers.empty(),
                          "Nemotron-H MTP requires one declared predictor stage");
        Mtp result;
        result.config = main;
        result.execution = &execution;
        result.executor = execution.executor().fork(model_parallel_primary_device(execution.config));
        result.embedding = std::move(embedding);
        result.primary = weights.placement.device;
        result.embedding_norm = weights.tensor("predictor.embedding_norm.weight");
        result.hidden_norm = weights.tensor("predictor.hidden_norm.weight");
        result.output_norm = weights.tensor("predictor.output_norm.weight");
        result.fusion = weights.weight("predictor.fusion.weight");
        TYLOINFER_RUNTIME_CHECK(result.fusion->size(1) == 2 * main.hidden_size &&
                              result.fusion->size(0) == main.hidden_size &&
                              result.embedding_norm.numel() == main.hidden_size &&
                              result.hidden_norm.numel() == main.hidden_size &&
                              result.output_norm.numel() == main.hidden_size,
                          "Nemotron-H MTP fusion/norm dimensions disagree with the backbone");
        for (std::size_t index = 0; index < main.mtp_mixers.size(); ++index) {
            const auto type =
                std::string(tyloi::models::nemotron_h::Config::mixer_name(main.mtp_mixers[index]));
            result.layers.push_back(load_layer(execution.executor().pipelines(), weights, main,
                                               static_cast<int>(index), type, "predictor"));
        }
        result.states = std::vector<LayerState>(result.layers.size());
        for (std::size_t index = 0; index < result.layers.size(); ++index) {
            const auto &layer = result.layers[index];
            if (const auto *full = std::get_if<FullAttention>(&layer.weights)) {
                auto &cache = result.states[index].emplace<KVCache>();
                if (runtime_options::kv_vram_preallocated()) {
                    MfqCudaGuard guard(layer.device);
                    cache = KVCache(1, full->kv_heads,
                        full->sliding ? full->attention_window : full->max_position_embeddings,
                        full->attention_head_dim, full->sliding,
                        ::tyloi::Device(::tyloi::kCUDA, layer.device),
                        full->official_bf16 ? ::tyloi::kBFloat16 :
                                              ::tyloi::kFloat16,
                        full->kv_quantization);
                }
            }
        }
        result.lengths.assign(1, 0);
        return result;
    }

    void reset(int64_t next_batch = 1) override {
        TYLOINFER_RUNTIME_CHECK(next_batch > 0, "Nemotron-H MTP batch must be positive");
        for (auto &state : states)
            if (auto *cache = std::get_if<KVCache>(&state))
                if (!cache->defined() || cache->batch_size() != next_batch)
                    *cache = KVCache();
        std::fill(lengths.begin(), lengths.end(), 0);
        batch = next_batch;
    }

    std::pair<DenseTensor, DenseTensor> evaluate(const DenseTensor &hidden,
                                       const DenseTensor &ids, int64_t depth = 0,
                                       const DenseTensor &supplied_positions = {}) {
        struct Ops {
            using DenseTensor = ::tyloi::DenseTensor;
            Mtp &model;
            Executor::Mode mode;
            static int rank(const DenseTensor &x) { return x.dim(); }
            static int64_t size(const DenseTensor &x, int axis) { return x.size(axis); }
            static int64_t elements(const DenseTensor &x) { return x.numel(); }
            static bool defined(const DenseTensor &x) { return x.defined(); }
            DenseTensor embed(DenseTensor ids, const DenseTensor &like) {
                return tyloi::cuda::embed(*model.embedding, std::move(ids),
                    model.execution->executor().collectives(), model.execution->executor().profiler(), model.primary)
                    .to(like.scalar_type());
            }
            DenseTensor normalize(const DenseTensor &x, tyloi::models::nemotron_h::PredictorNorm role) {
                using Norm = tyloi::models::nemotron_h::PredictorNorm;
                const auto &weight = role == Norm::embedding ? model.embedding_norm
                                     : role == Norm::hidden  ? model.hidden_norm
                                                             : model.output_norm;
                return qwen_rms_norm(x.reshape({-1, model.config.hidden_size})
                                         .to(::tyloi::kFloat32),
                                     weight, model.config.rms_norm_eps, 0.0)
                    .reshape(x.sizes());
            }
            DenseTensor positions(DenseTensor explicit_positions, int64_t start, int64_t tokens) {
                if (explicit_positions.defined())
                    return explicit_positions.to(::tyloi::kInt64).contiguous();
                // NoPE: positions only address the predictor's KV history.
                return ::tyloi::cuda::arange(start, start + tokens,
                                                  ::tyloi::TensorOptions()
                                                      .device(::tyloi::kCUDA)
                                                      .dtype(::tyloi::kInt64));
            }
            DenseTensor fuse(DenseTensor e, DenseTensor h) {
                return model.executor->projector(mode.ffn).run(
                    *model.fusion, ::tyloi::cuda::cat({e, h}, -1));
            }
            static DenseTensor lengths(int64_t batch, int64_t end, const DenseTensor &positions) {
                return ::tyloi::cuda::full({batch}, end, positions.options());
            }
            // One predictor stage: every declared sub-block adds its own
            // pre-normed mixer branch to the same residual stream.
            DenseTensor mixer_blocks(DenseTensor x, int64_t stage, const DenseTensor &pos) {
                const int64_t tokens = x.size(1);
                std::optional<DenseTensor> sequence_lengths;
                if (tokens == 1 && model.lengths[stage] > 0)
                    sequence_lengths = lengths(x.size(0), model.lengths[stage] + 1, pos);
                AttentionStep step{pos, sequence_lengths, mfq_nullopt, mfq_nullopt,
                                   model.lengths[stage], 0, 0};
                for (std::size_t index = 0; index < model.layers.size(); ++index)
                    x = forward_layer(model.layers[index], model.states[index], *model.executor,
                                      index, std::move(x), step, 0, mode);
                return x;
            }
        } ops{*this, execution->executor().forward_mode(execution->config.layer_placement.primary_device())};
        return tyloi::models::nemotron_h::mtp_predictor(ops, hidden, ids, supplied_positions, depth,
                                                     true);
    }

    DenseTensor forward(DenseTensor hidden, DenseTensor ids) override {
        return evaluate(hidden, ids).first;
    }

    // Depth chaining reuses the MTP's own final_layernorm output as the next
    // step's hidden state (llama.cpp/vLLM/SGLang), so the chain value is the
    // same normalized tensor the lm_head consumes. The backbone's output_norm
    // is not applied again.
    MtpStep step(DenseTensor hidden, DenseTensor ids) override {
        auto result = evaluate(hidden, ids);
        return {result.first, result.first};
    }

    MtpStep step_positioned(DenseTensor hidden, DenseTensor ids,
                            DenseTensor positions) override {
        auto result = evaluate(hidden, ids, 0, positions);
        return {result.first, result.first};
    }

    int64_t cache_position() const noexcept override {
        return lengths.empty() ? 0 : lengths.front();
    }

    void trim_cache_to(int64_t position) override {
        TYLOINFER_RUNTIME_CHECK(position >= 0 && position <= cache_position(),
                          "Nemotron-H MTP cache trim position is invalid");
        // The attention sub-block's KV storage is position addressed. Lowering
        // the logical boundary overwrites the rejected draft suffix without
        // copying history-sized tensors.
        lengths.front() = position;
    }

    bool teacher_forced_prompt_prime() const noexcept override { return true; }
    bool target_bootstrap_decode() const noexcept override { return false; }
    bool preserve_output_dtype() const noexcept override { return true; }
    bool retains_partial_target_prefix() const noexcept override { return true; }

    bool supports_session_state() const noexcept override { return true; }

    MtpSessionState
    capture_session_state(int64_t target_cache_position,
                          const ::tyloi::DenseTensor &last_target_hidden) const override {
        const int64_t predictor_position = target_cache_position - 1;
        if (predictor_position <= 0 || predictor_position > cache_position() ||
            !last_target_hidden.defined() || last_target_hidden.dim() != 3 ||
            last_target_hidden.size(0) != 1 || last_target_hidden.size(1) != 1 ||
            last_target_hidden.size(2) != config.hidden_size) {
            throw std::runtime_error("Nemotron-H MTP session boundary is unavailable");
        }
        MtpSessionState state;
        state.cache_pos = predictor_position;
        auto& snapshots = std::get<std::vector<KvSnapshot>>(state.payload);
        for (std::size_t index = 0; index < layers.size(); ++index) {
            if (!std::holds_alternative<FullAttention>(layers[index].weights))
                continue;
            MfqCudaGuard guard(layers[index].device);
            snapshots.push_back(
                capture_kv(std::get<KVCache>(states[index]), predictor_position, state.bytes));
        }
        state.last_target_hidden = last_target_hidden.clone();
        state.bytes += session_tensor_bytes(state.last_target_hidden);
        return state;
    }

    void restore_session_state(const MtpSessionState &state) override {
        std::vector<std::size_t> saved;
        for (std::size_t index = 0; index < layers.size(); ++index)
            if (std::holds_alternative<FullAttention>(layers[index].weights))
                saved.push_back(index);
        const auto *blocks = std::get_if<std::vector<KvSnapshot>>(&state.payload);
        if (state.cache_pos <= 0 || blocks == nullptr || blocks->size() != saved.size() ||
            !state.last_target_hidden.defined() || state.last_target_hidden.dim() != 3 ||
            state.last_target_hidden.size(0) != 1 || state.last_target_hidden.size(1) != 1 ||
            state.last_target_hidden.size(2) != config.hidden_size) {
            throw std::runtime_error("Nemotron-H MTP session state is incompatible");
        }
        for (std::size_t index = 0; index < saved.size(); ++index) {
            const auto layer_index = saved[index];
            MfqCudaGuard guard(layers[layer_index].device);
            const auto& attention = std::get<FullAttention>(layers[layer_index].weights);
            restore_kv(std::get<KVCache>(states[layer_index]), (*blocks)[index],
                attention.attention_head_dim,
                attention.official_bf16 ? ::tyloi::kBFloat16 : ::tyloi::kFloat16,
                attention.kv_quantization);
        }
        lengths.front() = state.cache_pos;
    }
};

} // namespace tyloi::cuda::nemotron_h
