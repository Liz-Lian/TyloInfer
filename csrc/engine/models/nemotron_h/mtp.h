#pragma once

#include "engine/weights/loader.h"
#include "engine/models/session/snapshot.h"
#include "engine/backends/cuda/executor/options.h"

#ifndef TYLOINFER_NATIVE_CUDA
#include "engine/backends/cuda/executor/execution.h"
#endif
#include "engine/backends/cuda/ops/norm/rms.h"
#include "engine/models/common/mtp.h"
#include "engine/models/nemotron_h/forward.h"
#include "engine/models/nemotron_h/load.h"
#include "engine/models/nemotron_h/predictor.h"
#include "engine/state/kv_view.h"

#include <algorithm>
#include <array>
#include <optional>
#include <string>
#include <utility>

namespace tyloi::cuda::nemotron_h {

struct PredictorState final : tyloi::models::MtpState {
    explicit PredictorState(const tyloi::models::Mtp& owner) : MtpState(owner) {}

    int64_t position() const noexcept override {
        return lengths.empty() ? 0 : lengths.front();
    }

    std::vector<LayerState> states;
    std::vector<int64_t> lengths;
    int64_t batch = 0;
};

// Nemotron-H's predictor is one multi-token-prediction head whose sub-blocks
// are listed in mtp_layers_block_type; both blocks run inside the single stage.
// The attention sub-block owns the
// predictor history; the MoE sub-block is stateless.
// The semantics follow llama.cpp (src/models/nemotron-h-moe.cpp), vLLM and
// SGLang rather than DeepSeek-V3: the backbone hands over its post-output_norm
// hidden and each further draft step
// reuses this module's final_layernorm output. The final norm stays RMSNorm as
// in vLLM/SGLang (llama.cpp uses LayerNorm; the real model will A/B the two).
struct Mtp final : tyloi::models::Mtp {
    using DenseTensor = ::tyloi::DenseTensor;

    tyloi::cuda::nemotron_h::Config config;
    ::tyloi::Tensor embedding;
    int primary = 0;
    ::tyloi::Tensor fusion;
#ifdef TYLOINFER_NATIVE_CUDA
    ::tyloi::Tensor embedding_norm, hidden_norm, output_norm;
#else
    DenseTensor embedding_norm, hidden_norm, output_norm;
#endif
    std::vector<Layer> layers;
#ifndef TYLOINFER_NATIVE_CUDA
    CudaExecutionContext *execution = nullptr;
    std::unique_ptr<Executor> executor;
#endif

    // The predictor history shares the backbone context limit, which the
    // loader may have overridden with --ctx-size.
    int64_t predictor_capacity() const noexcept { return config.max_position_embeddings; }

#ifdef TYLOINFER_NATIVE_CUDA
    static std::optional<Mtp> load_if_present(
#else
    static std::optional<Mtp> load_if_present(CudaExecutionContext &execution,
#endif
                                              const ::tyloi::weights::Loader& weights,
                                              const tyloi::models::nemotron_h::Config &main,
                                              ::tyloi::Tensor embedding) {
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
#ifndef TYLOINFER_NATIVE_CUDA
        result.execution = &execution;
        result.executor = execution.executor().fork(model_parallel_primary_device(execution.config));
#endif
        result.embedding = std::move(embedding);
        result.primary = weights.placement.device;
#ifdef TYLOINFER_NATIVE_CUDA
        result.embedding_norm = weights.tensor("predictor.embedding_norm.weight");
        result.hidden_norm = weights.tensor("predictor.hidden_norm.weight");
        result.output_norm = weights.tensor("predictor.output_norm.weight");
#else
        result.embedding_norm = weights.tensor("predictor.embedding_norm.weight").dense();
        result.hidden_norm = weights.tensor("predictor.hidden_norm.weight").dense();
        result.output_norm = weights.tensor("predictor.output_norm.weight").dense();
#endif
        result.fusion = weights.weight("predictor.fusion.weight");
#ifdef TYLOINFER_NATIVE_CUDA
        TYLOINFER_RUNTIME_CHECK(result.fusion.size(1) == 2 * main.hidden_size &&
                              result.fusion.size(0) == main.hidden_size &&
                              result.embedding_norm.numel() == main.hidden_size &&
                              result.hidden_norm.numel() == main.hidden_size &&
                              result.output_norm.numel() == main.hidden_size,
                          "Nemotron-H MTP fusion/norm dimensions disagree with the backbone");
#else
        TYLOINFER_RUNTIME_CHECK(result.fusion->size(1) == 2 * main.hidden_size &&
                              result.fusion->size(0) == main.hidden_size &&
                              result.embedding_norm.numel() == main.hidden_size &&
                              result.hidden_norm.numel() == main.hidden_size &&
                              result.output_norm.numel() == main.hidden_size,
                          "Nemotron-H MTP fusion/norm dimensions disagree with the backbone");
#endif
        for (std::size_t index = 0; index < main.mtp_mixers.size(); ++index) {
            const auto type =
                std::string(tyloi::models::nemotron_h::Config::mixer_name(main.mtp_mixers[index]));
            result.layers.push_back(load_layer(
                                               weights, main,
                                               static_cast<int>(index), type, "predictor"));
        }
        return result;
    }

    std::unique_ptr<tyloi::models::MtpState> make_state() const override {
        auto result = std::make_unique<PredictorState>(*this);
        result->states.resize(layers.size());
        for (std::size_t index = 0; index < layers.size(); ++index) {
            const auto &layer = layers[index];
            if (const auto *full = std::get_if<FullAttention>(&layer.weights)) {
                auto &cache = result->states[index].emplace<tyloi::state::KvView>();
                if (::tyloi::cuda::options::kv_vram_preallocated()) {
                    MfqCudaGuard guard(layer.device);
                    cache = tyloi::state::allocate_kv(1, full->kv_heads,
                        full->sliding ? full->attention_window : full->max_position_embeddings,
                        full->attention_head_dim, full->sliding,
                        ::tyloi::Device(::tyloi::kCUDA, layer.device),
                        full->official_bf16 ? ::tyloi::kBFloat16 :
                                              ::tyloi::kFloat16,
                        full->kv_quantization);
                }
            }
        }
        result->lengths.assign(1, 0);
        return result;
    }

    void reset(tyloi::models::MtpState& value, int64_t next_batch = 1) const override {
        auto& state = this->state<PredictorState>(value);
        TYLOINFER_RUNTIME_CHECK(next_batch > 0, "Nemotron-H MTP batch must be positive");
        for (auto &layer : state.states)
            if (auto *cache = std::get_if<tyloi::state::KvView>(&layer))
                if (!cache->defined() || cache->batch != next_batch)
                    *cache = {};
        std::fill(state.lengths.begin(), state.lengths.end(), 0);
        state.batch = next_batch;
    }

    std::pair<DenseTensor, DenseTensor> evaluate(tyloi::models::MtpState& value,
                                       const DenseTensor &hidden,
                                       const DenseTensor &ids, int64_t depth = 0,
                                       const DenseTensor &supplied_positions = {}) const {
        auto& state = this->state<PredictorState>(value);
        struct Ops {
            using DenseTensor = ::tyloi::DenseTensor;
            const Mtp &model;
            PredictorState& state;
#ifdef TYLOINFER_NATIVE_CUDA
            tyloi::models::ForwardOptions mode;
#else
            Executor::Mode mode;
#endif
            static int rank(const DenseTensor &x) { return x.dim(); }
            static int64_t size(const DenseTensor &x, int axis) { return x.size(axis); }
            static int64_t elements(const DenseTensor &x) { return x.numel(); }
            static bool defined(const DenseTensor &x) { return x.defined(); }
            DenseTensor embed(DenseTensor ids, const DenseTensor &like) {
#ifdef TYLOINFER_NATIVE_CUDA
                auto embedded = tyloi::ops::embedding(::tyloi::Tensor(std::move(ids)), model.embedding);
                return tyloi::ops::contiguous(tyloi::ops::cast(
                    embedded, like.scalar_type())).dense();
#else
                return tyloi::cuda::embed(model.embedding, std::move(ids),
                    model.execution->executor().collectives(), model.execution->executor().profiler(), model.primary)
                    .to(like.scalar_type());
#endif
            }
            DenseTensor normalize(const DenseTensor &x, tyloi::models::nemotron_h::PredictorNorm role) {
                using Norm = tyloi::models::nemotron_h::PredictorNorm;
                const auto &weight = role == Norm::embedding ? model.embedding_norm
                                     : role == Norm::hidden  ? model.hidden_norm
                                                             : model.output_norm;
#ifdef TYLOINFER_NATIVE_CUDA
                auto normalized = tyloi::ops::rms_norm(
                    ::tyloi::Tensor(x.reshape({-1, model.config.hidden_size})), weight,
                    model.config.rms_norm_eps, 0.0);
                return tyloi::ops::reshape(normalized, x.sizes()).dense();
#else
                auto normalized = qwen_rms_norm(x.reshape({-1, model.config.hidden_size})
                                         .to(::tyloi::kFloat32),
                                     weight, model.config.rms_norm_eps, 0.0);
                return normalized.reshape(x.sizes());
#endif
            }
            DenseTensor positions(DenseTensor explicit_positions, int64_t start, int64_t tokens) {
                if (explicit_positions.defined())
#ifdef TYLOINFER_NATIVE_CUDA
                    return tyloi::ops::contiguous(tyloi::ops::cast(
                        ::tyloi::Tensor(std::move(explicit_positions)),
                        tyloi::ScalarType::int64)).dense();
#else
                    return explicit_positions.to(::tyloi::kInt64).contiguous();
#endif
                // NoPE: positions only address the predictor's KV history.
#ifdef TYLOINFER_NATIVE_CUDA
                return tyloi::ops::arange(start, start + tokens,
                    tyloi::Device{tyloi::DeviceType::cuda, model.primary}).dense();
#else
                return ::tyloi::cuda::arange(start, start + tokens,
                                                  ::tyloi::TensorOptions()
                                                      .device(::tyloi::kCUDA)
                                                      .dtype(::tyloi::kInt64));
#endif
            }
            DenseTensor fuse(DenseTensor e, DenseTensor h) {
#ifdef TYLOINFER_NATIVE_CUDA
                auto joined = tyloi::ops::cat(std::array<::tyloi::Tensor, 2>{
                    ::tyloi::Tensor(std::move(e)), ::tyloi::Tensor(std::move(h))}, -1);
                return tyloi::ops::project(joined, model.fusion).dense();
#else
                return model.executor->projector(mode.ffn).run(
                    model.fusion, ::tyloi::cuda::cat({e, h}, -1));
#endif
            }
            static DenseTensor lengths(int64_t batch, int64_t end, const DenseTensor &positions) {
#ifdef TYLOINFER_NATIVE_CUDA
                const std::array<int64_t, 1> shape{batch};
                return tyloi::ops::full(shape, end, positions.device()).dense();
#else
                return ::tyloi::cuda::full({batch}, end, positions.options());
#endif
            }
            // One predictor stage: every declared sub-block adds its own
            // pre-normed mixer branch to the same residual stream.
            DenseTensor mixer_blocks(DenseTensor x, int64_t stage, const DenseTensor &pos) {
                const int64_t tokens = x.size(1);
                std::optional<DenseTensor> sequence_lengths;
                if (tokens == 1 && state.lengths[stage] > 0)
                    sequence_lengths = lengths(x.size(0), state.lengths[stage] + 1, pos);
#ifdef TYLOINFER_NATIVE_CUDA
                AttentionStep step{pos, sequence_lengths.value_or(::tyloi::DenseTensor()), ::tyloi::Tensor{},
                                   state.lengths[stage], 0, 0};
#else
                AttentionStep step{pos, sequence_lengths.value_or(::tyloi::DenseTensor()), mfq_nullopt,
                                   state.lengths[stage], 0, 0};
#endif
                for (std::size_t index = 0; index < model.layers.size(); ++index)
#ifdef TYLOINFER_NATIVE_CUDA
                    x = forward_layer(model.layers[index], state.states[index],
                                      std::move(x), step, 0, mode);
#else
                    x = forward_layer(model.layers[index], state.states[index], *model.executor,
                                      index, std::move(x), step, 0, mode);
#endif
                return x;
            }
#ifdef TYLOINFER_NATIVE_CUDA
        } ops{*this, state,
              tyloi::models::ForwardOptions{config.max_position_embeddings, nullptr, -1, primary}};
#else
        } ops{*this, state,
              execution->executor().forward_mode(execution->config.layer_placement.primary_device())};
#endif
        return tyloi::models::nemotron_h::mtp_predictor(ops, hidden, ids, supplied_positions, depth,
                                                     true);
    }

    // Depth chaining reuses the MTP's own final_layernorm output as the next
    // step's hidden state (llama.cpp/vLLM/SGLang), so the chain value is the
    // same normalized tensor the lm_head consumes. The backbone's output_norm
    // is not applied again.
    tyloi::models::MtpStep step(tyloi::models::MtpState& value,
                                const ::tyloi::Tensor& hidden, const ::tyloi::Tensor& ids,
                                const ::tyloi::Tensor& positions = {}) const override {
        auto result = evaluate(value, hidden.dense(),
                               ids.dense(), 0,
                               positions.defined() ? positions.dense()
                                                                 : DenseTensor{});
        auto output = std::make_shared<DenseTensor>(std::move(result.first));
        return {output, output};
    }

    void trim_cache_to(tyloi::models::MtpState& value, int64_t position) const override {
        auto& state = this->state<PredictorState>(value);
        TYLOINFER_RUNTIME_CHECK(position >= 0 && position <= state.position(),
                          "Nemotron-H MTP cache trim position is invalid");
        // The attention sub-block's KV storage is position addressed. Lowering
        // the logical boundary overwrites the rejected draft suffix without
        // copying history-sized tensors.
        state.lengths.front() = position;
    }

    bool teacher_forced_prompt_prime() const noexcept override { return true; }
    bool target_bootstrap_decode() const noexcept override { return false; }
    bool preserve_output_dtype() const noexcept override { return true; }
    bool retains_partial_target_prefix() const noexcept override { return true; }

    bool supports_session_state() const noexcept override { return true; }

    MtpSessionState
    capture_session_state(const tyloi::models::MtpState& value,
                          int64_t target_cache_position,
                          const ::tyloi::Tensor &last_target) const override {
        const auto& state = this->state<PredictorState>(value);
        const auto& last_target_hidden = last_target.dense();
        const int64_t predictor_position = target_cache_position - 1;
        if (predictor_position <= 0 || predictor_position > state.position() ||
            !last_target_hidden.defined() || last_target_hidden.dim() != 3 ||
            last_target_hidden.size(0) != 1 || last_target_hidden.size(1) != 1 ||
            last_target_hidden.size(2) != config.hidden_size) {
            throw std::runtime_error("Nemotron-H MTP session boundary is unavailable");
        }
        MtpSessionState result;
        result.cache_pos = predictor_position;
        auto& snapshots = std::get<std::vector<KvSnapshot>>(result.payload);
        for (std::size_t index = 0; index < layers.size(); ++index) {
            if (!std::holds_alternative<FullAttention>(layers[index].weights))
                continue;
            MfqCudaGuard guard(layers[index].device);
            snapshots.push_back(
                capture_kv(std::get<tyloi::state::KvView>(state.states[index]), predictor_position, result.bytes));
        }
        auto hidden = last_target_hidden.clone();
        result.bytes += session_tensor_bytes(hidden);
        result.last_target_hidden = std::make_shared<::tyloi::DenseTensor>(std::move(hidden));
        return result;
    }

    void restore_session_state(tyloi::models::MtpState& value,
                               const MtpSessionState &snapshot) const override {
        auto& state = this->state<PredictorState>(value);
        std::vector<std::size_t> saved;
        for (std::size_t index = 0; index < layers.size(); ++index)
            if (std::holds_alternative<FullAttention>(layers[index].weights))
                saved.push_back(index);
        const auto *blocks = std::get_if<std::vector<KvSnapshot>>(&snapshot.payload);
        const auto *hidden = snapshot.last_target_hidden.representation<::tyloi::DenseTensor>();
        if (snapshot.cache_pos <= 0 || blocks == nullptr || blocks->size() != saved.size() ||
            !hidden || !hidden->defined() || hidden->dim() != 3 ||
            hidden->size(0) != 1 || hidden->size(1) != 1 || hidden->size(2) != config.hidden_size) {
            throw std::runtime_error("Nemotron-H MTP session state is incompatible");
        }
        for (std::size_t index = 0; index < saved.size(); ++index) {
            const auto layer_index = saved[index];
            MfqCudaGuard guard(layers[layer_index].device);
            const auto& attention = std::get<FullAttention>(layers[layer_index].weights);
            restore_kv(std::get<tyloi::state::KvView>(state.states[layer_index]), (*blocks)[index],
                attention.attention_head_dim,
                attention.official_bf16 ? ::tyloi::kBFloat16 : ::tyloi::kFloat16,
                attention.kv_quantization);
        }
        state.lengths.front() = snapshot.cache_pos;
    }
};

} // namespace tyloi::cuda::nemotron_h
