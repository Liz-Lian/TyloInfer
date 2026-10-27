#pragma once

#include "backends/cuda/engine/executor/execution.h"
#include "backends/cuda/engine/executor/run.h"
#include "backends/cuda/engine/state/session/state.h"
#include "engine/models/qwen35/model.h"
#include "engine/state/qwen35.h"
#include "ops/attention/full.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

namespace mfq::cuda::qwen35 {

using Tensor = mfq_tensor_backend::Tensor;
struct Plan {
    std::int64_t kv_length = 0;
    std::int64_t attention_parts = 0;
};

class Run {
public:
    Run(std::shared_ptr<const Model> weights, CudaExecutionContext& execution);
    ~Run() = default;
    Run(const Run&) = delete;
    Run& operator=(const Run&) = delete;
    Run(Run&&) noexcept = default;
    Run& operator=(Run&&) noexcept = default;

    std::int64_t vocab_size() const noexcept;
    std::int64_t hidden_size() const noexcept;
    std::int64_t num_hidden_layers() const noexcept;
    std::int64_t num_attention_heads() const noexcept;
    std::int64_t num_key_value_heads() const noexcept;
    std::int64_t head_dim() const noexcept;
    std::int64_t max_position_embeddings() const noexcept;
    std::int64_t rotary_dim() const noexcept;
    double rope_base() const noexcept;
    double rms_norm_eps() const noexcept;
    double norm_weight_offset() const noexcept;
    double final_logit_softcapping() const noexcept;
    double embedding_scale() const noexcept;
    std::int64_t hc_mult() const noexcept;
    double hc_eps() const noexcept;
    bool tie_word_embeddings() const noexcept;
    std::int64_t num_experts() const noexcept;
    std::string_view model_type() const noexcept;
    std::string_view layer_type(std::int64_t layer) const;

    void reset(std::int64_t batch);
    void preallocate();
    Tensor embed_forward(Tensor ids) const;
    Tensor hidden_forward(
        Tensor ids,
        std::optional<Tensor> positions = {},
        std::optional<Tensor> lengths = {},
        std::vector<Tensor>* trace = nullptr,
        std::optional<Tensor> cache_positions = {},
        Tensor* raw_hidden = nullptr,
        std::int64_t confirmed_prefix = 0,
        Plan plan = {});
    Tensor hidden_forward_inputs(
        Tensor ids,
        Tensor embeddings,
        std::optional<Tensor> positions = {},
        std::optional<Tensor> lengths = {},
        std::vector<Tensor>* trace = nullptr,
        std::optional<Tensor> attention_mask = {},
        bool advance_cache_with_position_ids = false,
        std::optional<Tensor> cache_positions = {},
        Tensor* raw_hidden = nullptr,
        std::int64_t confirmed_prefix = 0,
        Plan plan = {});
    Tensor finalize_hidden(Tensor hidden, std::int64_t batch, std::int64_t tokens) const;
    Tensor apply_final_logit_softcap(Tensor logits) const;
    Tensor logits_from_hidden(Tensor hidden) const;
    Tensor next_token_from_hidden(Tensor hidden) const;
    Tensor forward(Tensor ids);
    Tensor forward_inputs(
        Tensor ids,
        Tensor embeddings,
        std::optional<Tensor> positions = {},
        std::optional<Tensor> lengths = {});
    Tensor last_logits(Tensor ids);
    Tensor next_token(Tensor ids);
    Tensor hidden_forward_static(Tensor ids, Tensor positions, Tensor lengths, Plan plan = {});
    Tensor last_logits_static(Tensor ids, Tensor positions, Tensor lengths, Plan plan = {});
    Tensor next_token_static(Tensor ids, Tensor positions, Tensor lengths, Plan plan = {});

    bool supports_speculation() const noexcept;
    bool supports_suffix_speculation() const noexcept { return false; }
    void commit_speculative();
    void rollback_speculative(std::int64_t accepted_suffix = 0);

    TextSessionStateKind text_session_state_kind() const;
    bool supports_text_session_state() const;
    bool supports_paged_text_session_state() const;
    TextSessionState capture_text_session_state(const std::vector<std::int64_t>& tokens) const;
    void restore_text_session_state(const TextSessionState& state);

    std::vector<Tensor*> graph_state();
    std::vector<Tensor> graph_storage() const;

    std::shared_ptr<const Model> weights;
    CudaExecutionContext* execution = nullptr;
    std::unique_ptr<mfq::cuda::Executor> executor;
    std::function<void()> restore_decode_workspace;
    std::vector<State> states;
    std::int64_t cache_pos = 0;
    std::int64_t decode_position_delta = 0;
    std::int64_t speculative_start = -1;
    std::int64_t speculative_confirmed = 0;
    std::int64_t batch = 0;

private:
    Tensor normalize_ids(Tensor ids) const;
    std::optional<Tensor> decode_lengths(const Tensor& ids) const;
    Tensor last_hidden(Tensor hidden) const;
};

Tensor forward_layer(
    CudaExecutionContext& execution,
    const Layer& layer,
    State& state,
    mfq::cuda::Executor& executor,
    std::size_t scratch_offset,
    Tensor input,
    const AttentionStep& step,
    std::int64_t confirmed_prefix,
    const RopeCache& rope);

} // namespace mfq::cuda::qwen35
