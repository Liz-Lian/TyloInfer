#pragma once

#include "backends/cuda/engine/executor/execution.h"
#include "backends/cuda/engine/executor/run.h"
#include "models/qwen35/model.h"
#include "models/qwen35/state.h"
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
    void rollback_speculative(std::int64_t accepted_suffix = 0);

    std::vector<Tensor> graph_storage() const;

    std::shared_ptr<const Model> weights;
    CudaExecutionContext* execution = nullptr;
    std::unique_ptr<mfq::cuda::Executor> executor;
    std::function<void()> restore_decode_workspace;
    State state;

private:
    Tensor normalize_ids(Tensor ids) const;
    std::optional<Tensor> decode_lengths(const Tensor& ids) const;
    Tensor last_hidden(Tensor hidden) const;
};

} // namespace mfq::cuda::qwen35
