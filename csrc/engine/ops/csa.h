#pragma once
#include "engine/tensor.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace tyloi::ops {
/** Immutable projection and normalization weights for compressed sparse attention. */
struct CsaWeights {
    Tensor query_a, query_b, key_value, output_a, output_b;
    Tensor compressor_key_value, compressor_gate, index_query, index_key, index_score;
    Tensor query_a_norm, key_value_norm, sinks, compressor_norm, index_key_norm;
};
/** Layer geometry and numerical policy for compressed sparse attention. */
struct CsaConfig {
    std::int64_t hidden = 0, n_heads = 0, head_dim = 0, o_groups = 0;
    std::int64_t sliding_window = 0, maximum = 0, index_n_heads = 0, index_head_dim = 0;
    std::int64_t index_topk = 0, candidate_source_layer = -1, candidate_topk_blocks = 0;
    std::int64_t candidate_block_size = 0, layer = 0, ratio = 0;
    int device = 0;
    double rms_eps = 0, rope_theta = 0, compress_rope_theta = 0;
    std::int64_t rope_original = 0;
    double rope_factor = 1, rope_beta_fast = 32, rope_beta_slow = 1;
    bool kv_source = false, index_source = false;
};
/** Execute compressed sparse attention and own its numerical policy. */
class Csa {
public:
    /** Own one layer's cache and transactional checkpoint as copyable tensor handles. */
    struct State {
        State() = default;
        /** Copy semantic state; unfinished backend consumers remain owned by the source. */
        State(const State&);
        /** Replace semantic state without copying the source's unfinished consumers. */
        State& operator=(const State&);
        State(State&&) noexcept = default;
        State& operator=(State&&) noexcept = default;
        Tensor local_kv, compressed_kv, index_k, partial_kv, partial_score;
        Tensor speculative_local_slots, speculative_slot_ids, speculative_partial_kv, speculative_partial_score;
        Tensor speculative_projected_kv, speculative_projected_score;
        std::int64_t position = 0, compressed_length = 0, partial_length = 0;
        std::int64_t speculative_position = -1, speculative_compressed_length = 0, speculative_partial_length = 0;
        std::int64_t prefilled_global_end = -1;
        bool speculative_pending = false, replay_local_window = false;
    private:
        friend class Csa;
        std::vector<Tensor> failed;
    };
    /** Own mathematical values published between attention layers. */
    struct Shared {
        Tensor compressed_kv, index_k, topk, candidate_blocks;
        std::int64_t compressed_length = 0, ratio = 0;
    };
    /** Construct a numerical layer using its immutable weights and geometry. */
    Csa(CsaWeights weights, CsaConfig config);
    ~Csa();
    Csa(Csa&&) noexcept;
    Csa& operator=(Csa&&) noexcept;
    Csa(const Csa&) = delete;
    Csa& operator=(const Csa&) = delete;
    /** Reset cache after retiring consumers, reusing compatible storage. */
    void reset(State&, std::int64_t batch) const;
    /** Estimate cache allocations needed by reset. */
    std::size_t reset_growth(const State&, std::int64_t batch) const;
    /** Estimate the speculative checkpoint allocation for a suffix. */
    std::size_t suffix_growth(const State&, std::int64_t batch, std::int64_t tokens) const;
    /** Estimate compressor projection retention during speculation. */
    std::size_t growth(const State&, std::int64_t batch, std::int64_t tokens, bool speculative) const;
    /** Save mutable cache slots before a speculative suffix. */
    void begin_speculative(State&, std::int64_t tokens) const;
    /** Commit a speculative checkpoint. */
    void commit_speculative(State&) const;
    /** Restore rejected writes while retaining the accepted prefix. */
    void rollback_speculative(State&, std::int64_t position) const;
    /** Enumerate owning cache and checkpoint tensors. */
    std::vector<Tensor> storage(const State&) const;
    /** Publish compressed source values for causal encoder prefill. */
    void publish_compressed(State&, Shared&, const Tensor& input, std::int64_t position) const;
    /** Run attention and advance numerical cache state. */
    Tensor forward(State&, Shared&, const Tensor& input, const Tensor& positions,
        std::int64_t position, const Tensor& sequence_lengths = {}, bool truncated_local = false) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace tyloi::ops
