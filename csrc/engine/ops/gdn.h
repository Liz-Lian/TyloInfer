#pragma once

#include "engine/tensor.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace tyloi::ops {
namespace detail { class GdnCore; }

/** Immutable projection and recurrence weights for a gated delta network. */
struct GdnWeights {
    std::vector<Tensor> input_projection;
    Tensor output, conv, dt_bias, a_log, norm;
    std::size_t qkv_count = 1;
};

/** Own gated delta network numerical policy and execute against request state. */
class Gdn {
public:
    /** Own recurrent values and rollback checkpoints for one request. */
    struct State {
        State();
        ~State();
        State(State&&) noexcept;
        State& operator=(State&&) noexcept;
        State(const State&) = delete;
        State& operator=(const State&) = delete;
        Tensor conv, recurrent, saved_conv, saved_state;
        std::vector<Tensor> saved_prefix_conv, saved_prefix_state;
        std::int64_t saved_prefix_position = -1;
    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
        friend class Gdn;
        friend class detail::GdnCore;
    };

    /** Preserve projection row geometry while evaluating a complete prefill layer. */
    class ProjectionScope {
    public:
        explicit ProjectionScope(std::int64_t rows);
        ~ProjectionScope();
        ProjectionScope(const ProjectionScope&) = delete;
        ProjectionScope& operator=(const ProjectionScope&) = delete;
    private:
        std::int64_t previous_;
    };

    /** Freeze backend numerical policy for the supplied weights and recurrence geometry. */
    Gdn(GdnWeights weights, std::int64_t key_heads, std::int64_t value_heads,
        std::int64_t width, std::int64_t kernel, double eps, bool silu_gate);
    ~Gdn();
    Gdn(Gdn&&) noexcept;
    Gdn& operator=(Gdn&&) noexcept;
    Gdn(const Gdn&) = delete;
    Gdn& operator=(const Gdn&) = delete;
    /** Reset request state, optionally retaining graph-stable storage. */
    void reset(State&, bool retain_storage = false) const;
    /** Commit a speculative state checkpoint. */
    void commit(State&) const;
    /** Restore a saved checkpoint or retained prefix. */
    void rollback(State&, std::int64_t keep = -1) const;
    /** Allocate persistent request workspace after admission. */
    void initialize_request_cache(State&, bool allocate_counters = true) const;
    /** Estimate missing recurrent backing. */
    std::size_t growth_bytes(const State&) const;
    /** Estimate recurrent backing and persistent workspace. */
    std::size_t request_cache_growth_bytes(const State&) const;
    /** Restore recurrent values into compatible request storage. */
    void restore_session(State&, const Tensor& convolution, const Tensor& recurrent, bool retain_storage = false) const;
    /** Return the logical recurrent matrix view. */
    Tensor recurrent_state(const State&) const;
    /** Return stable recurrent slots required by graph capture. */
    std::vector<Tensor*> graph_state(State&) const;
    /** Enumerate request-owned recurrent and private workspace handles. */
    std::vector<Tensor> storage(const State&) const;
    /** Project input rows using the configured grouped policy. */
    std::vector<Tensor> project_inputs(const Tensor&) const;
    /** Project a recurrent output into the model width. */
    Tensor project_output(const Tensor&) const;
    /** Prepare stable addresses for independent request rows when supported. */
    static Tensor prepare_independent_batch(const std::vector<std::pair<const Gdn*, State*>>& rows);
    /** Run one independent batch using prepared state addresses. */
    Tensor forward_independent_batch(State&, const Tensor&, const std::vector<Tensor>&, const Tensor&) const;
    /** Execute a graph window and update its supplied checkpoints. */
    Tensor forward_graph(State&, const Tensor&, std::int64_t confirmed,
        std::vector<Tensor>& checkpoint_conv, std::vector<Tensor>& checkpoint_state) const;
    /** Bind graph checkpoints for speculative rollback. */
    void bind_graph_checkpoint(State&, const std::vector<Tensor>& conv,
        const std::vector<Tensor>& recurrent, std::int64_t position) const;
    /** Execute a request sequence and maintain speculative checkpoints. */
    Tensor forward(State&, const Tensor&, bool cache, std::int64_t confirmed = 0,
        std::int64_t cache_position = -1) const;
    /** Execute a bounded sequence chunk with optional precomputed projections. */
    Tensor forward_chunk(State&, const Tensor&, bool cache, std::vector<Tensor> projections = {},
        bool project_output = true, Tensor checkpoint_conv = {}, Tensor checkpoint_state = {},
        bool* snapshot_written = nullptr) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
