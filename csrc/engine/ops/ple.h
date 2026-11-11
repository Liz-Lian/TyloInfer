#pragma once

#include "engine/tensor.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace tyloi::ops {
/** Immutable projection, normalization and convolution weights for PLE. */
struct PleWeights {
    Tensor key, value, key_norm, query_norm, conv_norm, conv;
};

/** Execute PLE numerical operations against request-owned convolution state. */
class Ple {
public:
    /** Own convolution history and speculative prefix checkpoints. */
    struct State {
        State();
        ~State();
        State(State&&) noexcept;
        State& operator=(State&&) noexcept;
        State(const State&) = delete;
        State& operator=(const State&) = delete;
        Tensor conv, saved_conv;
        std::vector<Tensor> saved_prefix_conv;
        std::int64_t saved_prefix_position = -1;
        bool saved = false;
    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
        friend class Ple;
    };
    /** Construct PLE from weights, channel geometry and numerical policy. */
    Ple(PleWeights weights, std::int64_t hidden, std::int64_t streams,
        std::int64_t dilation, double eps);
    ~Ple();
    Ple(Ple&&) noexcept;
    Ple& operator=(Ple&&) noexcept;
    Ple(const Ple&) = delete;
    Ple& operator=(const Ple&) = delete;
    /** Reset history after retiring consumers, optionally retaining stable graph storage. */
    void reset(State&, bool retain_storage = false) const;
    /** Commit and release a speculative checkpoint. */
    void commit(State&) const;
    /** Restore the saved checkpoint or a retained prefix. */
    void rollback(State&, std::int64_t keep = -1) const;
    /** Save the current convolution history for a speculative suffix. */
    void checkpoint(State&) const;
    /** Bind externally owned graph checkpoints to a speculative prefix. */
    void bind_graph_checkpoint(State&, const std::vector<Tensor>& conv, std::int64_t position) const;
    /** Restore session history, optionally retaining graph-stable storage. */
    void restore_session(State&, const Tensor& convolution, bool retain_storage = false) const;
    /** Return the request's convolution history. */
    const Tensor& conv_state(const State&) const;
    /** Return stable graph state slots. */
    std::vector<Tensor*> graph_state(State&) const;
    /** Enumerate state and retained checkpoint backings. */
    std::vector<Tensor> storage(const State&) const;
    /** Estimate missing convolution backing for a batch. */
    std::size_t growth_bytes(const State&, std::int64_t batch) const;
    /** Compute one chunk from prepared embeddings, optionally writing prefix snapshots. */
    Tensor forward_chunk(State&, const Tensor& input, const Tensor& embeddings, bool cache,
        std::vector<Tensor>* checkpoints = nullptr, std::size_t checkpoint_offset = 0) const;
    /** Compute a verification window and write convolution prefix checkpoints. */
    Tensor forward_graph(State&, const Tensor& input, const Tensor& embeddings, std::int64_t confirmed,
        std::vector<Tensor>& checkpoints) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace tyloi::ops
