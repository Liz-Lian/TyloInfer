#pragma once

#include "engine/tensor.h"
#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace tyloi::ops {
/** Immutable projections and numerical weights for recurrent linear attention. */
struct LinearWeights {
    Tensor attn_norm, conv_weight, conv_bias, dt_bias, a_log, linear_norm;
    std::vector<Tensor> in_proj, qkv_proj, ab_nint_proj, ab_proj, zab_proj;
    Tensor z_proj, out_proj, out_proj_dense;
};
/** Recurrent geometry, weight layout and execution target. */
struct LinearConfig {
    std::int64_t key_heads = 0, value_heads = 0, key_dim = 0, value_dim = 0, conv_kernel = 0;
    double rms_norm_eps = 1e-6, norm_weight_offset = 1.0;
    bool split_in_proj = false, split_dense_zab = false, dense_ab_tail = false;
    bool ab_is_nint = false, dense_out_proj = false, tiled_v_heads = false;
    Device target = Device(kCUDA, 0);
};
/** Own the prepared projections and numerical recurrence for one layer. */
class Linear {
public:
    /** Own recurrent payloads and a pending projected verification tape. */
    class State {
    public:
        /** Semantic recurrent payloads for sessions and graph staging. */
        struct Snapshot { Tensor convolution, recurrent; bool transposed = true; };
        /** Create an empty request recurrence. */
        State();
        ~State();
        State(State&&) noexcept;
        State& operator=(State&&) noexcept;
        State(const State&) = delete;
        State& operator=(const State&) = delete;
        /** Return whether both recurrent payloads exist. */
        bool defined() const noexcept;
        /** Return whether a speculative verification is pending. */
        bool pending() const noexcept;
        /** Return whether a projected replay tape exists. */
        bool has_tape() const noexcept;
        /** Borrow semantic payload handles without cloning their storage. */
        Snapshot view() const;
        /** Clone the confirmed recurrent payloads for a session. */
        Snapshot snapshot() const;
        /** Copy a session snapshot into suitable resident storage. */
        void restore(const Snapshot& value);
        /** Initialize an empty batched recurrence using an existing payload layout. */
        void initialize(const Snapshot& layout, std::int64_t batch);
        /** Reset recurrence, retaining compatible batch storage. */
        void reset(std::int64_t batch);
        /** Commit pending recurrence and discard the projected tape. */
        void commit() noexcept;
        /** Restore the verification checkpoint and return retained token count. */
        std::int64_t rewind(std::int64_t position);
        /** Restore and discard a pending verification checkpoint. */
        void abort();
        /** Return stable handles used by graph checkpointing. */
        std::vector<Tensor*> graph_state();
        /** Return retained recurrence and projected replay storage. */
        std::vector<Tensor> storage() const;
    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
        friend class Linear;
    };
    /** Prepare immutable weights for the supplied geometry and execution target. */
    Linear(LinearWeights weights, LinearConfig config);
    ~Linear();
    Linear(Linear&&) noexcept;
    Linear& operator=(Linear&&) noexcept;
    Linear(const Linear&) = delete;
    Linear& operator=(const Linear&) = delete;
    /** Execute attention, optionally checkpointing a confirmed prefix for replay. */
    std::array<Tensor, 2> forward(State& state, const Tensor& input,
                                  std::int64_t cache_position, std::int64_t confirmed) const;
    /** Replay retained projected tokens after State::rewind. */
    void replay(State& state, std::int64_t retained_tokens) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
