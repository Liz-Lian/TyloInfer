#pragma once

#include "engine/tensor.h"

#include <cstdint>
#include <vector>

namespace tyloi::cuda {

/** Own recurrent payloads and the checkpoint of an uncommitted verification window. */
class Recurrent {
public:
    struct State {
        ::tyloi::Tensor convolution, recurrent;
        bool transposed = true;

        /** Return whether both payloads have backing storage. */
        bool defined() const noexcept {
            return convolution.defined() && recurrent.defined();
        }
    };

    Recurrent() = default;
    Recurrent(const Recurrent&) = delete;
    Recurrent& operator=(const Recurrent&) = delete;
    /** Transfer the recurrent payloads and any pending checkpoint. */
    Recurrent(Recurrent&&) noexcept = default;
    /** Replace the recurrent payloads and any pending checkpoint by moving them. */
    Recurrent& operator=(Recurrent&&) noexcept = default;

    /** Borrow the owned payload objects without replacing their identity. */
    State& view() noexcept { return state_; }
    /** Borrow the current payload handles for read-only inspection. */
    const State& view() const noexcept { return state_; }
    /** Return whether the live recurrence has both payloads. */
    bool defined() const noexcept { return state_.defined(); }
    /** Return whether a verification checkpoint remains uncommitted. */
    bool pending() const noexcept { return start_ >= 0; }
    /** Return owning handles for live state and any speculative checkpoint. */
    std::vector<::tyloi::Tensor> storage() const;

    /** Zero compatible batch storage, releasing incompatible storage and speculative history. */
    void reset(int64_t batch);
    /** Clone confirmed payloads so subsequent computation cannot change the snapshot. */
    State snapshot() const;
    /** Copy a snapshot into compatible storage and retire speculative history. */
    void restore(const State& snapshot);
    /** Save confirmed state before executing a speculative verification window. */
    void begin(int64_t start, int64_t confirmed, int64_t tokens);
    /** Restore the checkpoint and return retained tokens, keeping it until replay commits. */
    int64_t rewind(int64_t position);
    /** Restore pending state and discard its verification window. */
    void abort();
    /** Discard speculative history after the owning model commits or resets. */
    void commit() noexcept;

private:
    void copy(const State& source);

    State state_, saved_;
    int64_t start_ = -1;
    int64_t confirmed_ = 0;
    int64_t tokens_ = 0;
};

} // namespace tyloi::cuda
