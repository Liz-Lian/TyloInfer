#pragma once

#include "backend.h"

#include <cstdint>

namespace mfq::cuda {

class Recurrent {
public:
    struct State {
        mfq_tensor_backend::Tensor convolution, recurrent;
        bool transposed = true;

        bool defined() const noexcept {
            return convolution.defined() && recurrent.defined();
        }
    };

    Recurrent() = default;
    Recurrent(const Recurrent&) = delete;
    Recurrent& operator=(const Recurrent&) = delete;

    State& view() noexcept { return state_; }
    const State& view() const noexcept { return state_; }
    bool defined() const noexcept { return state_.defined(); }
    bool pending() const noexcept { return start_ >= 0; }

    void reset(int64_t batch);
    State snapshot() const;
    void restore(const State& snapshot);
    void begin(int64_t start, int64_t confirmed, int64_t tokens);
    // Keep the checkpoint until replay succeeds, so abort can undo a failed replay.
    int64_t rewind(int64_t position);
    void abort();
    void commit() noexcept;

private:
    void copy(const State& source);

    State state_, saved_;
    int64_t start_ = -1;
    int64_t confirmed_ = 0;
    int64_t tokens_ = 0;
};

} // namespace mfq::cuda
