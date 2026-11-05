#pragma once

#include "mfq_tensor_backend.h"

#include <cstdint>

namespace mfq::cuda {

// Convolution and recurrent state owned by one linear-attention mixer.
// GatedDeltaNet and Mamba-2 keep the same lifetime for both tensors, so the
// container and its speculative checkpoint live in the backend instead of an
// architecture directory. The owner decides which native kernel updates the
// state and how a rollback replays the retained token prefix.
struct RecurrentMixerState {
    mfq_tensor_backend::Tensor convolution;
    mfq_tensor_backend::Tensor recurrent;
    // Transaction storage: initial snapshots, or an out-of-place recurrent
    // result when the owner uses checkpoint_convolution().
    mfq_tensor_backend::Tensor speculative_convolution;
    mfq_tensor_backend::Tensor speculative_recurrent;
    bool speculative_pending = false;

    bool defined() const noexcept {
        return convolution.defined() && recurrent.defined();
    }

    // Zero the warmed state for a new batch, reallocating when its geometry
    // changed. Native graph warmup keeps the storage addresses stable.
    void reset(int64_t batch) {
        clear_speculative();
        if (defined() && convolution.size(0) == batch && recurrent.size(0) == batch) {
            convolution.zero_();
            recurrent.zero_();
            return;
        }
        convolution = mfq_tensor_backend::Tensor();
        recurrent = mfq_tensor_backend::Tensor();
    }

    // Snapshot the confirmed state before a speculative window is evaluated.
    void checkpoint() {
        speculative_convolution =
            snapshot(std::move(speculative_convolution), convolution);
        speculative_recurrent = snapshot(std::move(speculative_recurrent), recurrent);
        speculative_pending = true;
    }

    // Verification can keep recurrence read-only and place its result in
    // transaction storage instead of copying the initial recurrent matrix.
    // The owner commits or discards that result; restore() applies only to
    // the full snapshot produced by checkpoint().
    void checkpoint_convolution() {
        speculative_convolution =
            snapshot(std::move(speculative_convolution), convolution);
        speculative_recurrent = mfq_tensor_backend::Tensor();
        speculative_pending = true;
    }

    // Revert the confirmed state to the snapshot taken by checkpoint().
    void restore() {
        convolution.copy_(speculative_convolution);
        recurrent.copy_(speculative_recurrent);
        clear_speculative();
    }

    void clear_speculative() noexcept {
        speculative_convolution = mfq_tensor_backend::Tensor();
        speculative_recurrent = mfq_tensor_backend::Tensor();
        speculative_pending = false;
    }

  private:
    static mfq_tensor_backend::Tensor
    snapshot(mfq_tensor_backend::Tensor saved, const mfq_tensor_backend::Tensor& state) {
        if (!saved.defined() || saved.sizes() != state.sizes() ||
            saved.scalar_type() != state.scalar_type())
            return state.clone();
        saved.copy_(state);
        return saved;
    }
};

} // namespace mfq::cuda
