#pragma once

#include "engine/backends/cuda/executor/decode_window.h"

namespace tyloi::cuda {
/** Retain and publish the profiling points used by one fused residual launch. */
class ResidualTrace {
public:
    explicit ResidualTrace(tyloi::ops::ResidualRole role) : window_(DecodeWindow::recording()) {
        if (window_) data_ = window_->prepare_residual_trace(role);
    }
    ~ResidualTrace() { if (window_) window_->commit_residual_trace(data_); }
    ResidualTrace(const ResidualTrace&) = delete;
    ResidualTrace& operator=(const ResidualTrace&) = delete;
    /** Publish recorded points and enforce the profiling capacity. */
    void finish() {
        if (!window_) return;
        auto* window=window_;
        window_=nullptr;
        window->commit_residual_trace(data_);
        window->validate_residual_trace();
    }
    ResidualTraceData& data() noexcept { return data_; }
private:
    DecodeWindow* window_;
    ResidualTraceData data_;
};
} // namespace tyloi::cuda
