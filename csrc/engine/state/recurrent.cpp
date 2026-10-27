#include "engine/state/recurrent.h"

#include <utility>

namespace mfq::cuda {

namespace {
mfq_tensor_backend::Tensor storage(const mfq_tensor_backend::Tensor& target,
                                   const mfq_tensor_backend::Tensor& source) {
    if (target.defined() && target.sizes() == source.sizes() &&
        target.scalar_type() == source.scalar_type() && target.device() == source.device())
        return target;
    return mfq_tensor_backend::empty(source.sizes(), source.options());
}
} // namespace

void Recurrent::reset(int64_t batch) {
    commit();
    if (defined() && state_.convolution.size(0) == batch && state_.recurrent.size(0) == batch) {
        state_.convolution.zero_();
        state_.recurrent.zero_();
    } else {
        state_ = {};
    }
}

Recurrent::State Recurrent::snapshot() const {
    TYLOINFER_RUNTIME_CHECK(defined() && !pending(),
        "recurrent snapshot requires confirmed state");
    return {state_.convolution.clone(), state_.recurrent.clone(), state_.transposed};
}

void Recurrent::copy(const State& source) {
    TYLOINFER_RUNTIME_CHECK(source.defined(), "recurrent snapshot is unavailable");
    // Prepare both allocations before changing either live tensor.
    auto convolution = storage(state_.convolution, source.convolution);
    auto recurrent = storage(state_.recurrent, source.recurrent);
    convolution.copy_(source.convolution);
    recurrent.copy_(source.recurrent);
    state_ = {std::move(convolution), std::move(recurrent), source.transposed};
}

void Recurrent::restore(const State& snapshot) {
    copy(snapshot);
    commit();
}

void Recurrent::begin(int64_t start, int64_t confirmed, int64_t tokens) {
    TYLOINFER_RUNTIME_CHECK(start >= 0 && confirmed > 0 && confirmed < tokens,
        "invalid recurrent verification window");
    auto saved = snapshot();
    saved_ = std::move(saved);
    start_ = start;
    confirmed_ = confirmed;
    tokens_ = tokens;
}

int64_t Recurrent::rewind(int64_t position) {
    TYLOINFER_RUNTIME_CHECK(pending() && position >= start_ &&
        position - start_ >= confirmed_ && position - start_ < tokens_,
        "invalid recurrent rollback position");
    copy(saved_);
    return position - start_;
}

void Recurrent::abort() {
    if (!pending()) return;
    copy(saved_);
    commit();
}

void Recurrent::commit() noexcept {
    saved_ = {};
    start_ = -1;
    confirmed_ = 0;
    tokens_ = 0;
}

} // namespace mfq::cuda
