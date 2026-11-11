#pragma once

#include "engine/tensor.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tyloi::models {

struct ForwardOptions;
struct Mtp;

class MtpState {
public:
    explicit MtpState(const Mtp& model) : model_(&model) {}
    virtual ~MtpState() = default;
    MtpState(const MtpState&) = delete;
    MtpState& operator=(const MtpState&) = delete;
    bool belongs_to(const Mtp& model) const noexcept { return model_ == &model; }
    virtual std::int64_t position() const noexcept = 0;
private:
    const Mtp* model_;
};

struct CausalState {
    int64_t position() const noexcept { return cache_pos; }
    void advance(int64_t tokens) noexcept { cache_pos += tokens; }
    void set_position_delta(int64_t delta) noexcept { decode_position_delta = delta; }

    int64_t cache_pos = 0;
    int64_t decode_position_delta = 0;
    int64_t speculative_start = -1;
    int64_t speculative_confirmed = 0;
    bool speculative_suffix_forward = false;
};

class State {
public:
    State() = default;
    State(State&&) noexcept = default;
    State& operator=(State&&) noexcept = default;
    State(const State&) = delete;
    State& operator=(const State&) = delete;
    virtual ~State() = default;
    std::unique_ptr<MtpState> predictor;
    virtual void preallocate() = 0;
    virtual void reset(std::int64_t batch) = 0;
    virtual std::int64_t position() const noexcept = 0;
    virtual void advance(std::int64_t tokens) noexcept = 0;
    virtual void set_position_delta(std::int64_t delta) noexcept = 0;
    virtual std::vector<std::pair<tyloi::Device, std::size_t>> growth(
        std::int64_t, const ForwardOptions&) const {
        throw std::logic_error("state allocation planning is not implemented");
    }
    // Owners keep storage alive; deduplicate using backing identity, not object identity.
    virtual std::vector<::tyloi::Tensor> storage() const = 0;
    // Borrow writable objects from this state; their lifetime remains tied to it.
    virtual std::vector<tyloi::Tensor*> graph_state() = 0;
    virtual bool supports_text_session_state() const = 0;
    virtual bool supports_paged_text_session_state() const = 0;
};

} // namespace tyloi::models
