#pragma once

#include "engine/tensor.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

struct TextSessionState;

namespace tyloi::models {

struct ForwardOptions;
struct Mtp;

/** Own validated restore payloads and staged storage until device consumers have drained. */
class Restore {
public:
    virtual ~Restore() = default;
    /** Return per-device backing that staging will newly allocate. */
    virtual std::vector<std::pair<Device, std::size_t>> growth() const = 0;
    /** Stage copies in the owner's quiescent execution scope without publishing positions. */
    virtual void stage() = 0;
    /** Publish the staged state after all copies have completed. */
    virtual void publish() noexcept = 0;
};

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
    /** Retire model-owned graphs before changing request backing or runtime policy. */
    virtual void release_graphs() {}
    virtual std::int64_t position() const noexcept = 0;
    virtual void advance(std::int64_t tokens) noexcept = 0;
    virtual void set_position_delta(std::int64_t delta) noexcept = 0;
    /** Commit a completed speculative verification pass. */
    virtual void commit_speculative() { throw std::logic_error("speculative state is unsupported by this model"); }
    /** Roll back a speculative pass while retaining the accepted suffix. */
    virtual void rollback_speculative(std::int64_t) {
        throw std::logic_error("speculative rollback is unsupported by this model");
    }
    /** Report whether the current transaction retains the accepted prefix. */
    virtual bool retains_partial_prefix(std::int64_t) const noexcept { return false; }
    /** Begin journaling an uncommitted speculative suffix. */
    virtual void begin_speculative_suffix(std::int64_t) {
        throw std::logic_error("speculative suffix is unsupported by this model");
    }
    /** Plan per-device allocation reservations for an ordinary or confirmed verification step. */
    virtual std::vector<std::pair<tyloi::Device, std::size_t>> growth(
        std::int64_t, const ForwardOptions&, std::int64_t confirmed = 0) const {
        throw std::logic_error("state allocation planning is not implemented");
    }
    /** Return actual device bytes that reset will newly allocate for the batch. */
    virtual std::vector<std::pair<tyloi::Device, std::size_t>> reset_growth(std::int64_t) const {
        return {};
    }
    /** Plan the checkpoint allocated before beginning an uncommitted speculative suffix. */
    virtual std::vector<std::pair<Device, std::size_t>> suffix_growth(std::int64_t) const {
        throw std::logic_error("speculative suffix allocation planning is unsupported");
    }
    // Owners keep storage alive; deduplicate using backing identity, not object identity.
    virtual std::vector<::tyloi::Tensor> storage() const = 0;
    // Borrow writable objects from this state; their lifetime remains tied to it.
    virtual std::vector<tyloi::Tensor*> graph_state() = 0;
    virtual bool supports_text_session_state() const = 0;
    virtual bool supports_paged_text_session_state() const = 0;
    /** Capture a text session snapshot for the committed tokens. */
    virtual TextSessionState capture_text_session_state(const std::vector<std::int64_t>& tokens) const;
    /** Return actual device bytes newly allocated when restoring the snapshot. */
    virtual std::vector<std::pair<tyloi::Device, std::size_t>> restore_growth(
        const TextSessionState& snapshot) const;
    /** Validate and own a staged restore, or return null when only direct restoration is available. */
    virtual std::unique_ptr<Restore> prepare_restore(const TextSessionState&) { return {}; }
    /** Restore a compatible text session snapshot into this state. */
    virtual void restore_text_session_state(const TextSessionState& snapshot);
};

} // namespace tyloi::models
