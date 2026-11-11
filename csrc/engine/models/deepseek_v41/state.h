#pragma once

#include "engine/models/state.h"
#include "engine/models/session/snapshot.h"
#include "engine/models/session/size.h"
#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/state/media.h"
#endif

#include <memory>
#include <vector>

namespace tyloi::models { struct Mtp; }
namespace tyloi::cuda::deepseek_v41_runtime { struct SharedState; }

namespace tyloi::cuda::deepseek_v41 {

struct Model;

// One request: the cross-layer shared scratch (with per-layer states), the
// logical cache position and the speculative journal. The model record owns
// weights only; everything mutable lives here.
class State : public tyloi::models::State {
public:
    explicit State(Model& model);
    State(const State&) = delete;
    State& operator=(const State&) = delete;
    State(State&&) = delete;
    State& operator=(State&&) = delete;
    ~State() override;

    /** Check that this request state was created by the given model. */
    bool belongs_to(const Model& model) const noexcept { return &model_ == &model; }

    void preallocate() override {}
    void reset(std::int64_t batch) override;
#ifdef TYLOINFER_NATIVE_CUDA
    /** Plan backing allocations required by reset. */
    std::vector<std::pair<tyloi::Device, std::size_t>> reset_growth(std::int64_t batch) const override;
    /** Conservatively plan retained speculative allocations for a forward pass. */
    std::vector<std::pair<tyloi::Device, std::size_t>> growth(
        std::int64_t tokens, const tyloi::models::ForwardOptions& options,
        std::int64_t confirmed = 0) const override;
    /** Conservatively plan the speculative journal before it is created. */
    std::vector<std::pair<tyloi::Device, std::size_t>> suffix_growth(std::int64_t tokens) const override;
#endif
    std::int64_t position() const noexcept override { return cache_pos; }
    void advance(std::int64_t tokens) noexcept override { cache_pos += tokens; }
    void set_position_delta(std::int64_t delta) noexcept override { decode_position_delta = delta; }
    void set_image_mask(::tyloi::Tensor mask);
    void begin_forward(bool capture_raw_hidden);
    std::vector<::tyloi::Tensor> storage() const override;
    std::vector<tyloi::Tensor*> graph_state() override { return {}; }

    bool supports_text_session_state() const override;
    bool supports_paged_text_session_state() const override;
    TextSessionStateKind text_session_state_kind() const;
    /** Capture a text session snapshot for the committed tokens. */
    TextSessionState capture_text_session_state(const std::vector<std::int64_t>& tokens) const override;
#ifndef TYLOINFER_NATIVE_CUDA
    /** Restore a compatible text session snapshot. */
    void restore_text_session_state(const TextSessionState& state) override;
#else
    /** Plan allocations needed to restore a compatible snapshot. */
    std::vector<std::pair<tyloi::Device, std::size_t>> restore_growth(const TextSessionState& state) const override;
    /** Prepare a staged restore without mutating the live state. */
    std::unique_ptr<tyloi::models::Restore> prepare_restore(const TextSessionState& state) override;
#endif

    // Speculative suffix journal; the block records hold the checkpoints.
    /** Begin journaling an uncommitted speculative suffix. */
    void begin_speculative_suffix(std::int64_t draft_tokens) override;
    /** Commit the completed speculative verification pass. */
    void commit_speculative() override;
    /** Roll back a speculative pass while retaining accepted tokens. */
    void rollback_speculative(std::int64_t accepted) override;

    std::unique_ptr<deepseek_v41_runtime::SharedState> shared;
#ifdef TYLOINFER_NATIVE_CUDA
    tyloi::cache::Media media;
#endif
    std::int64_t cache_pos = 0;
    std::int64_t decode_position_delta = 0;
    std::int64_t batch = 0;
    std::int64_t speculative_start = -1;
    std::int64_t speculative_confirmed = 0;
    bool speculative_suffix_forward = false;

private:
    friend class Candidate;
    friend tyloi::models::SnapshotSize snapshot_size(const State&, const tyloi::models::Mtp*);
    Model& model_;
};

tyloi::models::SnapshotSize snapshot_size(const State&, const tyloi::models::Mtp*);

} // namespace tyloi::cuda::deepseek_v41
