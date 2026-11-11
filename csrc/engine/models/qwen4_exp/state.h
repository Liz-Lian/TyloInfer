#pragma once

#include "engine/models/state.h"
#include "engine/models/session/snapshot.h"
#include "engine/models/session/size.h"
#include "engine/models/qwen4_exp/snapshot.h"
#include "engine/models/qwen4_exp/layer.h"
#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/state/media.h"
#endif

#include <memory>
#include <vector>

namespace tyloi::models { struct Mtp; }
namespace tyloi::cuda { struct Qwen4DecodeGraph; }
namespace tyloi::cuda::qwen4_exp {

struct Model;

// Per-request recurrent and attention storage for one shared layer.
struct LayerState {
    Gdn::State gdn;
    std::unique_ptr<Qsa::State> qsa;
    Ple::State ple;
    bool retain_graph_storage = false, retained_prefix = false;

    explicit LayerState(const Layer& layer);
    void reset(const Layer& layer);
    void commit(const Layer& layer);
    void rollback(const Layer& layer, std::int64_t keep);
    std::vector<::tyloi::Tensor*> graph_state(const Layer& layer);
};

/** Capture the committed layer caches and recurrent payloads. */
tyloi::models::qwen4_exp::LayerSnapshot capture_layer_session(
    const Layer& layer, const LayerState& state, std::int64_t position, std::size_t& bytes);
#ifndef TYLOINFER_NATIVE_CUDA
/** Restore a layer snapshot through the legacy Torch execution path. */
void restore_layer_session(
    const Layer& layer, LayerState& state,
    const tyloi::models::qwen4_exp::LayerSnapshot& snapshot, std::int64_t position);
#else
/** Validate a layer snapshot against its model and current request layout. */
void validate_layer_session(const Layer& layer, const LayerState& current,
    const tyloi::models::qwen4_exp::LayerSnapshot& snapshot, std::int64_t position);
/** Estimate fresh candidate storage, excluding managed KV admission. */
std::size_t layer_restore_growth(const Layer& layer, const LayerState& current,
    const tyloi::models::qwen4_exp::LayerSnapshot& snapshot, std::int64_t position);
/** Copy a validated snapshot into an already-owned independent layer candidate. */
void stage_layer_session(const Layer& layer, LayerState& candidate,
    const tyloi::models::qwen4_exp::LayerSnapshot& snapshot, std::int64_t position);
#endif

// One request owns its layer caches, decode graph, positions and journal.
class State : public tyloi::models::State {
public:
    explicit State(const Model& model);
    State(const State&) = delete;
    State& operator=(const State&) = delete;
    State(State&&) = delete;
    State& operator=(State&&) = delete;
    ~State() override;

    /** Check whether this request state was created by the model. */
    bool belongs_to(const Model& model) const noexcept { return &model_ == &model; }

    void preallocate() override {}
    void reset(std::int64_t batch) override;
#ifdef TYLOINFER_NATIVE_CUDA
    /** Plan per-device cache and checkpoint allocations for a forward window. */
    std::vector<std::pair<tyloi::Device, std::size_t>> growth(
        std::int64_t tokens, const tyloi::models::ForwardOptions& options,
        std::int64_t confirmed = 0) const override;
    /** Reset releases or clears existing backing without allocating new device storage. */
    std::vector<std::pair<tyloi::Device, std::size_t>> reset_growth(std::int64_t batch) const override;
    /** Plan recurrent and PLE checkpoints before beginning a suffix. */
    std::vector<std::pair<tyloi::Device, std::size_t>> suffix_growth(std::int64_t tokens) const override;
#endif
    std::int64_t position() const noexcept override { return cache_pos; }
    void advance(std::int64_t tokens) noexcept override { cache_pos += tokens; }
    void set_position_delta(std::int64_t delta) noexcept override { decode_position_delta = delta; }
    std::vector<::tyloi::Tensor> storage() const override;
    std::vector<tyloi::Tensor*> graph_state() override;

    bool supports_text_session_state() const override;
    bool supports_paged_text_session_state() const override;
    TextSessionStateKind text_session_state_kind() const;
    /** Capture a text session snapshot for the committed tokens. */
    TextSessionState capture_text_session_state(const std::vector<std::int64_t>& tokens) const override;
#ifndef TYLOINFER_NATIVE_CUDA
    /** Restore a compatible text session snapshot. */
    void restore_text_session_state(const TextSessionState& state) override;
#else
    /** Validate and prepare an independent staged session restore. */
    std::unique_ptr<tyloi::models::Restore> prepare_restore(const TextSessionState& state) override;
    /** Retire graph references before replacing request-owned layer storage. */
    void retire_graphs() noexcept;
#endif

    // Speculative suffix journal; the layer records hold the checkpoints.
    /** Begin journaling an uncommitted speculative suffix. */
    void begin_speculative_suffix(std::int64_t draft_tokens) override;
    /** Commit the completed speculative verification pass. */
    void commit_speculative() override;
    /** Roll back a speculative pass while retaining accepted tokens. */
    void rollback_speculative(std::int64_t accepted) override;
    void begin_forward();
    void finish_forward(const ::tyloi::DenseTensor& full_positions, std::int64_t batch,
                        std::int64_t tokens);

    std::int64_t cache_pos = 0;
    std::int64_t decode_position_delta = 0;
    std::int64_t batch = 0;
    std::int64_t speculative_start = -1;
    std::int64_t speculative_confirmed = 0;
    bool speculative_suffix_forward = false;
#ifdef TYLOINFER_NATIVE_CUDA
    ::tyloi::Tensor positions;
    /** Request-owned prepared media cache. */
    tyloi::cache::Media images;
#else
    ::tyloi::DenseTensor positions;
#endif
    std::vector<std::unique_ptr<LayerState>> layers;
    std::unique_ptr<Qwen4DecodeGraph> graph_registry;
    bool reuse_decode_graphs = false;
    std::uint64_t decode_graph_captures = 0;
    std::int64_t decode_reserve_until = 0;
    bool speculative_prefix_retention = false;

private:
    friend class Candidate;
    friend tyloi::models::SnapshotSize snapshot_size(const State&, const tyloi::models::Mtp*);
    const Model& model_;
};

tyloi::models::SnapshotSize snapshot_size(const State&, const tyloi::models::Mtp*);

} // namespace tyloi::cuda::qwen4_exp
