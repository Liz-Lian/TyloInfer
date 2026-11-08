#pragma once

#include "engine/models/state.h"
#include "engine/models/session/snapshot.h"
#include "engine/models/session/size.h"
#include "engine/models/qwen4_exp/snapshot.h"
#include "engine/models/qwen4_exp/layer.h"

#include <memory>
#include <vector>

namespace tyloi::models { struct Mtp; }
namespace tyloi::cuda { struct Qwen4DecodeGraph; }
namespace mfq::cuda { class RouterLookaheadInputs; }
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

tyloi::models::qwen4_exp::LayerSnapshot capture_layer_session(
    const Layer& layer, const LayerState& state, std::int64_t position, std::size_t& bytes);
void restore_layer_session(
    const Layer& layer, LayerState& state,
    const tyloi::models::qwen4_exp::LayerSnapshot& snapshot, std::int64_t position);

// One request owns its layer caches, decode graph, positions and journal.
class State : public tyloi::models::State {
public:
    explicit State(const Model& model);
    State(const State&) = delete;
    State& operator=(const State&) = delete;
    State(State&&) = delete;
    State& operator=(State&&) = delete;
    ~State() override;

    void preallocate() override {}
    void reset(std::int64_t batch) override;
    std::int64_t position() const noexcept override { return cache_pos; }
    void advance(std::int64_t tokens) noexcept override { cache_pos += tokens; }
    void set_position_delta(std::int64_t delta) noexcept override { decode_position_delta = delta; }
    std::vector<std::shared_ptr<const tyloi::Tensor>> storage() const override;
    std::vector<tyloi::Tensor*> graph_state() override;

    bool supports_text_session_state() const override;
    bool supports_paged_text_session_state() const override;
    TextSessionStateKind text_session_state_kind() const;
    TextSessionState capture_text_session_state(const std::vector<std::int64_t>& tokens) const;
    void restore_text_session_state(const TextSessionState& state);

    // Speculative suffix journal; the layer records hold the checkpoints.
    void begin_speculative_suffix(std::int64_t draft_tokens);
    void commit_speculative();
    void rollback_speculative(std::int64_t accepted);
    void begin_forward();
    void finish_forward(const ::tyloi::DenseTensor& full_positions, std::int64_t batch,
                        std::int64_t tokens);

    std::int64_t cache_pos = 0;
    std::int64_t decode_position_delta = 0;
    std::int64_t batch = 0;
    std::int64_t speculative_start = -1;
    std::int64_t speculative_confirmed = 0;
    bool speculative_suffix_forward = false;
    ::tyloi::DenseTensor positions;
    std::vector<std::unique_ptr<LayerState>> layers;
    std::shared_ptr<mfq::cuda::RouterLookaheadInputs> router_audit;
    std::unique_ptr<Qwen4DecodeGraph> graph_registry;
    bool reuse_decode_graphs = false;
    std::uint64_t decode_graph_captures = 0;
    std::int64_t decode_reserve_until = 0;
    bool speculative_prefix_retention = false;
    std::optional<bool> router_lookahead;

private:
    friend tyloi::models::SnapshotSize snapshot_size(const State&, const tyloi::models::Mtp*);
    const Model& model_;
};

tyloi::models::SnapshotSize snapshot_size(const State&, const tyloi::models::Mtp*);

} // namespace tyloi::cuda::qwen4_exp
