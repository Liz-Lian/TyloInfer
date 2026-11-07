#pragma once

#include "engine/models/state.h"
#include "engine/models/session.h"
#include "engine/models/common/session/size.h"

#include <memory>
#include <vector>

namespace tyloi::models { struct Mtp; }
namespace tyloi::cuda::qwen4_exp {

struct Model;

// One request: the logical cache position, the speculative journal and the
// session snapshot plumbing. The recurrent caches inside the layer records
// follow into the State when requests stop being single-flight.
class State : public tyloi::models::State {
public:
    explicit State(Model& model);
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

    std::int64_t cache_pos = 0;
    std::int64_t decode_position_delta = 0;
    std::int64_t batch = 0;
    std::int64_t speculative_start = -1;
    std::int64_t speculative_confirmed = 0;
    bool speculative_suffix_forward = false;

private:
    friend tyloi::models::SnapshotSize snapshot_size(const State&, const tyloi::models::Mtp*);
    Model& model_;
};

tyloi::models::SnapshotSize snapshot_size(const State&, const tyloi::models::Mtp*);

} // namespace tyloi::cuda::qwen4_exp
