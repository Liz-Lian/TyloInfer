#pragma once

#include "engine/models/session/snapshot.h"
#include "engine/models/state.h"
#include "engine/state/kv_view.h"

#include <vector>

namespace tyloi::cuda::minicpmo45::text {
struct Model;

class State
    : public tyloi::models::State
{
public:
    struct Restore {
        std::vector<tyloi::state::KvView> layers;
        int64_t cache_pos = 0, decode_position_delta = 0;
    };
    explicit State(const Model&);
    State(const State&) = delete;
    State& operator=(const State&) = delete;
    State(State&&) noexcept = default;
    State& operator=(State&&) noexcept = default;
    /** Check request ownership against the supplied model. */
    bool belongs_to(const Model& model) const noexcept { return model_ == &model; }
    /** Reuse a completed prefix without copying or replacing its KV backing. */
    void rewind(std::int64_t position) override;
    /** Plan KV allocation for the requested batch and token span. */
    std::vector<std::pair<tyloi::Device,std::size_t>> growth(std::int64_t batch, std::int64_t tokens,
        const tyloi::models::ForwardOptions& options, std::int64_t confirmed = 0) const override;
    /** Reset only releases incompatible KV backing. */
    std::vector<std::pair<tyloi::Device,std::size_t>> reset_growth(std::int64_t batch) const override;
    /** Prepare an independently owned session restore candidate. */
    std::unique_ptr<tyloi::models::Restore> prepare_restore(const TextSessionState&) override;

    void preallocate();
    void reset(int64_t batch);
    int64_t position() const noexcept { return cache_pos; }
    void advance(int64_t tokens) noexcept { cache_pos += tokens; }
    void set_position_delta(int64_t delta) noexcept { decode_position_delta = delta; }
    std::vector<::tyloi::Tensor> storage() const;
    std::vector<::tyloi::Tensor*> graph_state() { return {}; }
    TextSessionStateKind text_session_state_kind() const;
    bool supports_text_session_state() const;
    bool supports_paged_text_session_state() const;
    TextSessionState capture_text_session_state(const std::vector<int64_t>&) const;
    void restore_text_session_state(const TextSessionState&);
    /** Validate a session before allocating its restore candidate. */
    Restore prepare(const TextSessionState&) const;
    /** Stage a session in independently owned backing. */
    void stage(Restore&, const TextSessionState&) const;
    /** Publish the drained candidate while retaining replaced backing. */
    void publish(Restore&&) noexcept;

    std::vector<tyloi::state::KvView> layers;
    int64_t cache_pos = 0, decode_position_delta = 0, batch = 0;

private:
    struct Layout {
        int device;
        bool cpu, sliding;
        int64_t heads, head_dim, capacity;
        tyloi::state::KvQuantization quantization;
    };
    std::vector<Layout> layout_;
    const Model* model_;
    int64_t maximum_;
};
} // namespace tyloi::cuda::minicpmo45::text
