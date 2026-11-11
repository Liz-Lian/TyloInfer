#pragma once

#include "engine/models/state.h"
#include "engine/state/kv_view.h"

#include <cstdint>
#include <vector>

namespace tyloi::cuda::gemma4 {

struct Model;

class State : public tyloi::models::State {
public:
    explicit State(const Model&);
    State(const State&) = delete;
    State& operator=(const State&) = delete;
    State(State&&) noexcept = default;
    State& operator=(State&&) noexcept = default;

    void preallocate() override;
    void reset(std::int64_t batch) override;
    std::int64_t position() const noexcept override { return cache_pos; }
    void advance(std::int64_t tokens) noexcept override { cache_pos += tokens; }
    void set_position_delta(std::int64_t delta) noexcept override { decode_position_delta = delta; }
    std::vector<std::pair<tyloi::Device, std::size_t>> growth(
        std::int64_t tokens, const tyloi::models::ForwardOptions& options) const override;
    std::vector<::tyloi::Tensor> storage() const override;
    std::vector<tyloi::Tensor*> graph_state() override;

    bool supports_text_session_state() const override;
    bool supports_paged_text_session_state() const override;

    struct Layout {
        int device;
        std::int64_t heads, head_dim, capacity;
        bool sliding;
        tyloi::ScalarType dtype;
        tyloi::state::KvQuantization quantization;
    };
    const std::vector<Layout>& layout() const noexcept { return layout_; }

    std::vector<tyloi::state::KvView> layers;
    std::int64_t cache_pos = 0;
    std::int64_t decode_position_delta = 0;
    std::int64_t batch = 0;

private:
    std::vector<Layout> layout_;
    std::vector<::tyloi::Tensor> graph_views_;
};

} // namespace tyloi::cuda::gemma4
