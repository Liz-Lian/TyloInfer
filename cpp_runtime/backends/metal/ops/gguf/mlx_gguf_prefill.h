#pragma once
#include "mlx_gguf.h"
#include "mlx_gguf_routes.h"

namespace mfq::metal {
bool gguf_expert_prefill_available(const mlx::core::array& input);
bool gguf_short_sparse_routes(const mlx::core::array& ids, int experts);
mlx::core::array gguf_dense_prefill(const mlx::core::array& input, const mlx::core::array& weight);
int gguf_prefill_tile_rows(const MlxGgufWeight& gate, const MlxGgufWeight& up,
    const MlxGgufWeight& down, const mlx::core::array& ids);

class MlxGgufPrefillPlan {
public:
    MlxGgufPrefillPlan(const mlx::core::array& input, const mlx::core::array& ids, int experts,
        int tile_rows = 16, bool shared_input = true, bool include_shared = false);
    static MlxGgufPrefillPlan from_topk(const mlx::core::array& input, const mlx::core::array& ids,
        int experts, int tile_rows, bool include_shared = false);
    mlx::core::array gate_up(const MlxGgufWeight& gate, const MlxGgufWeight& up,
        const MlxGgufWeight* shared_gate = nullptr, const MlxGgufWeight* shared_up = nullptr) const;
    mlx::core::array project(const MlxGgufWeight& weight) const;
    mlx::core::array restore(const mlx::core::array& grouped) const;
    mlx::core::array down(const MlxGgufWeight& weight, const mlx::core::array& hidden,
        const mlx::core::array& probabilities, bool float_accumulation = false,
        const MlxGgufWeight* shared = nullptr) const;
private:
    MlxGgufPrefillPlan(const mlx::core::array& input, const mlx::core::array& ids, int experts,
        int tile_rows, bool shared_input, bool include_shared, bool unique_per_token);
    std::optional<MlxExpertRouteTiles> routes_;
    std::optional<MlxGgufGroupedRoutes> grouped_;
    struct Gather {
        mlx::core::array slots, offsets, params;
        int tiles;
    };
    std::optional<Gather> gather_;
    mlx::core::array gather_input(const mlx::core::array& input) const;
    mlx::core::array source_;
    mlx::core::array positions_, input_;
    mlx::core::Dtype dtype_;
    int tokens_, selected_, padded_rows_;
};
} // namespace mfq::metal
