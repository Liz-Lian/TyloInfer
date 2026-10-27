#pragma once

#include <mlx/mlx.h>
#include <optional>

namespace mfq::metal {
class MlxAffineWeight;
class MlxAffineRoutePlan;
class MlxAffineIndirectWeight;
struct MlxExpertRouteTiles;

class MlxAffineStaged {
public:
    static std::optional<mlx::core::array> project(const MlxAffineWeight& weight,
        const mlx::core::array& input, const MlxAffineRoutePlan& plan,
        bool shared_input, bool input_in_route_order, bool swiglu = false);
    static std::optional<mlx::core::array> project_indirect(const MlxAffineIndirectWeight& weight,
        const mlx::core::array& input, const mlx::core::array& ids,
        bool shared_input, const MlxExpertRouteTiles& plan);
};
}
