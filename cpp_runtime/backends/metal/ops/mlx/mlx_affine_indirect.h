#pragma once

#include "mlx_expert_route_tiles.h"
#include <array>
#include <memory>

namespace mfq::metal {
class MlxWeightResidency;
class MlxAffineIndirectWeight {
public:
    static MlxAffineIndirectWeight from_address_tables(mlx::core::Shape shape, int bits, int group_size,
        mlx::core::Dtype scale_dtype, std::array<mlx::core::array, 3> tables,
        std::shared_ptr<MlxWeightResidency> residency);
    mlx::core::array routed_matmul(const mlx::core::array& input, const mlx::core::array& ids,
        bool shared_input, const MlxExpertRouteTiles* plan = nullptr) const;
    mlx::core::array routed_swiglu(const MlxAffineIndirectWeight& up, const mlx::core::array& input,
        const mlx::core::array& ids, const MlxExpertRouteTiles* plan = nullptr) const;
    int input_size() const noexcept { return shape_[2]; }
    int output_size() const noexcept { return shape_[1]; }
    int expert_count() const noexcept { return shape_[0]; }
private:
    MlxAffineIndirectWeight(mlx::core::Shape shape, int bits, int group_size, mlx::core::Dtype dtype,
        std::array<mlx::core::array, 3> tables);
    mlx::core::Shape shape_;
    int bits_, group_size_;
    mlx::core::Dtype dtype_;
    std::array<mlx::core::array, 3> tables_;
    friend class MlxAffineStaged;
};
} // namespace mfq::metal
