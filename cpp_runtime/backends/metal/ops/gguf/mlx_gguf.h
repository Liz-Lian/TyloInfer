#pragma once

#include <mlx/mlx.h>
#include <array>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include "mlx_expert_route_tiles.h"

namespace mfq::metal {
class MfqContainer;
class MlxWeightResidency;
class MlxGgufPrefillPlan;
class MlxGgufStagedPrefill;
struct MlxNativeGgufWeight;
struct MlxGgufRotation;

bool is_gguf_dtype(std::string_view dtype) noexcept;
std::size_t gguf_packed_row_bytes(std::string_view dtype, int width);
using MlxGgufRoutePlan = MlxExpertRouteTiles;
MlxGgufRoutePlan gguf_route_plan(const mlx::core::array& ids, int experts, bool wide_mixed = false);

class MlxGgufWeight {
public:
    static MlxGgufWeight load(const MfqContainer& model, const std::string& name, bool native_projection = false);
    static bool supports_native(const MfqContainer& model, const std::string& name);
    const MlxNativeGgufWeight* native_weight_ref() const noexcept { return native_.get(); }
    static MlxGgufWeight from_bytes(std::string_view dtype,
        std::span<const std::uint8_t> bytes, mlx::core::Shape shape);
    static MlxGgufWeight from_experts(std::string_view dtype, mlx::core::Shape shape,
        const std::vector<std::int32_t>& ids, const std::vector<mlx::core::array>& packed,
        std::shared_ptr<MlxWeightResidency> residency = {});
    static MlxGgufWeight from_mixed_experts(mlx::core::Shape shape,
        const std::vector<std::int32_t>& ids, const std::vector<std::string>& dtypes,
        const std::vector<mlx::core::array>& packed,
        std::shared_ptr<MlxWeightResidency> residency = {});
    static MlxGgufWeight from_address_table(std::string_view dtype, mlx::core::Shape shape,
        mlx::core::array table, std::shared_ptr<MlxWeightResidency> residency);
    static MlxGgufWeight from_mixed_address_table(mlx::core::Shape shape,
        const std::vector<std::string>& dtypes, mlx::core::array table,
        std::shared_ptr<MlxWeightResidency> residency);
    mlx::core::array matmul(const mlx::core::array& input) const;
    mlx::core::array grouped_row_matmul(const mlx::core::array& input, int groups) const;
    mlx::core::array swiglu(const MlxGgufWeight& up, const mlx::core::array& input) const;
    mlx::core::array embedding(const mlx::core::array& ids,
        mlx::core::Dtype dtype = mlx::core::float16) const;
    mlx::core::array dequantize(mlx::core::Dtype dtype = mlx::core::float16) const;
    mlx::core::array routed_matmul(const mlx::core::array& input,
        const mlx::core::array& expert_ids, bool shared_input,
        const MlxGgufRoutePlan* plan = nullptr) const;
    mlx::core::array routed_swiglu(const MlxGgufWeight& up, const mlx::core::array& input,
        const mlx::core::array& expert_ids, const MlxGgufRoutePlan* plan = nullptr, float limit = 0.f) const;
    mlx::core::array routed_weighted_matmul(const mlx::core::array& input,
        const mlx::core::array& expert_ids, const mlx::core::array& probabilities,
        const MlxGgufRoutePlan* plan = nullptr) const;
    std::optional<std::vector<mlx::core::array>> gated_residual(const mlx::core::array& normalized,
        const MlxGgufWeight& up, const mlx::core::array* injection, int streams) const;
    int input_size() const noexcept { return shape_.back(); }
    int output_size() const noexcept { return shape_[shape_.size() - 2]; }
    int expert_count() const noexcept { return shape_.size() == 3 ? shape_[0] : 1; }
    int format() const noexcept { return format_; }
    bool has_staged_planes() const noexcept { return planes_.has_value(); }
    bool has_mixed_formats() const noexcept { return mixed_layout_.has_value(); }
    bool has_hadamard() const noexcept { return bool(rotation_); }
    bool same_rotation(const MlxGgufWeight& other) const;
    void configure_hadamard(const MfqContainer& model, const std::string& name);
    void prepare_grouped();
    std::size_t packed_nbytes() const noexcept;
private:
    friend class MlxGgufPrefillPlan;
    friend class MlxGgufStagedPrefill;
    static MlxGgufWeight allocate_packed(std::string_view dtype, mlx::core::Shape shape,
        std::size_t bytes);
    MlxGgufWeight(mlx::core::array bytes, mlx::core::Shape shape, int format, int row_bytes,
        bool indirect = false);
    mlx::core::array mixed_matmul(const mlx::core::array& input, const mlx::core::array& ids,
        bool shared_input, const MlxGgufRoutePlan* plan, const mlx::core::array* gathered_input = nullptr) const;
    mlx::core::array mixed_swiglu(const MlxGgufWeight& up, const mlx::core::array& input,
        const mlx::core::array& ids, float limit, const MlxGgufRoutePlan* plan) const;
    mlx::core::array rotate_input(const mlx::core::array& input) const;
    mlx::core::array unrotate_rows(const mlx::core::array& input, mlx::core::Dtype dtype) const;
    mlx::core::array bytes_;
    mlx::core::Shape shape_;
    int format_;
    int row_bytes_;
    bool indirect_;
    std::optional<std::array<mlx::core::array, 2>> planes_;
    std::shared_ptr<const MlxNativeGgufWeight> native_;
    std::optional<mlx::core::array> mixed_layout_;
    int mixed_mask_ = 0;
    std::size_t mixed_payload_bytes_ = 0;
    std::shared_ptr<const MlxGgufRotation> rotation_;
};
} // namespace mfq::metal
