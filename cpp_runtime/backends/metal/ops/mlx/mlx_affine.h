#pragma once

#include <mlx/mlx.h>
#include <array>
#include <optional>
#include <memory>
#include <string>
#include <utility>

namespace mfq::metal {
class MfqContainer;
struct MlxNativeGgufWeight;

class MlxAffineRoutePlan {
public:
    MlxAffineRoutePlan(const mlx::core::array& ids, int experts);
    static MlxAffineRoutePlan from_topk(const mlx::core::array& ids, int experts);
    mlx::core::array restore(const mlx::core::array& output) const;
private:
    MlxAffineRoutePlan(const mlx::core::array& ids, int experts, bool valid);
    mlx::core::array ids_;
    std::optional<mlx::core::array> valid_, order_, inverse_;
    mutable std::array<std::optional<std::pair<mlx::core::array, mlx::core::array>>, 3> tiles_;
    int tokens_, routes_, experts_;
    friend class MlxAffineWeight;
    friend class MlxAffineStaged;
};

class MlxAffineWeight {
public:
    void prepare_native();
    const MlxNativeGgufWeight* native_weight_ref() const noexcept { return native_.get(); }
    static bool supports_native(const MfqContainer& model, const std::string& name);
    static MlxAffineWeight load(const MfqContainer& model, const std::string& name);
    static MlxAffineWeight from_arrays(mlx::core::array values, mlx::core::array scales,
        mlx::core::array biases, int bits, int group_size);
    static std::optional<MlxAffineWeight> join_rows(const MlxAffineWeight& first, const MlxAffineWeight& second);
    mlx::core::array matmul(const mlx::core::array& input) const;
    mlx::core::array embedding(const mlx::core::array& ids,
        mlx::core::Dtype dtype = mlx::core::float16) const;
    mlx::core::array dequantize(mlx::core::Dtype dtype = mlx::core::float16) const;
    mlx::core::array routed_matmul(const mlx::core::array& input,
        const mlx::core::array& ids, bool shared_input) const;
    mlx::core::array routed_matmul(const mlx::core::array& input,
        const MlxAffineRoutePlan& plan, bool shared_input, bool input_in_route_order = false) const;
    mlx::core::array routed_swiglu(const mlx::core::array& input, const MlxAffineRoutePlan& plan) const;
    std::optional<std::vector<mlx::core::array>> gated_residual(const mlx::core::array& normalized,
        const MlxAffineWeight& up, const MlxAffineWeight* injection, int streams) const;
    int input_size() const noexcept { return width_; }
    int output_size() const noexcept { return values_.shape(-2); }
    int expert_count() const noexcept { return values_.ndim() == 3 ? values_.shape(0) : 1; }
    mlx::core::Dtype activation_dtype() const noexcept { return scales_.dtype(); }
    std::size_t packed_nbytes() const noexcept;
private:
    MlxAffineWeight(mlx::core::array values, mlx::core::array scales, mlx::core::array biases,
        int bits, int group_size, int width);
    mlx::core::array values_, scales_, biases_;
    int bits_, group_size_, width_;
    std::shared_ptr<const MlxNativeGgufWeight> native_;
    friend class MlxAffineStaged;
};

} // namespace mfq::metal
