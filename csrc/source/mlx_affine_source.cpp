#include "mlx_affine_source.h"

#include <limits>
#include <stdexcept>

namespace mfq {

MlxAffineBinding mlx_affine_binding(const HfSafetensorsSource& source,
    const nlohmann::json& quantization, const TensorMetadata& values) {
    if (!quantization.is_object() || values.dtype != "U32" || !values.shape ||
        values.shape->empty() || values.name.size() < 7 || values.name.substr(values.name.size() - 7) != ".weight")
        throw std::invalid_argument("MLX affine weight has no valid quantization configuration: " + values.name);
    const auto module = values.name.substr(0, values.name.size() - 7);
    const auto config = quantization.contains(module) ? quantization.at(module) : quantization;
    if (!config.is_object() || config.value("mode", std::string("affine")) != "affine")
        throw std::invalid_argument("unsupported MLX quantization mode: " + module);
    MlxAffineBinding binding;
    auto& layout = binding.layout;
    layout.bits = config.value("bits", quantization.value("bits", 4));
    layout.group_size = config.value("group_size", quantization.value("group_size", 64));
    if (layout.bits < 1 || values.shape->back() < 1 ||
        values.shape->back() > std::numeric_limits<std::int64_t>::max() / 32 ||
        values.shape->back() * 32 % layout.bits)
        throw std::invalid_argument("invalid MLX affine packed row: " + module);
    layout.shape = *values.shape;
    layout.shape.back() = layout.shape.back() * 32 / layout.bits;
    binding.scales = module + ".scales"; binding.biases = module + ".biases";
    const auto* scales = source.find_tensor(binding.scales);
    const auto* biases = source.find_tensor(binding.biases);
    if (!scales || !biases || !scales->shape || !biases->shape || biases->dtype != scales->dtype)
        throw std::invalid_argument("MLX affine scales/biases are missing or incompatible: " + module);
    layout.scale_dtype = scales->dtype;
    layout.validate();
    auto scale_shape = layout.shape; scale_shape.back() /= layout.group_size;
    if (*scales->shape != scale_shape || *biases->shape != scale_shape ||
        scales->nbytes != layout.scales_bytes || biases->nbytes != layout.scales_bytes ||
        values.nbytes != layout.values_bytes)
        throw std::invalid_argument("MLX affine packed/scales/biases extents disagree: " + module);
    binding.prefix = layout.prefix(); layout.prefix_bytes = binding.prefix.size(); layout.validate();
    return binding;
}

} // namespace mfq
