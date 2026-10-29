#include "mlx_grouped_linear.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace mfq::metal {
namespace {
using mlx::core::array;
using mlx::core::Shape;
using mlx::core::Dtype;
using mlx::core::CompileOptions;
using mlx::core::MathMode;
constexpr int kFamilyNint8Zero = 1;
constexpr int kFamilyMx = 3;
#include "mlx_native_projection_layout.inc"
#include "mlx_native_mx_projection_header.inc"
#include "mlx_dense_projection_kernels.inc"

int checked_int(std::size_t value, const char* label) {
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error(label);
    return static_cast<int>(value);
}

std::string direct_kernel_key(const std::vector<DirectProjectionLayout>& layouts) {
    std::string result;
    for (const auto& layout : layouts) result += "mx" + std::to_string(layout.bits) + "_";
    return result;
}

std::vector<std::string> direct_input_names(const std::vector<DirectProjectionLayout>& layouts) {
    std::vector<std::string> result;
    for (std::size_t index = 0; index < layouts.size(); ++index) {
        const auto suffix = std::to_string(index);
        result.push_back("mx_values_" + suffix);
        result.push_back("mx_scales_" + suffix);
    }
    result.push_back("x");
    return result;
}

#include "mlx_native_mx_projection_kernels.inc"
#include "mlx_native_mx_blockwise.inc"
#include "mlx_native_mx_generic.inc"

mlx::core::fast::CustomKernelFunction mx_generic_kernel(
    const std::vector<DirectProjectionLayout>& layouts, int rows, bool batch_rows) {
    static std::mutex mutex;
    static std::unordered_map<std::string, mlx::core::fast::CustomKernelFunction> kernels;
    const auto key = direct_kernel_key(layouts) + std::to_string(rows) + (batch_rows ? "_batch" : "_row");
    std::lock_guard lock(mutex);
    if (const auto found = kernels.find(key); found != kernels.end()) return found->second;
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    auto kernel = mlx::core::fast::metal_kernel(
        "tyloinfer_native_mx_generic_" + key, direct_input_names(layouts), {"y"},
        make_direct_source(layouts, batch_rows, rows), kNativeMxProjectionHeader, true, false, options);
    kernels.emplace(key, kernel);
    return kernel;
}

mlx::core::fast::CustomKernelFunction mx_blockwise_kernel(
    const std::vector<DirectProjectionLayout>& layouts) {
    static std::mutex mutex;
    static std::unordered_map<std::string, mlx::core::fast::CustomKernelFunction> kernels;
    const auto key = direct_kernel_key(layouts);
    std::lock_guard lock(mutex);
    if (const auto found = kernels.find(key); found != kernels.end()) return found->second;
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    auto kernel = mlx::core::fast::metal_kernel(
        "tyloinfer_native_mx_group_" + key, direct_input_names(layouts), {"y"},
        make_direct_small_m_blockwise_source(layouts, true, false),
        kNativeMxProjectionHeader, true, false, options);
    kernels.emplace(key, kernel);
    return kernel;
}
}

struct MlxGroupedLinear::Impl {
    std::vector<array> dense;
    std::vector<MlxMxWeight> mx;
    std::vector<DirectProjectionLayout> layouts;
    std::vector<array> inputs;
    std::vector<int> outputs;
    int width = 0;
    int total = 0;
    int tiles = 0;
    int single_tiles = 0;
    std::size_t bytes = 0;
    bool block32 = false;
    bool mx8 = false;
};

MlxGroupedLinear::MlxGroupedLinear(std::vector<MlxGroupedLinearWeightRef> weights)
    : impl_(std::make_shared<Impl>()) {
    if (weights.size() < 2 || weights.size() > 14)
        throw MlxGroupedLinearUnsupported("projection group requires two through fourteen weights");
    const bool dense = std::holds_alternative<const array*>(weights.front());
    for (const auto& ref : weights) {
        if (dense != std::holds_alternative<const array*>(ref))
            throw MlxGroupedLinearUnsupported("projection group requires one format family");
        int width = 0;
        int output = 0;
        if (dense) {
            const auto* weight = std::get<const array*>(ref);
            if (!weight || weight->ndim() != 2 || !weight->flags().row_contiguous ||
                (weight->dtype() != mlx::core::float16 && weight->dtype() != mlx::core::bfloat16) ||
                (!impl_->dense.empty() && weight->dtype() != impl_->dense.front().dtype()))
                throw MlxGroupedLinearUnsupported("dense group requires matching FP16/BF16 matrices");
            width = weight->shape(1);
            output = weight->shape(0);
            impl_->bytes += weight->nbytes();
            impl_->dense.push_back(*weight);
        } else {
            const auto* weight = std::get<const MlxMxWeight*>(ref);
            if (!weight) throw std::invalid_argument("null projection weight");
            width = weight->input_size();
            output = weight->output_size();
            const bool block32 = weight->bits() == 8 && weight->scale_column_block_size() == 32;
            if (!impl_->mx.empty() && block32 != impl_->block32)
                throw MlxGroupedLinearUnsupported("MX group requires matching scale layouts");
            impl_->block32 = block32;
            impl_->bytes += weight->packed_nbytes();
            impl_->mx.push_back(*weight);
            DirectProjectionLayout layout;
            layout.family = kFamilyMx;
            layout.bits = weight->bits();
            layout.output_size = output;
            layout.tile_begin = impl_->tiles;
            layout.output_offset = impl_->total;
            impl_->tiles = checked_int(static_cast<std::size_t>(impl_->tiles) + (output + 7) / 8, "projection tiles overflow");
            layout.tile_end = impl_->tiles;
            impl_->layouts.push_back(layout);
            impl_->single_tiles = std::max(impl_->single_tiles, (output + 15) / 16);
            impl_->inputs.push_back(weight->packed_values());
            impl_->inputs.push_back(weight->block_scales());
        }
        if (width <= 0 || output <= 0 || (impl_->width && impl_->width != width))
            throw std::invalid_argument("projection dimensions must be positive and share one input width");
        impl_->width = width;
        impl_->total = checked_int(static_cast<std::size_t>(impl_->total) + output, "projection output overflow");
        impl_->outputs.push_back(output);
    }
    impl_->mx8 = !impl_->mx.empty() && !impl_->block32 && impl_->width % 128 == 0 &&
        std::all_of(impl_->mx.begin(), impl_->mx.end(), [](const auto& weight) { return weight.bits() == 8; });
}

bool MlxGroupedLinear::supports(const array& input) const noexcept {
    if (input.ndim() == 0 || input.shape(-1) != impl_->width || input.size() == 0 ||
        (input.dtype() != mlx::core::float16 && input.dtype() != mlx::core::bfloat16 && input.dtype() != mlx::core::float32)) return false;
    const auto rows = input.size() / impl_->width;
    return rows <= (impl_->block32 ? 6 : max_rows()) &&
        (!impl_->block32 || input.dtype() != mlx::core::float32);
}

std::vector<array> MlxGroupedLinear::matmul(const array& input) const {
    if (!supports(input)) throw MlxGroupedLinearUnsupported("unsupported native projection group input");
    if (impl_->block32) return MlxMxWeight::projection_group_matmul(impl_->mx, input);
    const int rows = checked_int(input.size() / impl_->width, "projection rows overflow");
    const bool dense = !impl_->dense.empty();
    const auto dtype = dense ? impl_->dense.front().dtype() :
        (input.dtype() == mlx::core::float32 ? mlx::core::float32 : mlx::core::float16);
    auto source = mlx::core::contiguous(mlx::core::reshape(mlx::core::astype(input, dtype), Shape{rows, impl_->width}));
    auto inputs = dense ? impl_->dense : impl_->inputs;
    inputs.push_back(source);
    std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>> arguments{
        {"T", dtype}, {"M", rows}, {"ROWS", rows}, {"K", impl_->width}, {"TOTAL_OUT", impl_->total}};
    const bool single_mx8 = !dense && rows == 1 && impl_->mx8 && dtype == mlx::core::float16;
    const bool batch_rows = !dense && rows >= 2 && rows <= 6;
    const auto* layout = std::getenv("TYLOINFER_METAL_GROUPED_SMALL_M_LAYOUT");
    const bool blockwise = batch_rows && dtype == mlx::core::float16 &&
        (!layout || std::strcmp(layout, "scalar") != 0);
    int offset = 0;
    int tiles = 0;
    for (std::size_t index = 0; index < impl_->outputs.size(); ++index) {
        const auto name = "P" + std::to_string(index) + "_";
        const auto width = impl_->outputs[index];
        arguments.emplace_back(name + "OUT", width);
        if (!dense) arguments.emplace_back(name + "BITS", impl_->layouts[index].bits);
        arguments.emplace_back(name + "OFFSET", offset);
        arguments.emplace_back(name + "OUT_OFFSET", offset);
        arguments.emplace_back(name + "TILE_BEGIN", tiles);
        tiles += (width + (dense ? 3 : 7)) / (dense ? 4 : 8);
        arguments.emplace_back(name + "TILE_END", tiles);
        offset += width;
    }
    arguments.emplace_back("TOTAL_TILES", tiles);
    auto kernel = dense ? dense_projection_group_kernel(impl_->dense.size()) :
        (single_mx8 ? single_row_mxfp8_kernel(impl_->layouts) :
            (blockwise ? mx_blockwise_kernel(impl_->layouts) : mx_generic_kernel(impl_->layouts, rows, batch_rows)));
    const int threads = dense || single_mx8 ? 128 : 64;
    if (single_mx8) tiles = impl_->single_tiles;
    if (!dense && !single_mx8 && !batch_rows) tiles *= rows;
    auto combined = kernel(inputs, {Shape{rows, impl_->total}}, {dtype},
        {checked_int(static_cast<std::size_t>(tiles) * threads, "projection grid overflow"), 1, 1},
        {threads, 1, 1}, arguments, std::nullopt, false, {}).front();
    std::vector<array> results;
    offset = 0;
    for (const auto width : impl_->outputs) {
        auto shape = input.shape();
        shape.back() = width;
        auto value = mlx::core::reshape(mlx::core::slice(combined, Shape{0, offset}, Shape{rows, offset + width}), shape);
        if (!dense && input.dtype() == mlx::core::bfloat16) value = mlx::core::astype(value, input.dtype());
        results.push_back(std::move(value));
        offset += width;
    }
    return results;
}

bool MlxGroupedLinear::supports_single_row_swiglu(const array& input) const noexcept {
    return supports_small_m_swiglu(input) && input.size() == static_cast<std::size_t>(impl_->width);
}
bool MlxGroupedLinear::supports_small_m_swiglu(const array& input) const noexcept {
    if (impl_->block32) return MlxMxWeight::supports_projection_pair_swiglu(impl_->mx, input);
    return impl_->mx8 && impl_->outputs.size() == 2 && impl_->outputs[0] == impl_->outputs[1] &&
        supports(input) && input.dtype() != mlx::core::float32;
}
array MlxGroupedLinear::single_row_swiglu(const array& input, float limit) const {
    if (!supports_single_row_swiglu(input)) throw MlxGroupedLinearUnsupported("unsupported native SwiGLU input");
    return small_m_swiglu(input, limit);
}
array MlxGroupedLinear::small_m_swiglu(const array& input, float limit) const {
    if (!supports_small_m_swiglu(input)) throw MlxGroupedLinearUnsupported("unsupported native SwiGLU input");
    if (!std::isfinite(limit) || limit < 0) throw std::invalid_argument("SwiGLU limit must be finite and non-negative");
    if (impl_->block32) return MlxMxWeight::projection_pair_swiglu(impl_->mx, input, limit);
    const int rows = checked_int(input.size() / impl_->width, "SwiGLU rows overflow");
    auto inputs = impl_->inputs;
    inputs.push_back(mlx::core::contiguous(mlx::core::reshape(input, Shape{rows, impl_->width})));
    inputs.push_back(array({limit}, mlx::core::float32));
    auto result = single_row_mxfp8_pair_swiglu_kernel(input.dtype())(inputs,
        {Shape{rows, impl_->outputs[0]}}, {input.dtype()}, {((impl_->outputs[0] + 3) / 4) * 128, 1, 1}, {128, 1, 1},
        {{"K", impl_->width}, {"OUT", impl_->outputs[0]}, {"M", rows}}, std::nullopt, false, {}).front();
    auto shape = input.shape();
    shape.back() = impl_->outputs[0];
    return mlx::core::reshape(result, shape);
}
int MlxGroupedLinear::input_size() const noexcept { return impl_->width; }
int MlxGroupedLinear::total_output_size() const noexcept { return impl_->total; }
std::size_t MlxGroupedLinear::projection_count() const noexcept { return impl_->outputs.size(); }
const std::vector<int>& MlxGroupedLinear::output_sizes() const noexcept { return impl_->outputs; }
std::size_t MlxGroupedLinear::packed_nbytes() const noexcept { return impl_->bytes; }
bool MlxGroupedLinear::uses_zero_copy_storage() const noexcept { return true; }
std::size_t MlxGroupedLinear::copied_packed_nbytes() const noexcept { return 0; }
bool MlxGroupedLinear::supports_single_row_projection_fusion() const noexcept { return true; }
bool MlxGroupedLinear::has_projection_fusion() const noexcept { return true; }
bool MlxGroupedLinear::has_single_row_mxfp8_fast_path() const noexcept { return impl_->mx8; }
}
