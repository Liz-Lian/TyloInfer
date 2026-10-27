#include "qwen4_ops.h"
#include "mlx_kernel_prepare.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace mlx::core;

array reference_projection(const array& input, const array& weight) {
    const auto x = contiguous(astype(input, float32));
    const auto w = contiguous(astype(weight, float32));
    eval(x, w);
    const int width = weight.shape(1), rows = weight.shape(0);
    std::vector<float> values(rows);
    for (int row = 0; row < rows; ++row) {
        double sum = 0;
        for (int column = 0; column < width; ++column)
            sum += double(x.data<float>()[column]) * w.data<float>()[row * width + column];
        values[row] = float(sum);
    }
    return astype(array(values.begin(), {1, 1, rows}), promote_types(input.dtype(), weight.dtype()));
}

void compare(const array& actual, const array& expected, float tolerance) {
    if (actual.shape() != expected.shape() || actual.dtype() != expected.dtype())
        throw std::runtime_error("dense mHC changed output shape or dtype");
    const float error = max(abs(astype(actual, float32) - astype(expected, float32))).item<float>();
    if (!std::isfinite(error) || error > tolerance)
        throw std::runtime_error("dense mHC error=" + std::to_string(error));
}

void check(Dtype dtype, bool with_injection) {
    constexpr int hidden = 2560, streams = 4, low = 320, width = hidden * streams;
    const auto input = astype(reshape(sin(arange(width, float32) * .017f), {1, 1, width}), dtype);
    const auto norm = sin(arange(width, float32) * .013f) * .1f;
    const auto down = astype(reshape(sin(arange(low * width, float32) * .019f) * .002f, {low, width}), bfloat16);
    const auto up = astype(reshape(cos(arange(width * low, float32) * .023f) * .01f, {width, low}), bfloat16);
    const auto inject = astype(reshape(sin(arange(streams * width, float32) * .027f) * .01f, {streams, width}), bfloat16);
    const auto normalized = mfq::metal::qwen4_grouped_rms_norm(input, norm, hidden);
    const auto projected = reference_projection(normalized, down);
    const auto scaled = projected / array(float(streams), projected.dtype());
    const auto activated = scaled * sigmoid(scaled);
    const auto gates = sigmoid(reference_projection(activated, up));
    const auto expected = mean(reshape(gates * astype(normalized, gates.dtype()), {1, 1, streams, hidden}), -2);
    const auto actual = mfq::metal::qwen4_gated_residual_pre(input, norm, down, up,
        with_injection ? std::optional(inject) : std::nullopt, hidden, streams);
    const float tolerance = expected.dtype() == bfloat16 ? .008f : expected.dtype() == float16 ? .001f : 2e-5f;
    compare(actual.branch, expected, tolerance);
    if (bool(actual.injection) != with_injection)
        throw std::runtime_error("dense mHC changed injection presence");
    if (with_injection) {
        const auto result = reference_projection(normalized, inject);
        compare(*actual.injection, array(2.f, result.dtype()) * sigmoid(result / array(float(streams), result.dtype())), tolerance);
    }
}
void check_batch(Dtype dtype, bool with_injection, Dtype weight_dtype = bfloat16, bool offset_weight = false) {
    constexpr int hidden = 2560, streams = 4, low = 320, width = hidden * streams;
    const auto norm = sin(arange(width, float32) * .013f) * .1f;
    const auto down = astype(reshape(sin(arange(low * width, float32) * .019f) * .002f, {low, width}), weight_dtype);
    auto up = astype(reshape(cos(arange(width * low, float32) * .023f) * .01f, {width, low}), weight_dtype);
    if (offset_weight) {
        auto padded = concatenate({zeros({1}, weight_dtype), reshape(up, {-1})});
        up = reshape(slice(padded, {1}, {width * low + 1}), {width, low});
        eval(up);
    }
    const auto inject = astype(reshape(sin(arange(streams * width, float32) * .027f) * .01f, {streams, width}), weight_dtype);
    const std::optional<array> injection = with_injection ? std::optional(inject) : std::nullopt;
    std::string preparation_key;
    for (int rows = 1; rows <= 8; ++rows) for (bool batch_layout : {false, true}) {
        const Shape shape{batch_layout ? rows : 1, batch_layout ? 1 : rows, width};
        const auto input = astype(reshape(sin(arange(rows * width, float32) * .017f), shape), dtype);
        auto branch_shape = shape, gate_shape = shape;
        branch_shape.back() = hidden; gate_shape.back() = streams;
        const auto branch = astype(reshape(cos(arange(rows * hidden, float32) * .029f), branch_shape), float16);
        const auto gates = reshape(sin(arange(rows * streams, float32) * .031f), gate_shape);
        const auto flat = reshape(input, {rows, width});
        const auto flat_branch = reshape(branch, {rows, hidden}), flat_gates = reshape(gates, {rows, streams});
        for (bool chained : {false, true}) {
            const auto actual = chained
                ? mfq::metal::qwen4_gated_residual_pre_after(branch, input, gates, norm, down, up, injection, hidden, streams)
                : mfq::metal::qwen4_gated_residual_pre(input, norm, down, up, injection, hidden, streams);
            if (!chained) {
                const auto key = mfq::metal::mlx_kernel_preparation_key(actual.branch.primitive());
                if (preparation_key.empty()) preparation_key = key;
                if (key != preparation_key) throw std::runtime_error("dense HC specializes by runtime row count");
            }
            std::vector<array> expected_branches, expected_gates, expected_residuals;
            for (int row = 0; row < rows; ++row) {
                const auto x = reshape(slice(flat, {row, 0}, {row + 1, width}), {1, 1, width});
                const auto b = reshape(slice(flat_branch, {row, 0}, {row + 1, hidden}), {1, 1, hidden});
                const auto g = reshape(slice(flat_gates, {row, 0}, {row + 1, streams}), {1, 1, streams});
                const auto expected = chained
                    ? mfq::metal::qwen4_gated_residual_pre_after(b, x, g, norm, down, up, injection, hidden, streams)
                    : mfq::metal::qwen4_gated_residual_pre(x, norm, down, up, injection, hidden, streams);
                expected_branches.push_back(reshape(expected.branch, {1, hidden}));
                expected_residuals.push_back(reshape(expected.residual, {1, width}));
                if (with_injection) expected_gates.push_back(reshape(*expected.injection, {1, streams}));
            }
            compare(actual.branch, reshape(concatenate(expected_branches, 0), branch_shape), 0.f);
            compare(actual.residual, reshape(concatenate(expected_residuals, 0), shape), 0.f);
            if (with_injection) compare(*actual.injection, reshape(concatenate(expected_gates, 0), gate_shape), 0.f);
        }
    }
}

void check_prefill(Dtype weight_dtype, bool with_injection, int rows = 65, int hidden = 128, int low = 64) {
    constexpr int streams = 4;
    const int width = hidden * streams;
    const auto input = reshape(sin(arange(rows * width, float32) * .017f), {1, rows, width});
    const auto norm = sin(arange(width, float32) * .013f) * .1f;
    const mfq::metal::MlxLinear down(astype(reshape(sin(arange(low * width, float32) * .019f) * .002f, {low, width}), weight_dtype));
    const mfq::metal::MlxLinear up(astype(reshape(cos(arange(width * low, float32) * .023f) * .01f, {width, low}), weight_dtype));
    const auto inject = mfq::metal::MlxLinear(astype(reshape(sin(arange(streams * width, float32) * .027f) * .01f, {streams, width}), weight_dtype));
    const std::optional<mfq::metal::MlxLinear> injection = with_injection ? std::optional(inject) : std::nullopt;
    for (const float epsilon : {-1.f, std::numeric_limits<float>::quiet_NaN()}) {
        bool rejected = false;
        try { (void)mfq::metal::qwen4_gated_residual_pre(input, norm, down, up, injection, hidden, streams, epsilon); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) throw std::runtime_error("dense prefill mHC accepted invalid epsilon");
    }
    for (const bool chained : {false, true}) {
        const auto branch = reshape(cos(arange(rows * hidden, float32) * .029f), {1, rows, hidden});
        const auto gates = full({1, rows, streams}, .75f, float32);
        const auto residual = chained ? mfq::metal::qwen4_gated_residual_post(branch, input, gates, streams) : input;
        const auto normalized = mfq::metal::qwen4_grouped_rms_norm(residual, norm, hidden);
        const auto scaled = down(normalized) / array(float(streams), float32);
        const auto activated = scaled * sigmoid(scaled);
        const auto logits = up(activated);
        const auto expected = mean(reshape(sigmoid(logits) * normalized, {1, rows, streams, hidden}), -2);
        const auto actual = chained
            ? mfq::metal::qwen4_gated_residual_pre_after(branch, input, gates, norm, down, up, injection, hidden, streams)
            : mfq::metal::qwen4_gated_residual_pre(input, norm, down, up, injection, hidden, streams);
        compare(actual.branch, expected, 2e-6f);
        compare(actual.residual, residual, 0.f);
        if (bool(actual.injection) != with_injection)
            throw std::runtime_error("dense prefill mHC changed injection presence");
        if (with_injection)
            compare(*actual.injection, array(2.f, float32) * sigmoid(inject(normalized) / array(float(streams), float32)), 0.f);
    }
}
}

int main() {
    try {
        set_default_device(Device::gpu);
        for (const auto dtype : {float16, bfloat16, float32})
            for (bool injection : {false, true}) {
                check(dtype, injection);
                for (const auto weight_dtype : {float16, bfloat16, float32})
                    check_batch(dtype, injection, weight_dtype);
            }
        for (const auto dtype : {float16, bfloat16})
            for (bool injection : {false, true}) check_prefill(dtype, injection);
        check_prefill(bfloat16, true, 64, 192, 128);
        check_prefill(bfloat16, false, 65, 2560, 320);
        check_batch(float32, true, bfloat16, true);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
