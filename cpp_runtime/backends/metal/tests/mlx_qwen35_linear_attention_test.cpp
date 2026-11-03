#include "mlx_qwen35_linear_attention.h"

#include <cmath>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <mlx/mlx.h>
#include <mlx/primitives.h>

namespace {

using mlx::core::Shape;
using mlx::core::array;
using mfq::metal::MlxLinear;
using mfq::metal::MlxQwen35DenseSwiGlu;
using mfq::metal::MlxQwen35LinearAttentionBlock;
using mfq::metal::MlxRmsNorm;
using mfq::metal::Qwen35Config;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(
    float actual,
    float expected,
    float tolerance = 8e-4f) {
    if (!std::isfinite(actual) ||
        std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(
            "Qwen3.5 linear-attention mismatch: actual=" +
            std::to_string(actual) +
            " expected=" + std::to_string(expected));
    }
}

std::vector<float> patterned(
    int size,
    float scale,
    int period,
    int center) {
    std::vector<float> result;
    result.reserve(size);
    for (int index = 0; index < size; ++index) {
        result.push_back(
            static_cast<float>(index % period - center) * scale);
    }
    return result;
}

array matrix(
    int rows,
    int columns,
    const std::vector<float>& values) {
    require(
        static_cast<int>(values.size()) == rows * columns,
        "invalid linear-attention test matrix");
    return array(values.begin(), Shape{rows, columns});
}

array matrix_with_dtype(
    int rows,
    int columns,
    const std::vector<float>& values,
    mlx::core::Dtype dtype) {
    auto result = matrix(rows, columns, values);
    return dtype == mlx::core::float32
        ? result
        : mlx::core::astype(result, dtype);
}

array constant_vector(int size, float value) {
    const std::vector<float> values(
        static_cast<std::size_t>(size),
        value);
    return array(values.begin(), Shape{size});
}

Qwen35Config test_config(int dimension = 32) {
    Qwen35Config config;
    config.model_type = "qwen3_5";
    config.vocab_size = 32;
    config.hidden_size = 4;
    config.intermediate_size = 5;
    config.num_hidden_layers = 1;
    config.num_attention_heads = 2;
    config.num_key_value_heads = 1;
    config.max_position_embeddings = 32;
    config.head_dim = 2;
    config.rotary_dim = 2;
    config.rms_norm_eps = 1e-6;
    config.norm_weight_offset = 1.0;
    config.layer_types = {"linear_attention"};
    config.linear_conv_kernel_dim = 4;
    config.linear_key_head_dim = dimension;
    config.linear_value_head_dim = dimension;
    config.linear_num_key_heads = 1;
    config.linear_num_value_heads = 2;
    config.linear_a_is_log = true;
    return config;
}

MlxQwen35DenseSwiGlu make_ffn(
    mlx::core::Dtype dtype = mlx::core::float32) {
    return MlxQwen35DenseSwiGlu(
        MlxLinear(matrix_with_dtype(
            5,
            4,
            patterned(20, 0.021f, 9, 4),
            dtype)),
        MlxLinear(matrix_with_dtype(
            5,
            4,
            patterned(20, 0.018f, 7, 3),
            dtype)),
        MlxLinear(matrix_with_dtype(
            4,
            5,
            patterned(20, 0.016f, 11, 5),
            dtype)));
}

MlxQwen35LinearAttentionBlock make_block(
    bool split,
    bool gguf_layout,
    mlx::core::Dtype projection_dtype = mlx::core::float32,
    int dimension = 32,
    bool force_generic = false) {
    auto config = test_config(dimension);
    constexpr int hidden = 4;
    const int key_size = dimension;
    const int value_size = 2 * dimension;
    const int qk_size = 2 * key_size;
    const int channels = qk_size + value_size;
    constexpr int value_heads = 2;
    constexpr int kernel = 4;

    const auto qkv_values =
        patterned(channels * hidden, 0.008f, 13, 6);
    const std::vector<float> qk_values(
        qkv_values.begin(),
        qkv_values.begin() + qk_size * hidden);
    const std::vector<float> value_values(
        qkv_values.begin() + qk_size * hidden,
        qkv_values.end());

    std::vector<float> convolution(channels * kernel, 0.0f);
    for (int channel = 0; channel < channels; ++channel) {
        convolution[channel * kernel] = 0.04f;
        convolution[channel * kernel + 1] = -0.08f;
        convolution[channel * kernel + 2] = 0.16f;
        convolution[channel * kernel + 3] = 0.88f;
    }

    const array hidden_norm(
        {0.0f, 0.03f, -0.02f, 0.01f},
        Shape{hidden});
    std::vector<float> linear_norm_values(dimension);
    for (int index = 0; index < dimension; ++index) {
        linear_norm_values[index] =
            0.95f + 0.002f * static_cast<float>(index % 11);
    }

    return MlxQwen35LinearAttentionBlock(
        config,
        gguf_layout,
        MlxRmsNorm(
            hidden_norm,
            static_cast<float>(config.rms_norm_eps),
            static_cast<float>(config.norm_weight_offset)),
        split
            ? std::nullopt
            : std::optional<MlxLinear>(
                  MlxLinear(matrix_with_dtype(
                      channels,
                      hidden,
                      qkv_values,
                      projection_dtype))),
        split
            ? std::optional<MlxLinear>(
                  MlxLinear(matrix_with_dtype(
                      qk_size,
                      hidden,
                      qk_values,
                      projection_dtype)))
            : std::nullopt,
        split
            ? std::optional<MlxLinear>(
                  MlxLinear(matrix_with_dtype(
                      value_size,
                      hidden,
                      value_values,
                      projection_dtype)))
            : std::nullopt,
        MlxLinear(matrix_with_dtype(
            value_size,
            hidden,
            patterned(value_size * hidden, 0.006f, 17, 8),
            projection_dtype)),
        MlxLinear(matrix(
            value_heads,
            hidden,
            {
                0.07f, -0.04f, 0.03f, 0.02f,
                -0.02f, 0.06f, -0.05f, 0.04f,
            })),
        MlxLinear(matrix(
            value_heads,
            hidden,
            {
                -0.03f, 0.05f, 0.02f, -0.04f,
                0.04f, -0.02f, 0.06f, 0.01f,
            })),
        matrix(channels, kernel, convolution),
        force_generic ? std::optional<array>(mlx::core::zeros(Shape{channels}, mlx::core::float32)) : std::nullopt,
        array({-0.3f, 0.2f}, Shape{value_heads}),
        array(
            {std::log(0.65f), std::log(0.9f)},
            Shape{value_heads}),
        MlxRmsNorm(
            array(
                linear_norm_values.begin(),
                Shape{dimension}),
            static_cast<float>(config.rms_norm_eps)),
        MlxLinear(matrix_with_dtype(
            hidden,
            value_size,
            patterned(hidden * value_size, 0.004f, 19, 9),
            projection_dtype)),
        MlxRmsNorm(
            hidden_norm,
            static_cast<float>(config.rms_norm_eps),
            static_cast<float>(config.norm_weight_offset)),
        make_ffn(projection_dtype));
}

void require_array_close(
    const array& actual,
    const array& expected,
    float tolerance = 8e-4f) {
    require(actual.shape() == expected.shape(), "array shape mismatch");
    require(
        actual.dtype() == mlx::core::float32 &&
            expected.dtype() == mlx::core::float32,
        "test arrays must evaluate as float32");
    const auto* actual_values = actual.data<float>();
    const auto* expected_values = expected.data<float>();
    for (std::size_t index = 0; index < actual.size(); ++index) {
        require_close(
            actual_values[index],
            expected_values[index],
            tolerance);
    }
}

void test_combined_and_split_equivalence() {
    const array input(
        {
            0.25f, -0.5f, 0.75f, 0.1f,
            -0.2f, 0.4f, 0.3f, -0.7f,
            0.6f, 0.15f, -0.35f, 0.8f,
        },
        Shape{1, 3, 4});
    auto combined = make_block(false, true);
    auto split = make_block(true, false);
    auto combined_output = combined.forward(input, 0, false);
    auto split_output = split.forward(input, 0, false);
    mlx::core::eval(combined_output, split_output);

    require(!combined.split_input(), "combined layout was not selected");
    require(split.split_input(), "split layout was not selected");
    require(
        combined.uses_combined_alpha_beta() &&
            split.uses_combined_alpha_beta(),
        "dense alpha/beta projections were not combined");
    require_array_close(combined_output, split_output);
}

void test_prefill_decode_cache_equivalence() {
    const array full_input(
        {
            0.25f, -0.5f, 0.75f, 0.1f,
            -0.2f, 0.4f, 0.3f, -0.7f,
            0.6f, 0.15f, -0.35f, 0.8f,
        },
        Shape{1, 3, 4});
    const array prefix_input(
        {
            0.25f, -0.5f, 0.75f, 0.1f,
            -0.2f, 0.4f, 0.3f, -0.7f,
        },
        Shape{1, 2, 4});
    const array decode_input(
        {0.6f, 0.15f, -0.35f, 0.8f},
        Shape{1, 1, 4});

    auto full = make_block(false, true);
    auto full_output = full.forward(full_input, 0, true);

    auto decode = make_block(false, true);
    auto prefix_output = decode.forward(prefix_input, 0, true);
    auto decode_output = decode.forward(decode_input, 2, true);

    auto uncached = make_block(false, true);
    auto uncached_output =
        uncached.forward(full_input, 0, false);

    require(
        full.convolution_state().has_value() &&
            full.recurrent_state().has_value() &&
            decode.convolution_state().has_value() &&
            decode.recurrent_state().has_value(),
        "linear-attention cache state was not created");
    const auto full_conv = *full.convolution_state();
    const auto full_recurrent = *full.recurrent_state();
    const auto decode_conv = *decode.convolution_state();
    const auto decode_recurrent = *decode.recurrent_state();
    mlx::core::eval(
        full_output,
        prefix_output,
        decode_output,
        uncached_output,
        full_conv,
        full_recurrent,
        decode_conv,
        decode_recurrent);

    require(full.cache_position() == 3, "prefill cache position mismatch");
    require(decode.cache_position() == 3, "decode cache position mismatch");
    require(
        uncached.cache_position() == 0 &&
            !uncached.convolution_state().has_value() &&
            !uncached.recurrent_state().has_value(),
        "uncached call unexpectedly retained state");

    const auto* full_values = full_output.data<float>();
    const auto* decode_values = decode_output.data<float>();
    const auto* uncached_values = uncached_output.data<float>();
    for (int column = 0; column < 4; ++column) {
        require_close(
            decode_values[column],
            full_values[8 + column]);
    }
    for (int index = 0; index < 12; ++index) {
        require_close(
            uncached_values[index],
            full_values[index]);
    }
    require_array_close(full_conv, decode_conv);
    require_array_close(
        full_recurrent,
        decode_recurrent,
        1.5e-3f);
    require(
        std::fabs(full_values[8] - 0.6f) > 1e-5f,
        "linear-attention block did not update the residual");

    bool rejected_noncontiguous = false;
    try {
        (void)decode.forward(decode_input, 1, true);
    } catch (const std::runtime_error&) {
        rejected_noncontiguous = true;
    }
    require(
        rejected_noncontiguous,
        "non-contiguous linear-attention cache append was accepted");

    decode.clear_cache();
    require(
        decode.cache_position() == 0 &&
            decode.cache_batch() == 0 &&
            !decode.convolution_state().has_value() &&
            !decode.recurrent_state().has_value(),
        "clear_cache did not release linear-attention state");
}

void test_half_activation_dtype_preservation() {
    auto input = mlx::core::astype(
        array(
            {0.25f, -0.5f, 0.75f, 0.1f},
            Shape{1, 1, 4}),
        mlx::core::float16);
    auto block = make_block(
        false,
        true,
        mlx::core::float16);
    auto output = block.forward(input, 0, true);
    require(
        block.convolution_state().has_value() &&
            block.recurrent_state().has_value(),
        "half activation did not create linear-attention state");
    auto convolution_state = *block.convolution_state();
    auto recurrent_state = *block.recurrent_state();
    mlx::core::eval(
        output,
        convolution_state,
        recurrent_state);

    require(
        output.dtype() == mlx::core::float16,
        "FP16 linear-attention input was promoted at block output");
    require(
        convolution_state.dtype() == mlx::core::float32 &&
            recurrent_state.dtype() == mlx::core::float32,
        "linear-attention cache/state did not remain FP32");
}

void test_fused_decode_cache_equivalence() {
    using namespace mlx::core;
    for (const auto dtype : {float32, float16}) for (const bool tiled : {false, true})
    for (const bool split : {false, true}) {
        const auto input = astype(reshape(sin(arange(28, float32) * .17f) * .5f, {1, 7, 4}), dtype);
        auto full = make_block(split, tiled, dtype, 128);
        auto decoded = make_block(split, tiled, dtype, 128);
        auto restored = make_block(split, tiled, dtype, 128);
        const auto expected = astype(full.forward(input, 0, true), float32);
        auto prefix = decoded.forward(slice(input, {0, 0, 0}, {1, 2, 4}), 0, true);
        prefix.eval();
        restored.restore_cache(decoded.snapshot_cache());
        std::vector<array> steps;
        for (int token = 2; token < 7; ++token) {
            auto result = restored.forward(slice(input, {0, token, 0}, {1, token + 1, 4}), token, true);
            require(restored.recurrent_state()->has_primitive() &&
                std::string(restored.recurrent_state()->primitive().name()) == "GatedDeltaDecodeStep",
                "128-dimensional GDN decode did not use the fused kernel");
            steps.push_back(astype(result, float32));
        }
        const auto actual = concatenate(steps, 1);
        const auto tail = slice(expected, {0, 2, 0}, {1, 7, 4});
        eval(actual, tail, *full.convolution_state(), *restored.convolution_state(),
            *full.recurrent_state(), *restored.recurrent_state());
        require_array_close(actual, tail, dtype == float16 ? .003f : 1e-4f);
        require_array_close(*full.convolution_state(), *restored.convolution_state());
        require_array_close(*full.recurrent_state(), *restored.recurrent_state(), 1e-4f);
        require(restored.cache_position() == 7, "fused decode cache position mismatch");
    }
}

void test_fused_speculative_cache_equivalence() {
    using namespace mlx::core;
    const auto close = [](const array& actual, const array& expected, float tolerance) {
        const auto a = astype(actual, float32), b = astype(expected, float32);
        eval(a, b);
        require_array_close(a, b, tolerance);
    };
    for (const auto dtype : {float32, float16}) for (const bool tiled : {false, true})
    for (const bool split : {false, true}) for (const int tokens : {2, 3, 4, 5, 6, 8}) {
        const auto input = astype(reshape(sin(arange(48, float32) * .17f) * .5f, {1, 12, 4}), dtype);
        auto reference = make_block(split, tiled, dtype, 128, true);
        auto actual = make_block(split, tiled, dtype, 128);
        reference.forward(slice(input, {0, 0, 0}, {1, 2, 4}), 0, true).eval();
        const auto checkpoint = reference.snapshot_cache();
        const auto window = slice(input, {0, 2, 0}, {1, 2 + tokens, 4});
        const auto tolerance = dtype == float16 ? .003f : 1e-4f;
        for (const int accepted : {0, tokens / 2, tokens - 1}) {
            reference.restore_cache(checkpoint);
            actual.restore_cache(checkpoint);
            const auto expected = reference.forward(window, 2, true);
            const auto verified = actual.forward_speculative(window, 2, 1);
            require(actual.recurrent_state()->has_primitive() &&
                std::string(actual.recurrent_state()->primitive().name()) == "GatedDeltaDecodeStep",
                "GDN speculative window did not use the fused kernel");
            close(verified, expected, tolerance);
            if (accepted == tokens - 1) actual.commit_speculative();
            else actual.rollback_speculative(accepted);
            const int position = 3 + accepted;
            reference.restore_cache(checkpoint);
            reference.forward(slice(input, {0, 2, 0}, {1, position, 4}), 2, true).eval();
            require(actual.cache_position() == position, "fused GDN speculative cache position mismatch");
            close(*actual.convolution_state(), *reference.convolution_state(), 8e-4f);
            close(*actual.recurrent_state(), *reference.recurrent_state(), 1e-4f);
            const auto next = slice(input, {0, position, 0}, {1, position + 1, 4});
            close(actual.forward(next, position, true), reference.forward(next, position, true), tolerance);
            close(*actual.recurrent_state(), *reference.recurrent_state(), 1e-4f);
        }
    }
}

void test_borrowed_cache_snapshot() {
    using namespace mlx::core;
    for (const int dimension : {32, 128}) for (const bool split : {false, true})
        for (const bool tiled : {false, true}) for (const auto dtype : {float16, float32}) {
        auto block = make_block(split, tiled, dtype, dimension);
        const auto input = astype(array({.2f, -.4f, .1f, .3f}, Shape{1, 1, 4}), dtype);
        auto first = block.forward(input, true);
        const auto status = block.recurrent_state()->status();
        const auto borrowed = block.snapshot_cache(false);
        require(borrowed.convolution_state.id() == block.convolution_state()->id() &&
            borrowed.recurrent_state.id() == block.recurrent_state()->id() &&
            borrowed.recurrent_state.status() == status,
            "borrowed GDN snapshot copied or evaluated its cache");
        first.eval();
        const auto owned = block.snapshot_cache();
        auto discarded = block.forward(input, true);
        discarded.eval();
        require_array_close(borrowed.convolution_state, owned.convolution_state, 0.f);
        require_array_close(borrowed.recurrent_state, owned.recurrent_state, 0.f);
        block.restore_cache(borrowed, false);
        require(block.cache_position() == 1 && block.recurrent_state()->id() == borrowed.recurrent_state.id(),
            "borrowed GDN restore changed cache ownership or position");
        auto reference = make_block(split, tiled, dtype, dimension);
        reference.restore_cache(owned);
        const auto actual = astype(block.forward(input, true), float32);
        const auto expected = astype(reference.forward(input, true), float32);
        eval(actual, expected);
        require_array_close(actual, expected, 0.f);
    }
}

void test_shape_validation() {
    bool rejected = false;
    try {
        auto config = test_config();
        (void)MlxQwen35LinearAttentionBlock(
            config,
            false,
            MlxRmsNorm(array({0.0f, 0.0f, 0.0f, 0.0f}, Shape{4}), 1e-6f, 1.0f),
            MlxLinear(matrix(128, 4, patterned(512, 0.001f, 7, 3))),
            std::nullopt,
            std::nullopt,
            MlxLinear(matrix(63, 4, patterned(252, 0.001f, 7, 3))),
            MlxLinear(matrix(2, 4, patterned(8, 0.001f, 7, 3))),
            MlxLinear(matrix(2, 4, patterned(8, 0.001f, 7, 3))),
            matrix(128, 4, patterned(512, 0.001f, 7, 3)),
            std::nullopt,
            array({0.0f, 0.0f}, Shape{2}),
            array({0.0f, 0.0f}, Shape{2}),
            MlxRmsNorm(
                constant_vector(32, 1.0f),
                1e-6f),
            MlxLinear(matrix(4, 64, patterned(256, 0.001f, 7, 3))),
            MlxRmsNorm(array({0.0f, 0.0f, 0.0f, 0.0f}, Shape{4}), 1e-6f, 1.0f),
            make_ffn());
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "invalid linear-attention projection was accepted");
}

} // namespace

int main() {
    try {
#ifdef TYLOINFER_MLX_METALLIB_DEFAULT
        mlx::core::metal::set_metallib_path(
            TYLOINFER_MLX_METALLIB_DEFAULT);
        mlx::core::set_default_device(mlx::core::Device::gpu);
#endif
        test_combined_and_split_equivalence();
        test_prefill_decode_cache_equivalence();
        test_half_activation_dtype_preservation();
        test_fused_decode_cache_equivalence();
        test_fused_speculative_cache_equivalence();
        test_borrowed_cache_snapshot();
        test_shape_validation();
        std::cout
            << "MFQ C++ Qwen3.5 Gated DeltaNet block tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
