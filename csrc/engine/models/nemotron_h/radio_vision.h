#pragma once
#include "engine/generation/sequence.h"
#include "engine/models/common/prompt.h"
#include "engine/ops.h"

#include "engine/backends/cuda/ops/norm/rms.h"
#include "engine/backends/cuda/ops/vision/attention.h"
#include "engine/backends/cuda/ops/projection/bias.h"
#include "engine/backends/cuda/ops/norm/layer.h"
#include "engine/models/nemotron_h/vision.h"
#include "engine/models/nemotron_h/forward.h"
#include "engine/weights/loader.h"

#include "engine/requests/media.h"
#include "engine/backends/cuda/kernels/mfq_cuda_activation_ops.h"
#include "engine/models/common/model_graph.h"
#include "engine/state/session/prefix_cache.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfq::cuda::nemotron_h {

using Tensor = mfq_tensor_backend::Tensor;
using mfq::models::nemotron_h::RadioPositionInterpolation;
using mfq::models::nemotron_h::RadioVisionConfig;

// RADIO patch generator + encoder: normalized row-major patches of one image
// in, the tower's patch features out (cls/register prefix removed, exactly like
// `RadioModel.forward`'s `features`).
//
// The tower is a plain pre-LayerNorm ViT: biased q/k/v stored as one fused
// tensor, full self-attention over the patch grid, no positional rotation, no
// QK norm, and LayerScale folded away because released checkpoints carry no
// layer_scale parameter.
class RadioEncoder {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    using Tensor = mfq::Tensor;
#endif
  public:
    static RadioEncoder load(const mfq::cuda::weight_loader::Loader& weights,
                             const RadioVisionConfig &config) {
        config.validate();
        RadioEncoder result;
        result.patch_projection = weights.projection("vision.patch_embedding.weight");
        result.position_embedding =
            weights.native("vision.position_embedding.weight");
        result.cls_register_token =
            weights.native("vision.cls_register_token.weight");
        if (!result.patch_projection.is_dense()) {
            throw std::runtime_error(
                "Nemotron-H RADIO patch projection must stay dense: the tower is exported "
                "in BF16");
        }
        if (result.patch_projection.out() != config.hidden_size ||
            result.patch_projection.neuron_len() != config.patch_width()) {
            throw std::runtime_error(
                "Nemotron-H RADIO patch projection is not [hidden, patch_width]");
        }
        const auto side = config.position_side();
        if (result.position_embedding.sizes().vec() !=
            std::vector<int64_t>{1, side * side, config.hidden_size}) {
            throw std::runtime_error(
                "Nemotron-H RADIO position embedding does not match max_img_size");
        }
        if (result.cls_register_token.sizes().vec() !=
            std::vector<int64_t>{config.summary_tokens(), config.hidden_size}) {
            throw std::runtime_error(
                "Nemotron-H RADIO cls/register table disagrees with its config");
        }
        result.layers.reserve(static_cast<std::size_t>(config.depth));
        for (std::int64_t index = 0; index < config.depth; ++index) {
            result.layers.push_back(load_layer(weights, config, index));
        }
        result.config_ = config;
        return result;
    }

    // `patches` is [rows * columns, patch_width] row-major normalized pixels.
    Tensor encode(
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
                  const mfq::cuda::Projector& projector,
#endif
                  Tensor patches, std::int64_t rows,
                  std::int64_t columns) const {
        const auto& config = config_;
        if (rows <= 0 || columns <= 0 || rows > config.position_side() ||
            columns > config.position_side()) {
            throw std::runtime_error("Nemotron-H RADIO image patch grid is invalid");
        }
        const auto count = rows * columns;
        if (patches.dim() != 2 || patches.size(0) != count ||
            patches.size(1) != config.patch_width()) {
            throw std::runtime_error(
                "Nemotron-H RADIO pixel patches disagree with the image grid");
        }
        const auto dtype = layer_dtype();
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        auto cast_patches = mfq::ops::cast(patches, dtype);
        if (patch_projection.dense.dim() != 2) throw std::invalid_argument("linear weight must be a matrix");
        auto hidden = mfq::ops::contiguous(mfq::ops::cast(mfq::ops::matmul(
            cast_patches, mfq::ops::transpose(patch_projection.dense, 0, 1)), dtype));
        hidden = mfq::ops::add(hidden, mfq::ops::cast(interpolated_positions(rows, columns), dtype));
        auto prefix = mfq::ops::contiguous(mfq::ops::expand(mfq::ops::unsqueeze(
            mfq::ops::cast(cls_register_token, dtype), 0), {1, config.summary_tokens(), config.hidden_size}));
        hidden = mfq::ops::contiguous(mfq::ops::cat({prefix, mfq::ops::unsqueeze(hidden, 0)}, 1));
#else
        auto hidden =
            mfq_linear(patches.to(dtype), patch_projection.dense, std::nullopt)
                .to(dtype)
                .contiguous();
        hidden = hidden + interpolated_positions(rows, columns).to(dtype);
        auto prefix = cls_register_token.to(dtype)
                          .unsqueeze(0)
                          .expand({1, config.summary_tokens(), config.hidden_size})
                          .contiguous();
        hidden = mfq_tensor_backend::cat({prefix, hidden.unsqueeze(0)}, 1).contiguous();
#endif
        for (const auto& block : layers) {
            hidden = block.forward(
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
                projector,
#endif
                std::move(hidden));
        }
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        return mfq::ops::contiguous(mfq::ops::slice(hidden, 1, config.summary_tokens(), config.summary_tokens() + count));
#else
        return hidden.narrow(1, config.summary_tokens(), count).contiguous();
#endif
    }

    const RadioVisionConfig &config() const noexcept { return config_; }

  private:
    struct Layer {
        BiasWeight query;
        BiasWeight key;
        BiasWeight value;
        BiasWeight output;
        mfq_tensor_backend::Tensor norm1_weight;
        mfq_tensor_backend::Tensor norm1_bias;
        mfq_tensor_backend::Tensor norm2_weight;
        mfq_tensor_backend::Tensor norm2_bias;
        BiasWeight fc1;
        BiasWeight fc2;
        int64_t heads = 0;
        int64_t head_dim = 0;
        double layer_norm_eps = 1e-6;

        Tensor forward(
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
                const Projector& projector,
#endif
                Tensor input,
                MfqOptional<Tensor> mask = mfq_nullopt) const {
            if (heads <= 0 || head_dim <= 0) {
                throw std::runtime_error("vision block head geometry is invalid");
            }
            const int64_t batch = input.size(0);
            const int64_t tokens = input.size(1);
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
            auto split_heads = [&](Tensor value) {
                return mfq::ops::contiguous(mfq::ops::transpose(
                    mfq::ops::reshape(value, {batch, tokens, heads, head_dim}), 1, 2));
            };
#else
            auto split_heads = [&](Tensor value) {
                return value.reshape({batch, tokens, heads, head_dim}).transpose(1, 2).contiguous();
            };
#endif
            const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
            const auto norm = [&](const Tensor& x, const Tensor& weight, const Tensor& bias) {
                const auto dtype = x.scalar_type();
                return mfq::ops::contiguous(mfq::ops::cast(
                    mfq::ops::layer_norm(x, mfq::ops::cast(weight, dtype),
                                         mfq::ops::cast(bias, dtype), layer_norm_eps), dtype));
            };
            auto normalized = norm(input, norm1_weight, norm1_bias);
            auto q = split_heads(mfq::ops::linear(normalized, query.weight, query.bias));
            auto k = split_heads(mfq::ops::linear(normalized, key.weight, key.bias));
            auto v = split_heads(mfq::ops::linear(normalized, value.weight, value.bias));
            auto attended = mfq::ops::attention(std::move(q), std::move(k), std::move(v), mask, scale);
            auto branch = mfq::ops::linear(mfq::ops::reshape(mfq::ops::transpose(attended, 1, 2),
                {batch, tokens, heads * head_dim}), output.weight, output.bias);
            input = mfq::ops::contiguous(mfq::ops::add(input, mfq::ops::cast(branch, input.scalar_type())));
            normalized = norm(input, norm2_weight, norm2_bias);
            branch = mfq::ops::linear(mfq::ops::gelu(
                mfq::ops::linear(normalized, fc1.weight, fc1.bias)), fc2.weight, fc2.bias);
#else
            auto normalized = vision_layer_norm(input, norm1_weight, norm1_bias, layer_norm_eps);
            auto q = split_heads(mfq::cuda::biased(projector, query, normalized));
            auto k = split_heads(mfq::cuda::biased(projector, key, normalized));
            auto v = split_heads(mfq::cuda::biased(projector, value, normalized));
            auto attended = vision_non_causal_attention(std::move(q), std::move(k), std::move(v), mask, scale);
            auto branch = mfq::cuda::biased(projector, output, attended.transpose(1, 2).reshape(
                {batch, tokens, heads * head_dim}));
            input = (input + branch.to(input.scalar_type())).contiguous();
            normalized = vision_layer_norm(input, norm2_weight, norm2_bias, layer_norm_eps);
            branch = mfq::cuda::biased(projector, fc2,
                mfq_tensor_backend::gelu(mfq::cuda::biased(projector, fc1, normalized), "none"));
#endif
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
            return mfq::ops::contiguous(mfq::ops::add(input, mfq::ops::cast(branch, input.scalar_type())));
#else
            return (input + branch.to(input.scalar_type())).contiguous();
#endif
        }
    };

    static std::array<BiasWeight, 3> split_fused_qkv(const BiasWeight &fused, int64_t width) {
        if (width <= 0 || fused.weight.out() != 3 * width) {
            throw std::runtime_error("fused vision qkv projection is not [3*w, hidden]");
        }
        return {bias_rows(fused, 0, width), bias_rows(fused, width, width),
                bias_rows(fused, 2 * width, width)};
    }

    static Layer load_layer(const mfq::cuda::weight_loader::Loader& weights,
                            const RadioVisionConfig &config, std::int64_t index) {
        const std::string prefix = "vision.block." + std::to_string(index);
        const auto fused = weights.biased(prefix + ".attention.qkv.weight", prefix + ".attention.qkv.bias");
        const auto split = split_fused_qkv(fused, config.hidden_size);
        Layer block;
        block.query = split[0];
        block.key = split[1];
        block.value = split[2];
        block.output = weights.biased(prefix + ".attention.output.weight", prefix + ".attention.output.bias");
        block.norm1_weight = weights.native(prefix + ".norm1.weight");
        block.norm1_bias = weights.native(prefix + ".norm1.bias");
        block.norm2_weight = weights.native(prefix + ".norm2.weight");
        block.norm2_bias = weights.native(prefix + ".norm2.bias");
        block.fc1 = weights.biased(prefix + ".mlp.up.weight", prefix + ".mlp.up.bias");
        block.fc2 = weights.biased(prefix + ".mlp.down.weight", prefix + ".mlp.down.bias");
        block.heads = config.num_heads;
        block.head_dim = config.head_dim();
        block.layer_norm_eps = config.layer_norm_eps;
        const bool geometry_ok =
            block.output.weight.out() == config.hidden_size &&
            block.output.weight.neuron_len() == config.hidden_size &&
            block.fc1.weight.out() == config.intermediate_size &&
            block.fc1.weight.neuron_len() == config.hidden_size &&
            block.fc2.weight.out() == config.hidden_size &&
            block.fc2.weight.neuron_len() == config.intermediate_size;
        if (!geometry_ok) {
            throw std::runtime_error("Nemotron-H RADIO block " + std::to_string(index) +
                                     " geometry disagrees with its config");
        }
        return block;
    }

    // Cropped-position embedding: bilinear interpolation of the square table
    // onto this image's patch grid, gathered with the four-tap index/weight
    // table so no host-to-device reshape is needed.
    Tensor interpolated_positions(std::int64_t rows, std::int64_t columns) const {
        const auto interpolation = mfq::models::nemotron_h::make_radio_position_interpolation(
            static_cast<std::int32_t>(rows), static_cast<std::int32_t>(columns),
            static_cast<std::int32_t>(config_.position_side()));
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        const auto table = mfq::ops::reshape(position_embedding,
            {config_.position_side() * config_.position_side(), config_.hidden_size});
        Tensor index = mfq::ops::tensor(interpolation.indices, mfq::Device{mfq::DeviceType::cpu, 0});
        index = mfq::ops::cast(index, mfq_tensor_backend::kInt32, table.device());
        index = mfq::ops::cast(index, mfq_tensor_backend::kInt64);
        Tensor weights = mfq::ops::tensor(interpolation.weights, mfq::Device{mfq::DeviceType::cpu, 0});
        weights = mfq::ops::reshape(weights, {interpolation.patch_count, 4, 1});
        weights = mfq::ops::cast(weights, mfq_tensor_backend::kFloat32, table.device());
        return mfq::ops::contiguous(mfq::ops::sum(mfq::ops::multiply(mfq::ops::reshape(
            mfq::ops::index_select(table, 0, mfq::ops::reshape(index, {-1})),
            {interpolation.patch_count, 4, config_.hidden_size}), weights), 1));
#else
        const auto table = position_embedding.reshape(
            {config_.position_side() * config_.position_side(), config_.hidden_size});
        auto index =
            mfq_tensor_backend::tensor(interpolation.indices,
                                       mfq_tensor_backend::TensorOptions()
                                           .dtype(mfq_tensor_backend::kInt32)
                                           .device(mfq_tensor_backend::kCPU))
                .to(table.device())
                .to(mfq_tensor_backend::kInt64);
        auto weights =
            mfq_tensor_backend::tensor(interpolation.weights,
                                       mfq_tensor_backend::TensorOptions()
                                           .dtype(mfq_tensor_backend::kFloat32)
                                           .device(mfq_tensor_backend::kCPU))
                .reshape({interpolation.patch_count, 4, 1})
                .to(table.device());
        return mfq_tensor_backend::sum(
                   table.index_select(0, index.reshape({-1}))
                           .reshape({interpolation.patch_count, 4, config_.hidden_size}) *
                       weights,
                   1)
            .contiguous();
#endif
    }

    mfq_tensor_backend::ScalarType layer_dtype() const {
        return position_embedding.scalar_type();
    }

    QuantLinear patch_projection;
    Tensor position_embedding;
    Tensor cls_register_token;
    std::vector<Layer> layers;
    RadioVisionConfig config_;
};

// Vision-to-language projector: `vision_final_layernorm` -> pixel shuffle
// (downsample 0.5, `ps_version v2`) -> `mlp1` (RMSNorm, up, relu-squared, down).
class RadioProjector {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    using Tensor = mfq::Tensor;
#endif
  public:
    static RadioProjector load(const mfq::cuda::weight_loader::Loader& weights,
                               const RadioVisionConfig &config) {
        RadioProjector result;
        result.norm_weight =
            weights.native("vision.projector.norm.weight");
        result.norm_bias =
            weights.native("vision.projector.norm.bias");
        result.mlp_norm =
            weights.native("vision.projector.mlp.norm.weight");
        result.up = weights.biased("vision.projector.mlp.up.weight");
        result.down = weights.biased("vision.projector.mlp.down.weight");
        if (result.norm_weight.numel() != config.hidden_size ||
            result.norm_bias.numel() != config.hidden_size ||
            result.mlp_norm.numel() != config.shuffle_width() ||
            result.up.weight.neuron_len() != config.shuffle_width() ||
            result.up.weight.out() != config.projector_intermediate_size ||
            result.down.weight.neuron_len() != config.projector_intermediate_size ||
            result.down.weight.out() != config.text_hidden_size) {
            throw std::runtime_error(
                "Nemotron-H RADIO projector geometry disagrees with its config");
        }
        result.config_ = config;
        return result;
    }

    // `features` is [1, rows * columns, hidden]; the result is
    // [rows * columns / 4, text_hidden_size] in pixel-shuffle token order.
    Tensor project(
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
                   const mfq::cuda::Projector& projector,
#endif
                   Tensor features, std::int64_t rows,
                   std::int64_t columns) const {
        const auto& config = config_;
        const auto stride = config.downsample_factor;
        if (rows <= 0 || columns <= 0 || rows % stride || columns % stride) {
            throw std::runtime_error(
                "Nemotron-H RADIO patch grid does not divide the pixel shuffle factor");
        }
        const auto patches = rows * columns;
        if (features.dim() != 3 || features.size(0) != 1 || features.size(1) != patches ||
            features.size(2) != config.hidden_size) {
            throw std::runtime_error("Nemotron-H RADIO tower features have an invalid shape");
        }
        const auto dtype = features.scalar_type();
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        auto hidden = Tensor(mfq::ops::contiguous(mfq::ops::cast(
            mfq::ops::layer_norm(features, mfq::ops::cast(norm_weight, dtype),
                                 mfq::ops::cast(norm_bias, dtype), config.layer_norm_eps), dtype)));
#else
        auto hidden = vision_layer_norm(features, norm_weight, norm_bias, config.layer_norm_eps);
#endif
        // Pixel shuffle with `ps_version v2`: regroup each 2x2 patch block into
        // one token (4 * hidden features) in row-pair / column-pair order.  The
        // v1 variant skips the final permute and transposes the token grid, so
        // the trailing transpose is not redundant.
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        hidden = mfq::ops::reshape(hidden, {1, rows, columns / stride, stride * config.hidden_size});
        hidden = mfq::ops::contiguous(mfq::ops::transpose(hidden, 1, 2));
        hidden = mfq::ops::reshape(hidden,
            {1, columns / stride, rows / stride, stride * stride * config.hidden_size});
        hidden = mfq::ops::contiguous(mfq::ops::transpose(hidden, 1, 2));
        hidden = mfq::ops::reshape(hidden, {1, patches / (stride * stride), config.shuffle_width()});
#else
        hidden = hidden.reshape({1, rows, columns / stride, stride * config.hidden_size});
        hidden = hidden.transpose(1, 2).contiguous();
        hidden = hidden.reshape(
            {1, columns / stride, rows / stride, stride * stride * config.hidden_size});
        hidden = hidden.transpose(1, 2).contiguous();
        hidden = hidden.reshape({1, patches / (stride * stride), config.shuffle_width()});
#endif
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        auto normalized = mfq::ops::cast(mfq::ops::reshape(mfq::ops::rms_norm(
            mfq::ops::reshape(mfq::ops::cast(hidden, mfq_tensor_backend::kFloat32), {-1, config.shuffle_width()}),
            mfq::ops::cast(mlp_norm, mfq_tensor_backend::kFloat32), config.projector_norm_eps, 0.0),
            hidden.sizes()), dtype);
        auto projected = mfq::ops::relu_squared(mfq::ops::linear(normalized, up.weight, up.bias));
        return mfq::ops::contiguous(mfq::ops::reshape(
            mfq::ops::linear(std::move(projected), down.weight, down.bias),
            {patches / (stride * stride), config.text_hidden_size}));
#else
        auto normalized = qwen_rms_norm(hidden.to(mfq_tensor_backend::kFloat32).reshape(
                                            {-1, config.shuffle_width()}),
                                        mlp_norm.to(mfq_tensor_backend::kFloat32),
                                        config.projector_norm_eps, 0.0)
                              .reshape(hidden.sizes())
                              .to(dtype);
        auto projected = mfq::cuda::biased(projector, up, std::move(normalized));
        projected = relu_squared_cuda(projected.contiguous());
        return mfq::cuda::biased(projector, down, std::move(projected))
            .reshape({patches / (stride * stride), config.text_hidden_size})
            .contiguous();
#endif
    }

  private:
    Tensor norm_weight;
    Tensor norm_bias;
    Tensor mlp_norm;
    BiasWeight up;
    BiasWeight down;
    RadioVisionConfig config_;
};

// RADIO tower plus projector, i.e. everything between normalized pixels and
// the language model's placeholder rows.
class RadioVisionTower {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    using Tensor = mfq::Tensor;
#endif
  public:
    static RadioVisionTower load(const mfq::cuda::weight_loader::Loader& weights,
                                 const RadioVisionConfig &config) {
        return RadioVisionTower(RadioEncoder::load(weights, config),
                                RadioProjector::load(weights, config), config);
    }

    Tensor embed(
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
                 const mfq::cuda::Projector& projector,
#endif
                 Tensor patches, std::int64_t rows,
                 std::int64_t columns) const {
        auto features = encoder_.encode(
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
            projector,
#endif
            std::move(patches), rows, columns);
        return projector_.project(
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
            projector,
#endif
            std::move(features), rows, columns);
    }

    std::int64_t tokens(std::int64_t rows, std::int64_t columns) const {
        const auto stride = config_.downsample_factor;
        return rows * columns / (stride * stride);
    }

    const RadioVisionConfig &config() const noexcept { return config_; }

  private:
    RadioVisionTower(RadioEncoder encoder, RadioProjector projector,
                     RadioVisionConfig config)
        : encoder_(std::move(encoder)), projector_(std::move(projector)),
          config_(std::move(config)) {}

    RadioEncoder encoder_;
    RadioProjector projector_;
    RadioVisionConfig config_;
};

// One image's patch grid from the media payload.  Rows and columns are patch
// counts before the pixel shuffle.
struct RadioImageGrid {
    std::int64_t rows = 0;
    std::int64_t columns = 0;
};

// RADIO prompt component: encodes every image of a media payload and rewrites
// the `<image>` placeholder rows of the prompt embeddings.  Nemotron-H has no
// multimodal position policy (the text backbone is NoPE), so the returned
// positions are plain token offsets and only address the text cache.
class CudaRadioPromptComponent {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    using Tensor = mfq::Tensor;
#endif
  public:
    static CudaRadioPromptComponent load(const mfq::cuda::weight_loader::Loader& weights,
                                         const RadioVisionConfig &config,
                                         std::string input_contract,
                                         std::string position_policy) {
        if (input_contract != mfq::kRadioCpeInputContract) {
            throw std::invalid_argument("unsupported RADIO vision input contract: " +
                                        input_contract);
        }
        if (!position_policy.empty()) {
            throw std::invalid_argument("RADIO vision declares no position policy, got: " +
                                        position_policy);
        }
        if (config.image_token_id < 0) {
            throw std::invalid_argument("Nemotron-H RADIO needs the `<image>` placeholder id");
        }
        return CudaRadioPromptComponent(RadioVisionTower::load(weights, config), config,
                                        std::move(input_contract));
    }

    mfq::StepSequence<PreparedPrompt> prepare(Model &language,
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
                                                  Projector projector,
                                                  Executor &executor, Collectives &collectives,
#endif
                                                  int primary,
                                                  const std::vector<std::int64_t> &token_ids,
                                                  const MfqMultimodalInput &media) const {
        const auto grids = validated_grids(media);
        const auto runs = placeholder_runs(token_ids, grids);
        const auto cache_key = media_cache_key(media, grids);
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        Tensor pixels = mfq::ops::from_blob(
                          const_cast<float *>(media.pixel_values.data()), media.pixel_shape,
                          mfq::TensorOptions()
                              .dtype(mfq::ScalarType::float32)
                              .device(mfq::DeviceType::cpu));
        pixels = mfq::ops::clone(pixels);
        pixels = mfq::ops::cast(pixels, mfq_tensor_backend::kFloat32, mfq::Device{mfq_tensor_backend::kCUDA, 0});
#else
        auto pixels = mfq_tensor_backend::from_blob(
                          const_cast<float *>(media.pixel_values.data()), media.pixel_shape,
                          mfq_tensor_backend::TensorOptions()
                              .dtype(mfq_tensor_backend::kFloat32)
                              .device(mfq_tensor_backend::kCPU))
                          .clone()
                          .to(mfq_tensor_backend::kCUDA);
#endif
        co_yield mfq::StepState::advanced;

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        Tensor ids = mfq::ops::tensor(token_ids, mfq::Device{mfq::DeviceType::cpu, 0});
        ids = mfq::ops::reshape(ids, {1, static_cast<std::int64_t>(token_ids.size())});
        ids = mfq::ops::cast(ids, mfq_tensor_backend::kInt64, mfq::Device{mfq_tensor_backend::kCUDA, 0});
        auto embeddings = mfq::ops::contiguous(language.embedding(ids, primary));
#else
        auto ids = mfq_tensor_backend::tensor(
                       token_ids, mfq_tensor_backend::TensorOptions()
                                      .dtype(mfq_tensor_backend::kInt64)
                                      .device(mfq_tensor_backend::kCPU))
                       .reshape({1, static_cast<std::int64_t>(token_ids.size())})
                       .to(mfq_tensor_backend::kCUDA);
        auto embeddings = embed(language, executor, collectives, primary, ids).contiguous();
#endif
        std::int64_t patch_offset = 0;
        std::size_t run_index = 0;
        for (const auto& grid : grids) {
            const auto count = grid.rows * grid.columns;
            const auto expected = tower_.tokens(grid.rows, grid.columns);
            const auto& run = runs[run_index];
            if (static_cast<std::int64_t>(run.size()) != expected) {
                throw std::invalid_argument(
                    "Nemotron-H image placeholders disagree with the projected patch count");
            }
            auto projected =
                tower_.embed(
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
                             projector,
#endif
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                             mfq::ops::slice(pixels, 0, patch_offset, patch_offset + count),
#else
                             pixels.narrow(0, patch_offset, count),
#endif
                             grid.rows, grid.columns);
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
            Tensor index = mfq::ops::tensor(run, mfq::Device{mfq::DeviceType::cpu, 0});
            index = mfq::ops::cast(index, mfq_tensor_backend::kInt64, mfq::Device{mfq_tensor_backend::kCUDA, 0});
            mfq::ops::index_copy(embeddings, 1, index,
                mfq::ops::unsqueeze(mfq::ops::cast(projected, embeddings.scalar_type()), 0));
#else
            auto index = mfq_tensor_backend::tensor(
                             run, mfq_tensor_backend::TensorOptions()
                                      .dtype(mfq_tensor_backend::kInt64)
                                      .device(mfq_tensor_backend::kCPU))
                             .to(mfq_tensor_backend::kCUDA);
            embeddings.index_copy_(1, index,
                                   projected.to(embeddings.scalar_type()).unsqueeze(0));
#endif
            patch_offset += count;
            ++run_index;
            co_yield mfq::StepState::advanced;
        }
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        auto positions = mfq::ops::reshape(mfq::ops::arange(0,
            static_cast<std::int64_t>(token_ids.size()), mfq::Device{mfq_tensor_backend::kCUDA, 0}),
            {1, static_cast<std::int64_t>(token_ids.size())});
#else
        auto positions = mfq_tensor_backend::arange(
                             static_cast<std::int64_t>(token_ids.size()),
                             mfq_tensor_backend::TensorOptions()
                                 .dtype(mfq_tensor_backend::kInt64)
                                 .device(mfq_tensor_backend::kCUDA))
                             .reshape({1, static_cast<std::int64_t>(token_ids.size())});
#endif
        co_yield PreparedPrompt{token_ids, std::move(embeddings), std::move(positions), 0,
                                    cache_key};
    }

    const std::string &input_contract() const noexcept { return input_contract_; }
    const RadioVisionConfig &config() const noexcept { return config_; }

  private:
    CudaRadioPromptComponent(RadioVisionTower tower, RadioVisionConfig config,
                             std::string input_contract)
        : tower_(std::move(tower)), config_(std::move(config)),
          input_contract_(std::move(input_contract)) {}

    // Geometry and contract checks for one media payload: images only, one
    // `<image>` run per image, and row-major patches that divide the shuffle.
    std::vector<RadioImageGrid> validated_grids(const MfqMultimodalInput &media) const {
        const auto stride = config_.downsample_factor;
        if (media.processor != MfqMultimodalProcessor::grid_vision ||
            media.processor_name != input_contract_ || !media.video_grid.empty() ||
            media.vision_grid_shape.size() != 2 || media.vision_grid_shape[1] != 3 ||
            media.pixel_shape.size() != 2 || media.pixel_shape[0] <= 0 ||
            media.pixel_shape[1] != config_.patch_width() ||
            media.image_grid != media.vision_grid ||
            media.image_grid_shape != media.vision_grid_shape) {
            throw std::invalid_argument("Nemotron-H RADIO media geometry is invalid");
        }
        const auto images = media.vision_grid_shape[0];
        if (images <= 0 || media.vision_types.size() != static_cast<std::size_t>(images)) {
            throw std::invalid_argument("Nemotron-H RADIO media needs one grid per image");
        }
        std::vector<RadioImageGrid> grids;
        grids.reserve(static_cast<std::size_t>(images));
        std::int64_t patches = 0;
        for (std::int64_t image = 0; image < images; ++image) {
            if (media.vision_types[static_cast<std::size_t>(image)] != 1) {
                throw std::invalid_argument(
                    "Nemotron-H RADIO accepts images only; video is not supported");
            }
            const auto offset = static_cast<std::size_t>(3 * image);
            const auto temporal = media.vision_grid[offset];
            const RadioImageGrid grid{media.vision_grid[offset + 1],
                                      media.vision_grid[offset + 2]};
            if (temporal != 1 || grid.rows <= 0 || grid.columns <= 0 ||
                grid.rows % stride || grid.columns % stride) {
                throw std::invalid_argument(
                    "Nemotron-H RADIO patch grid must be a positive multiple of the pixel "
                    "shuffle factor");
            }
            patches += grid.rows * grid.columns;
            grids.push_back(grid);
        }
        if (patches != media.pixel_shape[0]) {
            throw std::invalid_argument(
                "Nemotron-H RADIO patch count disagrees with pixel_values");
        }
        return grids;
    }

    // Token positions of every `<image>` placeholder, grouped per image in media
    // order.  A run must be contiguous: the processor emits `<img>` + N*N_token
    // + `</img>` for each image, so a gap means the prompt and the payload
    // disagree about how many images were sent.
    std::vector<std::vector<std::int64_t>> placeholder_runs(
            const std::vector<std::int64_t> &token_ids,
            const std::vector<RadioImageGrid> &grids) const {
        std::vector<std::int64_t> positions;
        for (std::size_t index = 0; index < token_ids.size(); ++index) {
            if (token_ids[index] == config_.image_token_id) {
                positions.push_back(static_cast<std::int64_t>(index));
            }
        }
        std::vector<std::vector<std::int64_t>> runs;
        runs.reserve(grids.size());
        std::size_t offset = 0;
        for (const auto& grid : grids) {
            const auto count = static_cast<std::size_t>(tower_.tokens(grid.rows, grid.columns));
            if (offset + count > positions.size()) {
                throw std::invalid_argument(
                    "Nemotron-H prompt holds fewer `<image>` placeholders than the images "
                    "require");
            }
            std::vector<std::int64_t> run(positions.begin() + static_cast<std::ptrdiff_t>(offset),
                                          positions.begin() +
                                              static_cast<std::ptrdiff_t>(offset + count));
            for (std::size_t index = 1; index < run.size(); ++index) {
                if (run[index] != run[0] + static_cast<std::int64_t>(index)) {
                    throw std::invalid_argument(
                        "Nemotron-H `<image>` placeholders of one image must be contiguous");
                }
            }
            runs.push_back(std::move(run));
            offset += count;
        }
        if (offset != positions.size()) {
            throw std::invalid_argument(
                "Nemotron-H prompt holds more `<image>` placeholders than the images require");
        }
        return runs;
    }

    std::string media_cache_key(const MfqMultimodalInput &media,
                                const std::vector<RadioImageGrid> &grids) const {
        std::string key = input_contract_ + '|';
        key += mfq::cache::block_hash_hex(mfq::cache::sha256(
            media.pixel_values.data(), media.pixel_values.size() * sizeof(float)));
        for (const auto& grid : grids) {
            key.push_back('|');
            key += std::to_string(grid.rows);
            key.push_back('x');
            key += std::to_string(grid.columns);
        }
        return key;
    }

    RadioVisionTower tower_;
    RadioVisionConfig config_;
    std::string input_contract_;
};

} // namespace mfq::cuda::nemotron_h
