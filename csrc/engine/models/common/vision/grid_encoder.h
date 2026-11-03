#pragma once
#include "engine/models/common/prompt.h"

#include "engine/weights/fields.h"
#include "engine/weights/routed_cache.h"
#include "engine/weights/workspace.h"
#include "engine/backends/cuda/ops/norm/layer.h"
#include "engine/backends/cuda/ops/projection/bias.h"
#include "engine/backends/cuda/ops/vision/attention.h"
#include "engine/ops.h"
#include "engine/models/common/transformer_layer.h"
#include "engine/models/common/vision/grid_forward.h"
#include "engine/models/common/vision/grid_vision.h"
#include "engine/requests/media.h"
#include "engine/state/session/prefix_cache.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfq::cuda::weight_loader { struct Loader; }

namespace mfq::cuda {
class Collectives;
class Executor;
}

namespace mfq::cuda::grid_vision_runtime {

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
using Tensor = mfq::Tensor;
#else
using Tensor = mfq_tensor_backend::Tensor;
#endif

class CudaGridVisionEncoder {
  public:
    static CudaGridVisionEncoder load(const weight_loader::Loader& weights,
                                      const GridVisionConfig &config);

    mfq::StepSequence<Tensor> encode(
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
                                   Projector projector,
#endif
                                   Tensor pixels, const std::vector<GridShape> &grids) const;

    const GridVisionConfig &config() const noexcept { return config_; }
    bool host_weights() const noexcept { return !patch_weight_.is_cuda(); }

  private:
    struct Attention {
        BiasWeight qkv;
        BiasWeight output;
        std::int64_t hidden = 0;
        std::int64_t heads = 0;
        std::int64_t head_dim = 0;
        double theta = 10'000.0;

        Tensor operator()(
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
            const Projector& projector,
#endif
            const Tensor& input,
            const std::vector<std::int32_t>& positions,
            const std::vector<std::int32_t>& segment_lengths) const {
            const std::int64_t tokens = input.size(0);
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
            auto projected = mfq::ops::reshape(mfq::ops::affine(input, qkv.weight, qkv.bias), {tokens, 3, heads, head_dim});
            auto query = mfq::ops::axial_rope(mfq::ops::select(projected, 1, 0), positions, theta);
            auto key = mfq::ops::axial_rope(mfq::ops::select(projected, 1, 1), positions, theta);
            auto value = mfq::ops::select(projected, 1, 2);
            auto attended = mfq::ops::segmented_attention(query, key, value, segment_lengths);
            return mfq::ops::affine(mfq::ops::reshape(attended, {tokens, hidden}), output.weight, output.bias);
#else
            auto projected = mfq::cuda::affine(projector, qkv, input).reshape({tokens, 3, heads, head_dim});
            auto query = vision::apply_axial_rope(projected.select(1, 0), positions, theta);
            auto key = vision::apply_axial_rope(projected.select(1, 1), positions, theta);
            auto value = projected.select(1, 2);
            auto attended = vision::segmented_attention(query, key, value, segment_lengths);
            return mfq::cuda::affine(projector, output, attended.reshape({tokens, hidden}));
#endif
        }
    };

    struct Layer {
        vision::LayerNorm norm1;
        Attention attention;
        vision::LayerNorm norm2;
        BiasWeight mlp_up, mlp_down;

        template <class Visitor>
        void visit_tensors(Visitor visit) {
            fields::tensors(norm1, visit);
            fields::tensors(norm2, visit);
            fields::tensors(attention.qkv, visit);
            fields::tensors(attention.output, visit);
            fields::tensors(mlp_up, visit);
            fields::tensors(mlp_down, visit);
        }

        Tensor operator()(
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
                          const Projector& projector,
#endif
                          const Tensor& input,
                          const std::vector<std::int32_t>& positions,
                          const std::vector<std::int32_t>& segment_lengths) const {
            return mfq::models::pre_norm_layer(
                input, [&](const Tensor& x, int stage) { return stage == 0 ? norm1(x) : norm2(x); },
                [&](Tensor x) { return attention(
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
                    projector,
#endif
                    x, positions, segment_lengths); },
                [&](Tensor x) {
                    return mfq::ops::mlp(
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                        std::move(x), [&](Tensor value) { return mfq::ops::affine(std::move(value), mlp_up.weight, mlp_up.bias); },
                        [](Tensor value) { return mfq::ops::gelu(std::move(value), true); },
                        [&](Tensor value) { return mfq::ops::affine(std::move(value), mlp_down.weight, mlp_down.bias); });
#else
                        std::move(x), [&](Tensor value) { return mfq::cuda::affine(projector, mlp_up, std::move(value)); },
                        [](Tensor value) { return mfq_tensor_backend::gelu(value, "tanh"); },
                        [&](Tensor value) { return mfq::cuda::affine(projector, mlp_down, std::move(value)); });
#endif
                },
                [](Tensor residual, Tensor branch) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                    auto dtype = residual.scalar_type();
                    return mfq::ops::add(std::move(residual), mfq::ops::cast(std::move(branch), dtype));
#else
                    return residual + branch.to(residual.scalar_type());
#endif
                });
        }
    };

    CudaGridVisionEncoder(RoutedCache& weights, GridVisionConfig config, Tensor patch_weight, Tensor patch_bias,
                          Tensor position_weight, std::vector<Layer> blocks);

    RoutedCache& weights_;
    GridVisionConfig config_;
    Tensor patch_weight_;
    Tensor patch_bias_;
    Tensor position_weight_;
    std::vector<Layer> blocks_;
};

class CudaGridVisionPromptComponent {
  public:
    static CudaGridVisionPromptComponent
    load(const weight_loader::Loader& weights,
         const GridVisionConfig &vision, int64_t image_token_id, int64_t video_token_id,
         std::string input_contract, std::string position_policy);

    template <typename Model>
    mfq::StepSequence<PreparedPrompt> prepare(
        const Model &language, MediaCache &images,
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
        Projector projector, Collectives &collectives,
#endif
        int primary,
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
        Executor *executor,
#endif
        const std::vector<int64_t> &token_ids, const MfqMultimodalInput &media) const {
        if (media.processor != MfqMultimodalProcessor::grid_vision ||
            media.processor_name != input_contract_ ||
            media.vision_grid_shape != std::vector<int64_t>{1, 3} ||
            media.vision_grid.size() != 3 || media.vision_types != std::vector<int32_t>{1} ||
            !media.video_grid.empty()) {
            throw std::invalid_argument(
                "CUDA grid-Vision milestone accepts exactly one image and no video");
        }
        const GridShape grid{media.vision_grid[0], media.vision_grid[1], media.vision_grid[2]};
        const auto expected_pixel_shape =
            std::vector<int64_t>{static_cast<int64_t>(grid.temporal) * grid.height * grid.width,
                                 encoder_.config().patch_width()};
        if (grid.temporal != 1 || media.image_grid != media.vision_grid ||
            media.image_grid_shape != std::vector<int64_t>{1, 3} ||
            media.pixel_shape != expected_pixel_shape ||
            media.pixel_values.size() !=
                static_cast<size_t>(expected_pixel_shape[0] * expected_pixel_shape[1])) {
            throw std::invalid_argument("CUDA grid-Vision image geometry is invalid");
        }
        std::string cache_key =
            input_contract_ + '|' + position_policy_ + '|' +
            mfq::cache::block_hash_hex(mfq::cache::sha256(
                media.pixel_values.data(), media.pixel_values.size() * sizeof(float)));
        for (const auto value : media.pixel_shape) {
            cache_key.push_back('|');
            cache_key += std::to_string(value);
        }
        for (const auto value : media.vision_grid) {
            cache_key.push_back('|');
            cache_key += std::to_string(value);
        }

        Tensor merged = images.find(cache_key);
        if (!merged.defined()) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
            auto pixels = mfq::ops::from_blob(
                const_cast<float *>(media.pixel_values.data()), media.pixel_shape,
                mfq::TensorOptions().dtype(mfq::ScalarType::float32).device(mfq::DeviceType::cpu));
            pixels = mfq::ops::clone(std::move(pixels));
            pixels = mfq::ops::cast(std::move(pixels), mfq::ScalarType::float32, mfq::Device{mfq::DeviceType::cuda, 0});
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
            auto encoding = encoder_.encode(
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
                projector,
#endif
                std::move(pixels), {grid});
            Tensor patches;
            while (auto step = encoding.next()) {
                if (step.value)
                    patches = std::move(*step.value);
                else
                    co_yield step.state;
            }
            const auto merge = [&](const auto &norm, const auto &up, const auto &down) {
                return mfq::models::grid_vision::merge(
                    patches, patches.size(0), encoder_.config(),
                    [&](Tensor value) { return norm(value); },
                    [](Tensor value, int64_t rows, int64_t width) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                        return mfq::ops::reshape(std::move(value), {rows, width});
#else
                        return value.reshape({rows, width});
#endif
                    },
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                    [&](Tensor value) { return mfq::ops::affine(std::move(value), up.weight, up.bias); },
                    [](Tensor value) { return mfq::ops::gelu(std::move(value)); },
                    [&](Tensor value) { return mfq::ops::affine(std::move(value), down.weight, down.bias); });
#else
                    [&](Tensor value) { return mfq::cuda::affine(projector, up, std::move(value)); },
                    [](Tensor value) { return mfq_tensor_backend::gelu(value, "none"); },
                    [&](Tensor value) { return mfq::cuda::affine(projector, down, std::move(value)); });
#endif
            };
            if (encoder_.host_weights()) {
                const auto bytes = fields::bytes(merger_norm_) +
                                   fields::bytes(merger_up_) +
                                   fields::bytes(merger_down_);
                auto loan = weights_.borrow(bytes + patches.nbytes() * 12 + (std::size_t(16) << 20));
                Tensor output;
                {
                    auto norm = fields::to_cuda(merger_norm_);
                    auto up = fields::to_cuda(merger_up_);
                    auto down = fields::to_cuda(merger_down_);
                    output = merge(norm, up, down);
                }
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                merged = loan.finish(mfq_tensor_backend::Tensor(std::move(output)));
#else
                merged = loan.finish(std::move(output));
#endif
            } else {
                merged = merge(merger_norm_, merger_up_, merger_down_);
            }
            images.store(cache_key, merged);
            co_yield mfq::StepState::advanced;
        }

        auto prepared = mfq::models::grid_vision::prepare(
            token_ids, merged, merged.size(0), image_token_id_, video_token_id_, encoder_.config(),
            {grid},
            [&](const auto &tokens) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                auto ids = mfq::ops::tensor(tokens, mfq::Device{mfq::DeviceType::cuda, 0});
                Tensor shaped = mfq::ops::reshape(std::move(ids), {1, -1});
                if constexpr (requires(const Model &value, Tensor value_ids, int device) {
                                  value.embedding(std::move(value_ids), device);
                              }) {
                    return mfq::ops::contiguous(language.embedding(std::move(shaped), primary));
                } else {
                    return mfq::ops::contiguous(language.embed_forward(mfq_tensor_backend::Tensor(shaped)));
                }
#else
                auto ids =
                    mfq_tensor_backend::tensor(tokens, mfq_tensor_backend::TensorOptions()
                                                           .dtype(mfq_tensor_backend::kInt64)
                                                           .device(mfq_tensor_backend::kCUDA));
                ids = ids.reshape({1, -1});
                if constexpr (requires(const Model &value, mfq::cuda::Executor &value_executor,
                                       mfq::cuda::Collectives &collectives, int primary,
                                       Tensor value_ids) {
                                  embed(value, value_executor, collectives, primary,
                                        std::move(value_ids));
                              }) {
                    if (!executor)
                        throw std::invalid_argument(
                            "CUDA grid-Vision Qwen embedding requires an executor");
                    return embed(language, *executor, collectives, primary, std::move(ids))
                        .contiguous();
                } else {
                    return language.embed_forward(ids).contiguous();
                }
#endif
            },
            [](Tensor &embeddings, const Tensor &vision, const std::vector<int64_t> &indices) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                auto index = mfq::ops::tensor(indices, mfq::Device{mfq::DeviceType::cuda, 0});
                mfq::ops::index_copy(embeddings, 1, std::move(index),
                    mfq::ops::unsqueeze(mfq::ops::cast(vision, embeddings.scalar_type()), 0));
#else
                auto index =
                    mfq_tensor_backend::tensor(indices, mfq_tensor_backend::TensorOptions()
                                                            .dtype(mfq_tensor_backend::kInt64)
                                                            .device(mfq_tensor_backend::kCUDA));
                embeddings.index_copy_(1, index, vision.to(embeddings.scalar_type()).unsqueeze(0));
#endif
            },
            [](const GridMropePositions &positions) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                Tensor values = mfq::ops::tensor(positions.values, mfq::Device{mfq::DeviceType::cpu, 0});
                values = mfq::ops::reshape(std::move(values), {3, positions.token_count});
                values = mfq::ops::cast(std::move(values), mfq_tensor_backend::kInt32, mfq::Device{mfq::DeviceType::cuda, 0});
                return mfq::ops::cast(std::move(values), mfq_tensor_backend::kInt64);
#else
                return mfq_tensor_backend::tensor(positions.values,
                                                  mfq_tensor_backend::TensorOptions()
                                                      .dtype(mfq_tensor_backend::kInt32)
                                                      .device(mfq_tensor_backend::kCPU))
                    .reshape({3, positions.token_count})
                    .to(mfq_tensor_backend::kCUDA)
                    .to(mfq_tensor_backend::kInt64);
#endif
            });
        co_yield PreparedPrompt{token_ids, std::move(prepared.embeddings),
                                    std::move(prepared.positions), prepared.decode_delta,
                                    std::move(cache_key)};
    }

    const std::string &input_contract() const noexcept { return input_contract_; }
    const std::string &position_policy() const noexcept { return position_policy_; }

  private:
    CudaGridVisionPromptComponent(RoutedCache& weights, CudaGridVisionEncoder encoder, vision::LayerNorm merger_norm,
                                  BiasWeight merger_up, BiasWeight merger_down,
                                  int64_t image_token_id, int64_t video_token_id,
                                  std::string input_contract, std::string position_policy);

    RoutedCache& weights_;
    CudaGridVisionEncoder encoder_;
    vision::LayerNorm merger_norm_;
    BiasWeight merger_up_;
    BiasWeight merger_down_;
    int64_t image_token_id_ = -1;
    int64_t video_token_id_ = -1;
    std::string input_contract_;
    std::string position_policy_;
};

} // namespace mfq::cuda::grid_vision_runtime
