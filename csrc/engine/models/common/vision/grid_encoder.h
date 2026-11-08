#pragma once
#include "engine/models/common/prompt.h"

#include "engine/weights/fields.h"
#include "engine/weights/cache.h"
#include "engine/weights/workspace.h"
#include "engine/backends/cuda/ops/norm/layer.h"
#include "engine/backends/cuda/ops/projection/bias.h"
#include "engine/backends/cuda/ops/projection/projection.h"
#include "engine/backends/cuda/ops/vision/attention.h"
#include "engine/ops.h"
#include "engine/models/common/vision/grid_forward.h"
#include "engine/models/common/vision/grid_vision.h"
#include "engine/requests/media.h"
#include "engine/state/session/prefix_cache.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tyloi::weights { struct Loader; }

namespace tyloi::cuda {
class Collectives;
class Executor;
}

namespace tyloi::cuda::grid_vision_runtime {

#ifndef TYLOINFER_NATIVE_CUDA
using DenseTensor = ::tyloi::DenseTensor;
#endif

class CudaGridVisionEncoder {
  public:
    static CudaGridVisionEncoder load(const ::tyloi::weights::Loader& weights, ::tyloi::weights::Cache& cache,
                                      const GridVisionConfig &config);

#ifdef TYLOINFER_NATIVE_CUDA
    tyloi::StepSequence<std::shared_ptr<tyloi::Tensor>> encode(
                                   std::shared_ptr<tyloi::Tensor> pixels, const std::vector<GridShape> &grids) const;
#else
    tyloi::StepSequence<DenseTensor> encode(
                                   Projector projector,
                                   DenseTensor pixels, const std::vector<GridShape> &grids) const;
#endif

    const GridVisionConfig &config() const noexcept { return config_; }
    bool host_weights() const noexcept {
#ifdef TYLOINFER_NATIVE_CUDA
        return !patch_weight_->device().is_cuda();
#else
        return !patch_weight_.is_cuda();
#endif
    }

  private:
    struct Attention {
        weights::BiasWeight qkv;
        weights::BiasWeight output;
        std::int64_t hidden = 0;
        std::int64_t heads = 0;
        std::int64_t head_dim = 0;
        double theta = 10'000.0;

#ifdef TYLOINFER_NATIVE_CUDA
        std::shared_ptr<tyloi::Tensor> operator()(const tyloi::Tensor& input,
#else
        DenseTensor operator()(
            const Projector& projector,
            const DenseTensor& input,
#endif
            const std::vector<std::int32_t>& positions,
            const std::vector<std::int32_t>& segment_lengths) const {
            const std::int64_t tokens = input.size(0);
#ifdef TYLOINFER_NATIVE_CUDA
            auto projected = tyloi::ops::reshape(*tyloi::ops::affine(input, *qkv.weight, qkv.bias.get()), {tokens, 3, heads, head_dim});
            auto query = tyloi::ops::axial_rope(*tyloi::ops::select(*projected, 1, 0), positions, theta);
            auto key = tyloi::ops::axial_rope(*tyloi::ops::select(*projected, 1, 1), positions, theta);
            auto value = tyloi::ops::select(*projected, 1, 2);
            auto attended = tyloi::ops::segmented_attention(*query, *key, *value, segment_lengths);
            return tyloi::ops::affine(*tyloi::ops::reshape(*attended, {tokens, hidden}), *output.weight, output.bias.get());
#else
            auto projected = tyloi::cuda::affine(projector, qkv, input).reshape({tokens, 3, heads, head_dim});
            auto query = vision::apply_axial_rope(projected.select(1, 0), positions, theta);
            auto key = vision::apply_axial_rope(projected.select(1, 1), positions, theta);
            auto value = projected.select(1, 2);
            auto attended = vision::segmented_attention(query, key, value, segment_lengths);
            return tyloi::cuda::affine(projector, output, attended.reshape({tokens, hidden}));
#endif
        }
    };

    struct Layer {
        vision::LayerNorm norm1;
        Attention attention;
        vision::LayerNorm norm2;
        weights::BiasWeight mlp_up, mlp_down;

        template <class Visitor>
        void visit_tensors(Visitor visit) const {
            fields::tensors(norm1, visit);
            fields::tensors(norm2, visit);
            fields::tensors(attention.qkv, visit);
            fields::tensors(attention.output, visit);
            fields::tensors(mlp_up, visit);
            fields::tensors(mlp_down, visit);
        }

        void to_device(::tyloi::Device device) {
            norm1.to_device(device);
            norm2.to_device(device);
            fields::to_device(attention.qkv, device);
            fields::to_device(attention.output, device);
            fields::to_device(mlp_up, device);
            fields::to_device(mlp_down, device);
        }

#ifdef TYLOINFER_NATIVE_CUDA
        std::shared_ptr<tyloi::Tensor> operator()(std::shared_ptr<tyloi::Tensor> input,
#else
        DenseTensor operator()(
                          const Projector& projector,
                          const DenseTensor& input,
#endif
                          const std::vector<std::int32_t>& positions,
                          const std::vector<std::int32_t>& segment_lengths) const {
#ifdef TYLOINFER_NATIVE_CUDA
            auto branch = attention(norm1(input), positions, segment_lengths);
            input = tyloi::ops::add(input, tyloi::ops::cast(branch, input.scalar_type()));
            branch = tyloi::ops::affine(tyloi::ops::gelu(
                tyloi::ops::affine(norm2(input), mlp_up.weight, mlp_up.bias), true),
                mlp_down.weight, mlp_down.bias);
            return tyloi::ops::add(input, tyloi::ops::cast(branch, input.scalar_type()));
#else
            auto hidden = input;
            auto branch = attention(projector, norm1(hidden), positions, segment_lengths);
            hidden = hidden + branch.to(hidden.scalar_type());
            branch = tyloi::cuda::affine(projector, mlp_down, ::tyloi::cuda::gelu(
                tyloi::cuda::affine(projector, mlp_up, norm2(hidden)), "tanh"));
            return hidden + branch.to(hidden.scalar_type());
#endif
        }
    };

#ifdef TYLOINFER_NATIVE_CUDA
    CudaGridVisionEncoder(::tyloi::weights::Cache& weights, GridVisionConfig config,
                          std::shared_ptr<const tyloi::Tensor> patch_weight, std::shared_ptr<const tyloi::Tensor> patch_bias,
                          std::shared_ptr<const tyloi::Tensor> position_weight, std::vector<Layer> blocks);
#else
    CudaGridVisionEncoder(::tyloi::weights::Cache& weights, GridVisionConfig config, DenseTensor patch_weight, DenseTensor patch_bias,
                          DenseTensor position_weight, std::vector<Layer> blocks);
#endif

    ::tyloi::weights::Cache& weights_;
    GridVisionConfig config_;
#ifdef TYLOINFER_NATIVE_CUDA
    std::shared_ptr<const tyloi::Tensor> patch_weight_, patch_bias_, position_weight_;
#else
    DenseTensor patch_weight_, patch_bias_, position_weight_;
#endif
    std::vector<Layer> blocks_;
};

class CudaGridVisionPromptComponent {
  public:
    static CudaGridVisionPromptComponent
    load(const ::tyloi::weights::Loader& weights, ::tyloi::weights::Cache& cache,
         const GridVisionConfig &vision, int64_t image_token_id, int64_t video_token_id,
         std::string input_contract, std::string position_policy);

    template <typename Model>
    tyloi::StepSequence<PreparedPrompt> prepare(
        const Model &language, MediaCache &images,
#ifndef TYLOINFER_NATIVE_CUDA
        Projector projector, Collectives &collectives,
#endif
        int primary,
#ifndef TYLOINFER_NATIVE_CUDA
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
            tyloi::cache::block_hash_hex(tyloi::cache::sha256(
                media.pixel_values.data(), media.pixel_values.size() * sizeof(float)));
        for (const auto value : media.pixel_shape) {
            cache_key.push_back('|');
            cache_key += std::to_string(value);
        }
        for (const auto value : media.vision_grid) {
            cache_key.push_back('|');
            cache_key += std::to_string(value);
        }

#ifdef TYLOINFER_NATIVE_CUDA
        auto merged = images.find(cache_key);
        if (!merged) {
#else
        DenseTensor merged = images.find(cache_key);
        if (!merged.defined()) {
#endif
#ifdef TYLOINFER_NATIVE_CUDA
            auto pixels = tyloi::ops::from_blob(
                const_cast<float *>(media.pixel_values.data()), media.pixel_shape,
                tyloi::TensorOptions().dtype(tyloi::ScalarType::float32).device(tyloi::DeviceType::cpu));
            pixels = tyloi::ops::clone(*pixels);
            pixels = tyloi::ops::cast(*pixels, tyloi::ScalarType::float32, tyloi::Device{tyloi::kCUDA, primary});
#else
            auto pixels = ::tyloi::cuda::from_blob(
                              const_cast<float *>(media.pixel_values.data()), media.pixel_shape,
                              ::tyloi::TensorOptions()
                                  .dtype(::tyloi::kFloat32)
                                  .device(::tyloi::kCPU))
                              .clone()
                              .to(::tyloi::kCUDA);
#endif
            co_yield tyloi::StepState::advanced;
            auto encoding = encoder_.encode(
#ifndef TYLOINFER_NATIVE_CUDA
                projector,
#endif
                std::move(pixels), {grid});
#ifdef TYLOINFER_NATIVE_CUDA
            std::shared_ptr<tyloi::Tensor> patches;
#else
            DenseTensor patches;
#endif
            while (auto step = encoding.next()) {
                if (step.value)
                    patches = std::move(*step.value);
                else
                    co_yield step.state;
            }
            const auto merge = [&](const auto &norm, const auto &up, const auto &down) {
                return tyloi::models::grid_vision::merge(
#ifdef TYLOINFER_NATIVE_CUDA
                    patches, patches->size(0), encoder_.config(),
                    [&](std::shared_ptr<tyloi::Tensor> value) { return norm(*value); },
                    [](std::shared_ptr<tyloi::Tensor> value, int64_t rows, int64_t width) {
                        return tyloi::ops::reshape(*value, {rows, width});
                    },
                    [&](std::shared_ptr<tyloi::Tensor> value) { return tyloi::ops::affine(*value, *up.weight, up.bias.get()); },
                    [](std::shared_ptr<tyloi::Tensor> value) { return tyloi::ops::gelu(*value); },
                    [&](std::shared_ptr<tyloi::Tensor> value) { return tyloi::ops::affine(*value, *down.weight, down.bias.get()); });
#else
                    patches, patches.size(0), encoder_.config(),
                    [&](DenseTensor value) { return norm(value); },
                    [](DenseTensor value, int64_t rows, int64_t width) {
                        return value.reshape({rows, width});
                    },
                    [&](DenseTensor value) { return tyloi::cuda::affine(projector, up, std::move(value)); },
                    [](DenseTensor value) { return ::tyloi::cuda::gelu(value, "none"); },
                    [&](DenseTensor value) { return tyloi::cuda::affine(projector, down, std::move(value)); });
#endif
            };
            if (encoder_.host_weights()) {
                const auto bytes = fields::bytes(merger_norm_) +
                                   fields::bytes(merger_up_) +
                                   fields::bytes(merger_down_);
#ifdef TYLOINFER_NATIVE_CUDA
                tyloi::weights::Loan loan(weights_, bytes + patches->nbytes() * 12 + (std::size_t(16) << 20));
                std::shared_ptr<tyloi::Tensor> output;
#else
                tyloi::weights::Loan loan(weights_, bytes + patches.nbytes() * 12 + (std::size_t(16) << 20));
                DenseTensor output;
#endif
                {
                    // Stage copies of the merger weights on the primary device.
                    auto norm = merger_norm_;
                    auto up = merger_up_;
                    auto down = merger_down_;
                    const auto device = tyloi::Device{tyloi::kCUDA, primary};
                    fields::to_device(norm, device);
                    fields::to_device(up, device);
                    fields::to_device(down, device);
                    output = merge(norm, up, down);
                }
#ifdef TYLOINFER_NATIVE_CUDA
                merged = loan.finish(std::move(output));
#else
                merged = dynamic_cast<const DenseTensor&>(*loan.finish(std::make_shared<DenseTensor>(std::move(output))));
#endif
            } else {
                merged = merge(merger_norm_, merger_up_, merger_down_);
            }
            images.store(cache_key, merged);
            co_yield tyloi::StepState::advanced;
        }

        auto prepared = tyloi::models::grid_vision::prepare(
#ifdef TYLOINFER_NATIVE_CUDA
            token_ids, merged, merged->size(0), image_token_id_, video_token_id_, encoder_.config(),
#else
            token_ids, merged, merged.size(0), image_token_id_, video_token_id_, encoder_.config(),
#endif
            {grid},
            [&](const auto &tokens) {
#ifdef TYLOINFER_NATIVE_CUDA
                auto ids = tyloi::ops::tensor(tokens, tyloi::Device{tyloi::kCUDA, primary});
                auto shaped = tyloi::ops::reshape(*ids, {1, -1});
                return tyloi::ops::contiguous(*language.embedding(*shaped, primary));
#else
                auto ids =
                    ::tyloi::cuda::tensor(tokens, ::tyloi::TensorOptions()
                                                           .dtype(::tyloi::kInt64)
                                                           .device(::tyloi::kCUDA));
                ids = ids.reshape({1, -1});
                if constexpr (requires(const Model &value, tyloi::cuda::Executor &value_executor,
                                       tyloi::cuda::Collectives &collectives, int primary,
                                       DenseTensor value_ids) {
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
#ifdef TYLOINFER_NATIVE_CUDA
            [](std::shared_ptr<tyloi::Tensor>& embeddings, const std::shared_ptr<tyloi::Tensor>& vision, const std::vector<int64_t> &indices) {
                auto index = tyloi::ops::tensor(indices, embeddings->device());
                tyloi::ops::index_copy(*embeddings, 1, *index,
                    *tyloi::ops::unsqueeze(*tyloi::ops::cast(*vision, embeddings->scalar_type()), 0));
#else
            [](DenseTensor &embeddings, const DenseTensor &vision, const std::vector<int64_t> &indices) {
                auto index =
                    ::tyloi::cuda::tensor(indices, ::tyloi::TensorOptions()
                                                            .dtype(::tyloi::kInt64)
                                                            .device(::tyloi::kCUDA));
                embeddings.index_copy_(1, index, vision.to(embeddings.scalar_type()).unsqueeze(0));
#endif
            },
            [primary](const GridMropePositions &positions) {
#ifdef TYLOINFER_NATIVE_CUDA
                auto values = tyloi::ops::tensor(positions.values, tyloi::Device{tyloi::DeviceType::cpu, 0});
                values = tyloi::ops::reshape(*values, {3, positions.token_count});
                values = tyloi::ops::cast(*values, ::tyloi::kInt32, tyloi::Device{tyloi::kCUDA, primary});
                return tyloi::ops::cast(*values, ::tyloi::kInt64);
#else
                return ::tyloi::cuda::tensor(positions.values,
                                                  ::tyloi::TensorOptions()
                                                      .dtype(::tyloi::kInt32)
                                                      .device(::tyloi::kCPU))
                    .reshape({3, positions.token_count})
                    .to(::tyloi::kCUDA)
                    .to(::tyloi::kInt64);
#endif
            });
        co_yield PreparedPrompt{token_ids, std::move(prepared.embeddings),
                                    std::move(prepared.positions), prepared.decode_delta,
                                    std::move(cache_key)};
    }

    const std::string &input_contract() const noexcept { return input_contract_; }
    const std::string &position_policy() const noexcept { return position_policy_; }

  private:
    CudaGridVisionPromptComponent(::tyloi::weights::Cache& weights, CudaGridVisionEncoder encoder, vision::LayerNorm merger_norm,
                                  tyloi::weights::BiasWeight merger_up, tyloi::weights::BiasWeight merger_down,
                                  int64_t image_token_id, int64_t video_token_id,
                                  std::string input_contract, std::string position_policy);

    ::tyloi::weights::Cache& weights_;
    CudaGridVisionEncoder encoder_;
    vision::LayerNorm merger_norm_;
    weights::BiasWeight merger_up_;
    weights::BiasWeight merger_down_;
    int64_t image_token_id_ = -1;
    int64_t video_token_id_ = -1;
    std::string input_contract_;
    std::string position_policy_;
};

} // namespace tyloi::cuda::grid_vision_runtime
