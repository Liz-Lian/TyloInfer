#pragma once

#include "backends/cuda/engine/executor/vision.h"
#include "backends/cuda/engine/weights/vision.h"
#include "ops/vision/vision.h"
#include "engine/models/common/grid_vision_model.h"
#include "engine/models/common/vision/grid_vision.h"
#include "engine/requests/media.h"
#include "engine/state/prefix_cache.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfq { class ModelSource; }

namespace mfq::cuda::grid_vision_runtime {

using Tensor = mfq_tensor_backend::Tensor;

namespace detail {

using vision::Affine;
using vision::VisionAttention;
using vision::gelu_tanh;
using vision::to_cuda;
using vision::vision_workspace;
using vision::weight_bytes;

struct LayerNorm {
    Tensor weight;
    Tensor bias;
    double epsilon = 1e-6;

    template <class Visitor>
    void visit_tensors(Visitor visit) {
        visit(weight);
        visit(bias);
    }

    static LayerNorm load(CudaExecutionContext &execution, const mfq::ModelSource &model,
                          GridVisionTensorRole weight, GridVisionTensorRole bias, int64_t width,
                          double epsilon, size_t block = 0);

    Tensor operator()(const Tensor &input) const;
};

struct VisionBlock {
    LayerNorm norm1;
    VisionAttention attention;
    LayerNorm norm2;
    Affine mlp_up;
    Affine mlp_down;

    template <class Visitor>
    void visit_tensors(Visitor visit) {
        norm1.visit_tensors(visit);
        norm2.visit_tensors(visit);
        vision::visit_tensors(attention.qkv, visit);
        vision::visit_tensors(attention.output, visit);
        vision::visit_tensors(mlp_up, visit);
        vision::visit_tensors(mlp_down, visit);
    }

    Tensor operator()(CudaExecutionContext &execution, const Tensor &input,
                      const GridVisionLayout &layout) const;
};

} // namespace detail

using detail::Affine;

class CudaGridVisionEncoder {
  public:
    static CudaGridVisionEncoder load(CudaExecutionContext &execution,
                                      const mfq::ModelSource &model,
                                      const GridVisionConfig &config);

    mfq::StepSequence<Tensor> encode(CudaExecutionContext &execution, Tensor pixels,
                                      const std::vector<GridShape> &grids) const;

    const GridVisionConfig &config() const noexcept { return config_; }
    bool host_weights() const noexcept { return !patch_weight_.is_cuda(); }

  private:
    CudaGridVisionEncoder(GridVisionConfig config, Tensor patch_weight, Tensor patch_bias,
                          Tensor position_weight, std::vector<detail::VisionBlock> blocks);

    GridVisionConfig config_;
    Tensor patch_weight_;
    Tensor patch_bias_;
    Tensor position_weight_;
    std::vector<detail::VisionBlock> blocks_;
};

class CudaGridVisionPromptComponent {
  public:
    static CudaGridVisionPromptComponent
    load(CudaExecutionContext &execution, const mfq::ModelSource &model,
         const GridVisionConfig &vision, int64_t image_token_id, int64_t video_token_id,
         std::string input_contract, std::string position_policy);

    template <typename Model>
    mfq::StepSequence<CudaPreparedPrompt> prepare(Model &language,
                                                   const std::vector<int64_t> &token_ids,
                                                   const MfqMultimodalInput &media) const {
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

        Tensor merged;
        if (cached_vision_key_ == cache_key && cached_vision_.defined()) {
            merged = cached_vision_;
        } else {
            auto pixels = mfq_tensor_backend::from_blob(
                              const_cast<float *>(media.pixel_values.data()), media.pixel_shape,
                              mfq_tensor_backend::TensorOptions()
                                  .dtype(mfq_tensor_backend::kFloat32)
                                  .device(mfq_tensor_backend::kCPU))
                              .clone()
                              .to(mfq_tensor_backend::kCUDA);
            co_yield mfq::StepState::advanced;
            auto encoding = encoder_.encode(*language.execution, std::move(pixels), {grid});
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
                        return value.reshape({rows, width});
                    },
                    [&](Tensor value) { return up(*language.execution, std::move(value)); },
                    [](Tensor value) { return mfq_tensor_backend::gelu(value, "none"); },
                    [&](Tensor value) { return down(*language.execution, std::move(value)); });
            };
            if (encoder_.host_weights()) {
                const auto bytes = detail::weight_bytes(merger_norm_) +
                                   detail::weight_bytes(merger_up_) +
                                   detail::weight_bytes(merger_down_);
                merged = detail::vision_workspace(
                    *language.execution, bytes + patches.nbytes() * 12 + (std::size_t(16) << 20),
                    [&] {
                        auto norm = detail::to_cuda(merger_norm_);
                        auto up = detail::to_cuda(merger_up_);
                        auto down = detail::to_cuda(merger_down_);
                        return merge(norm, up, down);
                    });
            } else {
                merged = merge(merger_norm_, merger_up_, merger_down_);
            }
            // ponytail: one image is the current native contract; use an LRU
            // only when alternating concurrent images becomes measurable.
            cached_vision_key_ = cache_key;
            cached_vision_ = merged;
            co_yield mfq::StepState::advanced;
        }

        auto prepared = mfq::models::grid_vision::prepare(
            token_ids, merged, merged.size(0), image_token_id_, video_token_id_, encoder_.config(),
            {grid},
            [&](const auto &tokens) {
                auto ids =
                    mfq_tensor_backend::tensor(tokens, mfq_tensor_backend::TensorOptions()
                                                           .dtype(mfq_tensor_backend::kInt64)
                                                           .device(mfq_tensor_backend::kCUDA))
                        .reshape({1, -1});
                return language.embed_forward(ids).contiguous();
            },
            [](Tensor &embeddings, const Tensor &vision, const std::vector<int64_t> &indices) {
                auto index =
                    mfq_tensor_backend::tensor(indices, mfq_tensor_backend::TensorOptions()
                                                            .dtype(mfq_tensor_backend::kInt64)
                                                            .device(mfq_tensor_backend::kCUDA));
                embeddings.index_copy_(1, index, vision.to(embeddings.scalar_type()).unsqueeze(0));
            },
            [](const GridMropePositions &positions) {
                return mfq_tensor_backend::tensor(positions.values,
                                                  mfq_tensor_backend::TensorOptions()
                                                      .dtype(mfq_tensor_backend::kInt32)
                                                      .device(mfq_tensor_backend::kCPU))
                    .reshape({3, positions.token_count})
                    .to(mfq_tensor_backend::kCUDA)
                    .to(mfq_tensor_backend::kInt64);
            });
        co_yield CudaPreparedPrompt{token_ids, std::move(prepared.embeddings),
                                    std::move(prepared.positions), prepared.decode_delta,
                                    std::move(cache_key)};
    }

    const std::string &input_contract() const noexcept { return input_contract_; }
    const std::string &position_policy() const noexcept { return position_policy_; }

  private:
    CudaGridVisionPromptComponent(CudaGridVisionEncoder encoder, detail::LayerNorm merger_norm,
                                  detail::Affine merger_up, detail::Affine merger_down,
                                  int64_t image_token_id, int64_t video_token_id,
                                  std::string input_contract, std::string position_policy);

    CudaGridVisionEncoder encoder_;
    detail::LayerNorm merger_norm_;
    detail::Affine merger_up_;
    detail::Affine merger_down_;
    int64_t image_token_id_ = -1;
    int64_t video_token_id_ = -1;
    std::string input_contract_;
    std::string position_policy_;
    mutable std::string cached_vision_key_;
    mutable Tensor cached_vision_;
};

} // namespace mfq::cuda::grid_vision_runtime
