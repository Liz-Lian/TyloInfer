#pragma once
#include "engine/models/common/prompt.h"

#include "engine/weights/fields.h"
#include "engine/weights/workspace.h"
#include "engine/backends/cuda/ops/projection/bias.h"
#include "engine/backends/cuda/ops/vision/attention.h"
#include "engine/models/common/gated_mlp.h"
#include "engine/models/common/transformer_layer.h"
#include "engine/models/common/grid_vision_model.h"
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

using Tensor = mfq_tensor_backend::Tensor;

class CudaGridVisionEncoder {
  public:
    static CudaGridVisionEncoder load(const weight_loader::Loader& weights,
                                      const GridVisionConfig &config);

    mfq::StepSequence<Tensor> encode(Projector projector, std::shared_ptr<MoeExpertCache> cache,
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
            const Projector& projector,
            const Tensor& input,
            const std::vector<std::int32_t>& positions,
            const std::vector<std::int32_t>& segment_lengths) const {
            const std::int64_t tokens = input.size(0);
            auto projected = mfq::cuda::affine(projector, qkv, input).reshape({tokens, 3, heads, head_dim});
            auto query = vision::apply_axial_rope(projected.select(1, 0), positions, theta);
            auto key = vision::apply_axial_rope(projected.select(1, 1), positions, theta);
            auto value = projected.select(1, 2);
            auto attended = vision::segmented_attention(query, key, value, segment_lengths);
            return mfq::cuda::affine(projector, output, attended.reshape({tokens, hidden}));
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

        Tensor operator()(const Projector& projector, const Tensor& input,
                          const std::vector<std::int32_t>& positions,
                          const std::vector<std::int32_t>& segment_lengths) const {
            return mfq::models::pre_norm_layer(
                input, [&](const Tensor& x, int stage) { return stage == 0 ? norm1(x) : norm2(x); },
                [&](Tensor x) { return attention(projector, x, positions, segment_lengths); },
                [&](Tensor x) {
                    return mfq::models::mlp(
                        std::move(x), [&](Tensor value) { return mfq::cuda::affine(projector, mlp_up, std::move(value)); },
                        [](Tensor value) { return mfq_tensor_backend::gelu(value, "tanh"); },
                        [&](Tensor value) { return mfq::cuda::affine(projector, mlp_down, std::move(value)); });
                },
                [](Tensor residual, Tensor branch) {
                    return residual + branch.to(residual.scalar_type());
                });
        }
    };

    CudaGridVisionEncoder(GridVisionConfig config, Tensor patch_weight, Tensor patch_bias,
                          Tensor position_weight, std::vector<Layer> blocks);

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
        const Model &language, Projector projector, std::shared_ptr<MoeExpertCache> cache,
        Collectives &collectives, int primary, Executor *executor,
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
            auto encoding = encoder_.encode(projector, cache, std::move(pixels), {grid});
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
                    [&](Tensor value) { return mfq::cuda::affine(projector, up, std::move(value)); },
                    [](Tensor value) { return mfq_tensor_backend::gelu(value, "none"); },
                    [&](Tensor value) { return mfq::cuda::affine(projector, down, std::move(value)); });
            };
            if (encoder_.host_weights()) {
                const auto bytes = fields::bytes(merger_norm_) +
                                   fields::bytes(merger_up_) +
                                   fields::bytes(merger_down_);
                MoePrefillWorkspaceLoan loan(cache);
                loan.begin(bytes + patches.nbytes() * 12 + (std::size_t(16) << 20));
                Tensor output;
                {
                    auto norm = fields::to_cuda(merger_norm_);
                    auto up = fields::to_cuda(merger_up_);
                    auto down = fields::to_cuda(merger_down_);
                    output = merge(norm, up, down);
                }
                merged = loan.finish(std::move(output));
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
        co_yield PreparedPrompt{token_ids, std::move(prepared.embeddings),
                                    std::move(prepared.positions), prepared.decode_delta,
                                    std::move(cache_key)};
    }

    const std::string &input_contract() const noexcept { return input_contract_; }
    const std::string &position_policy() const noexcept { return position_policy_; }

  private:
    CudaGridVisionPromptComponent(CudaGridVisionEncoder encoder, vision::LayerNorm merger_norm,
                                  BiasWeight merger_up, BiasWeight merger_down,
                                  int64_t image_token_id, int64_t video_token_id,
                                  std::string input_contract, std::string position_policy);

    CudaGridVisionEncoder encoder_;
    vision::LayerNorm merger_norm_;
    BiasWeight merger_up_;
    BiasWeight merger_down_;
    int64_t image_token_id_ = -1;
    int64_t video_token_id_ = -1;
    std::string input_contract_;
    std::string position_policy_;
    mutable std::string cached_vision_key_;
    mutable Tensor cached_vision_;
};

} // namespace mfq::cuda::grid_vision_runtime
