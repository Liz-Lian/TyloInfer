#pragma once
#include "engine/models/common/prompt.h"

#include "engine/weights/fields.h"
#include "engine/backends/cuda/ops/norm/layer.h"
#include "engine/backends/cuda/ops/projection/bias.h"
#include "engine/backends/cuda/ops/projection/projection.h"
#include "engine/backends/cuda/ops/vision/attention.h"
#include "engine/ops.h"
#include "engine/models/common/vision/grid_forward.h"
#include "engine/models/common/vision/grid_vision.h"
#include "engine/requests/media.h"
#include "engine/state/session/prefix_cache.h"
#include "engine/state/media.h"

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


class CudaGridVisionEncoder {
  public:
    /** Load the grid encoder weights for the selected placement. */
    static CudaGridVisionEncoder load(const ::tyloi::weights::Loader& weights,
                                      const GridVisionConfig &config);

    tyloi::StepSequence<::tyloi::Tensor> encode(
                                   ::tyloi::Tensor pixels, std::vector<GridShape> grids) const;

    const GridVisionConfig &config() const noexcept { return config_; }
    bool host_weights() const noexcept {
        return !patch_weight_.device().is_cuda();
    }

  private:
    struct Attention {
        weights::BiasWeight qkv;
        weights::BiasWeight output;
        std::int64_t hidden = 0;
        std::int64_t heads = 0;
        std::int64_t head_dim = 0;
        double theta = 10'000.0;

        ::tyloi::Tensor operator()(const tyloi::Tensor& input,
            const std::vector<std::int32_t>& positions,
            const std::vector<std::int32_t>& segment_lengths) const {
            const std::int64_t tokens = input.size(0);
            auto projected = tyloi::ops::reshape(tyloi::ops::affine(input, qkv.weight, qkv.bias), {tokens, 3, heads, head_dim});
            auto query = tyloi::ops::axial_rope(tyloi::ops::select(projected, 1, 0), positions, theta);
            auto key = tyloi::ops::axial_rope(tyloi::ops::select(projected, 1, 1), positions, theta);
            auto value = tyloi::ops::select(projected, 1, 2);
            auto attended = tyloi::ops::segmented_attention(query, key, value, segment_lengths);
            return tyloi::ops::affine(tyloi::ops::reshape(attended, {tokens, hidden}), output.weight, output.bias);
        }
    };

    struct Layer {
        vision::LayerNorm norm1;
        Attention attention;
        vision::LayerNorm norm2;
        weights::BiasWeight mlp_up, mlp_down;

        template <class Visitor>
        void visit_tensors(Visitor visit) const {
            ::tyloi::weights::fields::tensors(norm1, visit);
            ::tyloi::weights::fields::tensors(norm2, visit);
            ::tyloi::weights::fields::tensors(attention.qkv, visit);
            ::tyloi::weights::fields::tensors(attention.output, visit);
            ::tyloi::weights::fields::tensors(mlp_up, visit);
            ::tyloi::weights::fields::tensors(mlp_down, visit);
        }

        void to_device(::tyloi::Device device) {
            norm1.to_device(device);
            norm2.to_device(device);
            ::tyloi::weights::fields::to_device(attention.qkv, device);
            ::tyloi::weights::fields::to_device(attention.output, device);
            ::tyloi::weights::fields::to_device(mlp_up, device);
            ::tyloi::weights::fields::to_device(mlp_down, device);
        }

        ::tyloi::Tensor operator()(::tyloi::Tensor input,
                          const std::vector<std::int32_t>& positions,
                          const std::vector<std::int32_t>& segment_lengths) const {
            auto branch = attention(norm1(input), positions, segment_lengths);
            input = tyloi::ops::add(input, tyloi::ops::cast(branch, input.scalar_type()));
            branch = tyloi::ops::affine(tyloi::ops::gelu(
                tyloi::ops::affine(norm2(input), mlp_up.weight, mlp_up.bias), true),
                mlp_down.weight, mlp_down.bias);
            return tyloi::ops::add(input, tyloi::ops::cast(branch, input.scalar_type()));
        }
    };

    CudaGridVisionEncoder(GridVisionConfig config,
                          ::tyloi::Tensor patch_weight, ::tyloi::Tensor patch_bias,
                          ::tyloi::Tensor position_weight, std::vector<Layer> blocks);

    GridVisionConfig config_;
    ::tyloi::Tensor patch_weight_, patch_bias_, position_weight_;
    std::vector<Layer> blocks_;
};

class CudaGridVisionPromptComponent {
  public:
    static CudaGridVisionPromptComponent
    load(const ::tyloi::weights::Loader& weights,
         const GridVisionConfig &vision, int64_t image_token_id, int64_t video_token_id,
         std::string input_contract, std::string position_policy);

    template <typename Model>
    tyloi::StepSequence<tyloi::models::Prepared> prepare(
        const Model &language, tyloi::cache::Media &images,
        int primary,
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

        auto merged = images.find(cache_key);
        if (!merged) {
            auto pixels = tyloi::ops::from_blob(
                const_cast<float *>(media.pixel_values.data()), media.pixel_shape,
                tyloi::TensorOptions().dtype(tyloi::ScalarType::float32).device(tyloi::DeviceType::cpu));
            pixels = tyloi::ops::clone(pixels);
            pixels = tyloi::ops::cast(pixels, tyloi::ScalarType::float32, tyloi::Device{tyloi::kCUDA, primary});
            co_yield tyloi::StepState::advanced;
            auto encoding = encoder_.encode(
                std::move(pixels), {grid});
            ::tyloi::Tensor patches;
            while (auto step = encoding.next()) {
                if (step.layer || step.graph) throw std::logic_error("vision encoding does not support layer or graph events");
                if (step.workspace) {
                    co_yield *step.workspace;
                    continue;
                }
                if (step.value)
                    patches = std::move(*step.value);
                else
                    co_yield step.state;
            }
            const auto merge = [&](const auto &norm, const auto &up, const auto &down) {
                return tyloi::models::grid_vision::merge(
                    patches, patches.size(0), encoder_.config(),
                    [&](::tyloi::Tensor value) { return norm(value); },
                    [](::tyloi::Tensor value, int64_t rows, int64_t width) {
                        return tyloi::ops::reshape(value, {rows, width});
                    },
                    [&](::tyloi::Tensor value) { return tyloi::ops::affine(value, up.weight, up.bias); },
                    [](::tyloi::Tensor value) { return tyloi::ops::gelu(value); },
                    [&](::tyloi::Tensor value) { return tyloi::ops::affine(value, down.weight, down.bias); });
            };
            if (encoder_.host_weights()) {
                const auto bytes = ::tyloi::weights::fields::bytes(merger_norm_) +
                                   ::tyloi::weights::fields::bytes(merger_up_) +
                                   ::tyloi::weights::fields::bytes(merger_down_);
                co_yield tyloi::Workspace{bytes + patches.nbytes() * 12 + (std::size_t(16) << 20)};
                ::tyloi::Tensor output;
                {
                    // Stage copies of the merger weights on the primary device.
                    auto norm = merger_norm_;
                    auto up = merger_up_;
                    auto down = merger_down_;
                    const auto device = tyloi::Device{tyloi::kCUDA, primary};
                    ::tyloi::weights::fields::to_device(norm, device);
                    ::tyloi::weights::fields::to_device(up, device);
                    ::tyloi::weights::fields::to_device(down, device);
                    output = merge(norm, up, down);
                }
                co_yield tyloi::Workspace{0, &output};
                merged = std::move(output);
            } else {
                merged = merge(merger_norm_, merger_up_, merger_down_);
            }
            images.store(cache_key, merged);
            co_yield tyloi::StepState::advanced;
        }

        auto prepared = tyloi::models::grid_vision::prepare(
            token_ids, merged, merged.size(0), image_token_id_, video_token_id_, encoder_.config(),
            {grid},
            [&](const auto &tokens) {
                auto ids = tyloi::ops::tensor(tokens, tyloi::Device{tyloi::kCUDA, primary});
                auto shaped = tyloi::ops::reshape(ids, {1, -1});
                return tyloi::ops::contiguous(language.embedding(shaped, primary));
            },
            [](::tyloi::Tensor& embeddings, const ::tyloi::Tensor& vision, const std::vector<int64_t> &indices) {
                auto index = tyloi::ops::tensor(indices, embeddings.device());
                tyloi::ops::index_copy(embeddings, 1, index,
                    tyloi::ops::unsqueeze(tyloi::ops::cast(vision, embeddings.scalar_type()), 0));
            },
            [primary](const GridMropePositions &positions) {
                auto values = tyloi::ops::tensor(positions.values, tyloi::Device{tyloi::DeviceType::cpu, 0});
                values = tyloi::ops::reshape(values, {3, positions.token_count});
                values = tyloi::ops::cast(values, ::tyloi::kInt32, tyloi::Device{tyloi::kCUDA, primary});
                return tyloi::ops::cast(values, ::tyloi::kInt64);
            });
        co_yield tyloi::models::Prepared{token_ids, std::move(prepared.embeddings),
                                    std::move(prepared.positions), prepared.decode_delta,
                                    std::move(cache_key)};
    }

    const std::string &input_contract() const noexcept { return input_contract_; }
    const std::string &position_policy() const noexcept { return position_policy_; }

  private:
    CudaGridVisionPromptComponent(
                                  CudaGridVisionEncoder encoder, vision::LayerNorm merger_norm,
                                  tyloi::weights::BiasWeight merger_up, tyloi::weights::BiasWeight merger_down,
                                  int64_t image_token_id, int64_t video_token_id,
                                  std::string input_contract, std::string position_policy);

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
