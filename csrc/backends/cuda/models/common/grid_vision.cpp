#include "models/common/grid_vision.h"

#include "backends/cuda/engine/models/vision.h"
#include "backends/cuda/engine/executor/options.h"
#include "backends/cuda/engine/executor/execution.h"
#include "backends/cuda/engine/weights/load.h"

#include <algorithm>
#include <stdexcept>

namespace mfq::cuda::grid_vision_runtime {
namespace {

Tensor required_vector(CudaExecutionContext &execution, const mfq::ModelSource &model,
                       GridVisionTensorRole role, int64_t width, size_t block = 0) {
    const auto name = grid_vision_canonical_name(role, block);
    auto value = load_dense_gpu(execution, model, name);
    if (value.dim() != 1 || value.size(0) != width) {
        throw std::runtime_error("grid-ViT vector shape mismatch: " + name);
    }
    return value;
}

vision::LayerNorm load_norm(CudaExecutionContext &execution, const mfq::ModelSource &model,
                          GridVisionTensorRole weight, GridVisionTensorRole bias, int64_t width,
                          double epsilon, size_t block = 0) {
    return {
        required_vector(execution, model, weight, width, block),
        required_vector(execution, model, bias, width, block),
        epsilon,
    };
}

} // namespace

CudaGridVisionEncoder CudaGridVisionEncoder::load(CudaExecutionContext &execution,
                                                  const mfq::ModelSource &model,
                                                  const GridVisionConfig &config) {
    config.validate();
    const auto hidden = config.hidden_size;
    auto patch_weight = load_dense_gpu(
        execution, model, grid_vision_canonical_name(GridVisionTensorRole::patch_weight));
    const std::vector<int64_t> expected_patch{hidden, config.in_channels,
                                              config.temporal_patch_size, config.patch_size,
                                              config.patch_size};
    if (patch_weight.sizes().vec() != expected_patch) {
        throw std::runtime_error("grid-ViT patch weight shape mismatch");
    }
    auto patch_bias =
        required_vector(execution, model, GridVisionTensorRole::patch_bias, hidden);
    auto position_weight = load_dense_gpu(
        execution, model, grid_vision_canonical_name(GridVisionTensorRole::position_weight));
    if (position_weight.dim() != 2 ||
        position_weight.size(0) != config.num_position_embeddings ||
        position_weight.size(1) != hidden) {
        throw std::runtime_error("grid-ViT learned position table shape mismatch");
    }
    std::vector<vision::Layer> blocks;
    blocks.reserve(static_cast<size_t>(config.depth));
    for (size_t index = 0; index < static_cast<size_t>(config.depth); ++index) {
        vision::Layer block{
            load_norm(execution, model, GridVisionTensorRole::block_norm1_weight,
                                    GridVisionTensorRole::block_norm1_bias, hidden,
                                    config.layer_norm_eps, index),
            {
                vision::load_affine(execution, model,
                             grid_vision_canonical_name(
                                 GridVisionTensorRole::block_attention_qkv_weight, index),
                             grid_vision_canonical_name(
                                 GridVisionTensorRole::block_attention_qkv_bias, index)),
                vision::load_affine(execution, model,
                             grid_vision_canonical_name(
                                 GridVisionTensorRole::block_attention_output_weight, index),
                             grid_vision_canonical_name(
                                 GridVisionTensorRole::block_attention_output_bias, index)),
                hidden,
                config.num_heads,
                config.head_dim(),
                config.rope_theta,
            },
            load_norm(execution, model, GridVisionTensorRole::block_norm2_weight,
                                    GridVisionTensorRole::block_norm2_bias, hidden,
                                    config.layer_norm_eps, index),
            vision::load_affine(execution, model,
                         grid_vision_canonical_name(
                             GridVisionTensorRole::block_mlp_up_weight, index),
                         grid_vision_canonical_name(
                             GridVisionTensorRole::block_mlp_up_bias, index)),
            vision::load_affine(execution, model,
                         grid_vision_canonical_name(
                             GridVisionTensorRole::block_mlp_down_weight, index),
                         grid_vision_canonical_name(
                             GridVisionTensorRole::block_mlp_down_bias, index)),
        };
        if (block.attention.qkv.linear.neuron_len() != hidden ||
            block.attention.qkv.linear.out() != 3 * hidden ||
            block.attention.output.linear.neuron_len() != hidden ||
            block.attention.output.linear.out() != hidden ||
            block.mlp_up.linear.neuron_len() != hidden ||
            block.mlp_up.linear.out() != config.intermediate_size ||
            block.mlp_down.linear.neuron_len() != config.intermediate_size ||
            block.mlp_down.linear.out() != hidden) {
            throw std::runtime_error("grid-ViT transformer block geometry mismatch");
        }
        blocks.push_back(std::move(block));
    }
    return CudaGridVisionEncoder(config, std::move(patch_weight), std::move(patch_bias),
                                 std::move(position_weight), std::move(blocks));
}

mfq::StepSequence<Tensor> CudaGridVisionEncoder::encode(
    Projector projector, std::shared_ptr<MoeExpertCache> cache, Tensor pixels,
    const std::vector<GridShape> &grids) const {
    return mfq::models::grid_vision::encode(
        std::move(pixels), config_, grids, blocks_,
        [this, cache](Tensor input, int64_t count) {
            if (input.dim() == 5)
                input = input.reshape({input.size(0), -1});
            if (input.dim() != 2 || input.size(0) != count ||
                input.size(1) != config_.patch_width())
                throw std::runtime_error(
                    "grid-ViT pixel values must be flattened CxTxHxW patches");
            auto run = [&] {
                auto weight = patch_weight_.to(mfq_tensor_backend::kCUDA)
                                  .reshape({config_.hidden_size, config_.patch_width()});
                auto output = mfq_tensor_backend::matmul(input.to(weight.scalar_type()),
                                                         weight.transpose(0, 1));
                return output + patch_bias_.to(mfq_tensor_backend::kCUDA)
                                   .to(output.scalar_type());
            };
            return host_weights()
                       ? vision::workspace(
                             cache,
                             patch_weight_.nbytes() + patch_bias_.nbytes() +
                                 std::size_t(count * config_.hidden_size) * 16,
                             run)
                       : run();
        },
        [this, cache](const LearnedPositionInterpolation &interpolation) {
            auto run = [&] {
                auto position_weight = position_weight_.to(mfq_tensor_backend::kCUDA);
                auto index = mfq_tensor_backend::tensor(
                                 interpolation.indices,
                                 mfq_tensor_backend::TensorOptions()
                                     .dtype(mfq_tensor_backend::kInt32)
                                     .device(mfq_tensor_backend::kCPU))
                                 .to(position_weight.device())
                                 .to(mfq_tensor_backend::kInt64);
                auto weights = mfq_tensor_backend::tensor(
                                   interpolation.weights,
                                   mfq_tensor_backend::TensorOptions()
                                       .dtype(mfq_tensor_backend::kFloat32)
                                       .device(mfq_tensor_backend::kCPU))
                                   .reshape({interpolation.patch_count, 4, 1})
                                   .to(position_weight.device());
                return mfq_tensor_backend::sum(
                    position_weight.index_select(0, index.reshape({-1}))
                            .reshape({interpolation.patch_count, 4, config_.hidden_size}) *
                        weights,
                    1);
            };
            return host_weights()
                       ? vision::workspace(
                             cache,
                             position_weight_.nbytes() +
                                 std::size_t(interpolation.patch_count * config_.hidden_size) *
                                     48,
                             run)
                       : run();
        },
        [](Tensor hidden, Tensor learned) { return hidden + learned.to(hidden.scalar_type()); },
        [this, projector, cache](const auto &block, Tensor value, const GridVisionLayout &layout) {
            // FP32 RoPE intermediates, Q/K/V copies and residual overlap;
            // FFN input/output plus a decoded FP16 projection. GELU is fused.
            const auto projection_scratch =
                std::size_t(2) * config_.hidden_size *
                std::max<int64_t>(3 * config_.hidden_size, config_.intermediate_size);
            if (host_weights())
                return vision::workspace(
                    cache,
                    vision::weight_bytes(block) + value.nbytes() * 16 + projection_scratch +
                        std::size_t(value.size(0) * config_.intermediate_size) * 12 +
                        (std::size_t(16) << 20),
                    [&] {
                        auto resident = vision::to_cuda(block);
                        return resident(projector, value, layout.positions, layout.segment_lengths);
                    });
            return block(projector, std::move(value), layout.positions, layout.segment_lengths);
        });
}

CudaGridVisionEncoder::CudaGridVisionEncoder(
    GridVisionConfig config, Tensor patch_weight, Tensor patch_bias, Tensor position_weight,
    std::vector<vision::Layer> blocks)
    : config_(std::move(config)),
      patch_weight_(std::move(patch_weight)),
      patch_bias_(std::move(patch_bias)),
      position_weight_(std::move(position_weight)),
      blocks_(std::move(blocks)) {}

CudaGridVisionPromptComponent CudaGridVisionPromptComponent::load(
    CudaExecutionContext &execution, const mfq::ModelSource &model, const GridVisionConfig &vision,
    int64_t image_token_id, int64_t video_token_id, std::string input_contract,
    std::string position_policy) {
    if (input_contract != kGridVisionInputContract) {
        throw std::invalid_argument("unsupported grid-Vision input contract: " + input_contract);
    }
    if (position_policy != kGridMropePositionPolicy) {
        throw std::invalid_argument("unsupported grid-Vision position policy: " + position_policy);
    }
    if (image_token_id < 0 || video_token_id < 0 || image_token_id == video_token_id) {
        throw std::invalid_argument("invalid grid-Vision placeholder token IDs");
    }
    vision.validate();
    struct LoadingScope {
        bool &value;
        bool previous;
        explicit LoadingScope(bool &flag, bool host) : value(flag), previous(flag) { value = host; }
        ~LoadingScope() { value = previous; }
    };
    const bool host = !runtime_options::vision_resident() && !execution.tensor_parallel.enabled();
    LoadingScope loading(execution.loading_cpu_layer, host);
    const int64_t unit = vision.spatial_merge_size * vision.spatial_merge_size;
    auto merger_norm = load_norm(
        execution, model, GridVisionTensorRole::merger_norm_weight,
        GridVisionTensorRole::merger_norm_bias, vision.hidden_size, vision.layer_norm_eps);
    auto merger_up =
        vision::load_affine(execution, model,
                     grid_vision_canonical_name(GridVisionTensorRole::merger_mlp_up_weight),
                     grid_vision_canonical_name(GridVisionTensorRole::merger_mlp_up_bias));
    auto merger_down =
        vision::load_affine(execution, model,
                     grid_vision_canonical_name(GridVisionTensorRole::merger_mlp_down_weight),
                     grid_vision_canonical_name(GridVisionTensorRole::merger_mlp_down_bias));
    if (merger_up.linear.neuron_len() != unit * vision.hidden_size ||
        merger_down.linear.neuron_len() != merger_up.linear.out() ||
        merger_down.linear.out() != vision.out_hidden_size) {
        throw std::runtime_error("grid-Vision merger geometry mismatch");
    }
    return CudaGridVisionPromptComponent(
        CudaGridVisionEncoder::load(execution, model, vision), std::move(merger_norm),
        std::move(merger_up), std::move(merger_down), image_token_id, video_token_id,
        std::move(input_contract), std::move(position_policy));
}

CudaGridVisionPromptComponent::CudaGridVisionPromptComponent(
    CudaGridVisionEncoder encoder, vision::LayerNorm merger_norm, vision::Affine merger_up,
    vision::Affine merger_down, int64_t image_token_id, int64_t video_token_id,
    std::string input_contract, std::string position_policy)
    : encoder_(std::move(encoder)),
      merger_norm_(std::move(merger_norm)),
      merger_up_(std::move(merger_up)),
      merger_down_(std::move(merger_down)),
      image_token_id_(image_token_id),
      video_token_id_(video_token_id),
      input_contract_(std::move(input_contract)),
      position_policy_(std::move(position_policy)) {}

} // namespace mfq::cuda::grid_vision_runtime
