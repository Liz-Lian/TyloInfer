#include "engine/models/common/vision/grid_encoder.h"

#include "engine/backends/cuda/executor/options.h"
#include "engine/models/common/model_graph.h"
#include "engine/weights/loader.h"

#include <algorithm>
#include <stdexcept>

namespace tyloi::cuda::grid_vision_runtime {
namespace {

::tyloi::DenseTensor required_vector(const weight_loader::Loader& weights,
                       GridVisionTensorRole role, int64_t width, size_t block = 0) {
    const auto name = grid_vision_canonical_name(role, block);
    auto value = weights.tensor(name);
    if (value.dim() != 1 || value.size(0) != width) {
        throw std::runtime_error("grid-ViT vector shape mismatch: " + name);
    }
    return value;
}

vision::LayerNorm load_norm(const weight_loader::Loader& weights,
                          GridVisionTensorRole weight, GridVisionTensorRole bias, int64_t width,
                          double epsilon, size_t block = 0) {
    return {
        required_vector(weights, weight, width, block),
        required_vector(weights, bias, width, block),
        epsilon,
    };
}

} // namespace

CudaGridVisionEncoder CudaGridVisionEncoder::load(const weight_loader::Loader& weights,
                                                  const GridVisionConfig &config) {
    config.validate();
    const auto hidden = config.hidden_size;
    auto patch_weight = weights.tensor(grid_vision_canonical_name(GridVisionTensorRole::patch_weight));
    const std::vector<int64_t> expected_patch{hidden, config.in_channels,
                                              config.temporal_patch_size, config.patch_size,
                                              config.patch_size};
    if (patch_weight.sizes().vec() != expected_patch) {
        throw std::runtime_error("grid-ViT patch weight shape mismatch");
    }
    auto patch_bias =
        required_vector(weights, GridVisionTensorRole::patch_bias, hidden);
    auto position_weight = weights.tensor(grid_vision_canonical_name(GridVisionTensorRole::position_weight));
    if (position_weight.dim() != 2 ||
        position_weight.size(0) != config.num_position_embeddings ||
        position_weight.size(1) != hidden) {
        throw std::runtime_error("grid-ViT learned position table shape mismatch");
    }
    std::vector<Layer> blocks;
    blocks.reserve(static_cast<size_t>(config.depth));
    for (size_t index = 0; index < static_cast<size_t>(config.depth); ++index) {
        Layer block{
            load_norm(weights, GridVisionTensorRole::block_norm1_weight,
                                    GridVisionTensorRole::block_norm1_bias, hidden,
                                    config.layer_norm_eps, index),
            {
                weights.affine(
                    grid_vision_canonical_name(GridVisionTensorRole::block_attention_qkv_weight, index),
                    grid_vision_canonical_name(GridVisionTensorRole::block_attention_qkv_bias, index)),
                weights.affine(
                    grid_vision_canonical_name(GridVisionTensorRole::block_attention_output_weight, index),
                    grid_vision_canonical_name(GridVisionTensorRole::block_attention_output_bias, index)),
                hidden,
                config.num_heads,
                config.head_dim(),
                config.rope_theta,
            },
            load_norm(weights, GridVisionTensorRole::block_norm2_weight,
                                    GridVisionTensorRole::block_norm2_bias, hidden,
                                    config.layer_norm_eps, index),
            weights.affine(
                grid_vision_canonical_name(GridVisionTensorRole::block_mlp_up_weight, index),
                grid_vision_canonical_name(GridVisionTensorRole::block_mlp_up_bias, index)),
            weights.affine(
                grid_vision_canonical_name(GridVisionTensorRole::block_mlp_down_weight, index),
                grid_vision_canonical_name(GridVisionTensorRole::block_mlp_down_bias, index)),
        };
        if (block.attention.qkv.weight->neuron_len() != hidden ||
            block.attention.qkv.weight->out() != 3 * hidden ||
            block.attention.output.weight->neuron_len() != hidden ||
            block.attention.output.weight->out() != hidden ||
            block.mlp_up.weight->neuron_len() != hidden ||
            block.mlp_up.weight->out() != config.intermediate_size ||
            block.mlp_down.weight->neuron_len() != config.intermediate_size ||
            block.mlp_down.weight->out() != hidden) {
            throw std::runtime_error("grid-ViT transformer block geometry mismatch");
        }
        blocks.push_back(std::move(block));
    }
    return CudaGridVisionEncoder(weights.cache, config, std::move(patch_weight), std::move(patch_bias),
                                 std::move(position_weight), std::move(blocks));
}

tyloi::StepSequence<DenseTensor> CudaGridVisionEncoder::encode(
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
    Projector projector,
#endif
    DenseTensor pixels,
    const std::vector<GridShape> &grids) const {
    // Everything this encoder stages follows the pixels' device.
    const auto device = pixels.device();
    return tyloi::models::grid_vision::encode(
        std::move(pixels), config_, grids, blocks_,
        [this, &device](DenseTensor input, int64_t count) {
            if (input.dim() == 5)
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                input = tyloi::ops::reshape(input, {input.size(0), -1});
#else
                input = input.reshape({input.size(0), -1});
#endif
            if (input.dim() != 2 || input.size(0) != count ||
                input.size(1) != config_.patch_width())
                throw std::runtime_error(
                    "grid-ViT pixel values must be flattened CxTxHxW patches");
            auto run = [&] {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                auto weight = tyloi::ops::cast(patch_weight_, patch_weight_.scalar_type(), device);
                weight = tyloi::ops::reshape(std::move(weight), {config_.hidden_size, config_.patch_width()});
                auto output = tyloi::ops::matmul(tyloi::ops::cast(input, weight.scalar_type()),
                                               tyloi::ops::transpose(weight, 0, 1));
                auto bias = tyloi::ops::cast(patch_bias_, patch_bias_.scalar_type(), device);
                bias = tyloi::ops::cast(std::move(bias), output.scalar_type());
                return tyloi::ops::add(std::move(output), std::move(bias));
#else
                auto weight = patch_weight_.to(::tyloi::kCUDA)
                                  .reshape({config_.hidden_size, config_.patch_width()});
                auto output = ::tyloi::cuda::matmul(input.to(weight.scalar_type()),
                                                         weight.transpose(0, 1));
                return output + patch_bias_.to(::tyloi::kCUDA)
                                   .to(output.scalar_type());
#endif
            };
            if (!host_weights()) return run();
            auto loan = weights_.borrow(patch_weight_.nbytes() + patch_bias_.nbytes() +
                       std::size_t(count * config_.hidden_size) * 16);
            return loan.finish(run());
        },
        [this, &device](const LearnedPositionInterpolation &interpolation) {
            auto run = [&] {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                auto position_weight = tyloi::ops::cast(position_weight_, position_weight_.scalar_type(), device);
                DenseTensor index = tyloi::ops::tensor(interpolation.indices, tyloi::Device{tyloi::DeviceType::cpu, 0});
                index = tyloi::ops::cast(std::move(index), ::tyloi::kInt32, position_weight.device());
                index = tyloi::ops::cast(std::move(index), ::tyloi::kInt64);
                DenseTensor weights = tyloi::ops::tensor(interpolation.weights, tyloi::Device{tyloi::DeviceType::cpu, 0});
                weights = tyloi::ops::reshape(std::move(weights), {interpolation.patch_count, 4, 1});
                weights = tyloi::ops::cast(std::move(weights), ::tyloi::kFloat32, position_weight.device());
                auto selected = tyloi::ops::index_select(position_weight, 0, tyloi::ops::reshape(std::move(index), {-1}));
                selected = tyloi::ops::reshape(std::move(selected), {interpolation.patch_count, 4, config_.hidden_size});
                return tyloi::ops::sum(tyloi::ops::multiply(std::move(selected), std::move(weights)), 1);
#else
                auto position_weight = position_weight_.to(::tyloi::kCUDA);
                auto index = ::tyloi::cuda::tensor(
                                 interpolation.indices,
                                 ::tyloi::TensorOptions()
                                     .dtype(::tyloi::kInt32)
                                     .device(::tyloi::kCPU))
                                 .to(position_weight.device())
                                 .to(::tyloi::kInt64);
                auto weights = ::tyloi::cuda::tensor(
                                   interpolation.weights,
                                   ::tyloi::TensorOptions()
                                       .dtype(::tyloi::kFloat32)
                                       .device(::tyloi::kCPU))
                                   .reshape({interpolation.patch_count, 4, 1})
                                   .to(position_weight.device());
                return ::tyloi::cuda::sum(
                    position_weight.index_select(0, index.reshape({-1}))
                            .reshape({interpolation.patch_count, 4, config_.hidden_size}) *
                        weights,
                    1);
#endif
            };
            if (!host_weights()) return run();
            auto loan = weights_.borrow(position_weight_.nbytes() +
                       std::size_t(interpolation.patch_count * config_.hidden_size) * 48);
            return loan.finish(run());
        },
        [](DenseTensor hidden, DenseTensor learned) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
            auto dtype = hidden.scalar_type();
            return tyloi::ops::add(std::move(hidden), tyloi::ops::cast(std::move(learned), dtype));
#else
            return hidden + learned.to(hidden.scalar_type());
#endif
        },
        [this
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
         , projector
#endif
        ](const auto &block, DenseTensor value, const GridVisionLayout &layout) {
            // FP32 RoPE intermediates, Q/K/V copies and residual overlap;
            // FFN input/output plus a decoded FP16 projection. GELU is fused.
            const auto projection_scratch =
                std::size_t(2) * config_.hidden_size *
                std::max<int64_t>(3 * config_.hidden_size, config_.intermediate_size);
            if (host_weights()) {
                auto loan = weights_.borrow(fields::bytes(block) + value.nbytes() * 16 + projection_scratch +
                           std::size_t(value.size(0) * config_.intermediate_size) * 12 +
                           (std::size_t(16) << 20));
                DenseTensor output;
                {
                    // Block weights stage on the same device as the hidden state.
                    auto resident = block;
                    fields::to_device(resident, value.device());
                    output = resident(
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
                        projector,
#endif
                        value, layout.positions, layout.segment_lengths);
                }
                return loan.finish(std::move(output));
            }
            return block(
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
                projector,
#endif
                std::move(value), layout.positions, layout.segment_lengths);
        });
}

CudaGridVisionEncoder::CudaGridVisionEncoder(
    RoutedCache& weights, GridVisionConfig config, DenseTensor patch_weight, DenseTensor patch_bias, DenseTensor position_weight,
    std::vector<Layer> blocks)
    : weights_(weights),
      config_(std::move(config)),
      patch_weight_(std::move(patch_weight)),
      patch_bias_(std::move(patch_bias)),
      position_weight_(std::move(position_weight)),
      blocks_(std::move(blocks)) {}

CudaGridVisionPromptComponent CudaGridVisionPromptComponent::load(
    const weight_loader::Loader& weights, const GridVisionConfig &vision,
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
    const bool host = !runtime_options::vision_resident() && !weights.placement.tensor.enabled();
    const auto placed = weights.for_layer(weights.placement.device, host);
    const int64_t unit = vision.spatial_merge_size * vision.spatial_merge_size;
    auto merger_norm = load_norm(
        placed, GridVisionTensorRole::merger_norm_weight,
        GridVisionTensorRole::merger_norm_bias, vision.hidden_size, vision.layer_norm_eps);
    auto merger_up = placed.affine(
        grid_vision_canonical_name(GridVisionTensorRole::merger_mlp_up_weight),
        grid_vision_canonical_name(GridVisionTensorRole::merger_mlp_up_bias));
    auto merger_down = placed.affine(
        grid_vision_canonical_name(GridVisionTensorRole::merger_mlp_down_weight),
        grid_vision_canonical_name(GridVisionTensorRole::merger_mlp_down_bias));
    if (merger_up.weight->neuron_len() != unit * vision.hidden_size ||
        merger_down.weight->neuron_len() != merger_up.weight->out() ||
        merger_down.weight->out() != vision.out_hidden_size) {
        throw std::runtime_error("grid-Vision merger geometry mismatch");
    }
    return CudaGridVisionPromptComponent(
        weights.cache, CudaGridVisionEncoder::load(placed, vision), std::move(merger_norm),
        std::move(merger_up), std::move(merger_down), image_token_id, video_token_id,
        std::move(input_contract), std::move(position_policy));
}

CudaGridVisionPromptComponent::CudaGridVisionPromptComponent(
    RoutedCache& weights, CudaGridVisionEncoder encoder, vision::LayerNorm merger_norm, BiasWeight merger_up,
    BiasWeight merger_down, int64_t image_token_id, int64_t video_token_id,
    std::string input_contract, std::string position_policy)
    : weights_(weights),
      encoder_(std::move(encoder)),
      merger_norm_(std::move(merger_norm)),
      merger_up_(std::move(merger_up)),
      merger_down_(std::move(merger_down)),
      image_token_id_(image_token_id),
      video_token_id_(video_token_id),
      input_contract_(std::move(input_contract)),
      position_policy_(std::move(position_policy)) {}

} // namespace tyloi::cuda::grid_vision_runtime
