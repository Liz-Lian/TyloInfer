#include "engine/models/common/vision/grid_encoder.h"

#include "engine/backends/cuda/executor/options.h"
#include "engine/models/source/model_graph.h"
#include "engine/weights/loader.h"

#include <algorithm>
#include <stdexcept>

namespace tyloi::cuda::grid_vision_runtime {
namespace {

#ifdef TYLOINFER_NATIVE_CUDA
::tyloi::Tensor required_vector(const ::tyloi::weights::Loader& weights,
#else
::tyloi::DenseTensor required_vector(const ::tyloi::weights::Loader& weights,
#endif
                       GridVisionTensorRole role, int64_t width, size_t block = 0) {
    const auto name = grid_vision_canonical_name(role, block);
#ifdef TYLOINFER_NATIVE_CUDA
    auto value = weights.tensor(name);
    if (value.dim() != 1 || value.size(0) != width) {
#else
    auto value = weights.tensor(name).dense();
    if (value.dim() != 1 || value.size(0) != width) {
#endif
        throw std::runtime_error("grid-ViT vector shape mismatch: " + name);
    }
    return value;
}

vision::LayerNorm load_norm(const ::tyloi::weights::Loader& weights,
                          GridVisionTensorRole weight, GridVisionTensorRole bias, int64_t width,
                          double epsilon, size_t block = 0) {
    return {
        required_vector(weights, weight, width, block),
        required_vector(weights, bias, width, block),
        epsilon,
    };
}

} // namespace

CudaGridVisionEncoder CudaGridVisionEncoder::load(const ::tyloi::weights::Loader& weights,
#ifndef TYLOINFER_NATIVE_CUDA
                                                  ::tyloi::weights::Cache& cache,
#endif
                                                  const GridVisionConfig &config) {
    config.validate();
    const auto hidden = config.hidden_size;
#ifdef TYLOINFER_NATIVE_CUDA
    auto patch_weight = weights.tensor(grid_vision_canonical_name(GridVisionTensorRole::patch_weight));
#else
    auto patch_weight = weights.tensor(grid_vision_canonical_name(GridVisionTensorRole::patch_weight)).dense();
#endif
    const std::vector<int64_t> expected_patch{hidden, config.in_channels,
                                              config.temporal_patch_size, config.patch_size,
                                              config.patch_size};
#ifdef TYLOINFER_NATIVE_CUDA
    if (patch_weight.dim() != 5 || patch_weight.size(0) != expected_patch[0] ||
        patch_weight.size(1) != expected_patch[1] || patch_weight.size(2) != expected_patch[2] ||
        patch_weight.size(3) != expected_patch[3] || patch_weight.size(4) != expected_patch[4]) {
#else
    if (patch_weight.sizes().vec() != expected_patch) {
#endif
        throw std::runtime_error("grid-ViT patch weight shape mismatch");
    }
    auto patch_bias =
        required_vector(weights, GridVisionTensorRole::patch_bias, hidden);
#ifdef TYLOINFER_NATIVE_CUDA
    auto position_weight = weights.tensor(grid_vision_canonical_name(GridVisionTensorRole::position_weight));
    if (position_weight.dim() != 2 ||
        position_weight.size(0) != config.num_position_embeddings ||
        position_weight.size(1) != hidden) {
#else
    auto position_weight = weights.tensor(grid_vision_canonical_name(GridVisionTensorRole::position_weight)).dense();
    if (position_weight.dim() != 2 ||
        position_weight.size(0) != config.num_position_embeddings ||
        position_weight.size(1) != hidden) {
#endif
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
        if (block.attention.qkv.weight.size(1) != hidden ||
            block.attention.qkv.weight.size(0) != 3 * hidden ||
            block.attention.output.weight.size(1) != hidden ||
            block.attention.output.weight.size(0) != hidden ||
            block.mlp_up.weight.size(1) != hidden ||
            block.mlp_up.weight.size(0) != config.intermediate_size ||
            block.mlp_down.weight.size(1) != config.intermediate_size ||
            block.mlp_down.weight.size(0) != hidden) {
            throw std::runtime_error("grid-ViT transformer block geometry mismatch");
        }
        blocks.push_back(std::move(block));
    }
    return CudaGridVisionEncoder(
#ifndef TYLOINFER_NATIVE_CUDA
                                 cache,
#endif
                                 config, std::move(patch_weight), std::move(patch_bias),
                                 std::move(position_weight), std::move(blocks));
}

#ifdef TYLOINFER_NATIVE_CUDA
tyloi::StepSequence<::tyloi::Tensor> CudaGridVisionEncoder::encode(
    ::tyloi::Tensor pixels, std::vector<GridShape> grids) const {
    const auto layout = make_grid_vision_layout(grids, static_cast<int32_t>(config_.spatial_merge_size));
    const auto side = static_cast<int32_t>(std::sqrt(static_cast<double>(config_.num_position_embeddings)));
    const auto interpolation = make_learned_position_interpolation(
        grids, static_cast<int32_t>(config_.spatial_merge_size), side);
    const auto device = pixels.device();
    if (pixels.dim() == 5) pixels = tyloi::ops::reshape(pixels, {pixels.size(0), -1});
    if (pixels.dim() != 2 || pixels.size(0) != layout.patch_count || pixels.size(1) != config_.patch_width())
        throw std::runtime_error("grid-ViT pixel values must be flattened CxTxHxW patches");
    const bool host = host_weights();
    if (host) co_yield tyloi::Workspace{patch_weight_.nbytes() + patch_bias_.nbytes() +
        std::size_t(layout.patch_count * config_.hidden_size) * 16};
    ::tyloi::Tensor hidden;
    {
        auto weight = tyloi::ops::cast(patch_weight_, patch_weight_.scalar_type(), device);
        weight = tyloi::ops::reshape(weight, {config_.hidden_size, config_.patch_width()});
        auto output = tyloi::ops::matmul(tyloi::ops::cast(pixels, weight.scalar_type()), tyloi::ops::transpose(weight, 0, 1));
        auto bias = tyloi::ops::cast(patch_bias_, patch_bias_.scalar_type(), device);
        bias = tyloi::ops::cast(bias, output.scalar_type());
        hidden = tyloi::ops::add(output, bias);
    }
    if (host) co_yield tyloi::Workspace{0, &hidden};
    pixels = {};
    if (host) co_yield tyloi::Workspace{position_weight_.nbytes() +
        std::size_t(interpolation.patch_count * config_.hidden_size) * 48};
    ::tyloi::Tensor learned;
    {
        auto weight = tyloi::ops::cast(position_weight_, position_weight_.scalar_type(), device);
        auto index = tyloi::ops::tensor(interpolation.indices, tyloi::Device{tyloi::DeviceType::cpu, 0});
        index = tyloi::ops::cast(index, ::tyloi::kInt32, weight.device());
        index = tyloi::ops::cast(index, ::tyloi::kInt64);
        auto weights = tyloi::ops::tensor(interpolation.weights, tyloi::Device{tyloi::DeviceType::cpu, 0});
        weights = tyloi::ops::reshape(weights, {interpolation.patch_count, 4, 1});
        weights = tyloi::ops::cast(weights, ::tyloi::kFloat32, weight.device());
        auto selected = tyloi::ops::index_select(weight, 0, tyloi::ops::reshape(index, {-1}));
        selected = tyloi::ops::reshape(selected, {interpolation.patch_count, 4, config_.hidden_size});
        learned = tyloi::ops::sum(tyloi::ops::multiply(selected, weights), 1);
    }
    if (host) co_yield tyloi::Workspace{0, &learned};
    hidden = tyloi::ops::add(hidden, tyloi::ops::cast(learned, hidden.scalar_type()));
    learned = {};
    co_yield tyloi::StepState::advanced;
    for (const auto& block : blocks_) {
        const auto scratch = std::size_t(2) * config_.hidden_size *
            std::max<int64_t>(3 * config_.hidden_size, config_.intermediate_size);
        if (host) {
            co_yield tyloi::Workspace{::tyloi::weights::fields::bytes(block) + hidden.nbytes() * 16 + scratch +
                std::size_t(hidden.size(0) * config_.intermediate_size) * 12 + (std::size_t(16) << 20)};
            ::tyloi::Tensor output;
            {
                auto resident = block;
                ::tyloi::weights::fields::to_device(resident, hidden.device());
                output = resident(hidden, layout.positions, layout.segment_lengths);
            }
            co_yield tyloi::Workspace{0, &output};
            hidden = std::move(output);
        } else {
            hidden = block(std::move(hidden), layout.positions, layout.segment_lengths);
        }
        co_yield tyloi::StepState::advanced;
    }
    co_yield std::move(hidden);
}
#else
tyloi::StepSequence<DenseTensor> CudaGridVisionEncoder::encode(
    Projector projector,
    DenseTensor pixels,
    const std::vector<GridShape> &grids) const {
    // Everything this encoder stages follows the pixels' device.
    const auto device = pixels.device();
    return tyloi::models::grid_vision::encode(
        std::move(pixels), config_, grids, blocks_,
        [this, device](auto input, int64_t count) {
            if (input.dim() == 5)
                input = input.reshape({input.size(0), -1});
            if (input.dim() != 2 || input.size(0) != count ||
                input.size(1) != config_.patch_width())
                throw std::runtime_error(
                    "grid-ViT pixel values must be flattened CxTxHxW patches");
            auto run = [&] {
                auto weight = patch_weight_.to(::tyloi::kCUDA)
                                  .reshape({config_.hidden_size, config_.patch_width()});
                auto output = ::tyloi::cuda::matmul(input.to(weight.scalar_type()),
                                                         weight.transpose(0, 1));
                return output + patch_bias_.to(::tyloi::kCUDA)
                                   .to(output.scalar_type());
            };
            if (!host_weights()) return run();
            tyloi::weights::Loan loan(weights_,
                       patch_weight_.nbytes() + patch_bias_.nbytes() +
                       std::size_t(count * config_.hidden_size) * 16);
            ::tyloi::Tensor result(run());
            loan.finish(result);
            return DenseTensor(result.dense());
        },
        [this, device](const LearnedPositionInterpolation &interpolation) {
            auto run = [&] {
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
            };
            if (!host_weights()) return run();
            tyloi::weights::Loan loan(weights_,
                       position_weight_.nbytes() +
                       std::size_t(interpolation.patch_count * config_.hidden_size) * 48);
            ::tyloi::Tensor result(run());
            loan.finish(result);
            return DenseTensor(result.dense());
        },
        [](auto hidden, auto learned) {
            return hidden + learned.to(hidden.scalar_type());
        },
        [this
         , projector
        ](const auto &block, auto value, const GridVisionLayout &layout) {
            // FP32 RoPE intermediates, Q/K/V copies and residual overlap;
            // FFN input/output plus a decoded FP16 projection. GELU is fused.
            const auto projection_scratch =
                std::size_t(2) * config_.hidden_size *
                std::max<int64_t>(3 * config_.hidden_size, config_.intermediate_size);
            if (host_weights()) {
                tyloi::weights::Loan loan(weights_, ::tyloi::weights::fields::bytes(block) + value.nbytes() * 16 + projection_scratch +
                           std::size_t(value.size(0) * config_.intermediate_size) * 12 +
                           (std::size_t(16) << 20));
                DenseTensor output;
                {
                    // Block weights stage on the same device as the hidden state.
                    auto resident = block;
                    ::tyloi::weights::fields::to_device(resident, value.device());
                    output = resident(
                        projector,
                        value, layout.positions, layout.segment_lengths);
                }
                ::tyloi::Tensor result(std::move(output));
                loan.finish(result);
                return DenseTensor(result.dense());
            }
            return block(
                projector,
                std::move(value), layout.positions, layout.segment_lengths);
        });
}
#endif

CudaGridVisionEncoder::CudaGridVisionEncoder(
#ifdef TYLOINFER_NATIVE_CUDA
    GridVisionConfig config,
    ::tyloi::Tensor patch_weight, ::tyloi::Tensor patch_bias,
    ::tyloi::Tensor position_weight,
#else
    ::tyloi::weights::Cache& weights, GridVisionConfig config, DenseTensor patch_weight, DenseTensor patch_bias, DenseTensor position_weight,
#endif
    std::vector<Layer> blocks)
    :
#ifndef TYLOINFER_NATIVE_CUDA
      weights_(weights),
#endif
      config_(std::move(config)),
      patch_weight_(std::move(patch_weight)),
      patch_bias_(std::move(patch_bias)),
      position_weight_(std::move(position_weight)),
      blocks_(std::move(blocks)) {}

CudaGridVisionPromptComponent CudaGridVisionPromptComponent::load(
    const ::tyloi::weights::Loader& weights,
#ifndef TYLOINFER_NATIVE_CUDA
    ::tyloi::weights::Cache& cache,
#endif
    const GridVisionConfig &vision,
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
    const bool host = !::tyloi::cuda::options::vision_resident() && !weights.placement.tensor.enabled();
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
    if (merger_up.weight.size(1) != unit * vision.hidden_size ||
        merger_down.weight.size(1) != merger_up.weight.size(0) ||
        merger_down.weight.size(0) != vision.out_hidden_size) {
        throw std::runtime_error("grid-Vision merger geometry mismatch");
    }
    return CudaGridVisionPromptComponent(
#ifndef TYLOINFER_NATIVE_CUDA
        cache, CudaGridVisionEncoder::load(placed, cache, vision),
#else
        CudaGridVisionEncoder::load(placed, vision),
#endif
        std::move(merger_norm),
        std::move(merger_up), std::move(merger_down), image_token_id, video_token_id,
        std::move(input_contract), std::move(position_policy));
}

CudaGridVisionPromptComponent::CudaGridVisionPromptComponent(
#ifndef TYLOINFER_NATIVE_CUDA
    ::tyloi::weights::Cache& weights,
#endif
    CudaGridVisionEncoder encoder, vision::LayerNorm merger_norm, tyloi::weights::BiasWeight merger_up,
    tyloi::weights::BiasWeight merger_down, int64_t image_token_id, int64_t video_token_id,
    std::string input_contract, std::string position_policy)
    :
#ifndef TYLOINFER_NATIVE_CUDA
      weights_(weights),
#endif
      encoder_(std::move(encoder)),
      merger_norm_(std::move(merger_norm)),
      merger_up_(std::move(merger_up)),
      merger_down_(std::move(merger_down)),
      image_token_id_(image_token_id),
      video_token_id_(video_token_id),
      input_contract_(std::move(input_contract)),
      position_policy_(std::move(position_policy)) {}

} // namespace tyloi::cuda::grid_vision_runtime
