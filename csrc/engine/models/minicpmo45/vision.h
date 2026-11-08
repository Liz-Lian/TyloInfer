#pragma once

#include "engine/backends/cuda/ops/vision/attention.h"
#include "engine/backends/cuda/ops/projection/bias.h"
#include "engine/backends/cuda/ops/norm/layer.h"
#include "engine/weights/format.h"
#include "engine/weights/loader.h"
#include "engine/models/minicpmo45/media.h"

#include "engine/generation/sequence.h"
#include "backend.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

inline constexpr const char *MINICPMO45_RESAMPLER_POS_EMBED_ASSET =
    "__mfq_asset__/minicpmo45-resampler-pos-embed-v1.bf16";


inline ::tyloi::DenseTensor
minicpmo45_embedding(const ::tyloi::Tensor &embedding, ::tyloi::DenseTensor ids,
                     ::tyloi::ScalarType output_dtype = ::tyloi::kBFloat16) {
    ids = ids.contiguous().to(::tyloi::kCUDA, ::tyloi::kInt64);
    auto output = quant_embedding_lookup(embedding, ids);
    return output.to(output_dtype).contiguous();
}

inline ::tyloi::DenseTensor minicpmo45_layer_norm(
#ifdef TYLOINFER_NATIVE_CUDA
        ::tyloi::DenseTensor input, const ::tyloi::Tensor &weight,
        const ::tyloi::Tensor &bias, double eps) {
    return tyloi::cuda::vision_layer_norm(std::move(input),
        dynamic_cast<const ::tyloi::DenseTensor&>(weight),
        dynamic_cast<const ::tyloi::DenseTensor&>(bias), eps);
#else
        ::tyloi::DenseTensor input, const ::tyloi::DenseTensor &weight,
        const ::tyloi::DenseTensor &bias, double eps) {
    return tyloi::cuda::vision_layer_norm(std::move(input), weight, bias, eps);
#endif
}

struct MiniCPMO45VisionEncoder {
    struct Layer {
        tyloi::weights::BiasWeight query;
        tyloi::weights::BiasWeight key;
        tyloi::weights::BiasWeight value;
        tyloi::weights::BiasWeight output;
#ifdef TYLOINFER_NATIVE_CUDA
        std::shared_ptr<const ::tyloi::Tensor> norm1_weight, norm1_bias, norm2_weight, norm2_bias;
#else
        ::tyloi::DenseTensor norm1_weight;
        ::tyloi::DenseTensor norm1_bias;
        ::tyloi::DenseTensor norm2_weight;
        ::tyloi::DenseTensor norm2_bias;
#endif
        tyloi::weights::BiasWeight fc1;
        tyloi::weights::BiasWeight fc2;
        int64_t heads = 0;
        int64_t head_dim = 0;
        double layer_norm_eps = 1e-6;

        ::tyloi::DenseTensor forward(
                const tyloi::cuda::Projector& projector, ::tyloi::DenseTensor input,
                MfqOptional<::tyloi::DenseTensor> mask = mfq_nullopt) const {
            using DenseTensor = ::tyloi::DenseTensor;
            if (heads <= 0 || head_dim <= 0) {
                throw std::runtime_error("vision block head geometry is invalid");
            }
            const int64_t batch = input.size(0);
            const int64_t tokens = input.size(1);
            auto split_heads = [&](DenseTensor value) {
                return value.reshape({batch, tokens, heads, head_dim}).transpose(1, 2).contiguous();
            };
            const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
#ifdef TYLOINFER_NATIVE_CUDA
            auto normalized = minicpmo45_layer_norm(input, *norm1_weight, *norm1_bias, layer_norm_eps);
#else
            auto normalized = minicpmo45_layer_norm(input, norm1_weight, norm1_bias, layer_norm_eps);
#endif
            auto q = split_heads(tyloi::cuda::biased(projector, query, normalized));
            auto k = split_heads(tyloi::cuda::biased(projector, key, normalized));
            auto v = split_heads(tyloi::cuda::biased(projector, value, normalized));
            auto attended = tyloi::cuda::vision_non_causal_attention(std::move(q), std::move(k), std::move(v), mask, scale);
            auto branch = tyloi::cuda::biased(projector, output, attended.transpose(1, 2).reshape(
                {batch, tokens, heads * head_dim}));
            input = (input + branch.to(input.scalar_type())).contiguous();
#ifdef TYLOINFER_NATIVE_CUDA
            normalized = minicpmo45_layer_norm(input, *norm2_weight, *norm2_bias, layer_norm_eps);
#else
            normalized = minicpmo45_layer_norm(input, norm2_weight, norm2_bias, layer_norm_eps);
#endif
            branch = tyloi::cuda::biased(projector, fc2,
                ::tyloi::cuda::gelu(tyloi::cuda::biased(projector, fc1, normalized), "tanh"));
            return (input + branch.to(input.scalar_type())).contiguous();
        }
    };

    static Layer load_layer(
            const tyloi::weights::Loader& weights, int index,
            int64_t heads, int64_t head_dim) {
        const std::string prefix = "vision.block." + std::to_string(index);
        Layer result;
        result.query = weights.biased(prefix + ".attention.query.weight", prefix + ".attention.query.bias");
        result.key = weights.biased(prefix + ".attention.key.weight", prefix + ".attention.key.bias");
        result.value = weights.biased(prefix + ".attention.value.weight", prefix + ".attention.value.bias");
        result.output = weights.biased(prefix + ".attention.output.weight", prefix + ".attention.output.bias");
#ifdef TYLOINFER_NATIVE_CUDA
        result.norm1_weight = weights.native(prefix + ".norm1.weight");
        result.norm1_bias = weights.native(prefix + ".norm1.bias");
        result.norm2_weight = weights.native(prefix + ".norm2.weight");
        result.norm2_bias = weights.native(prefix + ".norm2.bias");
#else
        result.norm1_weight = dynamic_cast<const ::tyloi::DenseTensor&>(*weights.native(prefix + ".norm1.weight"));
        result.norm1_bias = dynamic_cast<const ::tyloi::DenseTensor&>(*weights.native(prefix + ".norm1.bias"));
        result.norm2_weight = dynamic_cast<const ::tyloi::DenseTensor&>(*weights.native(prefix + ".norm2.weight"));
        result.norm2_bias = dynamic_cast<const ::tyloi::DenseTensor&>(*weights.native(prefix + ".norm2.bias"));
#endif
        result.fc1 = weights.biased(prefix + ".mlp.up.weight", prefix + ".mlp.up.bias");
        result.fc2 = weights.biased(prefix + ".mlp.down.weight", prefix + ".mlp.down.bias");
        result.heads = heads;
        result.head_dim = head_dim;
        result.layer_norm_eps = 1e-6;
        return result;
    }

#ifdef TYLOINFER_NATIVE_CUDA
    std::shared_ptr<const ::tyloi::Tensor> patch_weight, patch_bias, position_embedding;
#else
    ::tyloi::DenseTensor patch_weight;
    ::tyloi::DenseTensor patch_bias;
    ::tyloi::DenseTensor position_embedding;
#endif
    // SigLIP blocks: 16 heads of 1152/16 = 72 features each.
    std::vector<Layer> layers;
#ifdef TYLOINFER_NATIVE_CUDA
    std::shared_ptr<const ::tyloi::Tensor> post_norm_weight, post_norm_bias;
#else
    ::tyloi::DenseTensor post_norm_weight;
    ::tyloi::DenseTensor post_norm_bias;
#endif
    int64_t patch_size = 14;
    int64_t position_side = 70;

    static MiniCPMO45VisionEncoder load(const tyloi::weights::Loader& weights) {
        MiniCPMO45VisionEncoder result;
#ifdef TYLOINFER_NATIVE_CUDA
        result.patch_weight = weights.native("vision.patch_embedding.weight");
        result.patch_bias = weights.native("vision.patch_embedding.bias");
        result.position_embedding = weights.native("vision.position_embedding.weight");
        result.post_norm_weight = weights.native("vision.output_norm.weight");
        result.post_norm_bias = weights.native("vision.output_norm.bias");
#else
        result.patch_weight =
            dynamic_cast<const ::tyloi::DenseTensor&>(*weights.native("vision.patch_embedding.weight"));
        result.patch_bias = dynamic_cast<const ::tyloi::DenseTensor&>(*weights.native("vision.patch_embedding.bias"));
        result.position_embedding =
            dynamic_cast<const ::tyloi::DenseTensor&>(*weights.native("vision.position_embedding.weight"));
        result.post_norm_weight =
            dynamic_cast<const ::tyloi::DenseTensor&>(*weights.native("vision.output_norm.weight"));
        result.post_norm_bias = dynamic_cast<const ::tyloi::DenseTensor&>(*weights.native("vision.output_norm.bias"));
#endif
        result.layers.reserve(27);
        for (int index = 0; index < 27; ++index) {
            result.layers.push_back(load_layer(weights, index, 16, 72));
        }
#ifdef TYLOINFER_NATIVE_CUDA
        if (result.patch_weight->dim() != 4 || result.patch_weight->size(0) != 1152 ||
            result.patch_weight->size(1) != 3 || result.patch_weight->size(2) != 14 ||
            result.patch_weight->size(3) != 14 || result.position_embedding->dim() != 2 ||
            result.position_embedding->size(0) != 4900 || result.position_embedding->size(1) != 1152) {
#else
        if (result.patch_weight.sizes().vec() != std::vector<int64_t>({1152, 3, 14, 14}) ||
            result.position_embedding.sizes().vec() != std::vector<int64_t>({4900, 1152})) {
#endif
            throw std::runtime_error("MiniCPM-o SigLIP tensor shapes disagree with version 4.5");
        }
        return result;
    }

#ifdef TYLOINFER_NATIVE_CUDA
    tyloi::StepSequence<::tyloi::Tensor> forward_steps(
        tyloi::cuda::Projector projector, ::tyloi::Tensor pixels_value,
        ::tyloi::Tensor patch_mask_value, ::tyloi::Tensor target_sizes_value) const {
#else
    tyloi::StepSequence<::tyloi::DenseTensor> forward_steps(
        tyloi::cuda::Projector projector, ::tyloi::DenseTensor pixels_value,
        ::tyloi::DenseTensor patch_mask_value, ::tyloi::DenseTensor target_sizes_value) const {
#endif
#ifdef TYLOINFER_NATIVE_CUDA
        if (!pixels_value.defined() || !patch_mask_value.defined() || !target_sizes_value.defined())
            throw std::runtime_error("MiniCPM-o vision input geometry is invalid");
        auto pixels = pixels_value.dense();
        auto patch_mask = patch_mask_value.dense();
        auto target_sizes = target_sizes_value.dense();
#else
        auto pixels = pixels_value;
        auto patch_mask = patch_mask_value;
        auto target_sizes = target_sizes_value;
#endif
        if (patch_mask.dim() == 3) {
            patch_mask = patch_mask.flatten(1);
        }
        if (pixels.dim() != 4 || pixels.size(1) != 3 || patch_mask.dim() != 2 ||
            target_sizes.dim() != 2 || target_sizes.size(1) != 2 ||
            pixels.size(0) != patch_mask.size(0) || pixels.size(0) != target_sizes.size(0) ||
            target_sizes.device().is_cuda()) {
            throw std::runtime_error("MiniCPM-o vision input geometry is invalid");
        }
        using DenseTensor = ::tyloi::DenseTensor;
        MfqOptional<DenseTensor> attention_mask = mfq_nullopt;
        auto encoding = tyloi::models::minicpmo45::vision_encoder(
            pixels, layers,
            [&](DenseTensor pixels) {
#ifdef TYLOINFER_NATIVE_CUDA
                const auto& patch_weight_value = dynamic_cast<const DenseTensor&>(*patch_weight);
                const auto& patch_bias_value = dynamic_cast<const DenseTensor&>(*patch_bias);
#else
                const auto& patch_weight_value = patch_weight;
                const auto& patch_bias_value = patch_bias;
#endif
                return ::tyloi::cuda::conv2d(
                           pixels.to(patch_weight_value.scalar_type()), patch_weight_value, patch_bias_value,
                           std::vector<int64_t>{patch_size, patch_size}, std::vector<int64_t>{0, 0},
                           std::vector<int64_t>{1, 1}, 1)
                    .flatten(2)
                    .transpose(1, 2)
                    .contiguous();
            },
            [&](DenseTensor embedded) {
                if (embedded.size(1) != patch_mask.size(1)) {
                    throw std::runtime_error(
                        "MiniCPM-o patch mask length does not match patch convolution");
                }
                auto sizes = target_sizes.to(::tyloi::kCPU, ::tyloi::kInt64)
                                 .contiguous();
                auto mask_cpu =
                    patch_mask.to(::tyloi::kCPU, ::tyloi::kBool).contiguous();
                const auto *size_data = sizes.data_ptr<int64_t>();
                const auto *mask_data = mask_cpu.data_ptr<bool>();
                auto layout = tyloi::models::minicpmo45::patch_positions(
                    embedded.size(0), embedded.size(1), position_side, size_data, mask_data);
                auto &position_ids = layout.ids;
                const bool all_patches_active = layout.all_active;
                auto ids =
                    ::tyloi::cuda::from_blob(
                        position_ids.data(),
                        std::vector<int64_t>{embedded.size(0), embedded.size(1)},
                        ::tyloi::TensorOptions().dtype(::tyloi::kInt64))
                        .clone()
                        .to(::tyloi::kCUDA);
#ifdef TYLOINFER_NATIVE_CUDA
                const auto& position_value = dynamic_cast<const DenseTensor&>(*position_embedding);
#else
                const auto& position_value = position_embedding;
#endif
                embedded =
                    embedded +
                    position_value.index_select(0, ids.reshape({-1})).reshape(embedded.sizes());

                if (!all_patches_active) {
                    auto invalid =
                        patch_mask.to(::tyloi::kCUDA, ::tyloi::kBool)
                            .logical_not()
                            .unsqueeze(1)
                            .unsqueeze(2);
                    attention_mask =
                        ::tyloi::cuda::zeros(
                            {embedded.size(0), 1, embedded.size(1), embedded.size(1)},
                            embedded.options().dtype(::tyloi::kFloat32))
                            .masked_fill(invalid, -std::numeric_limits<float>::infinity());
                }
                return embedded;
            },
            [&](const auto &layer, DenseTensor hidden) {
                return layer.forward(projector, hidden, attention_mask);
            },
            [&](DenseTensor hidden) {
#ifdef TYLOINFER_NATIVE_CUDA
                return minicpmo45_layer_norm(hidden, *post_norm_weight, *post_norm_bias, 1e-6);
#else
                return minicpmo45_layer_norm(hidden, post_norm_weight, post_norm_bias, 1e-6);
#endif
            });
        while (auto step = encoding.next()) {
#ifdef TYLOINFER_NATIVE_CUDA
            if (step.value) co_yield ::tyloi::Tensor(std::move(*step.value));
#else
            if (step.value) co_yield std::move(*step.value);
#endif
            else co_yield step.state;
        }
    }

#ifdef TYLOINFER_NATIVE_CUDA
    ::tyloi::Tensor forward(const tyloi::cuda::Projector& projector,
        ::tyloi::Tensor pixels, ::tyloi::Tensor mask, ::tyloi::Tensor sizes) const {
#else
    ::tyloi::DenseTensor forward(const tyloi::cuda::Projector& projector,
        ::tyloi::DenseTensor pixels, ::tyloi::DenseTensor mask, ::tyloi::DenseTensor sizes) const {
#endif
        return tyloi::finish_steps(
            forward_steps(projector, std::move(pixels), std::move(mask), std::move(sizes)));
    }
};

struct MiniCPMO45Resampler {
#ifdef TYLOINFER_NATIVE_CUDA
    std::shared_ptr<const ::tyloi::Tensor> query, position_embedding;
#else
    ::tyloi::DenseTensor query;
    ::tyloi::DenseTensor position_embedding;
#endif
    tyloi::weights::BiasWeight kv_projection;
#ifdef TYLOINFER_NATIVE_CUDA
    std::shared_ptr<const ::tyloi::Tensor> q_norm_weight, q_norm_bias, kv_norm_weight, kv_norm_bias;
    std::shared_ptr<const ::tyloi::Tensor> post_norm_weight, post_norm_bias;
    std::shared_ptr<const ::tyloi::Tensor> in_projection_weight, in_projection_bias;
    std::shared_ptr<const ::tyloi::Tensor> out_projection_weight, out_projection_bias, final_projection;
#else
    ::tyloi::DenseTensor q_norm_weight;
    ::tyloi::DenseTensor q_norm_bias;
    ::tyloi::DenseTensor kv_norm_weight;
    ::tyloi::DenseTensor kv_norm_bias;
    ::tyloi::DenseTensor post_norm_weight;
    ::tyloi::DenseTensor post_norm_bias;
    ::tyloi::DenseTensor in_projection_weight;
    ::tyloi::DenseTensor in_projection_bias;
    ::tyloi::DenseTensor out_projection_weight;
    ::tyloi::DenseTensor out_projection_bias;
    ::tyloi::DenseTensor final_projection;
#endif
    int64_t heads = 32;
    int64_t head_dim = 128;

    static MiniCPMO45Resampler load(const tyloi::weights::Loader& weights) {
        MiniCPMO45Resampler result;
        if (!weights.source.has_asset(MINICPMO45_RESAMPLER_POS_EMBED_ASSET)) {
            throw std::runtime_error("MiniCPM-o MFQ is missing the exact Resampler position asset; "
                                     "repack it with the current runtime assets tool");
        }
        auto position_blob = tyloi::read_asset(weights.source, MINICPMO45_RESAMPLER_POS_EMBED_ASSET);
        constexpr size_t position_header_bytes = 20;
        constexpr size_t position_value_bytes =
            static_cast<size_t>(70) * 70 * 4096 * sizeof(uint16_t);
        if (position_blob.size() != position_header_bytes + position_value_bytes ||
            std::memcmp(position_blob.data(), "MFQRSPB1", 8) != 0) {
            throw std::runtime_error("MiniCPM-o Resampler position asset has an invalid format");
        }
        size_t position_offset = 8;
        const uint32_t position_height = read_u32_from(position_blob, position_offset);
        const uint32_t position_width = read_u32_from(position_blob, position_offset);
        const uint32_t position_dim = read_u32_from(position_blob, position_offset);
        if (position_height != 70 || position_width != 70 || position_dim != 4096 ||
            position_offset != position_header_bytes) {
            throw std::runtime_error("MiniCPM-o Resampler position asset shape is not 70x70x4096");
        }
        {
            MfqCudaGuard guard(weights.placement.device);
            auto position =
                ::tyloi::cuda::from_blob(
                    position_blob.data() + position_offset, std::vector<int64_t>{70, 70, 4096},
                    ::tyloi::TensorOptions().dtype(::tyloi::kBFloat16))
                    .clone()
                    .to(::tyloi::kCUDA)
                    .contiguous();
#ifdef TYLOINFER_NATIVE_CUDA
            result.position_embedding = std::make_shared<::tyloi::DenseTensor>(std::move(position));
#else
            result.position_embedding = std::move(position);
#endif
        }
#ifdef TYLOINFER_NATIVE_CUDA
        result.query = weights.native("vision.resampler.query");
        result.kv_projection = weights.biased("vision.resampler.key_value.weight");
        result.q_norm_weight = weights.native("vision.resampler.query_norm.weight");
        result.q_norm_bias = weights.native("vision.resampler.query_norm.bias");
        result.kv_norm_weight = weights.native("vision.resampler.key_value_norm.weight");
        result.kv_norm_bias = weights.native("vision.resampler.key_value_norm.bias");
        result.post_norm_weight = weights.native("vision.resampler.output_norm.weight");
        result.post_norm_bias = weights.native("vision.resampler.output_norm.bias");
        result.in_projection_weight = weights.native("vision.resampler.attention.qkv.weight");
        result.in_projection_bias = weights.native("vision.resampler.attention.qkv.bias");
        result.out_projection_weight = weights.native("vision.resampler.attention.output.weight");
        result.out_projection_bias = weights.native("vision.resampler.attention.output.bias");
        result.final_projection = weights.native("vision.resampler.output.weight");
        if (result.query->dim() != 2 || result.query->size(0) != 64 || result.query->size(1) != 4096 ||
            result.in_projection_weight->dim() != 2 || result.in_projection_weight->size(0) != 12288 ||
            result.in_projection_weight->size(1) != 4096) {
#else
        result.query = dynamic_cast<const ::tyloi::DenseTensor&>(*weights.native("vision.resampler.query"));
        result.kv_projection =
            weights.biased("vision.resampler.key_value.weight");
        result.q_norm_weight =
            dynamic_cast<const ::tyloi::DenseTensor&>(*weights.native("vision.resampler.query_norm.weight"));
        result.q_norm_bias =
            dynamic_cast<const ::tyloi::DenseTensor&>(*weights.native("vision.resampler.query_norm.bias"));
        result.kv_norm_weight =
            dynamic_cast<const ::tyloi::DenseTensor&>(*weights.native("vision.resampler.key_value_norm.weight"));
        result.kv_norm_bias =
            dynamic_cast<const ::tyloi::DenseTensor&>(*weights.native("vision.resampler.key_value_norm.bias"));
        result.post_norm_weight =
            dynamic_cast<const ::tyloi::DenseTensor&>(*weights.native("vision.resampler.output_norm.weight"));
        result.post_norm_bias =
            dynamic_cast<const ::tyloi::DenseTensor&>(*weights.native("vision.resampler.output_norm.bias"));
        result.in_projection_weight =
            dynamic_cast<const ::tyloi::DenseTensor&>(*weights.native("vision.resampler.attention.qkv.weight"));
        result.in_projection_bias =
            dynamic_cast<const ::tyloi::DenseTensor&>(*weights.native("vision.resampler.attention.qkv.bias"));
        result.out_projection_weight =
            dynamic_cast<const ::tyloi::DenseTensor&>(*weights.native("vision.resampler.attention.output.weight"));
        result.out_projection_bias =
            dynamic_cast<const ::tyloi::DenseTensor&>(*weights.native("vision.resampler.attention.output.bias"));
        result.final_projection =
            dynamic_cast<const ::tyloi::DenseTensor&>(*weights.native("vision.resampler.output.weight"));
        if (result.query.sizes().vec() != std::vector<int64_t>({64, 4096}) ||
            result.in_projection_weight.sizes().vec() != std::vector<int64_t>({12288, 4096})) {
#endif
            throw std::runtime_error("MiniCPM-o Resampler tensor shapes disagree with version 4.5");
        }
        return result;
    }

#ifdef TYLOINFER_NATIVE_CUDA
    ::tyloi::Tensor forward(const tyloi::cuda::Projector& projector,
                            ::tyloi::Tensor input_value, ::tyloi::Tensor target_sizes_value) const {
#else
    ::tyloi::DenseTensor forward(const tyloi::cuda::Projector& projector,
                                 ::tyloi::DenseTensor input_value,
                                 ::tyloi::DenseTensor target_sizes_value) const {
#endif
#ifdef TYLOINFER_NATIVE_CUDA
        if (!input_value.defined() || !target_sizes_value.defined())
            throw std::runtime_error("MiniCPM-o Resampler input geometry is invalid");
        const auto& input = input_value.dense();
        const auto& target_sizes = target_sizes_value.dense();
#else
        const auto& input = input_value;
        const auto& target_sizes = target_sizes_value;
#endif
        if (input.dim() != 3 || input.size(2) != 1152 || target_sizes.dim() != 2 ||
            target_sizes.size(1) != 2 || target_sizes.size(0) != input.size(0) ||
            target_sizes.device().is_cuda() ||
            input.scalar_type() != ::tyloi::kBFloat16) {
            throw std::runtime_error("MiniCPM-o Resampler input geometry is invalid");
        }
        const int64_t batch = input.size(0);
        const int64_t length = input.size(1);
        auto sizes =
            target_sizes.to(::tyloi::kCPU, ::tyloi::kInt64).contiguous();
        const auto *size_data = sizes.data_ptr<int64_t>();
        std::vector<::tyloi::DenseTensor> positions;
        positions.reserve(static_cast<size_t>(batch));
        std::vector<uint8_t> key_padding(static_cast<size_t>(batch * length), uint8_t{1});
        for (int64_t index = 0; index < batch; ++index) {
            const int64_t height = size_data[2 * index];
            const int64_t width = size_data[2 * index + 1];
            const int64_t patches = height * width;
#ifdef TYLOINFER_NATIVE_CUDA
            if (height <= 0 || width <= 0 || height > position_embedding->size(0) ||
                width > position_embedding->size(1) || patches > length) {
#else
            if (height <= 0 || width <= 0 || height > position_embedding.size(0) ||
                width > position_embedding.size(1) || patches > length) {
#endif
                throw std::runtime_error("MiniCPM-o Resampler target size is invalid");
            }
#ifdef TYLOINFER_NATIVE_CUDA
            const auto& position_value = dynamic_cast<const ::tyloi::DenseTensor&>(*position_embedding);
#else
            const auto& position_value = position_embedding;
#endif
            auto position = position_value.narrow(0, 0, height)
                                .narrow(1, 0, width)
                                .reshape({patches, 4096});
            if (patches < length) {
                position = ::tyloi::cuda::cat(
                    {position,
                     ::tyloi::cuda::zeros({length - patches, 4096}, position.options())},
                    0);
            }
            positions.push_back(position);
            for (int64_t patch = 0; patch < patches; ++patch) {
                key_padding[static_cast<size_t>(index * length + patch)] = 0;
            }
        }
        auto position = ::tyloi::cuda::stack(positions, 0);
        auto key_padding_mask =
            ::tyloi::cuda::from_blob(
                key_padding.data(), std::vector<int64_t>{batch, length},
                ::tyloi::TensorOptions().dtype(::tyloi::kUInt8))
                .clone()
                .to(::tyloi::kCUDA, ::tyloi::kBool);
        const auto dtype = input.scalar_type();
        auto project = [&](::tyloi::DenseTensor value, int64_t offset) {
#ifdef TYLOINFER_NATIVE_CUDA
            const auto& projection_weight = dynamic_cast<const ::tyloi::DenseTensor&>(*in_projection_weight);
            const auto& projection_bias = dynamic_cast<const ::tyloi::DenseTensor&>(*in_projection_bias);
#else
            const auto& projection_weight = in_projection_weight;
            const auto& projection_bias = in_projection_bias;
#endif
            auto weight = projection_weight.narrow(0, offset, 4096);
            auto bias = projection_bias.narrow(0, offset, 4096);
            return mfq_linear(value.to(weight.scalar_type()), weight, bias).to(dtype).contiguous();
        };
        using DenseTensor = ::tyloi::DenseTensor;
        return tyloi::models::minicpmo45::resample(
            [&] { return tyloi::cuda::biased(projector, kv_projection, input); },
            [&](DenseTensor kv) {
#ifdef TYLOINFER_NATIVE_CUDA
                return minicpmo45_layer_norm(kv, *kv_norm_weight, *kv_norm_bias, 1e-6);
#else
                return minicpmo45_layer_norm(kv, kv_norm_weight, kv_norm_bias, 1e-6);
#endif
            },
            [&] {
#ifdef TYLOINFER_NATIVE_CUDA
                auto normalized = minicpmo45_layer_norm(dynamic_cast<const DenseTensor&>(*query), *q_norm_weight, *q_norm_bias, 1e-6);
#else
                auto normalized = minicpmo45_layer_norm(query, q_norm_weight, q_norm_bias, 1e-6);
#endif
                return normalized.unsqueeze(0)
                    .expand({batch, normalized.size(0), normalized.size(1)})
                    .contiguous();
            },
            [&](DenseTensor repeated_query) {
                return project(repeated_query, 0)
                    .reshape({batch, 64, heads, head_dim})
                    .transpose(1, 2)
                    .contiguous();
            },
            [&](DenseTensor kv) {
                return project(kv + position, 4096)
                    .reshape({batch, length, heads, head_dim})
                    .transpose(1, 2)
                    .contiguous();
            },
            [&](DenseTensor kv) {
                return project(kv, 8192)
                    .reshape({batch, length, heads, head_dim})
                    .transpose(1, 2)
                    .contiguous();
            },
            [&](DenseTensor q, DenseTensor k, DenseTensor v) {
                auto mask = ::tyloi::cuda::zeros({batch, 1, 1, length},
                                                      ::tyloi::TensorOptions()
                                                          .device(::tyloi::kCUDA)
                                                          .dtype(dtype))
                                .masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2),
                                             -std::numeric_limits<float>::infinity())
                                .expand({batch, heads, 1, length})
                                .reshape({batch * heads, 1, length});
                auto q_flat = q.reshape({batch * heads, 64, head_dim});
                auto k_flat = k.reshape({batch * heads, length, head_dim});
                auto v_flat = v.reshape({batch * heads, length, head_dim});
                auto attention_weights = ::tyloi::cuda::baddbmm(
                    mask, q_flat * std::sqrt(1.0 / static_cast<double>(head_dim)),
                    k_flat.transpose(1, 2));
                attention_weights = ::tyloi::cuda::softmax(attention_weights, -1);
                return ::tyloi::cuda::bmm(attention_weights, v_flat)
                    .reshape({batch, heads, 64, head_dim})
                    .transpose(1, 2)
                    .reshape({batch, 64, 4096});
            },
            [&](DenseTensor attended) {
#ifdef TYLOINFER_NATIVE_CUDA
                const auto& weight = dynamic_cast<const DenseTensor&>(*out_projection_weight);
                const auto& bias = dynamic_cast<const DenseTensor&>(*out_projection_bias);
#else
                const auto& weight = out_projection_weight;
                const auto& bias = out_projection_bias;
#endif
                return mfq_linear(attended.to(weight.scalar_type()), weight, bias)
                    .to(dtype)
                    .contiguous();
            },
            [&](DenseTensor attended) {
#ifdef TYLOINFER_NATIVE_CUDA
                return minicpmo45_layer_norm(attended, *post_norm_weight, *post_norm_bias, 1e-6);
#else
                return minicpmo45_layer_norm(attended, post_norm_weight, post_norm_bias, 1e-6);
#endif
            },
            [&](DenseTensor attended) {
#ifdef TYLOINFER_NATIVE_CUDA
                const auto& weight = dynamic_cast<const DenseTensor&>(*final_projection);
#else
                const auto& weight = final_projection;
#endif
                return ::tyloi::cuda::matmul(attended.to(weight.scalar_type()), weight)
                    .to(dtype)
                    .contiguous();
            });
    }
};
