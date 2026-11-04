#pragma once

#include "engine/models/minicpmo45/media.h"
#include "engine/ops.h"
#include "vision.h"

struct MiniCPMO45WhisperAttention {
    tyloi::cuda::BiasWeight q;
    tyloi::cuda::BiasWeight k;
    tyloi::cuda::BiasWeight v;
    tyloi::cuda::BiasWeight output;
    int64_t heads = 16;
    int64_t head_dim = 64;

    static MiniCPMO45WhisperAttention load(const tyloi::cuda::weight_loader::Loader& weights, const std::string &prefix) {
        MiniCPMO45WhisperAttention result;
        result.q = weights.biased(prefix + ".query.weight", prefix + ".query.bias");
        result.k = weights.biased(prefix + ".key.weight");
        result.v = weights.biased(prefix + ".value.weight", prefix + ".value.bias");
        result.output = weights.biased(prefix + ".output.weight", prefix + ".output.bias");
        return result;
    }

    ::tyloi::DenseTensor forward(const tyloi::cuda::Projector& projector,
                                       ::tyloi::DenseTensor input,
                                       MfqOptional<::tyloi::DenseTensor> mask,
                                       ::tyloi::DenseTensor *key_cache = nullptr,
                                       ::tyloi::DenseTensor *value_cache = nullptr) const {
        const int64_t batch = input.size(0);
        const int64_t tokens = input.size(1);
        auto reshape = [&](::tyloi::DenseTensor value) {
            return value.reshape({batch, tokens, heads, head_dim}).transpose(1, 2).contiguous();
        };
        using DenseTensor = ::tyloi::DenseTensor;
        return tyloi::models::minicpmo45::encoder_attention(
            input, key_cache && value_cache,
            [&](const DenseTensor &x) { return reshape(tyloi::cuda::biased(projector, q, x)); },
            [&](const DenseTensor &x) { return reshape(tyloi::cuda::biased(projector, k, x)); },
            [&](const DenseTensor &x) { return reshape(tyloi::cuda::biased(projector, v, x)); },
            [&](DenseTensor &key, DenseTensor &value) {
                if (key_cache->defined()) {
                    key = ::tyloi::cuda::cat({*key_cache, key}, 2).contiguous();
                    value = ::tyloi::cuda::cat({*value_cache, value}, 2).contiguous();
                }
                *key_cache = key;
                *value_cache = value;
            },
            [&](DenseTensor query, DenseTensor key, DenseTensor value) {
                std::optional<::tyloi::DenseTensor> attention_mask = std::nullopt;
                if (mask.has_value()) {
                    attention_mask = mask.value().to(query.scalar_type());
                }
                return mfq_scaled_dot_product_attention(query, key, value, attention_mask, 0.0,
                                                        false, std::nullopt, false)
                    .contiguous();
            },
            [&](DenseTensor attended) {
                attended = attended.transpose(1, 2).reshape({batch, tokens, heads * head_dim});
                return tyloi::cuda::biased(projector, output, attended);
            });
    }
};

struct MiniCPMO45WhisperLayer {
    MiniCPMO45WhisperAttention attention;
    ::tyloi::DenseTensor attention_norm_weight;
    ::tyloi::DenseTensor attention_norm_bias;
    ::tyloi::DenseTensor final_norm_weight;
    ::tyloi::DenseTensor final_norm_bias;
    tyloi::cuda::BiasWeight fc1;
    tyloi::cuda::BiasWeight fc2;
    ::tyloi::DenseTensor key_cache;
    ::tyloi::DenseTensor value_cache;

    static MiniCPMO45WhisperLayer load(const tyloi::cuda::weight_loader::Loader& weights,
                                       int index) {
        const std::string prefix = "audio.block." + std::to_string(index);
        MiniCPMO45WhisperLayer result;
        result.attention = MiniCPMO45WhisperAttention::load(weights, prefix + ".attention");
        result.attention_norm_weight =
            weights.native(prefix + ".attention.norm.weight");
        result.attention_norm_bias =
            weights.native(prefix + ".attention.norm.bias");
        result.final_norm_weight =
            weights.native(prefix + ".mlp.norm.weight");
        result.final_norm_bias = weights.native(prefix + ".mlp.norm.bias");
        result.fc1 = weights.biased(prefix + ".mlp.up.weight", prefix + ".mlp.up.bias");
        result.fc2 = weights.biased(prefix + ".mlp.down.weight", prefix + ".mlp.down.bias");
        return result;
    }

    void reset() {
        key_cache = ::tyloi::DenseTensor();
        value_cache = ::tyloi::DenseTensor();
    }

    ::tyloi::DenseTensor forward(const tyloi::cuda::Projector& projector,
                                       ::tyloi::DenseTensor input,
                                       MfqOptional<::tyloi::DenseTensor> mask,
                                       bool use_cache) {
        using DenseTensor = ::tyloi::DenseTensor;
        return tyloi::models::pre_norm_layer(
            input,
            [&](const DenseTensor &x, int stage) {
                return minicpmo45_layer_norm(
                    x, stage == 0 ? attention_norm_weight : final_norm_weight,
                    stage == 0 ? attention_norm_bias : final_norm_bias, 1e-5);
            },
            [&](DenseTensor x) {
                return attention.forward(projector, x, mask, use_cache ? &key_cache : nullptr,
                                         use_cache ? &value_cache : nullptr);
            },
            [&](DenseTensor x) {
                return tyloi::cuda::biased(projector, fc2,
                    ::tyloi::cuda::gelu(tyloi::cuda::biased(projector, fc1, x)));
            },
            [](DenseTensor x, DenseTensor branch) { return (x + branch).contiguous(); });
    }
};

struct MiniCPMO45AudioEncoder {
    ::tyloi::DenseTensor conv1_weight;
    ::tyloi::DenseTensor conv1_bias;
    ::tyloi::DenseTensor conv2_weight;
    ::tyloi::DenseTensor conv2_bias;
    ::tyloi::DenseTensor position_embedding;
    std::vector<MiniCPMO45WhisperLayer> layers;
    ::tyloi::DenseTensor final_norm_weight;
    ::tyloi::DenseTensor final_norm_bias;
    tyloi::cuda::BiasWeight projector1;
    tyloi::cuda::BiasWeight projector2;

    static MiniCPMO45AudioEncoder load(const tyloi::cuda::weight_loader::Loader& weights) {
        MiniCPMO45AudioEncoder result;
        result.conv1_weight =
            weights.native("audio.patch_embedding.conv1.weight");
        result.conv1_bias =
            weights.native("audio.patch_embedding.conv1.bias");
        result.conv2_weight =
            weights.native("audio.patch_embedding.conv2.weight");
        result.conv2_bias =
            weights.native("audio.patch_embedding.conv2.bias");
        result.position_embedding =
            weights.native("audio.position_embedding.weight");
        result.final_norm_weight =
            weights.native("audio.output_norm.weight");
        result.final_norm_bias = weights.native("audio.output_norm.bias");
        result.projector1 = weights.biased("audio.projector.input.weight", "audio.projector.input.bias");
        result.projector2 = weights.biased("audio.projector.output.weight", "audio.projector.output.bias");
        result.layers.reserve(24);
        for (int index = 0; index < 24; ++index) {
            result.layers.push_back(MiniCPMO45WhisperLayer::load(weights, index));
        }
        if (result.conv1_weight.sizes().vec() != std::vector<int64_t>({1024, 80, 3}) ||
            result.conv2_weight.sizes().vec() != std::vector<int64_t>({1024, 1024, 3}) ||
            result.position_embedding.sizes().vec() != std::vector<int64_t>({1500, 1024})) {
            throw std::runtime_error("MiniCPM-o Whisper tensor shapes disagree with version 4.5");
        }
        return result;
    }

    void reset() {
        for (auto &layer : layers)
            layer.reset();
    }

    int64_t cache_length() const {
        if (layers.empty() || !layers.front().key_cache.defined())
            return 0;
        return layers.front().key_cache.size(2);
    }

    ::tyloi::DenseTensor forward_streaming(const tyloi::cuda::Projector& projector,
                                                 ::tyloi::DenseTensor features,
                                                 int64_t prefix_extra_frames,
                                                 int64_t suffix_extra_frames) {
        if (features.dim() != 3 || features.size(0) != 1 || features.size(1) != 80 ||
            prefix_extra_frames < 0 || suffix_extra_frames < 0) {
            throw std::runtime_error("MiniCPM-o streaming audio expects [1,80,frames] and "
                                     "non-negative extra-frame counts");
        }
        const int64_t conv_tokens_before_crop = (features.size(2) - 1) / 2 + 1;
        if (cache_length() + conv_tokens_before_crop >= position_embedding.size(0)) {
            reset();
        }

        using DenseTensor = ::tyloi::DenseTensor;
        return tyloi::finish_steps(tyloi::models::minicpmo45::audio_encoder(
            features, layers,
            [&](DenseTensor features) {
                return ::tyloi::cuda::conv1d(
                    features.to(conv1_weight.scalar_type()), conv1_weight, conv1_bias,
                    std::vector<int64_t>{1}, std::vector<int64_t>{1}, std::vector<int64_t>{1}, 1);
            },
            [](DenseTensor x) { return ::tyloi::cuda::gelu(x); },
            [&](DenseTensor hidden) {
                return ::tyloi::cuda::conv1d(hidden, conv2_weight, conv2_bias,
                                                  std::vector<int64_t>{2}, std::vector<int64_t>{1},
                                                  std::vector<int64_t>{1}, 1);
            },
            [&](DenseTensor hidden) {
                const int64_t prefix_to_remove =
                    prefix_extra_frames > 0 ? (prefix_extra_frames + 1) / 2 : 0;
                const int64_t suffix_to_remove =
                    suffix_extra_frames > 0 ? (suffix_extra_frames + 1) / 2 : 0;
                if (prefix_to_remove + suffix_to_remove >= hidden.size(2)) {
                    throw std::runtime_error(
                        "MiniCPM-o streaming audio extra context removes every frame");
                }
                if (prefix_to_remove > 0) {
                    hidden = hidden.narrow(2, prefix_to_remove, hidden.size(2) - prefix_to_remove);
                }
                if (suffix_to_remove > 0) {
                    hidden = hidden.narrow(2, 0, hidden.size(2) - suffix_to_remove);
                }
                hidden = hidden.transpose(1, 2).contiguous();

                const int64_t past = cache_length();
                const int64_t tokens = hidden.size(1);
                if (past + tokens > position_embedding.size(0)) {
                    throw std::runtime_error(
                        "MiniCPM-o streaming Whisper position cache exceeds 1500 frames");
                }
                hidden = hidden + position_embedding.narrow(0, past, tokens);
                auto attention_mask =
                    ::tyloi::cuda::zeros({1, 1, tokens, past + tokens}, hidden.options());
                return std::array<DenseTensor, 2>{hidden, attention_mask};
            },
            [&](auto &layer, DenseTensor hidden, DenseTensor mask) {
                return layer.forward(projector, hidden, mask, true);
            },
            [&](DenseTensor hidden) {
                return minicpmo45_layer_norm(hidden, final_norm_weight, final_norm_bias, 1e-5);
            },
            [&](DenseTensor hidden) { return tyloi::cuda::biased(projector, projector1, hidden); },
            [](DenseTensor hidden) { return ::tyloi::cuda::relu(hidden); },
            [&](DenseTensor hidden) { return tyloi::cuda::biased(projector, projector2, hidden); },
            [&](DenseTensor hidden) {
                if (hidden.size(1) < 5) {
                    throw std::runtime_error(
                        "MiniCPM-o streaming audio chunk is too short for stride-5 pooling");
                }
                return ::tyloi::cuda::avg_pool1d(
                           hidden.transpose(1, 2), std::vector<int64_t>{5}, std::vector<int64_t>{5})
                    .transpose(1, 2)
                    .contiguous();
            }));
    }

    static std::vector<int64_t> pooled_lengths(::tyloi::DenseTensor raw_lengths) {
        auto lengths =
            raw_lengths.to(::tyloi::kCPU, ::tyloi::kInt64).contiguous();
        const auto *values = lengths.data_ptr<int64_t>();
        std::vector<int64_t> result(static_cast<size_t>(lengths.numel()));
        for (int64_t index = 0; index < lengths.numel(); ++index) {
            result[static_cast<size_t>(index)] =
                tyloi::models::minicpmo45::pooled_audio_length(values[index]);
        }
        return result;
    }

    tyloi::StepSequence<::tyloi::DenseTensor> forward_steps(
        tyloi::cuda::Projector projector,
        ::tyloi::DenseTensor features,
        ::tyloi::DenseTensor raw_lengths,
        bool use_cache = false) {
        if (features.dim() != 3 || features.size(1) != 80 || raw_lengths.dim() != 1 ||
            raw_lengths.size(0) != features.size(0)) {
            throw std::runtime_error("MiniCPM-o audio input geometry is invalid");
        }
        using DenseTensor = ::tyloi::DenseTensor;
        auto encoding = tyloi::models::minicpmo45::audio_encoder(
            features, layers,
            [&](DenseTensor features) {
                return ::tyloi::cuda::conv1d(
                    features.to(conv1_weight.scalar_type()), conv1_weight, conv1_bias,
                    std::vector<int64_t>{1}, std::vector<int64_t>{1}, std::vector<int64_t>{1}, 1);
            },
            [](DenseTensor x) { return ::tyloi::cuda::gelu(x); },
            [&](DenseTensor hidden) {
                return ::tyloi::cuda::conv1d(hidden, conv2_weight, conv2_bias,
                                                  std::vector<int64_t>{2}, std::vector<int64_t>{1},
                                                  std::vector<int64_t>{1}, 1);
            },
            [&](DenseTensor hidden) {
                hidden = hidden.transpose(1, 2).contiguous();
                const int64_t tokens = hidden.size(1);
                int64_t past = 0;
                if (use_cache && !layers.empty() && layers.front().key_cache.defined()) {
                    past = layers.front().key_cache.size(2);
                }
                if (past + tokens > position_embedding.size(0)) {
                    throw std::runtime_error(
                        "MiniCPM-o Whisper position cache exceeds 1500 frames");
                }
                hidden = hidden + position_embedding.narrow(0, past, tokens);

                // Preserve the official graph: its padding comparison uses the raw
                // Whisper feature lengths against the post-convolution sequence.
                auto length_tensor =
                    raw_lengths.to(::tyloi::kCUDA, ::tyloi::kInt64)
                        .contiguous();
                auto key_positions = ::tyloi::cuda::arange(
                    past + tokens, ::tyloi::TensorOptions()
                                       .device(::tyloi::kCUDA)
                                       .dtype(::tyloi::kInt64));
                auto valid_keys = key_positions.unsqueeze(0) < (length_tensor + past).unsqueeze(1);
                auto query_positions =
                    ::tyloi::cuda::arange(past, past + tokens,
                                               ::tyloi::TensorOptions()
                                                   .device(::tyloi::kCUDA)
                                                   .dtype(::tyloi::kInt64));
                // Official MiniCPM-o 4.5 uses audio_chunk_length=1.0. Whisper emits
                // 50 encoder frames per second, and every query can see its current
                // 50-frame chunk plus all preceding chunks.
                auto chunk_visible =
                    key_positions.unsqueeze(0) <
                    ((query_positions / tyloi::models::minicpmo45::audio_chunk_frames + 1) *
                     tyloi::models::minicpmo45::audio_chunk_frames)
                        .unsqueeze(1);
                auto visible = valid_keys.unsqueeze(1) & chunk_visible.unsqueeze(0);
                auto attention_mask =
                    ::tyloi::cuda::zeros({features.size(0), 1, tokens, past + tokens},
                                              ::tyloi::TensorOptions()
                                                  .device(::tyloi::kCUDA)
                                                  .dtype(::tyloi::kFloat32))
                        .masked_fill(visible.logical_not().unsqueeze(1),
                                     -std::numeric_limits<float>::infinity());
                return std::array<DenseTensor, 2>{hidden, attention_mask};
            },
            [&](auto &layer, DenseTensor hidden, DenseTensor mask) {
                return layer.forward(projector, hidden, mask, use_cache);
            },
            [&](DenseTensor hidden) {
                return minicpmo45_layer_norm(hidden, final_norm_weight, final_norm_bias, 1e-5);
            },
            [&](DenseTensor hidden) { return tyloi::cuda::biased(projector, projector1, hidden); },
            [](DenseTensor hidden) { return ::tyloi::cuda::relu(hidden); },
            [&](DenseTensor hidden) { return tyloi::cuda::biased(projector, projector2, hidden); },
            [&](DenseTensor hidden) {
                return ::tyloi::cuda::avg_pool1d(
                           hidden.transpose(1, 2), std::vector<int64_t>{5}, std::vector<int64_t>{5})
                    .transpose(1, 2)
                    .contiguous();
            });
        while (auto step = encoding.next()) co_yield std::move(step);
    }

    ::tyloi::DenseTensor forward(const tyloi::cuda::Projector& projector,
        ::tyloi::DenseTensor features, ::tyloi::DenseTensor lengths, bool use_cache = false) {
        return tyloi::finish_steps(
            forward_steps(projector, std::move(features), std::move(lengths), use_cache));
    }
};
