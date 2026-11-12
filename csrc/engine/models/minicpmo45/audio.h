#pragma once

#include "engine/models/minicpmo45/media.h"
#include "engine/ops.h"
#include "vision.h"
#include <stdexcept>

struct MiniCPMO45WhisperAttention {
    tyloi::weights::BiasWeight q;
    tyloi::weights::BiasWeight k;
    tyloi::weights::BiasWeight v;
    tyloi::weights::BiasWeight output;
    int64_t heads = 16;
    int64_t head_dim = 64;

    static MiniCPMO45WhisperAttention load(const tyloi::weights::Loader& weights, const std::string &prefix) {
        MiniCPMO45WhisperAttention result;
        result.q = weights.biased(prefix + ".query.weight", prefix + ".query.bias");
        result.k = weights.biased(prefix + ".key.weight");
        result.v = weights.biased(prefix + ".value.weight", prefix + ".value.bias");
        result.output = weights.biased(prefix + ".output.weight", prefix + ".output.bias");
        return result;
    }

    ::tyloi::DenseTensor forward(
#ifndef TYLOINFER_NATIVE_CUDA
                                       const tyloi::cuda::Projector& projector,
#endif
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
#ifdef TYLOINFER_NATIVE_CUDA
            [&](const DenseTensor &x) { return reshape(tyloi::ops::linear(x, q.weight, q.bias).dense()); },
            [&](const DenseTensor &x) { return reshape(tyloi::ops::linear(x, k.weight, k.bias).dense()); },
            [&](const DenseTensor &x) { return reshape(tyloi::ops::linear(x, v.weight, v.bias).dense()); },
#else
            [&](const DenseTensor &x) { return reshape(tyloi::cuda::biased(projector, q, x)); },
            [&](const DenseTensor &x) { return reshape(tyloi::cuda::biased(projector, k, x)); },
            [&](const DenseTensor &x) { return reshape(tyloi::cuda::biased(projector, v, x)); },
#endif
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
#ifdef TYLOINFER_NATIVE_CUDA
                return tyloi::ops::linear(attended, output.weight, output.bias).dense();
#else
                return tyloi::cuda::biased(projector, output, attended);
#endif
            });
    }
};

struct MiniCPMO45WhisperLayer {
    MiniCPMO45WhisperAttention attention;
#ifdef TYLOINFER_NATIVE_CUDA
    ::tyloi::Tensor attention_norm_weight, attention_norm_bias;
    ::tyloi::Tensor final_norm_weight, final_norm_bias;
#else
    ::tyloi::DenseTensor attention_norm_weight;
    ::tyloi::DenseTensor attention_norm_bias;
    ::tyloi::DenseTensor final_norm_weight;
    ::tyloi::DenseTensor final_norm_bias;
#endif
    tyloi::weights::BiasWeight fc1;
    tyloi::weights::BiasWeight fc2;
#ifndef TYLOINFER_NATIVE_CUDA
    ::tyloi::DenseTensor key_cache;
    ::tyloi::DenseTensor value_cache;
#endif

    static MiniCPMO45WhisperLayer load(const tyloi::weights::Loader& weights,
                                       int index) {
        const std::string prefix = "audio.block." + std::to_string(index);
        MiniCPMO45WhisperLayer result;
        result.attention = MiniCPMO45WhisperAttention::load(weights, prefix + ".attention");
#ifdef TYLOINFER_NATIVE_CUDA
        result.attention_norm_weight = weights.native(prefix + ".attention.norm.weight");
        result.attention_norm_bias = weights.native(prefix + ".attention.norm.bias");
        result.final_norm_weight = weights.native(prefix + ".mlp.norm.weight");
        result.final_norm_bias = weights.native(prefix + ".mlp.norm.bias");
#else
        result.attention_norm_weight =
            weights.native(prefix + ".attention.norm.weight").dense();
        result.attention_norm_bias =
            weights.native(prefix + ".attention.norm.bias").dense();
        result.final_norm_weight =
            weights.native(prefix + ".mlp.norm.weight").dense();
        result.final_norm_bias = weights.native(prefix + ".mlp.norm.bias").dense();
#endif
        result.fc1 = weights.biased(prefix + ".mlp.up.weight", prefix + ".mlp.up.bias");
        result.fc2 = weights.biased(prefix + ".mlp.down.weight", prefix + ".mlp.down.bias");
        return result;
    }

#ifndef TYLOINFER_NATIVE_CUDA
    void reset() {
        key_cache = ::tyloi::DenseTensor();
        value_cache = ::tyloi::DenseTensor();
    }
#endif

    ::tyloi::DenseTensor forward(
#ifdef TYLOINFER_NATIVE_CUDA
                                       ::tyloi::DenseTensor& key_cache,
                                       ::tyloi::DenseTensor& value_cache,
#else
                                       const tyloi::cuda::Projector& projector,
#endif
                                       ::tyloi::DenseTensor input,
                                       MfqOptional<::tyloi::DenseTensor> mask,
                                       bool use_cache)
#ifdef TYLOINFER_NATIVE_CUDA
        const
#endif
        {
#ifdef TYLOINFER_NATIVE_CUDA
        auto normalized = minicpmo45_layer_norm(input, attention_norm_weight, attention_norm_bias, 1e-5);
#else
        auto normalized = minicpmo45_layer_norm(input, attention_norm_weight, attention_norm_bias, 1e-5);
#endif
        auto branch = attention.forward(
#ifndef TYLOINFER_NATIVE_CUDA
                                        projector,
#endif
                                        normalized, mask, use_cache ? &key_cache : nullptr,
                                        use_cache ? &value_cache : nullptr);
        input = (input + branch).contiguous();
#ifdef TYLOINFER_NATIVE_CUDA
        normalized = minicpmo45_layer_norm(input, final_norm_weight, final_norm_bias, 1e-5);
#else
        normalized = minicpmo45_layer_norm(input, final_norm_weight, final_norm_bias, 1e-5);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
        branch = tyloi::ops::linear(::tyloi::cuda::gelu(
            tyloi::ops::linear(normalized, fc1.weight, fc1.bias).dense()), fc2.weight, fc2.bias).dense();
#else
        branch = tyloi::cuda::biased(projector, fc2,
            ::tyloi::cuda::gelu(tyloi::cuda::biased(projector, fc1, normalized)));
#endif
        return (input + branch).contiguous();
    }
};

struct MiniCPMO45AudioEncoder {
#ifdef TYLOINFER_NATIVE_CUDA
    /** Own one stream's independent Whisper key and value history. */
    struct State {
        struct Layer { ::tyloi::DenseTensor key, value; };
        std::vector<Layer> layers;
        explicit State(std::size_t count) : layers(count) {}
        /** Discard accumulated streaming history. */
        void reset() { for (auto& layer : layers) { layer.key = {}; layer.value = {}; } }
        /** Return the number of cached encoder frames. */
        std::int64_t length() const {
            return layers.empty() || !layers.front().key.defined() ? 0 : layers.front().key.size(2);
        }
    };
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    ::tyloi::Tensor conv1_weight, conv1_bias, conv2_weight, conv2_bias, position_embedding;
#else
    ::tyloi::DenseTensor conv1_weight;
    ::tyloi::DenseTensor conv1_bias;
    ::tyloi::DenseTensor conv2_weight;
    ::tyloi::DenseTensor conv2_bias;
    ::tyloi::DenseTensor position_embedding;
#endif
    std::vector<MiniCPMO45WhisperLayer> layers;
#ifdef TYLOINFER_NATIVE_CUDA
    ::tyloi::Tensor final_norm_weight, final_norm_bias;
#else
    ::tyloi::DenseTensor final_norm_weight;
    ::tyloi::DenseTensor final_norm_bias;
#endif
    tyloi::weights::BiasWeight projector1;
    tyloi::weights::BiasWeight projector2;

    static MiniCPMO45AudioEncoder load(const tyloi::weights::Loader& weights) {
        MiniCPMO45AudioEncoder result;
#ifdef TYLOINFER_NATIVE_CUDA
        result.conv1_weight = weights.native("audio.patch_embedding.conv1.weight");
        result.conv1_bias = weights.native("audio.patch_embedding.conv1.bias");
        result.conv2_weight = weights.native("audio.patch_embedding.conv2.weight");
        result.conv2_bias = weights.native("audio.patch_embedding.conv2.bias");
        result.position_embedding = weights.native("audio.position_embedding.weight");
        result.final_norm_weight = weights.native("audio.output_norm.weight");
        result.final_norm_bias = weights.native("audio.output_norm.bias");
#else
        result.conv1_weight =
            weights.native("audio.patch_embedding.conv1.weight").dense();
        result.conv1_bias =
            weights.native("audio.patch_embedding.conv1.bias").dense();
        result.conv2_weight =
            weights.native("audio.patch_embedding.conv2.weight").dense();
        result.conv2_bias =
            weights.native("audio.patch_embedding.conv2.bias").dense();
        result.position_embedding =
            weights.native("audio.position_embedding.weight").dense();
        result.final_norm_weight =
            weights.native("audio.output_norm.weight").dense();
        result.final_norm_bias = weights.native("audio.output_norm.bias").dense();
#endif
        result.projector1 = weights.biased("audio.projector.input.weight", "audio.projector.input.bias");
        result.projector2 = weights.biased("audio.projector.output.weight", "audio.projector.output.bias");
        result.layers.reserve(24);
        for (int index = 0; index < 24; ++index) {
            result.layers.push_back(MiniCPMO45WhisperLayer::load(weights, index));
        }
#ifdef TYLOINFER_NATIVE_CUDA
        if (result.conv1_weight.dim() != 3 || result.conv1_weight.size(0) != 1024 ||
            result.conv1_weight.size(1) != 80 || result.conv1_weight.size(2) != 3 ||
            result.conv2_weight.dim() != 3 || result.conv2_weight.size(0) != 1024 ||
            result.conv2_weight.size(1) != 1024 || result.conv2_weight.size(2) != 3 ||
            result.position_embedding.dim() != 2 || result.position_embedding.size(0) != 1500 ||
            result.position_embedding.size(1) != 1024) {
#else
        if (result.conv1_weight.sizes().vec() != std::vector<int64_t>({1024, 80, 3}) ||
            result.conv2_weight.sizes().vec() != std::vector<int64_t>({1024, 1024, 3}) ||
            result.position_embedding.sizes().vec() != std::vector<int64_t>({1500, 1024})) {
#endif
            throw std::runtime_error("MiniCPM-o Whisper tensor shapes disagree with version 4.5");
        }
        return result;
    }

#ifndef TYLOINFER_NATIVE_CUDA
    void reset() {
        for (auto &layer : layers)
            layer.reset();
    }

    int64_t cache_length() const {
        if (layers.empty() || !layers.front().key_cache.defined())
            return 0;
        return layers.front().key_cache.size(2);
    }
#endif

#ifdef TYLOINFER_NATIVE_CUDA
    /** Encode one audio chunk and retain the supplied stream history. */
    ::tyloi::Tensor forward_streaming(State& state,
                                      ::tyloi::Tensor features_value,
                                      int64_t prefix_extra_frames, int64_t suffix_extra_frames) const {
#else
    ::tyloi::DenseTensor forward_streaming(const tyloi::cuda::Projector& projector,
                                           ::tyloi::DenseTensor features_value,
                                           int64_t prefix_extra_frames, int64_t suffix_extra_frames) {
#endif
#ifdef TYLOINFER_NATIVE_CUDA
        if (!features_value.defined())
            throw std::runtime_error("MiniCPM-o streaming audio expects [1,80,frames] and "
                                     "non-negative extra-frame counts");
        if (state.layers.size() != layers.size())
            throw std::invalid_argument("MiniCPM-o audio state has incompatible layers");
        const auto& features = features_value.dense();
#else
        const auto& features = features_value;
#endif
        if (features.dim() != 3 || features.size(0) != 1 || features.size(1) != 80 ||
            prefix_extra_frames < 0 || suffix_extra_frames < 0) {
            throw std::runtime_error("MiniCPM-o streaming audio expects [1,80,frames] and "
                                     "non-negative extra-frame counts");
        }
        const int64_t conv_tokens_before_crop = (features.size(2) - 1) / 2 + 1;
#ifdef TYLOINFER_NATIVE_CUDA
        if (state.length() + conv_tokens_before_crop >= position_embedding.size(0)) {
#else
        if (cache_length() + conv_tokens_before_crop >= position_embedding.size(0)) {
#endif
#ifdef TYLOINFER_NATIVE_CUDA
            state.reset();
#else
            reset();
#endif
        }

        using DenseTensor = ::tyloi::DenseTensor;
        return tyloi::finish_steps(tyloi::models::minicpmo45::audio_encoder(
            features, layers,
            [&](DenseTensor features) {
#ifdef TYLOINFER_NATIVE_CUDA
                const auto& weight = conv1_weight.dense();
                const auto& bias = conv1_bias.dense();
#else
                const auto& weight = conv1_weight;
                const auto& bias = conv1_bias;
#endif
                return ::tyloi::cuda::conv1d(
                    features.to(weight.scalar_type()), weight, bias,
                    std::vector<int64_t>{1}, std::vector<int64_t>{1}, std::vector<int64_t>{1}, 1);
            },
            [](DenseTensor x) { return ::tyloi::cuda::gelu(x); },
            [&](DenseTensor hidden) {
#ifdef TYLOINFER_NATIVE_CUDA
                const auto& weight = conv2_weight.dense();
                const auto& bias = conv2_bias.dense();
#else
                const auto& weight = conv2_weight;
                const auto& bias = conv2_bias;
#endif
                return ::tyloi::cuda::conv1d(hidden, weight, bias,
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

#ifdef TYLOINFER_NATIVE_CUDA
                const int64_t past = state.length();
#else
                const int64_t past = cache_length();
#endif
                const int64_t tokens = hidden.size(1);
#ifdef TYLOINFER_NATIVE_CUDA
                if (past + tokens > position_embedding.size(0)) {
#else
                if (past + tokens > position_embedding.size(0)) {
#endif
                    throw std::runtime_error(
                        "MiniCPM-o streaming Whisper position cache exceeds 1500 frames");
                }
#ifdef TYLOINFER_NATIVE_CUDA
                const auto& position = position_embedding.dense();
#else
                const auto& position = position_embedding;
#endif
                hidden = hidden + position.narrow(0, past, tokens);
                auto attention_mask =
                    ::tyloi::cuda::zeros({1, 1, tokens, past + tokens}, hidden.options());
                return std::array<DenseTensor, 2>{hidden, attention_mask};
            },
            [&](auto &layer, DenseTensor hidden, DenseTensor mask) {
#ifdef TYLOINFER_NATIVE_CUDA
                auto& cache = state.layers[std::size_t(&layer - layers.data())];
                return layer.forward(cache.key, cache.value, hidden, mask, true);
#else
                return layer.forward(projector, hidden, mask, true);
#endif
            },
            [&](DenseTensor hidden) {
#ifdef TYLOINFER_NATIVE_CUDA
                return minicpmo45_layer_norm(hidden, final_norm_weight, final_norm_bias, 1e-5);
#else
                return minicpmo45_layer_norm(hidden, final_norm_weight, final_norm_bias, 1e-5);
#endif
            },
#ifdef TYLOINFER_NATIVE_CUDA
            [&](DenseTensor hidden) { return tyloi::ops::linear(hidden, projector1.weight, projector1.bias).dense(); },
#else
            [&](DenseTensor hidden) { return tyloi::cuda::biased(projector, projector1, hidden); },
#endif
            [](DenseTensor hidden) { return ::tyloi::cuda::relu(hidden); },
#ifdef TYLOINFER_NATIVE_CUDA
            [&](DenseTensor hidden) { return tyloi::ops::linear(hidden, projector2.weight, projector2.bias).dense(); },
#else
            [&](DenseTensor hidden) { return tyloi::cuda::biased(projector, projector2, hidden); },
#endif
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

#ifdef TYLOINFER_NATIVE_CUDA
    static std::vector<int64_t> pooled_lengths(::tyloi::Tensor raw_lengths) {
#else
    static std::vector<int64_t> pooled_lengths(::tyloi::DenseTensor raw_lengths) {
#endif
#ifdef TYLOINFER_NATIVE_CUDA
        auto lengths = raw_lengths.dense().to(::tyloi::kCPU, ::tyloi::kInt64).contiguous();
#else
        auto lengths = raw_lengths.to(::tyloi::kCPU, ::tyloi::kInt64).contiguous();
#endif
        const auto *values = lengths.data_ptr<int64_t>();
        std::vector<int64_t> result(static_cast<size_t>(lengths.numel()));
        for (int64_t index = 0; index < lengths.numel(); ++index) {
            result[static_cast<size_t>(index)] =
                tyloi::models::minicpmo45::pooled_audio_length(values[index]);
        }
        return result;
    }

#ifdef TYLOINFER_NATIVE_CUDA
    /** Encode audio while retaining optional history in the supplied state. */
    tyloi::StepSequence<::tyloi::Tensor> forward_steps(
        State& state, ::tyloi::Tensor features_value,
        ::tyloi::Tensor raw_lengths_value, bool use_cache = false) const {
#else
    tyloi::StepSequence<::tyloi::DenseTensor> forward_steps(
        tyloi::cuda::Projector projector, ::tyloi::DenseTensor features_value,
        ::tyloi::DenseTensor raw_lengths_value, bool use_cache = false) {
#endif
#ifdef TYLOINFER_NATIVE_CUDA
        if (!features_value.defined() || !raw_lengths_value.defined())
            throw std::runtime_error("MiniCPM-o audio input geometry is invalid");
        if (state.layers.size() != layers.size())
            throw std::invalid_argument("MiniCPM-o audio state has incompatible layers");
        const auto& features = features_value.dense();
        const auto& raw_lengths = raw_lengths_value.dense();
#else
        const auto& features = features_value;
        const auto& raw_lengths = raw_lengths_value;
#endif
        if (features.dim() != 3 || features.size(1) != 80 || raw_lengths.dim() != 1 ||
            raw_lengths.size(0) != features.size(0)) {
            throw std::runtime_error("MiniCPM-o audio input geometry is invalid");
        }
        using DenseTensor = ::tyloi::DenseTensor;
        auto encoding = tyloi::models::minicpmo45::audio_encoder(
            features, layers,
            [&](DenseTensor features) {
#ifdef TYLOINFER_NATIVE_CUDA
                const auto& weight = conv1_weight.dense();
                const auto& bias = conv1_bias.dense();
#else
                const auto& weight = conv1_weight;
                const auto& bias = conv1_bias;
#endif
                return ::tyloi::cuda::conv1d(
                    features.to(weight.scalar_type()), weight, bias,
                    std::vector<int64_t>{1}, std::vector<int64_t>{1}, std::vector<int64_t>{1}, 1);
            },
            [](DenseTensor x) { return ::tyloi::cuda::gelu(x); },
            [&](DenseTensor hidden) {
#ifdef TYLOINFER_NATIVE_CUDA
                const auto& weight = conv2_weight.dense();
                const auto& bias = conv2_bias.dense();
#else
                const auto& weight = conv2_weight;
                const auto& bias = conv2_bias;
#endif
                return ::tyloi::cuda::conv1d(hidden, weight, bias,
                                                  std::vector<int64_t>{2}, std::vector<int64_t>{1},
                                                  std::vector<int64_t>{1}, 1);
            },
            [&](DenseTensor hidden) {
                hidden = hidden.transpose(1, 2).contiguous();
                const int64_t tokens = hidden.size(1);
                int64_t past = 0;
#ifdef TYLOINFER_NATIVE_CUDA
                if (use_cache) past = state.length();
#else
                if (use_cache && !layers.empty() && layers.front().key_cache.defined()) {
                    past = layers.front().key_cache.size(2);
                }
#endif
#ifdef TYLOINFER_NATIVE_CUDA
                if (past + tokens > position_embedding.size(0)) {
#else
                if (past + tokens > position_embedding.size(0)) {
#endif
                    throw std::runtime_error(
                        "MiniCPM-o Whisper position cache exceeds 1500 frames");
                }
#ifdef TYLOINFER_NATIVE_CUDA
                const auto& position = position_embedding.dense();
#else
                const auto& position = position_embedding;
#endif
                hidden = hidden + position.narrow(0, past, tokens);

                // Preserve the official graph: its padding comparison uses the raw
                // Whisper feature lengths against the post-convolution sequence.
                auto length_tensor =
                    raw_lengths.to(hidden.device(), ::tyloi::kInt64)
                        .contiguous();
                auto key_positions = ::tyloi::cuda::arange(
                    past + tokens, ::tyloi::TensorOptions()
                                       .device(hidden.device())
                                       .dtype(::tyloi::kInt64));
                auto valid_keys = key_positions.unsqueeze(0) < (length_tensor + past).unsqueeze(1);
                auto query_positions =
                    ::tyloi::cuda::arange(past, past + tokens,
                                               ::tyloi::TensorOptions()
                                                   .device(hidden.device())
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
                                                  .device(hidden.device())
                                                  .dtype(::tyloi::kFloat32))
                        .masked_fill(visible.logical_not().unsqueeze(1),
                                     -std::numeric_limits<float>::infinity());
                return std::array<DenseTensor, 2>{hidden, attention_mask};
            },
            [&](auto &layer, DenseTensor hidden, DenseTensor mask) {
#ifdef TYLOINFER_NATIVE_CUDA
                auto& cache = state.layers[std::size_t(&layer - layers.data())];
                return layer.forward(cache.key, cache.value, hidden, mask, use_cache);
#else
                return layer.forward(projector, hidden, mask, use_cache);
#endif
            },
            [&](DenseTensor hidden) {
#ifdef TYLOINFER_NATIVE_CUDA
                return minicpmo45_layer_norm(hidden, final_norm_weight, final_norm_bias, 1e-5);
#else
                return minicpmo45_layer_norm(hidden, final_norm_weight, final_norm_bias, 1e-5);
#endif
            },
#ifdef TYLOINFER_NATIVE_CUDA
            [&](DenseTensor hidden) { return tyloi::ops::linear(hidden, projector1.weight, projector1.bias).dense(); },
#else
            [&](DenseTensor hidden) { return tyloi::cuda::biased(projector, projector1, hidden); },
#endif
            [](DenseTensor hidden) { return ::tyloi::cuda::relu(hidden); },
#ifdef TYLOINFER_NATIVE_CUDA
            [&](DenseTensor hidden) { return tyloi::ops::linear(hidden, projector2.weight, projector2.bias).dense(); },
#else
            [&](DenseTensor hidden) { return tyloi::cuda::biased(projector, projector2, hidden); },
#endif
            [&](DenseTensor hidden) {
                return ::tyloi::cuda::avg_pool1d(
                           hidden.transpose(1, 2), std::vector<int64_t>{5}, std::vector<int64_t>{5})
                    .transpose(1, 2)
                    .contiguous();
            });
        while (auto step = encoding.next()) {
            if (step.layer || step.graph) throw std::logic_error("audio encoding does not support layer or graph events");
#ifdef TYLOINFER_NATIVE_CUDA
            if (step.value) co_yield ::tyloi::Tensor(std::move(*step.value));
#else
            if (step.value) co_yield std::move(*step.value);
#endif
            else co_yield step.state;
        }
    }

#ifdef TYLOINFER_NATIVE_CUDA
    /** Encode audio to completion using the supplied state. */
    ::tyloi::Tensor forward(State& state,
        ::tyloi::Tensor features, ::tyloi::Tensor lengths, bool use_cache = false) const {
#else
    ::tyloi::DenseTensor forward(const tyloi::cuda::Projector& projector,
        ::tyloi::DenseTensor features, ::tyloi::DenseTensor lengths, bool use_cache = false) {
#endif
        return tyloi::finish_steps(
#ifdef TYLOINFER_NATIVE_CUDA
            forward_steps(state, std::move(features), std::move(lengths), use_cache));
#else
            forward_steps(projector, std::move(features), std::move(lengths), use_cache));
#endif
    }
};
