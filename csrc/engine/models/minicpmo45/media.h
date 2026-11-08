#pragma once

#include "engine/generation/sequence.h"
#include "engine/ops.h"
#include "engine/models/common/causal_forward.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <span>
#include <vector>

namespace tyloi::models::minicpmo45 {

struct MediaBound {
    int64_t batch = 0, source = 0, begin = 0, end = 0;
};

inline std::vector<MediaBound> media_bounds(std::span<const int64_t> values) {
    require_model(values.size() % 4 == 0, "media bounds require four columns");
    std::vector<MediaBound> result;
    result.reserve(values.size() / 4);
    for (size_t i = 0; i < values.size(); i += 4) {
        MediaBound bound{values[i], values[i + 1], values[i + 2], values[i + 3]};
        require_model(bound.batch >= 0 && bound.source >= 0 && bound.begin >= 0 &&
                          bound.end > bound.begin,
                      "invalid media bound");
        result.push_back(bound);
    }
    return result;
}

template <class DenseTensor> struct MultimodalInputs {
    DenseTensor ids, pixels, patch_mask, target_sizes, audio_features, audio_lengths;
    std::vector<MediaBound> images, audios;
};
template <class DenseTensor> struct MultimodalResult {
    DenseTensor vision_states, image_embeddings, audio_embeddings, input_embeddings;
};

// The model owns modality order, resampling, and replacement geometry. Device
// copies and the actual slice/scatter operations are supplied by the backend.
template <class Ops>
tyloi::StepSequence<MultimodalResult<typename Ops::DenseTensor>>
encode(Ops &ops, MultimodalInputs<typename Ops::DenseTensor> &input) {
    if (ops.rank(input.ids) == 1)
        input.ids = ops.batch_ids(std::move(input.ids));
    require_model(ops.rank(input.ids) == 2, "MiniCPM-o input_ids must have shape [batch,tokens]");
    input.ids = ops.device_ids(std::move(input.ids));
    MultimodalResult<typename Ops::DenseTensor> result;
    result.input_embeddings = ops.embed(input.ids);
    co_yield tyloi::StepState::advanced;
    auto check_bound = [&](const MediaBound &bound, const auto &encoded, int64_t length) {
        require_model(bound.batch >= 0 && bound.batch < ops.size(input.ids, 0) &&
                          bound.source >= 0 && bound.source < ops.size(encoded, 0) &&
                          bound.begin >= 0 && bound.end > bound.begin &&
                          bound.end <= ops.size(input.ids, 1) && bound.end - bound.begin == length,
                      "MiniCPM-o media bound does not match encoded length");
    };
    if (!input.images.empty()) {
        require_model(ops.defined(input.pixels) && ops.defined(input.patch_mask) &&
                          ops.defined(input.target_sizes),
                      "image bounds require image tensors");
        auto vision = ops.vision(input.pixels, input.patch_mask, input.target_sizes);
        while (auto step = vision.next()) {
            if (step.value) result.vision_states = std::move(*step.value);
            else co_yield step.state;
        }
        result.image_embeddings = ops.resample(result.vision_states, input.target_sizes);
        co_yield tyloi::StepState::advanced;
        for (const auto &bound : input.images) {
            check_bound(bound, result.image_embeddings, ops.size(result.image_embeddings, 1));
            ops.scatter(result.input_embeddings, result.image_embeddings, bound);
            co_yield tyloi::StepState::advanced;
        }
    }
    if (!input.audios.empty()) {
        require_model(ops.defined(input.audio_features) && ops.defined(input.audio_lengths),
                      "audio bounds require audio tensors");
        ops.reset_audio();
        auto audio = ops.audio(input.audio_features, input.audio_lengths);
        while (auto step = audio.next()) {
            if (step.value) result.audio_embeddings = std::move(*step.value);
            else co_yield step.state;
        }
        auto lengths = ops.audio_lengths(input.audio_lengths);
        for (const auto &bound : input.audios) {
            require_model(bound.source >= 0 && size_t(bound.source) < lengths.size(),
                          "audio bound source is out of range");
            check_bound(bound, result.audio_embeddings, lengths[bound.source]);
            ops.scatter(result.input_embeddings, result.audio_embeddings, bound);
            co_yield tyloi::StepState::advanced;
        }
    }
    co_yield std::move(result);
}

struct TtsSampling {
    int64_t eos_token = 6561, minimum_steps = 50, top_k = 25, minimum_keep = 0;
    double temperature = 0.8, top_p = 0.85, repetition_penalty = 1.05, min_p = 0.0;
    void validate(int64_t steps, int64_t vocabulary) const {
        require_model(steps > 0 && minimum_steps >= 0 && eos_token >= 0 && eos_token < vocabulary &&
                          std::isfinite(temperature) && temperature > 0.0 && top_p > 0.0 &&
                          top_p <= 1.0 && min_p >= 0.0 && min_p <= 1.0 && top_k >= 3 &&
                          top_k <= vocabulary && std::isfinite(repetition_penalty) &&
                          repetition_penalty > 0.0 && minimum_keep >= 0 && minimum_keep <= top_k,
                      "MiniCPM-o TTS generation limits are invalid");
    }
};

// Official TTS filters before temperature scaling; the reference evaluator
// scales first and keeps at least minimum_keep tokens. These orders differ
// from the text sampler and must not be interchanged.
template <class Ops>
auto sample_tts(Ops &ops, typename Ops::DenseTensor logits, const TtsSampling &config,
                std::span<const typename Ops::DenseTensor> history, int64_t step, bool evaluator) {
    if (evaluator)
        logits = ops.temperature(std::move(logits), config.temperature);
    if (!history.empty() && config.repetition_penalty != 1.0)
        logits =
            ops.penalties(std::move(logits), history.last(std::min<size_t>(16, history.size())),
                          config.repetition_penalty);
    if (step < config.minimum_steps)
        ops.mask_eos(logits, config.eos_token);
    if (evaluator)
        return ops.reference(std::move(logits), config.top_k, config.top_p, config.minimum_keep);
    logits = ops.top_k(std::move(logits), config.top_k);
    if (config.top_p < 1.0)
        logits = ops.top_p(std::move(logits), config.top_p);
    if (config.min_p > 0.0)
        logits = ops.min_p(std::move(logits), config.min_p);
    return ops.sample(ops.temperature(std::move(logits), config.temperature));
}

struct PatchPositions {
    std::vector<int64_t> ids;
    bool all_active = true;
};
inline PatchPositions patch_positions(int64_t batch, int64_t patches, int64_t side,
                                      const int64_t *sizes, const bool *mask) {
    require_model(batch > 0 && patches > 0 && side > 0, "invalid patch geometry");
    PatchPositions result{std::vector<int64_t>(batch * patches, 0), true};
    for (int64_t b = 0; b < batch; ++b) {
        const auto height = sizes[2 * b], width = sizes[2 * b + 1];
        require_model(height > 0 && width > 0 && height <= patches / width,
                      "MiniCPM-o target patch size is invalid");
        int64_t active = 0;
        for (int64_t patch = 0; patch < patches; ++patch) {
            if (!mask[b * patches + patch]) {
                result.all_active = false;
                continue;
            }
            require_model(active < height * width,
                          "MiniCPM-o patch mask has too many active entries");
            const auto row = std::min(side - 1, (active / width) * side / height);
            const auto column = std::min(side - 1, (active % width) * side / width);
            result.ids[b * patches + patch] = row * side + column;
            ++active;
        }
        require_model(active == height * width,
                      "MiniCPM-o patch mask active count disagrees with target size");
    }
    return result;
}

inline int64_t pooled_audio_length(int64_t frames) {
    const auto convolved = (frames - 1) / 2 + 1;
    const auto pooled = (convolved - 5) / 5 + 1;
    require_model(frames > 0 && pooled > 0, "MiniCPM-o audio length is too short");
    return pooled;
}
inline constexpr int64_t audio_chunk_frames = 50;

template <class DenseTensor, class Query, class Key, class Value, class Cache, class Attend,
          class Output>
auto encoder_attention(DenseTensor hidden, bool cached, Query query, Key key, Value value, Cache cache,
                       Attend attend, Output output) {
    auto q = query(hidden);
    auto k = key(hidden);
    auto v = value(hidden);
    if (cached)
        cache(k, v);
    return output(attend(std::move(q), std::move(k), std::move(v)));
}

template <class DenseTensor, class Layers, class Conv1, class Gelu, class Conv2, class Prepare,
          class Layer, class Normalize, class Project1, class Relu, class Project2, class Pool>
tyloi::StepSequence<DenseTensor> audio_encoder(DenseTensor features, Layers &layers, Conv1 conv1, Gelu gelu, Conv2 conv2,
                   Prepare prepare, Layer layer, Normalize normalize, Project1 project1, Relu relu,
                   Project2 project2, Pool pool) {
    auto hidden = gelu(conv1(std::move(features)));
    hidden = gelu(conv2(std::move(hidden)));
    auto prepared = prepare(std::move(hidden));
    hidden = std::move(prepared[0]);
    co_yield tyloi::StepState::advanced;
    for (auto& block : layers) {
        hidden = layer(block, std::move(hidden), prepared[1]);
        co_yield tyloi::StepState::advanced;
    }
    co_yield pool(project2(relu(project1(normalize(std::move(hidden))))));
}

template <class DenseTensor, class Layers, class Patch, class Prepare, class Layer, class Normalize>
tyloi::StepSequence<DenseTensor> vision_encoder(DenseTensor pixels, Layers &layers, Patch patch, Prepare prepare, Layer layer,
                    Normalize normalize) {
    auto hidden = prepare(patch(std::move(pixels)));
    co_yield tyloi::StepState::advanced;
    for (const auto& block : layers) {
        hidden = layer(block, std::move(hidden));
        co_yield tyloi::StepState::advanced;
    }
    co_yield normalize(std::move(hidden));
}

template <class ProjectKv, class NormKv, class NormQ, class Query, class Key, class Value,
          class Attend, class Output, class NormOutput, class Final>
auto resample(ProjectKv project_kv, NormKv norm_kv, NormQ norm_q, Query query, Key key, Value value,
              Attend attend, Output output, NormOutput norm_output, Final final) {
    auto kv = norm_kv(project_kv());
    auto q = query(norm_q());
    auto k = key(kv);
    auto v = value(kv);
    return final(norm_output(output(attend(std::move(q), std::move(k), std::move(v)))));
}

} // namespace tyloi::models::minicpmo45
