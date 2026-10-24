#pragma once

#include <cstdint>
#include <string>
#include <vector>

enum class MfqMultimodalProcessor {
    minicpmo,
    deepseek_v4,
    deepseek_v41,
    grid_vision,
};

// Architecture-neutral named media payload. Family-specific fields are
// validated before this reaches a runtime; prompt-position dependent metadata
// is produced after tokenizing.
struct MfqMultimodalInput {
    MfqMultimodalProcessor processor = MfqMultimodalProcessor::minicpmo;
    std::string processor_name;
    std::vector<float> pixel_values;
    std::vector<int64_t> pixel_shape;
    std::vector<uint8_t> patch_mask;
    std::vector<int64_t> patch_mask_shape;
    std::vector<int32_t> target_sizes;
    std::vector<int64_t> target_sizes_shape;
    std::vector<int32_t> vision_grid;
    std::vector<int64_t> vision_grid_shape;
    std::vector<int32_t> vision_types;
    std::vector<int32_t> image_grid;
    std::vector<int64_t> image_grid_shape;
    std::vector<int32_t> video_grid;
    std::vector<int64_t> video_grid_shape;
    std::vector<int64_t> image_bounds;
    std::vector<int64_t> image_permutation;
    std::vector<int64_t> image_permutation_offsets;
    std::vector<float> audio_features;
    std::vector<int64_t> audio_features_shape;
    std::vector<int64_t> audio_lengths;
    std::vector<int64_t> audio_bounds;
};

// Source compatibility for out-of-tree MiniCPM-o integrations.
using MfqVisionInput = MfqMultimodalInput;

struct MfqDuplexSessionParams {
    std::vector<int64_t> system_prefix;
    std::vector<int64_t> system_suffix;
    std::vector<float> reference_audio_features;
    int32_t reference_audio_frames = 0;
    std::vector<int64_t> special_ids;
    std::vector<int64_t> forbidden_ids;
    bool greedy = false;
    double temperature = 0.7;
    int32_t top_k = 100;
    double top_p = 0.8;
    double listen_probability_scale = 1.0;
    double repetition_penalty = 1.05;
    int32_t repetition_window = 512;
    double length_penalty = 1.0;
    double tts_temperature = 0.8;
    double tts_repetition_penalty = 1.05;
    uint64_t seed = 0;
};

struct MfqDuplexStepInput {
    std::vector<float> audio_features;
    std::vector<int64_t> text_tokens;
    int32_t audio_frames = 0;
    int64_t audio_prefix_extra_frames = 0;
    int64_t audio_suffix_extra_frames = 0;
    int32_t max_new_speak_tokens = 20;
    bool force_listen = false;
    bool force_speak = false;
};
