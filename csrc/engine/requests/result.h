#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

struct MfqDuplexStepResult {
    std::vector<int64_t> generated_tokens;
    std::vector<int32_t> audio_tokens;
    bool is_listen = false;
    bool end_of_turn = false;
    bool tts_force_flush = false;
    int64_t audio_chunk_index = 0;
    int64_t language_cache_position = 0;
    int64_t audio_cache_position = 0;
    int64_t tts_cache_position = 0;
    double inference_ms = 0.0;
};

struct MfqPrefillTiming {
    size_t prompt_tokens = 0;
    // Language-model prompt evaluation only.
    double llm_ms = 0.0;
    // Vision/audio encoder and multimodal projector work preceding the LLM.
    double multimodal_ms = 0.0;
    // Complete model-side prefill wall time.
    double model_ms = 0.0;
};
