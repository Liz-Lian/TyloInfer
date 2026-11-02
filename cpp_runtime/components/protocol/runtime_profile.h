#pragma once

#include "engine.h"

#include <optional>
#include <string>

struct MfqChatSamplingProfile {
    std::optional<int32_t> max_tokens;
    std::optional<double> temperature;
    std::optional<int32_t> top_k;
    std::optional<double> top_p;
    std::optional<double> presence_penalty;
    std::optional<double> frequency_penalty;
    std::optional<double> repetition_penalty;
    std::optional<bool> enable_thinking;
    std::optional<bool> enable_vision;
    std::optional<bool> enable_mtp;
    std::optional<int32_t> mtp_max_draft_tokens;
};

struct MfqDuplexSamplingProfile {
    std::optional<std::string> system_prompt;
    std::optional<std::string> decode_mode;
    std::optional<double> temperature;
    std::optional<int32_t> top_k;
    std::optional<double> top_p;
    std::optional<double> text_repetition_penalty;
    std::optional<int32_t> text_repetition_window_size;
    std::optional<double> length_penalty;
    std::optional<double> listen_prob_scale;
    std::optional<int32_t> force_listen_count;
    std::optional<int32_t> max_new_speak_tokens_per_chunk;
};

struct MfqTtsSamplingProfile {
    std::optional<double> temperature;
    std::optional<double> repetition_penalty;
    std::optional<int32_t> token2wav_steps;
};

struct MfqRuntimeProfile {
    MfqChatSamplingProfile chat;
    MfqDuplexSamplingProfile duplex;
    MfqTtsSamplingProfile tts;
    std::string source = "generic-defaults";
};

struct MfqRuntimeTransportConfig {
    std::string model_name = "mfq-model";
    std::string model_type;
    int64_t max_context = 0;
    int64_t context_capacity = 0;
    int64_t vocab_size = 0;
    std::optional<MfqModelCapabilities> model_capabilities;
    MfqRuntimeProfile runtime_profile;
};

MfqRuntimeProfile resolve_mfq_runtime_profile(
    const std::string & mfq_path,
    const std::string & model_architecture,
    const std::string & model_type,
    const std::string & model_name,
    const std::string & embedded_profile_json = {},
    const std::string & model_config_json = {},
    const std::string & explicit_profile_path = {});

MfqSamplingParams default_sampling_params(const MfqRuntimeProfile& profile);
