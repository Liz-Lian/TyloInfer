#include "engine/backends/cuda/executor/execution.h"
#include "components.h"
#include "runtime.h"

#include "engine/backends/cuda/executor/sampling.h"
#include "backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

MiniCPMO45Runtime MiniCPMO45Runtime::load(
        const mfq::cuda::minicpmo45::text::Model& language,
        mfq::cuda::minicpmo45::text::State& language_state,
        mfq::cuda::Executor& language_executor,
        const mfq::cuda::weight_loader::Loader& weights,
        CudaExecutionContext& execution) {
    auto &placement = execution.config.layer_placement;
    MfqCudaGuard guard(placement.primary_device());
    auto projector = execution.executor().projector(execution.executor().mode().ffn, model_parallel_primary_device(execution.config));
    return {language, language_state, language_executor, execution,
            MiniCPMO45VisionEncoder::load(weights),
            MiniCPMO45Resampler::load(weights),
            MiniCPMO45AudioEncoder::load(weights),
            mfq::cuda::minicpmo45::Tts::load(weights),
            execution.executor().fork(model_parallel_primary_device(execution.config)), execution.executor().forward_mode(execution.config.layer_placement.primary_device()),
            projector};
}

namespace mfq::cuda::minicpmo45 {

struct Components::State {
    MiniCPMO45Runtime runtime;
    std::optional<MiniCPMO45DuplexSession> duplex_session;

    State(const text::Model& language, text::State& language_state,
          Executor& language_executor, const weight_loader::Loader& weights,
          CudaExecutionContext& execution)
        : runtime(MiniCPMO45Runtime::load(language, language_state, language_executor,
                                         weights, execution)) {}
};

Components::Components(const text::Model& language, text::State& language_state,
                       Executor& language_executor, const weight_loader::Loader& weights,
                       CudaExecutionContext& execution)
    : state_(std::make_unique<State>(language, language_state, language_executor, weights,
                                     execution)) {}

Components::~Components() = default;

mfq::StepSequence<PreparedPrompt> Components::prepare(
        const std::vector<int64_t>& prompt, const MfqMultimodalInput& vision) {
    const auto cpu_i64 =
        mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCPU);
    const auto cuda_i64 =
        mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCUDA);
    auto input_ids = mfq_tensor_backend::tensor(prompt, cuda_i64)
        .reshape({1, -1}).contiguous();
    mfq_tensor_backend::Tensor pixels;
    mfq_tensor_backend::Tensor patch_mask;
    mfq_tensor_backend::Tensor target_sizes;
    mfq_tensor_backend::Tensor image_bounds;
    if (!vision.image_bounds.empty()) {
        pixels = mfq_tensor_backend::from_blob(
            const_cast<float *>(vision.pixel_values.data()),
            vision.pixel_shape,
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32).device(mfq_tensor_backend::kCPU))
            .clone();
        patch_mask = mfq_tensor_backend::from_blob(
            const_cast<uint8_t *>(vision.patch_mask.data()),
            vision.patch_mask_shape,
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kUInt8).device(mfq_tensor_backend::kCPU))
            .clone().to(mfq_tensor_backend::kBool);
        target_sizes = mfq_tensor_backend::from_blob(
            const_cast<int32_t *>(vision.target_sizes.data()),
            vision.target_sizes_shape,
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32).device(mfq_tensor_backend::kCPU))
            .clone();
        image_bounds = mfq_tensor_backend::from_blob(
            const_cast<int64_t *>(vision.image_bounds.data()),
            std::vector<int64_t>{
                static_cast<int64_t>(vision.image_bounds.size() / 4), 4},
            cpu_i64).clone();
    }
    mfq_tensor_backend::Tensor audio_features;
    mfq_tensor_backend::Tensor audio_lengths;
    mfq_tensor_backend::Tensor audio_bounds;
    if (!vision.audio_bounds.empty()) {
        audio_features = mfq_tensor_backend::from_blob(
            const_cast<float *>(vision.audio_features.data()),
            vision.audio_features_shape,
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32).device(mfq_tensor_backend::kCPU))
            .clone();
        audio_lengths = mfq_tensor_backend::tensor(
            vision.audio_lengths, cpu_i64).contiguous();
        audio_bounds = mfq_tensor_backend::from_blob(
            const_cast<int64_t *>(vision.audio_bounds.data()),
            std::vector<int64_t>{
                static_cast<int64_t>(vision.audio_bounds.size() / 4), 4},
            cpu_i64).clone();
    }


    co_yield mfq::StepState::advanced;
    auto encoding = state_->runtime.encode(input_ids, pixels, patch_mask,
        target_sizes, image_bounds, audio_features, audio_lengths, audio_bounds);
    mfq::models::minicpmo45::MultimodalResult<mfq_tensor_backend::Tensor> result;
    while (auto step = encoding.next()) {
        if (step.value) result = std::move(*step.value);
        else co_yield step.state;
    }
    PreparedPrompt prepared;
    prepared.token_ids = prompt;
    prepared.embeddings = std::move(result.input_embeddings);
    prepared.positions = mfq_tensor_backend::arange(static_cast<int64_t>(prompt.size()), cuda_i64);
    co_yield std::move(prepared);
}

void Components::start(const MfqDuplexSessionParams& parameters) {
        if (parameters.special_ids.size() != 15) {
            throw std::invalid_argument(
                "MiniCPM-o duplex requires 15 special token IDs");
        }
        if ((!parameters.greedy &&
                (!std::isfinite(parameters.temperature) ||
                 parameters.temperature <= 0.0)) ||
                parameters.top_k < 0 ||
                parameters.top_k >
                    std::min<int64_t>(
                        state_->runtime.language.metadata.vocab_size, 1024) ||
                !std::isfinite(parameters.top_p) ||
                parameters.top_p <= 0.0 || parameters.top_p > 1.0 ||
                !std::isfinite(parameters.listen_probability_scale) ||
                parameters.listen_probability_scale < 0.0 ||
                !std::isfinite(parameters.repetition_penalty) ||
                parameters.repetition_penalty <= 0.0 ||
                parameters.repetition_window <= 0 ||
                !std::isfinite(parameters.length_penalty) ||
                parameters.length_penalty <= 0.0 ||
                !std::isfinite(parameters.tts_temperature) ||
                parameters.tts_temperature <= 0.0 ||
                !std::isfinite(parameters.tts_repetition_penalty) ||
                parameters.tts_repetition_penalty <= 0.0) {
            throw std::invalid_argument(
                "MiniCPM-o duplex sampling configuration is invalid");
        }
        const int64_t audio_bos = parameters.special_ids.back();
        if (audio_bos < 0 || audio_bos >= 152064 ||
                std::any_of(
                    parameters.special_ids.begin(),
                    parameters.special_ids.end() - 1,
                    [&](int64_t token) {
                        return token < 0 ||
                            token >= state_->runtime.language.metadata.vocab_size;
                    }) ||
                std::any_of(
                    parameters.forbidden_ids.begin(),
                    parameters.forbidden_ids.end(),
                    [&](int64_t token) {
                        return token < 0 ||
                            token >= state_->runtime.language.metadata.vocab_size;
                    })) {
            throw std::invalid_argument(
                "MiniCPM-o duplex token ID is out of range");
        }
        if ((parameters.reference_audio_frames == 0) !=
                parameters.reference_audio_features.empty() ||
                parameters.reference_audio_frames < 0 ||
                (!parameters.reference_audio_features.empty() &&
                 (parameters.reference_audio_frames < 3 ||
                  parameters.reference_audio_features.size() !=
                    static_cast<size_t>(
                        parameters.reference_audio_frames) * 80))) {
            throw std::invalid_argument(
                "MiniCPM-o reference Mel geometry is invalid");
        }

        MfqCudaGuard guard(
            state_->runtime.execution.config.layer_placement.primary_device());
        mfq_tensor_backend::manual_seed(static_cast<int64_t>(parameters.seed));
        mfq_cuda_manual_seed_all(parameters.seed);
        auto special_ids = MiniCPMO45DuplexSpecialIds::from_tensor(
            mfq_tensor_backend::tensor(
                parameters.special_ids,
                mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)));
        if (state_->duplex_session) mfq_cuda_synchronize();
        state_->duplex_session.reset();
        state_->duplex_session.emplace(
            state_->runtime, special_ids, parameters.forbidden_ids,
            parameters.greedy);
        state_->duplex_session->temperature = parameters.temperature;
        state_->duplex_session->top_k = parameters.top_k;
        state_->duplex_session->top_p = parameters.top_p;
        state_->duplex_session->listen_probability_scale =
            parameters.listen_probability_scale;
        state_->duplex_session->repetition_penalty = parameters.repetition_penalty;
        state_->duplex_session->repetition_window = parameters.repetition_window;
        state_->duplex_session->length_penalty = parameters.length_penalty;
        state_->duplex_session->tts_temperature = parameters.tts_temperature;
        state_->duplex_session->tts_repetition_penalty =
            parameters.tts_repetition_penalty;

        const auto ids_tensor = [](const std::vector<int64_t> & values) {
            return values.empty()
                ? mfq_tensor_backend::Tensor()
                : mfq_tensor_backend::tensor(
                    values,
                    mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64));
        };
        mfq_tensor_backend::Tensor reference_features;
        if (!parameters.reference_audio_features.empty()) {
            reference_features = mfq_tensor_backend::tensor(
                parameters.reference_audio_features,
                mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32))
                .reshape({1, 80, parameters.reference_audio_frames});
        }
        state_->duplex_session->prepare(
            ids_tensor(parameters.system_prefix),
            reference_features,
            ids_tensor(parameters.system_suffix));
        mfq_cuda_synchronize();
}

MfqDuplexStepResult Components::step(const MfqDuplexStepInput& input) {
        const bool has_audio = input.audio_frames > 0;
        const bool has_text = !input.text_tokens.empty();
        if (has_audio && input.audio_features.size() !=
                static_cast<size_t>(input.audio_frames) * 80) {
            throw std::invalid_argument(
                "MiniCPM-o duplex Mel geometry is invalid");
        }
        if (!has_audio && !has_text) {
            throw std::invalid_argument(
                "MiniCPM-o duplex step has no input");
        }
        if (input.max_new_speak_tokens < 2) {
            throw std::invalid_argument(
                "MiniCPM-o duplex generation requires at least two token slots");
        }

        MfqCudaGuard guard(
            state_->runtime.execution.config.layer_placement.primary_device());
        if (!state_->duplex_session) {
            throw std::runtime_error(
                "MiniCPM-o duplex session is not prepared");
        }
        mfq_tensor_backend::Tensor audio_features;
        if (has_audio) {
            audio_features = mfq_tensor_backend::tensor(
                input.audio_features,
                mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32))
                .reshape({1, 80, input.audio_frames});
        }
        mfq_tensor_backend::Tensor text_ids;
        if (has_text) {
            text_ids = mfq_tensor_backend::tensor(
                input.text_tokens,
                mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64))
                .reshape({1, static_cast<int64_t>(input.text_tokens.size())});
        }
        const auto started = std::chrono::steady_clock::now();
        auto result = state_->duplex_session->run_step(
            {}, {}, {}, {}, audio_features,
            input.audio_prefix_extra_frames,
            input.audio_suffix_extra_frames,
            text_ids,
            input.max_new_speak_tokens,
            input.force_listen,
            input.force_speak);

        MfqDuplexStepResult response;
        auto generated = result.generated_ids
            .to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).contiguous().reshape({-1});
        const auto * generated_data = generated.template data_ptr<int64_t>();
        response.generated_tokens.assign(
            generated_data, generated_data + generated.numel());
        auto codes = result.tts_codes
            .to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt32).contiguous().reshape({-1});
        const auto * code_data = codes.template data_ptr<int32_t>();
        response.audio_tokens.assign(
            code_data, code_data + codes.numel());
        response.is_listen = result.is_listen;
        response.end_of_turn = result.end_of_turn;
        response.tts_force_flush = result.tts_force_flush;
        response.audio_chunk_index = state_->duplex_session->audio_chunk_index;
        response.language_cache_position = state_->runtime.language_state.cache_pos;
        response.audio_cache_position = state_->runtime.audio.cache_length();
        response.tts_cache_position = state_->duplex_session->tts_state.position;
        response.inference_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
        return response;
}

void Components::stop() {
        MfqCudaGuard guard(
            state_->runtime.execution.config.layer_placement.primary_device());
        mfq_cuda_synchronize();
        state_->duplex_session.reset();
        state_->runtime.language_state.reset(1);
        state_->runtime.audio.reset();
        mfq_cuda_synchronize();
}
} // namespace mfq::cuda::minicpmo45
