#ifndef TYLOINFER_NATIVE_CUDA
#include "engine/backends/cuda/executor/execution.h"
#endif
#include "components.h"
#include "runtime.h"

#include "engine/backends/cuda/executor/sampling.h"
#include "backend.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#ifndef TYLOINFER_NATIVE_CUDA
MiniCPMO45Runtime MiniCPMO45Runtime::load(
        const tyloi::cuda::minicpmo45::text::Model& language,
        tyloi::cuda::minicpmo45::text::State& language_state,
        tyloi::cuda::Executor& language_executor,
        const tyloi::weights::Loader& weights,
        CudaExecutionContext& execution) {
    auto &placement = execution.config.layer_placement;
    MfqCudaGuard guard(placement.primary_device());
    auto projector = execution.executor().projector(execution.executor().mode().ffn, model_parallel_primary_device(execution.config));
    return {language, language_state, language_executor, execution,
            MiniCPMO45VisionEncoder::load(weights),
            MiniCPMO45Resampler::load(weights),
            MiniCPMO45AudioEncoder::load(weights),
            tyloi::cuda::minicpmo45::Tts::load(weights),
            execution.executor().fork(model_parallel_primary_device(execution.config)), execution.executor().forward_mode(execution.config.layer_placement.primary_device()),
            projector};
}
#endif

namespace tyloi::cuda::minicpmo45 {

using ::tyloi::indexing::Slice;

#ifndef TYLOINFER_NATIVE_CUDA
std::unique_ptr<Components> load_components(text::Model& model, text::State& state,
        Executor& executor, const tyloi::weights::Loader& weights,
        bool load_optional_components, CudaExecutionContext& execution) {
    if (!load_optional_components || model.plan.vision == tyloi::models::Vision::none)
        return nullptr;
    if (model.plan.vision != tyloi::models::Vision::minicpmo45)
        throw std::runtime_error("unsupported MiniCPM-o CUDA vision adapter");
    return std::make_unique<Components>(model, state, executor, weights, execution);
}

struct Components::State {
    MiniCPMO45Runtime runtime;
    std::optional<MiniCPMO45DuplexSession> duplex_session;

    State(const text::Model& language, text::State& language_state,
          Executor& language_executor, const tyloi::weights::Loader& weights,
          CudaExecutionContext& execution)
        : runtime(MiniCPMO45Runtime::load(language, language_state, language_executor,
                                         weights, execution)) {}
};

Components::Components(const text::Model& language, text::State& language_state,
                       Executor& language_executor, const tyloi::weights::Loader& weights,
                       CudaExecutionContext& execution)
    : state_(std::make_unique<State>(language, language_state, language_executor, weights,
                                     execution)) {}

Components::~Components() = default;

tyloi::StepSequence<tyloi::models::Prepared> Components::prepare(
        const std::vector<int64_t>& prompt, const MfqMultimodalInput& vision) {
    const auto cpu_i64 =
        ::tyloi::TensorOptions().dtype(::tyloi::kInt64).device(::tyloi::kCPU);
    const auto cuda_i64 =
        ::tyloi::TensorOptions().dtype(::tyloi::kInt64).device(::tyloi::kCUDA);
    auto input_ids = ::tyloi::cuda::tensor(prompt, cuda_i64)
        .reshape({1, -1}).contiguous();
    ::tyloi::DenseTensor pixels;
    ::tyloi::DenseTensor patch_mask;
    ::tyloi::DenseTensor target_sizes;
    ::tyloi::DenseTensor image_bounds;
    if (!vision.image_bounds.empty()) {
        pixels = ::tyloi::cuda::from_blob(
            const_cast<float *>(vision.pixel_values.data()),
            vision.pixel_shape,
            ::tyloi::TensorOptions().dtype(::tyloi::kFloat32).device(::tyloi::kCPU))
            .clone();
        patch_mask = ::tyloi::cuda::from_blob(
            const_cast<uint8_t *>(vision.patch_mask.data()),
            vision.patch_mask_shape,
            ::tyloi::TensorOptions().dtype(::tyloi::kUInt8).device(::tyloi::kCPU))
            .clone().to(::tyloi::kBool);
        target_sizes = ::tyloi::cuda::from_blob(
            const_cast<int32_t *>(vision.target_sizes.data()),
            vision.target_sizes_shape,
            ::tyloi::TensorOptions().dtype(::tyloi::kInt32).device(::tyloi::kCPU))
            .clone();
        image_bounds = ::tyloi::cuda::from_blob(
            const_cast<int64_t *>(vision.image_bounds.data()),
            std::vector<int64_t>{
                static_cast<int64_t>(vision.image_bounds.size() / 4), 4},
            cpu_i64).clone();
    }
    ::tyloi::DenseTensor audio_features;
    ::tyloi::DenseTensor audio_lengths;
    ::tyloi::DenseTensor audio_bounds;
    if (!vision.audio_bounds.empty()) {
        audio_features = ::tyloi::cuda::from_blob(
            const_cast<float *>(vision.audio_features.data()),
            vision.audio_features_shape,
            ::tyloi::TensorOptions().dtype(::tyloi::kFloat32).device(::tyloi::kCPU))
            .clone();
        audio_lengths = ::tyloi::cuda::tensor(
            vision.audio_lengths, cpu_i64).contiguous();
        audio_bounds = ::tyloi::cuda::from_blob(
            const_cast<int64_t *>(vision.audio_bounds.data()),
            std::vector<int64_t>{
                static_cast<int64_t>(vision.audio_bounds.size() / 4), 4},
            cpu_i64).clone();
    }


    co_yield tyloi::StepState::advanced;
    auto encoding = state_->runtime.encode(input_ids, pixels, patch_mask,
        target_sizes, image_bounds, audio_features, audio_lengths, audio_bounds);
#ifdef TYLOINFER_NATIVE_CUDA
    tyloi::models::minicpmo45::MultimodalResult<::tyloi::Tensor> result;
#else
    tyloi::models::minicpmo45::MultimodalResult<::tyloi::DenseTensor> result;
#endif
    while (auto step = encoding.next()) {
        if (step.layer || step.graph) throw std::logic_error("media encoding does not support layer or graph events");
        if (step.value) result = std::move(*step.value);
        else co_yield step.state;
    }
    tyloi::models::Prepared prepared;
    prepared.token_ids = prompt;
#ifdef TYLOINFER_NATIVE_CUDA
    prepared.embeddings = std::move(result.input_embeddings);
    prepared.positions = std::make_shared<::tyloi::DenseTensor>(
        ::tyloi::cuda::arange(static_cast<int64_t>(prompt.size()), cuda_i64));
#else
    prepared.embeddings = std::move(result.input_embeddings);
    prepared.positions = ::tyloi::cuda::arange(static_cast<int64_t>(prompt.size()), cuda_i64);
#endif
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
        ::tyloi::cuda::manual_seed(static_cast<int64_t>(parameters.seed));
        mfq_cuda_manual_seed_all(parameters.seed);
        auto special_ids = MiniCPMO45DuplexSpecialIds::from_tensor(
            ::tyloi::cuda::tensor(
                parameters.special_ids,
                ::tyloi::TensorOptions().dtype(::tyloi::kInt64)));
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
                ? ::tyloi::DenseTensor()
                : ::tyloi::cuda::tensor(
                    values,
                    ::tyloi::TensorOptions().dtype(::tyloi::kInt64));
        };
        ::tyloi::DenseTensor reference_features;
        if (!parameters.reference_audio_features.empty()) {
            reference_features = ::tyloi::cuda::tensor(
                parameters.reference_audio_features,
                ::tyloi::TensorOptions().dtype(::tyloi::kFloat32))
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
        ::tyloi::DenseTensor audio_features;
        if (has_audio) {
            audio_features = ::tyloi::cuda::tensor(
                input.audio_features,
                ::tyloi::TensorOptions().dtype(::tyloi::kFloat32))
                .reshape({1, 80, input.audio_frames});
        }
        ::tyloi::DenseTensor text_ids;
        if (has_text) {
            text_ids = ::tyloi::cuda::tensor(
                input.text_tokens,
                ::tyloi::TensorOptions().dtype(::tyloi::kInt64))
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
            .to(::tyloi::kCPU, ::tyloi::kInt64).contiguous().reshape({-1});
        const auto * generated_data = generated.template data_ptr<int64_t>();
        response.generated_tokens.assign(
            generated_data, generated_data + generated.numel());
        auto codes = result.tts_codes
            .to(::tyloi::kCPU, ::tyloi::kInt32).contiguous().reshape({-1});
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
#endif

#ifdef TYLOINFER_NATIVE_CUDA
struct Components::State {
    int primary;
    MiniCPMO45VisionEncoder vision;
    MiniCPMO45Resampler resampler;
    MiniCPMO45AudioEncoder audio;
    Tts tts;
    State(const tyloi::weights::Loader& weights, int device)
        : primary(device), vision(MiniCPMO45VisionEncoder::load(weights)),
          resampler(MiniCPMO45Resampler::load(weights)), audio(MiniCPMO45AudioEncoder::load(weights)),
          tts(Tts::load(weights)) {}
};

Components::Components(std::unique_ptr<State> value) : state_(std::move(value)) {}
Components::~Components() = default;

std::unique_ptr<Components> Components::load(const tyloi::weights::Loader& weights, int primary) {
    return std::unique_ptr<Components>(new Components(std::make_unique<State>(weights, primary)));
}

std::unique_ptr<tyloi::models::Media::Request> Components::encode(
        const text::Model& language, tyloi::models::MediaInput source) const {
    return media_request(language, std::move(source), MediaMode::prompt);
}

std::unique_ptr<tyloi::models::Media::Request> Components::images(
        const text::Model& language, Tensor pixels, Tensor mask, Tensor sizes) const {
    tyloi::models::MediaInput input;
    input.pixels = std::move(pixels);
    input.patch_mask = std::move(mask);
    input.target_sizes = std::move(sizes);
    return media_request(language, std::move(input), MediaMode::images);
}

std::unique_ptr<tyloi::models::Media::Request> Components::audios(
        const text::Model& language, Tensor features, Tensor lengths) const {
    tyloi::models::MediaInput input;
    input.audio_features = std::move(features);
    input.audio_lengths = std::move(lengths);
    return media_request(language, std::move(input), MediaMode::audios);
}

std::unique_ptr<tyloi::models::Media::Request> Components::media_request(
        const text::Model& language, tyloi::models::MediaInput source, MediaMode mode) const {
    struct EncodeOps {
        using DenseTensor = tyloi::Tensor;
        const text::Model& language;
        const State& weights;
        MiniCPMO45AudioEncoder::State audio_state;
        EncodeOps(const text::Model& model, const State& value)
            : language(model), weights(value), audio_state(value.audio.layers.size()) {}
        static int64_t rank(const DenseTensor& value) { return value.dim(); }
        static int64_t size(const DenseTensor& value, int axis) { return value.size(axis); }
        static bool defined(const DenseTensor& value) { return value.defined(); }
        static DenseTensor batch_ids(DenseTensor value) { return tyloi::ops::unsqueeze(value, 0); }
        DenseTensor device_ids(DenseTensor value) const {
            return tyloi::ops::contiguous(tyloi::ops::cast(value, kInt64, Device(kCUDA, weights.primary)));
        }
        DenseTensor embed(DenseTensor ids) const { return text::embed(language, weights.primary, ids); }
        auto vision(DenseTensor pixels, DenseTensor mask, DenseTensor sizes) const {
            return weights.vision.forward_steps(tyloi::ops::copy(pixels, Device(kCUDA, weights.primary)), mask, sizes);
        }
        DenseTensor resample(DenseTensor hidden, DenseTensor sizes) const {
            return weights.resampler.forward(hidden, sizes);
        }
        void reset_audio() { audio_state.reset(); }
        auto audio(DenseTensor features, DenseTensor lengths) {
            return weights.audio.forward_steps(audio_state, tyloi::ops::copy(features, Device(kCUDA, weights.primary)), lengths, false);
        }
        static auto audio_lengths(DenseTensor lengths) { return MiniCPMO45AudioEncoder::pooled_lengths(lengths); }
        static void scatter(DenseTensor& target, const DenseTensor& source,
                            const tyloi::models::minicpmo45::MediaBound& bound) {
            auto& output = target.dense();
            const auto& input = source.dense();
            output.index({bound.batch, Slice(bound.begin, bound.end), Slice()})
                .copy_(input.index({bound.source, Slice(0, bound.end - bound.begin), Slice()})
                    .to(output.scalar_type()));
        }
    };
    struct Request final : tyloi::models::Media::Request {
        tyloi::models::MediaInput source;
        EncodeOps ops;
        MediaMode mode;
        tyloi::models::minicpmo45::MultimodalInputs<Tensor> input;
        tyloi::StepSequence<tyloi::models::minicpmo45::MultimodalResult<Tensor>> encoding;
        tyloi::StepSequence<Tensor> modality;
        tyloi::models::MediaResult result;
        bool started = false, finished = false, failed = false;
        Request(const text::Model& model, const State& weights, tyloi::models::MediaInput value, MediaMode kind)
            : source(std::move(value)), ops(model, weights), mode(kind),
              input{source.ids, source.pixels, source.patch_mask, source.target_sizes,
                  source.audio_features, source.audio_lengths,
                  source.image_bounds ? minicpmo45_parse_bounds(source.image_bounds.dense(), "image")
                                      : std::vector<tyloi::models::minicpmo45::MediaBound>{},
                  source.audio_bounds ? minicpmo45_parse_bounds(source.audio_bounds.dense(), "audio")
                                      : std::vector<tyloi::models::minicpmo45::MediaBound>{}},
              encoding(mode == MediaMode::prompt ? tyloi::models::minicpmo45::encode(ops, input)
                                                  : tyloi::StepSequence<tyloi::models::minicpmo45::MultimodalResult<Tensor>>{}) {}
        tyloi::StepResult<tyloi::models::MediaResult> next() override {
            if (failed) throw std::logic_error("media request already failed");
            if (finished) return {};
            try { return advance(); }
            catch (...) { failed = true; throw; }
        }
        tyloi::StepResult<tyloi::models::MediaResult> advance() {
            if (mode != MediaMode::prompt) {
                tyloi::ops::FfnScope scope(tyloi::ops::FfnScope::Role::independent);
                if (!started) {
                    started = true;
                    if (mode == MediaMode::images)
                        modality = ops.vision(source.pixels, source.patch_mask, source.target_sizes);
                    else {
                        ops.reset_audio();
                        modality = ops.audio(source.audio_features, source.audio_lengths);
                    }
                }
                auto step = modality.next();
                if (!step) throw std::logic_error("media encoding completed without a result");
                if (step.layer || step.graph || step.workspace)
                    throw std::logic_error("media encoding produced an unsupported event");
                if (!step.value) return {step.state, {}};
                if (mode == MediaMode::images) {
                    result.vision_states = std::move(*step.value);
                    result.image_embeddings = ops.resample(result.vision_states, source.target_sizes);
                } else result.audio_embeddings = std::move(*step.value);
                finished = true;
                return {step.state, result};
            }
            tyloi::StepResult<tyloi::models::minicpmo45::MultimodalResult<Tensor>> step;
            { tyloi::ops::FfnScope scope(tyloi::ops::FfnScope::Role::independent); step = encoding.next(); }
            if (!step) { finished = true; return {}; }
            if (step.layer || step.graph || step.workspace)
                throw std::logic_error("media encoding produced an unsupported event");
            if (!step.value) return {step.state, {}};
            auto& value = *step.value;
            result = {std::move(value.vision_states), std::move(value.image_embeddings),
                std::move(value.audio_embeddings), std::move(value.input_embeddings)};
            return {step.state, result};
        }
    };
    return std::make_unique<Request>(language, *state_, std::move(source), mode);
}

tyloi::StepSequence<tyloi::models::Prepared> Components::prepare(
        const text::Model& language, const std::vector<int64_t>& prompt,
        const MfqMultimodalInput& media) const {
    const auto cpu_i64 = TensorOptions().dtype(kInt64).device(kCPU);
    const auto cuda_i64 = TensorOptions().dtype(kInt64).device(Device(kCUDA, state_->primary));
    auto ids = tyloi::cuda::tensor(prompt, cuda_i64).reshape({1, -1}).contiguous();
    DenseTensor pixels, mask, sizes, image_bounds, features, lengths, audio_bounds;
    if (!media.image_bounds.empty()) {
        pixels = tyloi::cuda::from_blob(const_cast<float*>(media.pixel_values.data()), media.pixel_shape,
            TensorOptions().dtype(kFloat32).device(kCPU)).clone();
        mask = tyloi::cuda::from_blob(const_cast<uint8_t*>(media.patch_mask.data()), media.patch_mask_shape,
            TensorOptions().dtype(kUInt8).device(kCPU)).clone().to(kBool);
        sizes = tyloi::cuda::from_blob(const_cast<int32_t*>(media.target_sizes.data()), media.target_sizes_shape,
            TensorOptions().dtype(kInt32).device(kCPU)).clone();
        image_bounds = tyloi::cuda::from_blob(const_cast<int64_t*>(media.image_bounds.data()),
            std::vector<int64_t>{static_cast<int64_t>(media.image_bounds.size() / 4), 4}, cpu_i64).clone();
    }
    if (!media.audio_bounds.empty()) {
        features = tyloi::cuda::from_blob(const_cast<float*>(media.audio_features.data()), media.audio_features_shape,
            TensorOptions().dtype(kFloat32).device(kCPU)).clone();
        lengths = tyloi::cuda::tensor(media.audio_lengths, cpu_i64).contiguous();
        audio_bounds = tyloi::cuda::from_blob(const_cast<int64_t*>(media.audio_bounds.data()),
            std::vector<int64_t>{static_cast<int64_t>(media.audio_bounds.size() / 4), 4}, cpu_i64).clone();
    }
    co_yield tyloi::StepState::advanced;
    auto encoding = encode(language, {ids, pixels, mask, sizes, image_bounds,
        features, lengths, audio_bounds});
    tyloi::models::MediaResult result;
    for (;;) {
        auto step = encoding->next();
        if (!step) break;
        if (step.value) result = std::move(*step.value);
        else co_yield step.state;
    }
    tyloi::models::Prepared prepared;
    prepared.token_ids = prompt;
    prepared.embeddings = std::move(result.input_embeddings);
    prepared.positions = tyloi::ops::arange(0, static_cast<int64_t>(prompt.size()), Device(kCUDA, state_->primary));
    co_yield std::move(prepared);
}

tyloi::Tensor Components::condition(const tyloi::Tensor& ids, const tyloi::Tensor& hidden) const {
    return tyloi::Tensor(state_->tts.condition(ids.dense(), hidden.dense()));
}

std::unique_ptr<tyloi::models::Speech::Request> Components::generate(
        tyloi::Tensor condition, tyloi::models::SpeechOptions options, bool capture_logits,
        std::mt19937* evaluator_rng) const {
    struct Request final : tyloi::models::Speech::Request {
        const State& weights;
        Tts::State state;
        bool finished = false, failed = false;
        tyloi::models::SpeechOptions options;
        bool capture;
        std::mt19937* rng;
        std::vector<DenseTensor> trace;
        tyloi::models::SpeechResult result;
        std::unique_ptr<Tts::OfficialGeneration> generation;
        Request(const State& value, tyloi::Tensor input, tyloi::models::SpeechOptions config,
                bool capture_logits, std::mt19937* evaluator_rng)
            : weights(value), state(value.tts), options(config), capture(capture_logits), rng(evaluator_rng) {
            result.condition = std::move(input);
        }
        tyloi::StepResult<tyloi::models::SpeechResult> next() override {
            if (failed) throw std::logic_error("speech request already failed");
            if (finished) return {};
            try {
                if (!generation) {
                    tyloi::models::ForwardOptions mode;
                    mode.primary = weights.primary;
                    tyloi::models::minicpmo45::TtsSampling sampling{options.eos_token,
                        options.minimum_steps, options.top_k, options.minimum_keep, options.temperature, options.top_p,
                        options.repetition_penalty, options.min_p};
                    generation = std::make_unique<Tts::OfficialGeneration>(weights.tts, state, mode,
                        result.condition.dense(), options.steps, sampling, capture ? &trace : nullptr, rng);
                }
                generation->next();
                if (!generation->done) return {tyloi::StepState::advanced, {}};
                result.codes = tyloi::Tensor(generation->codes());
                if (!trace.empty()) result.logits = tyloi::Tensor(tyloi::cuda::stack(trace, 1));
                finished = true;
                return {tyloi::StepState::advanced, result};
            } catch (...) {
                failed = true;
                throw;
            }
        }
    };
    return std::make_unique<Request>(*state_, std::move(condition), options, capture_logits, evaluator_rng);
}

std::unique_ptr<tyloi::models::Duplex> Components::duplex(const text::Model& model, text::State& language_state) const {
    struct Session final : tyloi::models::Duplex {
        const State& weights;
        const text::Model& model;
        text::State& language;
        std::unique_ptr<MiniCPMO45DuplexSession> session;
        DenseTensor system_prefix, system_suffix, reference_audio;
        StepInput retained_input;
        StepOutput retained_output;
        Session(const State& value, const text::Model& language_model, text::State& state)
            : weights(value), model(language_model), language(state) {}
        static void validate(const MfqDuplexSessionParams& p, int64_t vocab) {
            if (p.special_ids.size() != 15 || (!p.greedy && (!std::isfinite(p.temperature) || p.temperature <= 0)) ||
                p.top_k < 0 || p.top_k > std::min<int64_t>(vocab, 1024) || !std::isfinite(p.top_p) || p.top_p <= 0 || p.top_p > 1 ||
                !std::isfinite(p.listen_probability_scale) || p.listen_probability_scale < 0 ||
                !std::isfinite(p.repetition_penalty) || p.repetition_penalty <= 0 || p.repetition_window <= 0 ||
                !std::isfinite(p.length_penalty) || p.length_penalty <= 0 || !std::isfinite(p.tts_temperature) || p.tts_temperature <= 0 ||
                !std::isfinite(p.tts_repetition_penalty) || p.tts_repetition_penalty <= 0)
                throw std::invalid_argument("MiniCPM-o duplex sampling configuration is invalid");
            if (std::any_of(p.special_ids.begin(), p.special_ids.end(), [](int64_t id) { return id < 0; }))
                throw std::invalid_argument("MiniCPM-o duplex token ID is out of range");
            if ((p.reference_audio_frames == 0) != p.reference_audio_features.empty() || p.reference_audio_frames < 0 ||
                (!p.reference_audio_features.empty() && (p.reference_audio_frames < 3 ||
                 p.reference_audio_features.size() != static_cast<std::size_t>(p.reference_audio_frames) * 80)))
                throw std::invalid_argument("MiniCPM-o reference Mel geometry is invalid");
        }
        static void validate(const MfqDuplexStepInput& input) {
            const bool audio = input.audio_frames > 0;
            if (audio && input.audio_features.size() != static_cast<std::size_t>(input.audio_frames) * 80)
                throw std::invalid_argument("MiniCPM-o duplex Mel geometry is invalid");
            if (!audio && input.text_tokens.empty()) throw std::invalid_argument("MiniCPM-o duplex step has no input");
            if (input.max_new_speak_tokens < 2) throw std::invalid_argument("MiniCPM-o duplex generation requires at least two token slots");
        }
        static void add(std::vector<std::pair<Device,std::size_t>>& result, Device device, std::size_t bytes) {
            if (!device.is_cuda() || !bytes) return;
            auto found = std::find_if(result.begin(), result.end(), [device](const auto& item) { return item.first == device; });
            if (found == result.end()) result.emplace_back(device, bytes);
            else {
                if (found->second > std::numeric_limits<std::size_t>::max() - bytes) throw std::overflow_error("MiniCPM-o duplex growth overflow");
                found->second += bytes;
            }
        }
        std::vector<std::pair<Device,std::size_t>> language_growth(int64_t tokens,
                const tyloi::models::ForwardOptions& options, bool reset) const {
            std::vector<std::pair<Device,std::size_t>> result;
            if (tokens <= 0) return result;
            const auto position = reset ? int64_t{0} : language.position();
            if (position < 0 || tokens > model.metadata.max_position_embeddings ||
                position > model.metadata.max_position_embeddings - tokens)
                throw std::invalid_argument("MiniCPM-o duplex language context exceeds capacity");
            for (std::size_t i = 0; i < model.layers.size(); ++i) {
                const auto& layer = model.layers[i];
                const auto& cache = language.layers[i];
                if (cache.defined() && (!reset || cache.batch == 1)) continue;
                const auto& attention = layer.attention;
                const auto capacity = attention.sliding ? attention.attention_window :
                    (options.kv_capacity > 0 ? std::max(position + tokens, options.kv_capacity) : attention.max_position_embeddings);
                const auto bytes = tyloi::state::kv_bytes(1, attention.kv_heads, capacity,
                    attention.attention_head_dim, attention.sliding, kBFloat16, attention.kv_quantization);
                if (!layer.cpu) add(result, Device(kCUDA, layer.device), bytes);
            }
            return result;
        }
        std::vector<std::pair<Device,std::size_t>> growth(const MfqDuplexSessionParams& p,
                const tyloi::models::ForwardOptions& options) const override {
            validate(p, model.metadata.vocab_size);
            int64_t first = static_cast<int64_t>(p.system_prefix.size());
            if (!first && p.reference_audio_frames > 0) first = tyloi::models::minicpmo45::pooled_audio_length(p.reference_audio_frames);
            if (!first) first = static_cast<int64_t>(p.system_suffix.size());
            return language_growth(first, options, true);
        }
        std::vector<std::pair<Device,std::size_t>> growth(const MfqDuplexStepInput& p,
                const tyloi::models::ForwardOptions& options) const override {
            validate(p);
            if (!session) throw std::runtime_error("MiniCPM-o duplex session is not prepared");
            auto result = language_growth(1, options, false);
            const auto device = Device(kCUDA, weights.primary);
            if (p.audio_frames > 0) {
                if (p.audio_prefix_extra_frames < 0 || p.audio_suffix_extra_frames < 0)
                    throw std::invalid_argument("MiniCPM-o streaming audio extra context is invalid");
                const auto frames = (int64_t(p.audio_frames) - 1) / 2 + 1;
                const auto prefix = p.audio_prefix_extra_frames / 2 + p.audio_prefix_extra_frames % 2;
                const auto suffix = p.audio_suffix_extra_frames / 2 + p.audio_suffix_extra_frames % 2;
                if (prefix >= frames || suffix >= frames - prefix)
                    throw std::invalid_argument("MiniCPM-o streaming audio extra context removes every frame");
                const auto past = session->audio_state.length() + frames >= 1500 ? 0 : session->audio_state.length();
                const auto tokens = past + frames - prefix - suffix;
                // F32 bounds both BF16 and F32 projection caches without counting temporary workspace.
                const auto bytes = std::size_t(weights.audio.layers.size()) * 2 * std::size_t(tokens) * 1024 * sizeof(float);
                add(result, device, bytes);
            }
            const auto& tts = weights.tts;
            for (std::size_t i = 0; i < tts.layers.size(); ++i) {
                const auto& attention = tts.layers[i].attention;
                const auto& cache = session->tts_state.kv[i];
                if (cache.defined() && cache.batch == 1) continue;
                const auto capacity = options.kv_capacity > 0
                    ? std::max<int64_t>(options.kv_capacity, int64_t(p.max_new_speak_tokens) + 1)
                    : tts.config.max_position_embeddings;
                add(result, device, tyloi::state::kv_bytes(1, attention.kv_heads, capacity,
                    attention.attention_head_dim, false, kFloat16, attention.kv_quantization));
            }
            return result;
        }
        void start(const MfqDuplexSessionParams& p, const tyloi::models::ForwardOptions& options) override {
            validate(p, model.metadata.vocab_size);
            tyloi::ops::FfnScope scope(tyloi::ops::FfnScope::Role::independent);
            MfqCudaGuard guard(weights.primary);
            tyloi::cuda::manual_seed(static_cast<int64_t>(p.seed));
            mfq_cuda_manual_seed_all(p.seed);
            auto special = MiniCPMO45DuplexSpecialIds::from_tensor(tyloi::cuda::tensor(p.special_ids, TensorOptions().dtype(kInt64)));
            session = std::make_unique<MiniCPMO45DuplexSession>(model, language, weights.vision, weights.resampler,
                weights.audio, weights.tts, special, p.forbidden_ids, p.greedy);
            session->options = options;
            session->temperature=p.temperature; session->top_k=p.top_k; session->top_p=p.top_p;
            session->listen_probability_scale=p.listen_probability_scale; session->repetition_penalty=p.repetition_penalty;
            session->repetition_window=p.repetition_window; session->length_penalty=p.length_penalty;
            session->tts_temperature=p.tts_temperature; session->tts_repetition_penalty=p.tts_repetition_penalty;
            const auto ids = [](const std::vector<int64_t>& values) {
                return values.empty() ? DenseTensor{} : tyloi::cuda::tensor(values, TensorOptions().dtype(kInt64));
            };
            system_prefix = ids(p.system_prefix);
            system_suffix = ids(p.system_suffix);
            reference_audio = {};
            if (!p.reference_audio_features.empty()) reference_audio=tyloi::cuda::tensor(p.reference_audio_features,
                TensorOptions().dtype(kFloat32)).reshape({1,80,p.reference_audio_frames});
            session->prepare(system_prefix, reference_audio, system_suffix);
        }
        MfqDuplexStepResult step(const MfqDuplexStepInput& input, const tyloi::models::ForwardOptions& options) override {
            validate(input);
            StepInput request;
            request.audio_prefix_extra_frames = input.audio_prefix_extra_frames;
            request.audio_suffix_extra_frames = input.audio_suffix_extra_frames;
            request.max_new_speak_tokens = input.max_new_speak_tokens;
            request.force_listen = input.force_listen;
            request.force_speak = input.force_speak;
            if (input.audio_frames > 0) request.audio_features=tyloi::Tensor(tyloi::cuda::tensor(input.audio_features,
                TensorOptions().dtype(kFloat32)).reshape({1,80,input.audio_frames}));
            if (!input.text_tokens.empty()) request.text_ids=tyloi::Tensor(tyloi::cuda::tensor(input.text_tokens,
                TensorOptions().dtype(kInt64)).reshape({1,static_cast<int64_t>(input.text_tokens.size())}));
            return step(std::move(request), options).result;
        }
        StepOutput step(StepInput input, const tyloi::models::ForwardOptions& options) override {
            if (!session) throw std::runtime_error("MiniCPM-o duplex session is not prepared");
            retained_input = std::move(input);
            retained_output = {};
            tyloi::ops::FfnScope scope(tyloi::ops::FfnScope::Role::independent);
            MfqCudaGuard guard(weights.primary);
            session->options=options;
            const auto dense = [](const Tensor& value) { return value ? value.dense() : DenseTensor{}; };
            const auto started=std::chrono::steady_clock::now();
            auto result=session->run_step(dense(retained_input.pixels), dense(retained_input.patch_mask),
                dense(retained_input.target_sizes), dense(retained_input.image_slice_counts),
                dense(retained_input.audio_features), retained_input.audio_prefix_extra_frames,
                retained_input.audio_suffix_extra_frames, dense(retained_input.text_ids),
                retained_input.max_new_speak_tokens, retained_input.force_listen, retained_input.force_speak);
            retained_output.decision_logits = Tensor(result.decision_logits);
            retained_output.audio_embeddings = Tensor(result.audio_embeddings);
            retained_output.generated_ids = Tensor(result.generated_ids);
            retained_output.tts_codes = Tensor(result.tts_codes);
            auto& response = retained_output.result;
            auto generated=result.generated_ids.to(kCPU,kInt64).contiguous().reshape({-1});
            const auto* generated_data=generated.data_ptr<int64_t>();
            response.generated_tokens.assign(generated_data,generated_data+generated.numel());
            auto codes=result.tts_codes.to(kCPU,kInt32).contiguous().reshape({-1});
            const auto* code_data=codes.data_ptr<int32_t>();
            response.audio_tokens.assign(code_data,code_data+codes.numel());
            response.is_listen=result.is_listen; response.end_of_turn=result.end_of_turn;
            response.tts_force_flush=result.tts_force_flush; response.audio_chunk_index=session->audio_chunk_index;
            response.language_cache_position=language.cache_pos; response.audio_cache_position=session->audio_state.length();
            response.tts_cache_position=session->tts_state.position;
            response.inference_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
            return retained_output;
        }
        void reset(const tyloi::models::ForwardOptions& options) override {
            if (!session) throw std::runtime_error("MiniCPM-o duplex session is not prepared");
            tyloi::ops::FfnScope scope(tyloi::ops::FfnScope::Role::independent);
            MfqCudaGuard guard(weights.primary);
            session->options = options;
            session->prepare(system_prefix, reference_audio, system_suffix);
        }
        void stop() override { session.reset(); }
    };
    return std::make_unique<Session>(*state_, model, language_state);
}
#endif
} // namespace tyloi::cuda::minicpmo45
