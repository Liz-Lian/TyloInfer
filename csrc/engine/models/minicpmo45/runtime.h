#pragma once

#include "engine/backends/cuda/executor/execution.h"
#include "text/forward.h"
#include "audio.h"
#include "tts.h"

inline auto minicpmo45_parse_bounds(::tyloi::DenseTensor bounds, const char *label) {
    if (!bounds.defined() || bounds.numel() == 0)
        return std::vector<tyloi::models::minicpmo45::MediaBound>{};
    if (bounds.dim() != 2 || bounds.size(1) != 4)
        throw std::runtime_error(std::string("MiniCPM-o ") + label +
                                 " bounds must have shape [count,4]");
    bounds = bounds.to(::tyloi::kCPU, ::tyloi::kInt64).contiguous();
    return tyloi::models::minicpmo45::media_bounds(
        std::span<const int64_t>(bounds.data_ptr<int64_t>(), bounds.numel()));
}

struct MiniCPMO45Runtime {
    const tyloi::cuda::minicpmo45::text::Model& language;
    tyloi::cuda::minicpmo45::text::State& language_state;
    tyloi::cuda::Executor& language_executor;
    CudaExecutionContext& execution;
    MiniCPMO45VisionEncoder vision;
    MiniCPMO45Resampler resampler;
    MiniCPMO45AudioEncoder audio;
    tyloi::cuda::minicpmo45::Tts tts;
    std::unique_ptr<tyloi::cuda::Executor> tts_executor;
    const tyloi::cuda::Executor::Mode tts_mode;
    tyloi::cuda::Projector projector;

    static MiniCPMO45Runtime load(const tyloi::cuda::minicpmo45::text::Model& language,
                                  tyloi::cuda::minicpmo45::text::State& language_state,
                                  tyloi::cuda::Executor& language_executor,
                                  const tyloi::weights::Loader& weights,
                                  CudaExecutionContext& execution);

    struct EncodeOps {
#ifdef TYLOINFER_NATIVE_CUDA
        using DenseTensor = ::tyloi::Tensor;
#else
        using DenseTensor = ::tyloi::DenseTensor;
#endif
        MiniCPMO45Runtime &runtime;
        static int64_t rank(const DenseTensor &t) { return t.dim(); }
        static int64_t size(const DenseTensor &t, int axis) { return t.size(axis); }
        static bool defined(const DenseTensor &t) { return t.defined(); }
        static DenseTensor batch_ids(DenseTensor t) {
#ifdef TYLOINFER_NATIVE_CUDA
            return t.dense().unsqueeze(0);
#else
            return t.unsqueeze(0);
#endif
        }
        static DenseTensor device_ids(DenseTensor t) {
#ifdef TYLOINFER_NATIVE_CUDA
            return t.dense().to(::tyloi::kCUDA, ::tyloi::kInt64).contiguous();
#else
            return t.to(::tyloi::kCUDA, ::tyloi::kInt64).contiguous();
#endif
        }
        DenseTensor embed(DenseTensor ids) {
            return tyloi::cuda::minicpmo45::text::embed(
                runtime.language, runtime.language_executor, runtime.execution.executor().collectives(),
                runtime.execution.config.layer_placement.primary_device(), std::move(ids),
                runtime.execution.executor().forward_mode(runtime.execution.config.layer_placement.primary_device()));
        }
        auto vision(DenseTensor pixels, DenseTensor mask, DenseTensor sizes) {
#ifdef TYLOINFER_NATIVE_CUDA
            pixels = pixels.dense().to(::tyloi::kCUDA);
#else
            pixels = pixels.to(::tyloi::kCUDA);
#endif
            return runtime.vision.forward_steps(runtime.projector, pixels, mask, sizes);
        }
        DenseTensor resample(DenseTensor hidden, DenseTensor sizes) {
            return runtime.resampler.forward(runtime.projector, hidden, sizes);
        }
        void reset_audio() { runtime.audio.reset(); }
        auto audio(DenseTensor features, DenseTensor lengths) {
#ifdef TYLOINFER_NATIVE_CUDA
            features = features.dense().to(::tyloi::kCUDA);
#else
            features = features.to(::tyloi::kCUDA);
#endif
            return runtime.audio.forward_steps(runtime.projector, features, lengths, false);
        }
        static auto audio_lengths(DenseTensor lengths) {
            return MiniCPMO45AudioEncoder::pooled_lengths(lengths);
        }
        static void scatter(DenseTensor &target, const DenseTensor &source,
                            const tyloi::models::minicpmo45::MediaBound &bound) {
#ifdef TYLOINFER_NATIVE_CUDA
            auto& target_dense = target.dense();
            const auto& source_dense = source.dense();
#else
            auto& target_dense = target;
            const auto& source_dense = source;
#endif
            target_dense.index({bound.batch, Slice(bound.begin, bound.end), Slice()})
                .copy_(source_dense.index({bound.source, Slice(0, bound.end - bound.begin), Slice()})
                           .to(target_dense.scalar_type()));
        }
    };

#ifdef TYLOINFER_NATIVE_CUDA
    tyloi::StepSequence<tyloi::models::minicpmo45::MultimodalResult<::tyloi::Tensor>>
    encode(::tyloi::Tensor input_ids, ::tyloi::Tensor pixels,
           ::tyloi::Tensor patch_mask, ::tyloi::Tensor target_sizes,
           ::tyloi::Tensor image_bounds, ::tyloi::Tensor audio_features,
           ::tyloi::Tensor audio_lengths, ::tyloi::Tensor audio_bounds) {
#else
    tyloi::StepSequence<tyloi::models::minicpmo45::MultimodalResult<::tyloi::DenseTensor>>
    encode(::tyloi::DenseTensor input_ids, ::tyloi::DenseTensor pixels,
           ::tyloi::DenseTensor patch_mask, ::tyloi::DenseTensor target_sizes,
           ::tyloi::DenseTensor image_bounds, ::tyloi::DenseTensor audio_features,
           ::tyloi::DenseTensor audio_lengths, ::tyloi::DenseTensor audio_bounds) {
#endif
        EncodeOps ops{*this};
#ifdef TYLOINFER_NATIVE_CUDA
        tyloi::models::minicpmo45::MultimodalInputs<::tyloi::Tensor> input{
#else
        tyloi::models::minicpmo45::MultimodalInputs<::tyloi::DenseTensor> input{
#endif
            input_ids,
            pixels,
            patch_mask,
            target_sizes,
            audio_features,
            audio_lengths,
#ifdef TYLOINFER_NATIVE_CUDA
            image_bounds.defined() ? minicpmo45_parse_bounds(image_bounds.dense(), "image")
                                   : std::vector<tyloi::models::minicpmo45::MediaBound>{},
            audio_bounds.defined() ? minicpmo45_parse_bounds(audio_bounds.dense(), "audio")
                                   : std::vector<tyloi::models::minicpmo45::MediaBound>{}};
#else
            minicpmo45_parse_bounds(image_bounds, "image"),
            minicpmo45_parse_bounds(audio_bounds, "audio")};
#endif
        auto encoding = tyloi::models::minicpmo45::encode(ops, input);
        while (auto step = encoding.next()) co_yield std::move(step);
    }
};

struct MiniCPMO45DuplexSpecialIds {
    int64_t unit_start = -1;
    int64_t unit_end = -1;
    int64_t image_start = -1;
    int64_t image_end = -1;
    int64_t slice_start = -1;
    int64_t slice_end = -1;
    int64_t listen = -1;
    int64_t speak = -1;
    int64_t tts_bos = -1;
    int64_t tts_eos = -1;
    int64_t chunk_eos = -1;
    int64_t chunk_tts_eos = -1;
    int64_t turn_eos = -1;
    int64_t tts_pad = -1;
    int64_t audio_bos = -1;

    static MiniCPMO45DuplexSpecialIds from_tensor(::tyloi::DenseTensor value) {
        value = value.to(::tyloi::kCPU, ::tyloi::kInt64).contiguous().reshape({-1});
        if (value.numel() != 15) {
            throw std::runtime_error(
                "MiniCPM-o duplex special_ids must contain 15 token IDs");
        }
        const auto * ids = value.data_ptr<int64_t>();
        MiniCPMO45DuplexSpecialIds result;
        result.unit_start = ids[0];
        result.unit_end = ids[1];
        result.image_start = ids[2];
        result.image_end = ids[3];
        result.slice_start = ids[4];
        result.slice_end = ids[5];
        result.listen = ids[6];
        result.speak = ids[7];
        result.tts_bos = ids[8];
        result.tts_eos = ids[9];
        result.chunk_eos = ids[10];
        result.chunk_tts_eos = ids[11];
        result.turn_eos = ids[12];
        result.tts_pad = ids[13];
        result.audio_bos = ids[14];
        const int64_t minimum = *std::min_element(ids, ids + 15);
        if (minimum < 0) {
            throw std::runtime_error(
                "MiniCPM-o duplex special token IDs must be non-negative");
        }
        return result;
    }

    bool is_chunk_terminator(int64_t token) const {
        return token == listen || token == chunk_eos ||
            token == chunk_tts_eos;
    }

    bool is_special(int64_t token) const {
        const std::array<int64_t, 15> values = {
            unit_start, unit_end, image_start, image_end,
            slice_start, slice_end, listen, speak, tts_bos,
            tts_eos, chunk_eos, chunk_tts_eos, turn_eos,
            tts_pad, audio_bos};
        return std::find(values.begin(), values.end(), token) != values.end();
    }
};

struct MiniCPMO45DuplexStepResult {
    ::tyloi::DenseTensor decision_logits;
    ::tyloi::DenseTensor audio_embeddings;
    ::tyloi::DenseTensor generated_ids;
    ::tyloi::DenseTensor tts_codes;
    bool is_listen = false;
    bool end_of_turn = false;
    bool tts_force_flush = false;
};

struct MiniCPMO45DuplexSession {
    MiniCPMO45Runtime & runtime;
    tyloi::cuda::minicpmo45::Tts::State tts_state;
    MiniCPMO45DuplexSpecialIds ids;
    std::vector<int64_t> forbidden_ids;
    std::vector<int64_t> generated_text_ids;
    int64_t audio_chunk_index = 0;
    int64_t tts_text_start_position = 0;
    bool current_turn_ended = true;
    bool greedy = false;
    double temperature = 0.7;
    int64_t top_k = 100;
    double top_p = 0.8;
    double listen_probability_scale = 1.0;
    double repetition_penalty = 1.05;
    int64_t repetition_window = 512;
    double length_penalty = 1.0;
    double tts_temperature = 0.8;
    double tts_repetition_penalty = 1.05;

    MiniCPMO45DuplexSession(
            MiniCPMO45Runtime & runtime_,
            MiniCPMO45DuplexSpecialIds ids_,
            std::vector<int64_t> forbidden_ids_,
            bool greedy_)
        : runtime(runtime_), tts_state(runtime_.tts), ids(ids_),
          forbidden_ids(std::move(forbidden_ids_)), greedy(greedy_) {
        if (std::find(forbidden_ids.begin(), forbidden_ids.end(),
                ids.chunk_eos) == forbidden_ids.end()) {
            forbidden_ids.push_back(ids.chunk_eos);
        }
        if (std::find(forbidden_ids.begin(), forbidden_ids.end(),
                ids.tts_pad) == forbidden_ids.end()) {
            forbidden_ids.push_back(ids.tts_pad);
        }
    }

    void prepare(
            ::tyloi::DenseTensor system_prefix_ids,
            ::tyloi::DenseTensor reference_audio_features = ::tyloi::DenseTensor(),
            ::tyloi::DenseTensor system_suffix_ids = ::tyloi::DenseTensor()) {
        runtime.language_state.reset(1);
        runtime.audio.reset();
        tts_state.reset(1);
        generated_text_ids.clear();
        audio_chunk_index = 0;
        tts_text_start_position = 0;
        current_turn_ended = true;
        if (system_prefix_ids.defined() &&
                system_prefix_ids.numel() > 0) {
            feed_ids(system_prefix_ids);
        }
        if (reference_audio_features.defined()) {
            if (reference_audio_features.dim() != 3 ||
                    reference_audio_features.size(0) != 1 ||
                    reference_audio_features.size(1) != 80 ||
                    reference_audio_features.size(2) < 3) {
                throw std::runtime_error(
                    "MiniCPM-o duplex reference audio expects [1,80,frames]");
            }
            auto raw_lengths = ::tyloi::cuda::tensor(
                std::vector<int64_t>{reference_audio_features.size(2)},
                ::tyloi::TensorOptions().dtype(::tyloi::kInt64));
            auto embeddings = runtime.audio.forward(
                runtime.projector,
                reference_audio_features.to(::tyloi::kCUDA),
                raw_lengths, false);
#ifdef TYLOINFER_NATIVE_CUDA
            feed_embeddings(embeddings.dense());
#else
            feed_embeddings(embeddings);
#endif
            runtime.audio.reset();
        }
        if (system_suffix_ids.defined() &&
                system_suffix_ids.numel() > 0) {
            feed_ids(system_suffix_ids);
        }
    }

    std::pair<::tyloi::DenseTensor, ::tyloi::DenseTensor> feed_embeddings(
            ::tyloi::DenseTensor embeddings,
            ::tyloi::DenseTensor token_ids = ::tyloi::DenseTensor()) {
        if (embeddings.dim() == 2) embeddings = embeddings.unsqueeze(0);
        if (embeddings.dim() != 3 || embeddings.size(0) != 1 ||
                embeddings.size(2) != runtime.language.metadata.hidden_size) {
            throw std::runtime_error(
                "MiniCPM-o duplex embeddings must have shape [1,tokens,4096]");
        }
        if (!token_ids.defined()) {
            token_ids = ::tyloi::cuda::zeros(
                {1, embeddings.size(1)},
                ::tyloi::TensorOptions().dtype(::tyloi::kInt64)
                    .device(::tyloi::kCUDA));
        } else {
            if (token_ids.dim() == 1) token_ids = token_ids.unsqueeze(0);
            if (token_ids.dim() != 2 || token_ids.size(0) != 1 ||
                    token_ids.size(1) != embeddings.size(1)) {
                throw std::runtime_error(
                    "MiniCPM-o duplex token IDs do not match embeddings");
            }
            token_ids = token_ids.to(::tyloi::kCUDA, ::tyloi::kInt64).contiguous();
        }
        auto hidden = tyloi::cuda::minicpmo45::text::forward(
            runtime.language, runtime.language_state, runtime.language_executor,
            runtime.execution.executor().collectives(), runtime.execution.executor().forward_mode(runtime.execution.config.layer_placement.primary_device()),
            tyloi::models::Input{token_ids, embeddings.to(::tyloi::kBFloat16).contiguous()});
        auto logits = tyloi::cuda::minicpmo45::text::logits(
#ifdef TYLOINFER_NATIVE_CUDA
            runtime.language, runtime.language_executor, tyloi::ops::select(hidden, 1, -1),
#else
            runtime.language, runtime.language_executor, hidden.index({Slice(), -1, Slice()}),
#endif
            true, runtime.execution.executor().forward_mode(runtime.execution.config.layer_placement.primary_device()));
#ifdef TYLOINFER_NATIVE_CUDA
        return {logits.dense(), hidden.dense()};
#else
        return {logits, hidden};
#endif
    }

    std::pair<::tyloi::DenseTensor, ::tyloi::DenseTensor> feed_ids(::tyloi::DenseTensor token_ids) {
        if (token_ids.dim() == 0) token_ids = token_ids.reshape({1});
        if (token_ids.dim() == 1) token_ids = token_ids.unsqueeze(0);
        token_ids = token_ids.to(::tyloi::kCUDA, ::tyloi::kInt64).contiguous();
#ifdef TYLOINFER_NATIVE_CUDA
        return feed_embeddings(
            MiniCPMO45Runtime::EncodeOps{runtime}.embed(token_ids).dense(), token_ids);
#else
        return feed_embeddings(
            MiniCPMO45Runtime::EncodeOps{runtime}.embed(token_ids), token_ids);
#endif
    }

    std::pair<::tyloi::DenseTensor, ::tyloi::DenseTensor> feed_id(int64_t token) {
        return feed_ids(::tyloi::cuda::tensor(
            std::vector<int64_t>{token},
            ::tyloi::TensorOptions().dtype(::tyloi::kInt64)
                .device(::tyloi::kCUDA)));
    }

    ::tyloi::DenseTensor apply_text_sampling_filters(::tyloi::DenseTensor logits) const {
        logits = logits.clone().to(::tyloi::kFloat32);
        if (!generated_text_ids.empty() && repetition_penalty != 1.0) {
            const size_t begin = generated_text_ids.size() >
                    static_cast<size_t>(repetition_window)
                ? generated_text_ids.size() -
                    static_cast<size_t>(repetition_window)
                : 0;
            std::vector<int64_t> recent(
                generated_text_ids.begin() +
                    static_cast<std::ptrdiff_t>(begin),
                generated_text_ids.end());
            std::sort(recent.begin(), recent.end());
            recent.erase(std::unique(recent.begin(), recent.end()), recent.end());
            auto recent_tensor = ::tyloi::cuda::tensor(
                recent, ::tyloi::TensorOptions().dtype(::tyloi::kInt64)
                    .device(logits.device()));
            auto selected = logits.index_select(1, recent_tensor);
            selected = repetition_penalty > 1.0
                ? selected / repetition_penalty
                : selected * (1.0 / repetition_penalty);
            logits.index_copy_(1, recent_tensor, selected);
        }
        if (length_penalty != 1.0) {
            auto selected = logits.index({Slice(), ids.turn_eos});
            selected = ::tyloi::cuda::where(
                selected > 0,
                selected / length_penalty,
                selected * length_penalty);
            logits.index_put_({Slice(), ids.turn_eos}, selected);
        }
        if (listen_probability_scale != 1.0) {
            logits.index_put_(
                {Slice(), ids.listen},
                logits.index({Slice(), ids.listen}) *
                    listen_probability_scale);
        }
        if (!forbidden_ids.empty()) {
            auto forbidden = ::tyloi::cuda::tensor(
                forbidden_ids,
                ::tyloi::TensorOptions().dtype(::tyloi::kInt64)
                    .device(logits.device()));
            logits.index_fill_(
                1, forbidden,
                -std::numeric_limits<float>::infinity());
        }
        if (greedy) return logits;
        logits = logits / temperature;
        if (top_k > 0 && top_k < logits.size(1)) {
            auto values = std::get<0>(
                ::tyloi::cuda::topk(
                    logits, top_k, -1, true, true));
            auto threshold = values.select(1, top_k - 1).unsqueeze(1);
            logits = logits.masked_fill(
                logits < threshold,
                -std::numeric_limits<float>::infinity());
        }
        if (top_p < 1.0) {
            auto sorted = ::tyloi::cuda::sort(logits, -1, true);
            auto sorted_logits = std::get<0>(sorted);
            auto sorted_indices = std::get<1>(sorted);
            auto cumulative = ::tyloi::cuda::cumsum(
                ::tyloi::cuda::softmax(sorted_logits, -1), -1);
            auto remove = cumulative > top_p;
            auto shifted = ::tyloi::cuda::zeros_like(remove);
            shifted.narrow(1, 1, remove.size(1) - 1)
                .copy_(remove.narrow(1, 0, remove.size(1) - 1));
            sorted_logits = sorted_logits.masked_fill(
                shifted, -std::numeric_limits<float>::infinity());
            logits = ::tyloi::cuda::full_like(
                logits, -std::numeric_limits<float>::infinity());
            logits.scatter_(1, sorted_indices, sorted_logits);
        }
        return logits;
    }

    int64_t sample_text_token(::tyloi::DenseTensor logits, bool force_listen) const {
        if (force_listen) return ids.listen;
        auto original = logits.to(::tyloi::kFloat32);
        auto first = greedy
            ? ::tyloi::cuda::argmax(original, -1)
            : ::tyloi::cuda::multinomial(::tyloi::cuda::softmax(original, -1), 1)
                .reshape({1});
        const int64_t first_id = first.item<int64_t>();
        if (first_id == ids.chunk_eos) return first_id;
        auto filtered = apply_text_sampling_filters(original);
        auto sampled = greedy
            ? ::tyloi::cuda::argmax(filtered, -1)
            : ::tyloi::cuda::multinomial(::tyloi::cuda::softmax(filtered, -1), 1)
                .reshape({1});
        return sampled.item<int64_t>();
    }

    MiniCPMO45DuplexStepResult run_step(
            ::tyloi::DenseTensor pixels,
            ::tyloi::DenseTensor patch_mask,
            ::tyloi::DenseTensor target_sizes,
            ::tyloi::DenseTensor image_slice_counts,
            ::tyloi::DenseTensor audio_features,
            int64_t audio_prefix_extra_frames,
            int64_t audio_suffix_extra_frames,
            ::tyloi::DenseTensor text_ids,
            int64_t max_new_speak_tokens,
            bool force_listen,
            bool force_speak = false) {
        if (max_new_speak_tokens < 2) {
            throw std::runtime_error(
                "MiniCPM-o duplex generation requires at least two token slots");
        }
        auto pending = feed_id(ids.unit_start);
        ::tyloi::DenseTensor generation_logits;
        bool has_content = false;
        ::tyloi::DenseTensor audio_embeddings;

        if (pixels.defined()) {
            if (!patch_mask.defined() || !target_sizes.defined()) {
                throw std::runtime_error(
                    "MiniCPM-o duplex image pixels require patch mask and target sizes");
            }
            auto vision_states = runtime.vision.forward(
                runtime.projector,
                pixels.to(::tyloi::kCUDA), patch_mask, target_sizes);
            auto image_embeddings = runtime.resampler.forward(
                runtime.projector, vision_states, target_sizes);
#ifdef TYLOINFER_NATIVE_CUDA
            const auto& image_embeddings_dense = image_embeddings.dense();
#else
            const auto& image_embeddings_dense = image_embeddings;
#endif
            std::vector<int64_t> counts;
            if (image_slice_counts.defined()) {
                auto count_tensor = image_slice_counts
                    .to(::tyloi::kCPU, ::tyloi::kInt64).contiguous().reshape({-1});
                const auto * values = count_tensor.data_ptr<int64_t>();
                counts.assign(values, values + count_tensor.numel());
            } else {
                counts.assign(
                    static_cast<size_t>(image_embeddings.size(0)), 1);
            }
            int64_t offset = 0;
            for (const int64_t count : counts) {
                if (count <= 0 || offset + count > image_embeddings.size(0)) {
                    throw std::runtime_error(
                        "MiniCPM-o duplex image slice counts are invalid");
                }
                pending = feed_id(ids.image_start);
                pending = feed_embeddings(image_embeddings_dense.index({offset}));
                pending = feed_id(ids.image_end);
                ++offset;
                for (int64_t slice = 1; slice < count; ++slice) {
                    pending = feed_id(ids.slice_start);
                    pending = feed_embeddings(image_embeddings_dense.index({offset}));
                    pending = feed_id(ids.slice_end);
                    ++offset;
                }
            }
            if (offset != image_embeddings.size(0)) {
                throw std::runtime_error(
                    "MiniCPM-o duplex image slice counts do not cover embeddings");
            }
            generation_logits = pending.first;
            has_content = true;
        }

        if (audio_features.defined()) {
            auto encoded_audio = runtime.audio.forward_streaming(
                runtime.projector,
                audio_features.to(::tyloi::kCUDA),
                audio_prefix_extra_frames,
                audio_suffix_extra_frames);
#ifdef TYLOINFER_NATIVE_CUDA
            audio_embeddings = encoded_audio.dense();
#else
            audio_embeddings = std::move(encoded_audio);
#endif
            pending = feed_embeddings(audio_embeddings);
            generation_logits = pending.first;
            ++audio_chunk_index;
            has_content = true;
        }

        if (text_ids.defined() && text_ids.numel() > 0) {
            pending = feed_ids(text_ids);
            if (!generation_logits.defined()) {
                generation_logits = pending.first;
            }
            has_content = true;
        }
        if (!has_content) {
            throw std::runtime_error(
                "MiniCPM-o duplex step contains no image, audio, or text input");
        }
        if (pixels.defined() && !audio_features.defined()) {
            ++audio_chunk_index;
        }

        MiniCPMO45DuplexStepResult result;
        result.decision_logits = generation_logits;
        result.audio_embeddings = audio_embeddings;
        std::vector<int64_t> generated;
        std::vector<int64_t> spoken_ids;
        std::vector<::tyloi::DenseTensor> spoken_hidden;
        auto logits = generation_logits;
        bool force_current = force_listen;
        for (int64_t index = 0; index < max_new_speak_tokens; ++index) {
            if (index == max_new_speak_tokens - 1) {
                feed_id(ids.chunk_eos);
                generated.push_back(ids.chunk_eos);
                break;
            }
            const bool forced_decision = force_current;
            int64_t token = sample_text_token(logits, force_current);
            force_current = false;
            if (!forced_decision && token != ids.chunk_eos) {
                generated_text_ids.push_back(token);
            }
            if (!forced_decision && token == ids.listen &&
                    (!current_turn_ended || force_speak)) {
                token = ids.tts_bos;
            }
            generated.push_back(token);
            result.is_listen = token == ids.listen;
            if (ids.is_chunk_terminator(token)) {
                pending = feed_id(token);
                break;
            }

            current_turn_ended = false;
            pending = feed_id(token);
            logits = pending.first;
            result.end_of_turn = token == ids.turn_eos;
            if (result.end_of_turn) current_turn_ended = true;
            if (index != 0) {
                spoken_ids.push_back(token);
                spoken_hidden.push_back(
                    pending.second.index({Slice(), -1, Slice()}));
            }
        }
        feed_id(ids.unit_end);

        result.generated_ids = ::tyloi::cuda::tensor(
            generated,
            ::tyloi::TensorOptions().dtype(::tyloi::kInt64)
                .device(::tyloi::kCUDA));
        if (!result.is_listen) {
            ::tyloi::DenseTensor text_id_tensor;
            ::tyloi::DenseTensor hidden_tensor;
            if (spoken_ids.empty()) {
                text_id_tensor = ::tyloi::cuda::empty(
                    {1, 0}, ::tyloi::TensorOptions().dtype(::tyloi::kInt64)
                        .device(::tyloi::kCUDA));
                hidden_tensor = ::tyloi::cuda::empty(
                    {1, 0, runtime.language.metadata.hidden_size},
                    ::tyloi::TensorOptions().dtype(::tyloi::kBFloat16)
                        .device(::tyloi::kCUDA));
            } else {
                text_id_tensor = ::tyloi::cuda::tensor(
                    spoken_ids,
                    ::tyloi::TensorOptions().dtype(::tyloi::kInt64)
                        .device(::tyloi::kCUDA)).unsqueeze(0);
                hidden_tensor = ::tyloi::cuda::stack(spoken_hidden, 1)
                    .to(::tyloi::kBFloat16).contiguous();
            }
            auto condition = runtime.tts.duplex_condition(
                runtime.projector, text_id_tensor, hidden_tensor, ids.audio_bos);
            const bool first_tts_chunk = tts_text_start_position == 0;
            if (first_tts_chunk) {
                tts_state.reset(1);
                result.tts_force_flush = true;
            }
            if (tts_state.position != tts_text_start_position) {
                throw std::runtime_error(
                    "MiniCPM-o duplex TTS cache position is inconsistent");
            }
            const int64_t minimum_codes =
                result.end_of_turn || first_tts_chunk ? 0 : 26;
            auto tts_result = runtime.tts.generate_duplex_chunk(
                tts_state, *runtime.tts_executor, runtime.tts_mode, condition, 26, minimum_codes, 6561,
                tts_temperature, tts_repetition_penalty);
            result.tts_codes = tts_result.codes;
            if (result.end_of_turn) {
                tts_state.reset(1);
                tts_text_start_position = 0;
            } else {
                tts_text_start_position +=
                    condition.size(1) + result.tts_codes.size(1);
                if (tts_state.position != tts_text_start_position) {
                    throw std::runtime_error(
                        "MiniCPM-o duplex TTS cache did not advance exactly");
                }
            }
        } else {
            result.tts_codes = ::tyloi::cuda::empty(
                {1, 0, 1},
                ::tyloi::TensorOptions().dtype(::tyloi::kInt64)
                    .device(::tyloi::kCUDA));
        }
        return result;
    }
};
