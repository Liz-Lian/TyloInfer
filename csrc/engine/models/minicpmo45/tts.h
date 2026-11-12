#pragma once

#include "engine/weights/loader.h"
#include "engine/models/minicpmo45/media.h"

#include "vision.h"
#include <random>
#include "engine/requests/policy.h"
#include "engine/generation/sampling.h"
#include "engine/models/common/load.h"
#include "engine/models/common/attention.h"
#include "engine/models/common/ffn.h"
#include "engine/models/common/mlp.h"
#ifndef TYLOINFER_NATIVE_CUDA
#include "engine/backends/cuda/executor/executor.h"
#endif
#include "engine/backends/cuda/ops/rope/rope.h"
#include "engine/state/kv_view.h"

#include <array>

struct MiniCPMO45TtsSamplingOps {
    using Slice = ::tyloi::indexing::Slice;
    using DenseTensor = ::tyloi::DenseTensor;
    std::mt19937 *evaluator_rng = nullptr;
    int device = 0;
    static DenseTensor temperature(DenseTensor scores, double value) { return scores / value; }
    static DenseTensor penalties(DenseTensor scores, std::span<const DenseTensor> history, double penalty) {
        auto counts = ::tyloi::cuda::zeros_like(scores);
        for (const auto &token : history)
            counts.scatter_add_(1, token.reshape({1, 1}),
                                ::tyloi::cuda::ones({1, 1}, counts.options()));
        auto alpha =
            ::tyloi::cuda::pow(::tyloi::cuda::full_like(counts, penalty), counts);
        return ::tyloi::cuda::where(scores < 0, scores * alpha, scores / alpha);
    }
    static void mask_eos(DenseTensor &scores, int64_t token) {
        scores.index_put_({Slice(), token}, -std::numeric_limits<float>::infinity());
    }
    DenseTensor reference(DenseTensor scores, int64_t top_k, double top_p, int64_t minimum_keep) const {
        auto host = scores.reshape({-1})
                        .to(::tyloi::kCPU, ::tyloi::kFloat32)
                        .contiguous();
        auto token = tyloi::engine::sample_top_k_top_p(
            std::span<const float>(host.data_ptr<float>(), host.numel()), top_k, top_p,
            minimum_keep, *evaluator_rng);
        return ::tyloi::cuda::tensor(std::vector<int64_t>{token},
                                          ::tyloi::TensorOptions()
                                              .device(::tyloi::Device(::tyloi::kCUDA, device))
                                              .dtype(::tyloi::kInt64));
    }
    static DenseTensor top_k(DenseTensor scores, int64_t count) {
        if (count >= scores.size(1))
            return scores;
        auto values = std::get<0>(::tyloi::cuda::topk(scores, count, -1, true, true));
        auto threshold = values.select(1, count - 1).unsqueeze(1);
        return scores.masked_fill(scores < threshold, -std::numeric_limits<float>::infinity());
    }
    static DenseTensor top_p(DenseTensor scores, double value) {
        auto sorted = ::tyloi::cuda::sort(scores, -1, true);
        auto logits = std::get<0>(sorted);
        auto indices = std::get<1>(sorted);
        auto cumulative = ::tyloi::cuda::cumsum(::tyloi::cuda::softmax(logits, -1), -1);
        auto keep = cumulative <= value;
        keep.narrow(1, 1, keep.size(1) - 1).copy_(keep.narrow(1, 0, keep.size(1) - 1).clone());
        keep.index_put_({Slice(), 0}, true);
        logits = logits.masked_fill(keep.logical_not(), -std::numeric_limits<float>::infinity());
        scores = ::tyloi::cuda::full_like(scores, -std::numeric_limits<float>::infinity());
        scores.scatter_(1, indices, logits);
        return scores;
    }
    static DenseTensor min_p(DenseTensor scores, double value) {
        auto maximum = std::get<0>(::tyloi::cuda::max(scores, -1, true));
        return scores.masked_fill(scores < maximum + std::log(value),
                                  -std::numeric_limits<float>::infinity());
    }
    static DenseTensor sample(DenseTensor scores) {
        return ::tyloi::cuda::multinomial(::tyloi::cuda::softmax(scores, -1), 1)
            .reshape({1})
            .to(::tyloi::kInt64);
    }
};

namespace tyloi::cuda::minicpmo45 {

using ::tyloi::indexing::Slice;

struct Tts {
    struct Layer {
        FullAttention attention;
#ifdef TYLOINFER_NATIVE_CUDA
        ::tyloi::Tensor ffn_norm;
#else
        ::tyloi::DenseTensor ffn_norm;
#endif
        tyloi::models::Mlp ffn;
    };

    struct State {
        explicit State(const Tts& model) : kv(model.layers.size()) {}
        std::vector<tyloi::state::KvView> kv;
        int64_t position = 0;

        void reset(int64_t batch) {
            position = 0;
            for (auto& cache : kv)
                if (cache.defined() && cache.batch != batch) cache = {};
        }
    };

    tyloi::models::ModelConfig config;
    int device = 0;
#ifdef TYLOINFER_NATIVE_CUDA
    tyloi::ops::RopeTable rope;
#else
    RopeCache rope;
#endif
    ::tyloi::Tensor text_embedding;
    ::tyloi::Tensor code_embedding;
    tyloi::weights::BiasWeight semantic_projector1;
    tyloi::weights::BiasWeight semantic_projector2;
    tyloi::weights::BiasWeight speaker_projector1;
    tyloi::weights::BiasWeight speaker_projector2;
    std::vector<Layer> layers;
#ifdef TYLOINFER_NATIVE_CUDA
    ::tyloi::Tensor output_norm, code_head;
#else
    ::tyloi::DenseTensor output_norm;
    ::tyloi::DenseTensor code_head;
#endif
    static Tts load(const tyloi::weights::Loader& weights) {
        MfqCudaGuard guard(weights.placement.device);
        Tts result;
        result.device = weights.placement.device;
        result.config = tyloi::models::minicpmo45::tts_decoder_config();
        RopeCache rope(
            result.config.max_position_embeddings,
            result.config.rotary_dim,
            result.config.rope_base, 0, -1,
            ::tyloi::Device(::tyloi::kCUDA, result.device));
#ifdef TYLOINFER_NATIVE_CUDA
        result.rope = rope.view();
#else
        result.rope = std::move(rope);
#endif
        result.text_embedding = weights.weight("tts.text_embedding.weight");
        result.code_embedding = weights.weight("tts.code_embedding.0.weight");
        result.semantic_projector1 = weights.biased("tts.semantic_projector.input.weight", "tts.semantic_projector.input.bias");
        result.semantic_projector2 = weights.biased("tts.semantic_projector.output.weight", "tts.semantic_projector.output.bias");
        result.speaker_projector1 = weights.biased("tts.speaker_projector.input.weight", "tts.speaker_projector.input.bias");
        result.speaker_projector2 = weights.biased("tts.speaker_projector.output.weight", "tts.speaker_projector.output.bias");
#ifdef TYLOINFER_NATIVE_CUDA
        result.output_norm = weights.native("tts.output_norm.weight");
        auto head_g_tensor = weights.native("tts.code_output.0.weight_norm.magnitude");
        auto head_v_tensor = weights.native("tts.code_output.0.weight_norm.direction");
        if (head_g_tensor.dim() != 2 || head_g_tensor.size(0) != 6562 || head_g_tensor.size(1) != 1 ||
                head_v_tensor.dim() != 2 || head_v_tensor.size(0) != 6562 || head_v_tensor.size(1) != 768 ||
#else
        result.output_norm = weights.native("tts.output_norm.weight").dense();
        auto head_g = weights.native("tts.code_output.0.weight_norm.magnitude").dense();
        auto head_v = weights.native("tts.code_output.0.weight_norm.direction").dense();
        if (head_g.sizes().vec() != std::vector<int64_t>({6562, 1}) ||
                head_v.sizes().vec() != std::vector<int64_t>({6562, 768}) ||
#endif
                result.text_embedding.size(0) != 152064 ||
                result.text_embedding.size(1) != 768 ||
                result.code_embedding.size(0) != 6562 ||
                result.code_embedding.size(1) != 768) {
            throw std::runtime_error(
                "MiniCPM-o TTS tensor shapes disagree with version 4.5");
        }
#ifdef TYLOINFER_NATIVE_CUDA
        const auto& head_g = head_g_tensor.dense();
        const auto& head_v = head_v_tensor.dense();
#endif
        auto head_v_f32 = head_v.to(::tyloi::kFloat32);
        auto row_norm = ::tyloi::cuda::sqrt(
            ::tyloi::cuda::sum(head_v_f32.square(), -1, true))
            .clamp_min(1e-12);
        auto code_head = (
            head_g.to(::tyloi::kFloat32) * head_v_f32 /
            row_norm).to(head_v.scalar_type()).contiguous();
#ifdef TYLOINFER_NATIVE_CUDA
        result.code_head = std::make_shared<::tyloi::DenseTensor>(std::move(code_head));
#else
        result.code_head = std::move(code_head);
#endif
        result.layers.reserve(20);
        TransformerWeightLoader loader{weights};
        for (int index = 0; index < 20; ++index) {
            auto& layer = result.layers.emplace_back();
            const auto prefix = "tts.block." + std::to_string(index) + ".";
            tyloi::models::load_attention_mixer(layer.attention, loader, result.config, index,
                                             prefix, 0.0);
#ifdef TYLOINFER_NATIVE_CUDA
            layer.ffn_norm = weights.tensor(prefix + "mlp.norm.weight");
#else
            layer.ffn_norm = loader.dense(prefix + "mlp.norm.weight");
#endif
            layer.ffn = tyloi::models::load_dense_ffn<tyloi::models::Mlp>(loader, result.config, prefix + "mlp.");
        }
        return result;
    }

    /** Project language hidden states into normalized speech conditioning. */
    ::tyloi::DenseTensor semantic_projection(
#ifndef TYLOINFER_NATIVE_CUDA
                                                  const Projector& projector,
#endif
                                                  ::tyloi::DenseTensor hidden) const {
#ifdef TYLOINFER_NATIVE_CUDA
        auto projected = tyloi::ops::linear(hidden, semantic_projector1.weight, semantic_projector1.bias).dense();
        projected = ::tyloi::cuda::relu(projected);
        projected = tyloi::ops::linear(projected, semantic_projector2.weight, semantic_projector2.bias).dense();
#else
        auto projected = tyloi::cuda::biased(projector, semantic_projector1, hidden);
        projected = ::tyloi::cuda::relu(projected);
        projected = tyloi::cuda::biased(projector, semantic_projector2, projected);
#endif
        auto norm = ::tyloi::cuda::sqrt(
            ::tyloi::cuda::sum(projected.to(::tyloi::kFloat32).square(), -1, true))
            .clamp_min(1e-12);
        return (projected / norm.to(projected.scalar_type())).contiguous();
    }

    /** Project speaker conditioning into the speech model width. */
    ::tyloi::DenseTensor speaker_projection(
#ifndef TYLOINFER_NATIVE_CUDA
                                                 const Projector& projector,
#endif
                                                 ::tyloi::DenseTensor hidden) const {
#ifdef TYLOINFER_NATIVE_CUDA
        return tyloi::ops::linear(::tyloi::cuda::relu(
            tyloi::ops::linear(hidden, speaker_projector1.weight, speaker_projector1.bias).dense()),
            speaker_projector2.weight, speaker_projector2.bias).dense();
#else
        return tyloi::cuda::biased(projector, speaker_projector2, ::tyloi::cuda::relu(
                tyloi::cuda::biased(projector, speaker_projector1, hidden)));
#endif
    }

    /** Build speech conditioning from aligned text and language hidden states. */
    ::tyloi::DenseTensor condition(
#ifndef TYLOINFER_NATIVE_CUDA
            const Projector& projector,
#endif
            ::tyloi::DenseTensor text_ids,
            ::tyloi::DenseTensor language_hidden) const {
        MfqCudaGuard guard(device);
        if (text_ids.dim() == 1) text_ids = text_ids.unsqueeze(0);
        if (language_hidden.dim() == 2) {
            language_hidden = language_hidden.unsqueeze(0);
        }
        if (text_ids.dim() != 2 || language_hidden.dim() != 3 ||
                text_ids.size(0) != 1 || language_hidden.size(0) != 1 ||
                text_ids.size(1) != language_hidden.size(1) ||
                language_hidden.size(2) != 4096) {
            throw std::runtime_error(
                "MiniCPM-o TTS condition expects one aligned text/hidden span");
        }
        auto text = minicpmo45_embedding(text_embedding, text_ids);
        auto semantic = semantic_projection(
#ifndef TYLOINFER_NATIVE_CUDA
            projector,
#endif
            language_hidden)
            .to(text.scalar_type());
        auto merged = (text + semantic).contiguous();
        auto suffix_ids = ::tyloi::cuda::tensor(
            std::vector<int64_t>{151692, 151687},
            ::tyloi::TensorOptions().dtype(::tyloi::kInt64)
                .device(::tyloi::Device(::tyloi::kCUDA, device))).reshape({1, 2});
        auto suffix = minicpmo45_embedding(
            text_embedding, suffix_ids, text.scalar_type());
        return ::tyloi::cuda::cat({merged, suffix}, 1).contiguous();
    }

    /** Build streaming speech conditioning and its audio start token. */
    ::tyloi::DenseTensor duplex_condition(
#ifndef TYLOINFER_NATIVE_CUDA
            const Projector& projector,
#endif
            ::tyloi::DenseTensor text_ids,
            ::tyloi::DenseTensor language_hidden,
            int64_t audio_bos_token) const {
        MfqCudaGuard guard(device);
        if (text_ids.dim() == 1) text_ids = text_ids.unsqueeze(0);
        if (language_hidden.dim() == 2) {
            language_hidden = language_hidden.unsqueeze(0);
        }
        if (text_ids.dim() != 2 || language_hidden.dim() != 3 ||
                text_ids.size(0) != 1 || language_hidden.size(0) != 1 ||
                text_ids.size(1) != language_hidden.size(1) ||
                language_hidden.size(2) != 4096 ||
                audio_bos_token < 0 || audio_bos_token >= text_embedding.size(0)) {
            throw std::runtime_error(
                "MiniCPM-o duplex TTS condition has invalid geometry or audio BOS");
        }
        ::tyloi::DenseTensor condition;
        if (text_ids.size(1) > 0) {
            auto text = minicpmo45_embedding(text_embedding, text_ids);
            auto semantic = semantic_projection(
#ifndef TYLOINFER_NATIVE_CUDA
                projector,
#endif
                language_hidden)
                .to(text.scalar_type());
            condition = (text + semantic).contiguous();
        }
        auto bos_ids = ::tyloi::cuda::tensor(
            std::vector<int64_t>{audio_bos_token},
            ::tyloi::TensorOptions().dtype(::tyloi::kInt64)
                .device(::tyloi::Device(::tyloi::kCUDA, device))).reshape({1, 1});
        auto bos = minicpmo45_embedding(
            text_embedding, bos_ids, ::tyloi::kBFloat16);
        return condition.defined()
            ? ::tyloi::cuda::cat({condition, bos}, 1).contiguous()
            : bos.contiguous();
    }

    /** Advance speech decoder cache using request execution options. */
    ::tyloi::DenseTensor hidden_forward(State& state,
#ifdef TYLOINFER_NATIVE_CUDA
            const tyloi::models::ForwardOptions& mode,
#else
            Executor& executor, const Executor::Mode& mode,
#endif
            ::tyloi::DenseTensor input_embeddings) const {
        MfqCudaGuard guard(device);
        if (input_embeddings.dim() != 3 ||
                input_embeddings.size(2) != config.hidden_size) {
            throw std::runtime_error(
                "MiniCPM-o TTS embeddings must be [batch,tokens,768]");
        }
        const int64_t batch = input_embeddings.size(0);
        const int64_t tokens = input_embeddings.size(1);
        TYLOINFER_RUNTIME_CHECK(state.kv.size() == layers.size(), "MiniCPM-o TTS state layer count differs");
        tyloi::models::require_model(
            batch > 0 && tokens > 0 && state.position >= 0 &&
                state.position <= config.max_position_embeddings &&
                tokens <= config.max_position_embeddings - state.position,
            "MiniCPM-o TTS context exceeds capacity or has invalid geometry");
        if (state.position == 0) state.reset(batch);
        auto positions = ::tyloi::cuda::arange(
            state.position, state.position + tokens,
            ::tyloi::TensorOptions().device(::tyloi::Device(::tyloi::kCUDA, device)).dtype(::tyloi::kInt64));
        MfqOptional<::tyloi::DenseTensor> sequence_length = mfq_nullopt;
        if (state.position > 0) {
            sequence_length = ::tyloi::cuda::full(
                {batch}, state.position + tokens,
                ::tyloi::TensorOptions().device(::tyloi::Device(::tyloi::kCUDA, device)).dtype(::tyloi::kInt64));
        }
        auto hidden = input_embeddings.to(::tyloi::kBFloat16).contiguous();
        input_embeddings = {};
        for (const auto& layer : layers) {
            const auto& attention = layer.attention;
            auto& cache = state.kv.at(attention.layer);
            if (!cache.defined()) {
                const auto capacity = mode.kv_capacity > 0
                    ? std::max(state.position + tokens, mode.kv_capacity)
                    : attention.max_position_embeddings;
                cache = tyloi::state::allocate_kv(
                    batch, attention.kv_heads, capacity,
                    attention.attention_head_dim, false, hidden.device(),
                    ::tyloi::kFloat16, attention.kv_quantization);
            }
            cache.quantization = attention.kv_quantization;
            auto* layer_trace = mode.trace_layer == attention.layer ? mode.trace : nullptr;
#ifdef TYLOINFER_NATIVE_CUDA
            AttentionStep step{
                ::tyloi::Tensor(positions),
                sequence_length ? ::tyloi::Tensor(*sequence_length) : ::tyloi::Tensor{},
                {}, state.position, 0, 0};
            auto output = attention.forward(::tyloi::Tensor(std::move(hidden)), step,
                cache, rope, layer_trace);
            hidden = ::tyloi::models::full_ffn(
                layer.ffn, layer.ffn_norm, output[0], output[1],
                attention.rms_norm_eps, attention.norm_weight_offset,
                attention.official_bf16, attention.attention_output_gate, layer_trace).dense();
#else
            AttentionStep step{positions, sequence_length, {}, state.position, 0, 0};
            auto output = attention.forward(executor.projector(mode.ffn), executor.profiler(),
                executor.attention_options(), std::move(hidden), step, cache,
                executor.attention_scratch(2 * attention.layer), rope, layer_trace);
            hidden = ::tyloi::models::full_ffn(
                layer.ffn, layer.ffn_norm, std::move(output[0]), std::move(output[1]),
                attention.rms_norm_eps, attention.norm_weight_offset,
                attention.official_bf16, attention.attention_output_gate, layer_trace).dense();
#endif
        }
        state.position += tokens;
        auto flat = hidden.reshape({batch * tokens, config.hidden_size});
        auto normalized_f32 = flat.to(::tyloi::kFloat32);
        normalized_f32 = normalized_f32 * ::tyloi::cuda::rsqrt(
            ::tyloi::cuda::mean(normalized_f32.square(), -1, true) + config.rms_norm_eps);
#ifdef TYLOINFER_NATIVE_CUDA
        const auto& weight = output_norm.dense();
#else
        const auto& weight = output_norm;
#endif
        return (normalized_f32.to(flat.scalar_type()) * weight.to(flat.scalar_type()))
            .reshape({batch, tokens, config.hidden_size})
            .contiguous();
    }

    ::tyloi::DenseTensor logits(::tyloi::DenseTensor hidden) const {
#ifdef TYLOINFER_NATIVE_CUDA
        const auto& weight = code_head.dense();
#else
        const auto& weight = code_head;
#endif
        return ::tyloi::cuda::matmul(
            hidden.to(weight.scalar_type()),
            weight.transpose(0, 1));
    }

    struct ChunkResult {
        ::tyloi::DenseTensor codes;
        bool finished = false;
    };

    /** Generate one streaming speech token chunk. */
    ChunkResult generate_duplex_chunk(
            State& state,
#ifdef TYLOINFER_NATIVE_CUDA
            const tyloi::models::ForwardOptions& mode,
#else
            Executor& executor, const Executor::Mode& mode,
#endif
            ::tyloi::DenseTensor condition_embeddings,
            int64_t max_new_tokens,
            int64_t minimum_new_tokens,
            int64_t eos_token,
            double temperature,
            double repetition_penalty) const {
        MfqCudaGuard guard(device);
        if (condition_embeddings.dim() != 3 ||
                condition_embeddings.size(0) != 1 ||
                condition_embeddings.size(2) != config.hidden_size ||
                max_new_tokens <= 0 || minimum_new_tokens < 0 ||
                eos_token < 0 || eos_token >= code_embedding.size(0) ||
                temperature <= 0.0 || repetition_penalty <= 0.0) {
            throw std::runtime_error(
                "MiniCPM-o duplex TTS generation limits are invalid");
        }
        std::vector<::tyloi::DenseTensor> sampled;
        sampled.reserve(static_cast<size_t>(max_new_tokens));
        auto current = condition_embeddings.to(::tyloi::kBFloat16).contiguous();
        bool finished = false;
        for (int64_t step = 0; step < max_new_tokens; ++step) {
            auto hidden = hidden_forward(state,
#ifndef TYLOINFER_NATIVE_CUDA
                executor,
#endif
                mode, current);
            auto step_logits = logits(
                hidden.index({Slice(), -1, Slice()}))
                .to(::tyloi::kFloat32) / temperature;
            if (!sampled.empty() && repetition_penalty != 1.0) {
                const size_t begin = sampled.size() > 16
                    ? sampled.size() - 16 : 0;
                auto counts = ::tyloi::cuda::zeros_like(step_logits);
                for (size_t index = begin; index < sampled.size(); ++index) {
                    counts.scatter_add_(
                        1, sampled[index].reshape({1, 1}),
                        ::tyloi::cuda::ones({1, 1}, counts.options()));
                }
                auto alpha = ::tyloi::cuda::pow(
                    ::tyloi::cuda::full_like(counts, repetition_penalty), counts);
                step_logits = ::tyloi::cuda::where(
                    step_logits < 0,
                    step_logits * alpha,
                    step_logits / alpha);
            }
            if (step < minimum_new_tokens) {
                step_logits.index_put_(
                    {Slice(), eos_token},
                    -std::numeric_limits<float>::infinity());
            }
            auto token = ::tyloi::cuda::multinomial(
                ::tyloi::cuda::softmax(step_logits, -1), 1)
                .reshape({1}).to(::tyloi::kInt64);
            sampled.push_back(token);
            finished = token.eq(eos_token).all().item<bool>();
            if (finished) break;
            current = minicpmo45_embedding(
                code_embedding, token.unsqueeze(1));
        }
        const int64_t returned = std::max<int64_t>(
            0, static_cast<int64_t>(sampled.size()) - 1);
        ::tyloi::DenseTensor codes;
        if (returned > 0) {
            codes = ::tyloi::cuda::stack(sampled, 1)
                .narrow(1, 0, returned)
                .unsqueeze(-1).contiguous();
        } else {
            codes = ::tyloi::cuda::empty(
                {1, 0, 1},
                ::tyloi::TensorOptions().device(::tyloi::Device(::tyloi::kCUDA, device))
                    .dtype(::tyloi::kInt64));
        }
        return {codes, finished};
    }

    /** Retain the state and sampling history for bounded official generation steps. */
    struct OfficialGeneration {
        const Tts& model;
        State& state;
#ifdef TYLOINFER_NATIVE_CUDA
        tyloi::models::ForwardOptions mode;
#else
        Executor& executor;
        Executor::Mode mode;
#endif
        tyloi::models::minicpmo45::TtsSampling config;
        int64_t steps;
        ::tyloi::DenseTensor current;
        std::vector<::tyloi::DenseTensor> generated;
        std::vector<::tyloi::DenseTensor>* logits_trace;
        MiniCPMO45TtsSamplingOps sampling_ops;
        bool done = false, hit_eos = false;

        OfficialGeneration(const Tts& model_, State& state_,
#ifdef TYLOINFER_NATIVE_CUDA
            const tyloi::models::ForwardOptions& mode_,
#else
            Executor& executor_, const Executor::Mode& mode_,
#endif
            ::tyloi::DenseTensor condition, int64_t steps_,
            tyloi::models::minicpmo45::TtsSampling config_,
            std::vector<::tyloi::DenseTensor>* trace, std::mt19937* rng)
            : model(model_), state(state_),
#ifndef TYLOINFER_NATIVE_CUDA
              executor(executor_),
#endif
              mode(mode_), config(config_), steps(steps_), current(std::move(condition)),
              logits_trace(trace), sampling_ops{rng, model_.device} {
            MfqCudaGuard guard(model.device);
            config.validate(steps, 6562);
            if (current.dim() != 3 || current.size(0) != 1)
                throw std::runtime_error("official MiniCPM-o TTS generation requires batch size one");
            state.reset(1);
            generated.reserve(static_cast<size_t>(steps));
        }

        /** Generate one token and retain all request state for the next step. */
        bool next() {
            if (done) return false;
            MfqCudaGuard guard(model.device);
            auto hidden = model.hidden_forward(state,
#ifndef TYLOINFER_NATIVE_CUDA
                executor,
#endif
                mode, current);
            auto scores = model.logits(hidden.index({Slice(), -1, Slice()})).to(::tyloi::kFloat32);
            if (logits_trace) logits_trace->push_back(scores.clone());
            auto token = tyloi::models::minicpmo45::sample_tts(sampling_ops, std::move(scores), config,
                std::span<const ::tyloi::DenseTensor>(generated), static_cast<int64_t>(generated.size()),
                sampling_ops.evaluator_rng != nullptr);
            generated.push_back(token);
            hit_eos = token.eq(config.eos_token).all().item<bool>();
            done = hit_eos || static_cast<int64_t>(generated.size()) == steps;
            if (!done) current = minicpmo45_embedding(model.code_embedding, token.unsqueeze(1));
            return true;
        }

        /** Return sampled codes after generation has completed. */
        ::tyloi::DenseTensor codes() const {
            if (!done) throw std::logic_error("speech generation is incomplete");
            MfqCudaGuard guard(model.device);
            auto sampled = ::tyloi::cuda::stack(generated, 1);
            return sampled.narrow(1, 0, static_cast<int64_t>(generated.size()) - (hit_eos ? 1 : 0))
                .unsqueeze(-1).contiguous();
        }
    };

    ::tyloi::DenseTensor generate_official(
        State& state,
#ifdef TYLOINFER_NATIVE_CUDA
        const tyloi::models::ForwardOptions& mode,
#else
        Executor& executor, const Executor::Mode& mode,
#endif
        ::tyloi::DenseTensor condition_embeddings, int64_t steps, int64_t eos_token = 6561,
        int64_t minimum_steps = 50, double temperature = 0.8, double top_p = 0.85,
        int64_t top_k = 25, double repetition_penalty = 1.05,
        std::vector<::tyloi::DenseTensor> *logits_trace = nullptr, double min_p = 0.0,
        std::mt19937 *evaluator_rng = nullptr, int64_t evaluator_min_keep = 0) const {
        const tyloi::models::minicpmo45::TtsSampling config{eos_token,          minimum_steps, top_k,
                                                          evaluator_min_keep, temperature,   top_p,
                                                          repetition_penalty, min_p};
        OfficialGeneration generation(*this, state,
#ifndef TYLOINFER_NATIVE_CUDA
            executor,
#endif
            mode, std::move(condition_embeddings), steps, config, logits_trace, evaluator_rng);
        while (generation.next()) {}
        return generation.codes();
    }
};

} // namespace tyloi::cuda::minicpmo45
