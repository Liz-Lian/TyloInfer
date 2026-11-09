#pragma once

#include "engine/weights/loader.h"
#include "engine/models/minicpmo45/media.h"

#include "vision.h"
#include <random>
#include "engine/requests/policy.h"
#include "engine/generation/sampling.h"
#include "engine/models/common/load.h"
#include "engine/models/common/attention.h"
#include "engine/backends/cuda/ops/ffn/ffn.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/ops/rope/rope.h"
#include "engine/state/kv_view.h"

#include <array>

using ::tyloi::indexing::Slice;

struct MiniCPMO45TtsSamplingOps {
    using DenseTensor = ::tyloi::DenseTensor;
    std::mt19937 *evaluator_rng = nullptr;
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
                                              .device(::tyloi::kCUDA)
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

struct Tts {
    struct Layer {
        FullAttention attention;
#ifdef TYLOINFER_NATIVE_CUDA
        std::shared_ptr<const ::tyloi::Tensor> ffn_norm;
#else
        ::tyloi::DenseTensor ffn_norm;
#endif
        Ffn ffn;
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
    std::shared_ptr<const ::tyloi::Tensor> text_embedding;
    std::shared_ptr<const ::tyloi::Tensor> code_embedding;
    tyloi::weights::BiasWeight semantic_projector1;
    tyloi::weights::BiasWeight semantic_projector2;
    tyloi::weights::BiasWeight speaker_projector1;
    tyloi::weights::BiasWeight speaker_projector2;
    std::vector<Layer> layers;
#ifdef TYLOINFER_NATIVE_CUDA
    std::shared_ptr<const ::tyloi::Tensor> output_norm, code_head;
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
        if (head_g_tensor->dim() != 2 || head_g_tensor->size(0) != 6562 || head_g_tensor->size(1) != 1 ||
                head_v_tensor->dim() != 2 || head_v_tensor->size(0) != 6562 || head_v_tensor->size(1) != 768 ||
#else
        result.output_norm = dynamic_cast<const ::tyloi::DenseTensor&>(*weights.native("tts.output_norm.weight"));
        auto head_g = dynamic_cast<const ::tyloi::DenseTensor&>(*weights.native("tts.code_output.0.weight_norm.magnitude"));
        auto head_v = dynamic_cast<const ::tyloi::DenseTensor&>(*weights.native("tts.code_output.0.weight_norm.direction"));
        if (head_g.sizes().vec() != std::vector<int64_t>({6562, 1}) ||
                head_v.sizes().vec() != std::vector<int64_t>({6562, 768}) ||
#endif
                result.text_embedding->size(0) != 152064 ||
                result.text_embedding->size(1) != 768 ||
                result.code_embedding->size(0) != 6562 ||
                result.code_embedding->size(1) != 768) {
            throw std::runtime_error(
                "MiniCPM-o TTS tensor shapes disagree with version 4.5");
        }
#ifdef TYLOINFER_NATIVE_CUDA
        const auto& head_g = dynamic_cast<const ::tyloi::DenseTensor&>(*head_g_tensor);
        const auto& head_v = dynamic_cast<const ::tyloi::DenseTensor&>(*head_v_tensor);
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
            layer.ffn = tyloi::models::load_dense_ffn<Ffn>(loader, result.config, prefix + "mlp.");
        }
        return result;
    }

    ::tyloi::DenseTensor semantic_projection(const Projector& projector,
                                                  ::tyloi::DenseTensor hidden) const {
        auto projected = tyloi::cuda::biased(projector, semantic_projector1, hidden);
        projected = ::tyloi::cuda::relu(projected);
        projected = tyloi::cuda::biased(projector, semantic_projector2, projected);
        auto norm = ::tyloi::cuda::sqrt(
            ::tyloi::cuda::sum(projected.to(::tyloi::kFloat32).square(), -1, true))
            .clamp_min(1e-12);
        return (projected / norm.to(projected.scalar_type())).contiguous();
    }

    ::tyloi::DenseTensor speaker_projection(const Projector& projector,
                                                 ::tyloi::DenseTensor hidden) const {
        return tyloi::cuda::biased(projector, speaker_projector2, ::tyloi::cuda::relu(
                tyloi::cuda::biased(projector, speaker_projector1, hidden)));
    }

    ::tyloi::DenseTensor condition(
            const Projector& projector,
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
        auto text = minicpmo45_embedding(*text_embedding, text_ids);
        auto semantic = semantic_projection(projector, language_hidden)
            .to(text.scalar_type());
        auto merged = (text + semantic).contiguous();
        auto suffix_ids = ::tyloi::cuda::tensor(
            std::vector<int64_t>{151692, 151687},
            ::tyloi::TensorOptions().dtype(::tyloi::kInt64)
                .device(::tyloi::kCUDA)).reshape({1, 2});
        auto suffix = minicpmo45_embedding(
            *text_embedding, suffix_ids, text.scalar_type());
        return ::tyloi::cuda::cat({merged, suffix}, 1).contiguous();
    }

    ::tyloi::DenseTensor duplex_condition(
            const Projector& projector,
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
                audio_bos_token < 0 || audio_bos_token >= text_embedding->size(0)) {
            throw std::runtime_error(
                "MiniCPM-o duplex TTS condition has invalid geometry or audio BOS");
        }
        ::tyloi::DenseTensor condition;
        if (text_ids.size(1) > 0) {
            auto text = minicpmo45_embedding(*text_embedding, text_ids);
            auto semantic = semantic_projection(projector, language_hidden)
                .to(text.scalar_type());
            condition = (text + semantic).contiguous();
        }
        auto bos_ids = ::tyloi::cuda::tensor(
            std::vector<int64_t>{audio_bos_token},
            ::tyloi::TensorOptions().dtype(::tyloi::kInt64)
                .device(::tyloi::kCUDA)).reshape({1, 1});
        auto bos = minicpmo45_embedding(
            *text_embedding, bos_ids, ::tyloi::kBFloat16);
        return condition.defined()
            ? ::tyloi::cuda::cat({condition, bos}, 1).contiguous()
            : bos.contiguous();
    }

    ::tyloi::DenseTensor hidden_forward(State& state, Executor& executor,
            const Executor::Mode& mode, ::tyloi::DenseTensor input_embeddings) const {
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
            ::tyloi::TensorOptions().device(::tyloi::kCUDA).dtype(::tyloi::kInt64));
        MfqOptional<::tyloi::DenseTensor> sequence_length = mfq_nullopt;
        if (state.position > 0) {
            sequence_length = ::tyloi::cuda::full(
                {batch}, state.position + tokens,
                ::tyloi::TensorOptions().device(::tyloi::kCUDA).dtype(::tyloi::kInt64));
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
            const std::array<tyloi::state::KvView, 1> kv{cache};
            const std::array<Task, 2> tasks{{
                {task::Attention{0, 1, 2, attention, 0}, attention.layer},
#ifdef TYLOINFER_NATIVE_CUDA
                {task::FullFfn{1, 2, 0, layer.ffn, *layer.ffn_norm, attention}, attention.layer},
#else
                {task::FullFfn{1, 2, 0, layer.ffn, layer.ffn_norm, attention}, attention.layer},
#endif
            }};
            Inputs inputs;
            inputs.values = {std::move(hidden), {}, {}};
            inputs.kv = kv;
            inputs.attention.positions = positions;
            inputs.attention.sequence_lengths = sequence_length;
            inputs.attention.cache_position = state.position;
#ifdef TYLOINFER_NATIVE_CUDA
            inputs.rope = rope;
#else
            inputs.rope = &rope;
#endif
            inputs.trace = mode.trace;
            inputs.trace_scope = mode.trace_layer;
            inputs.scratch_offset = 2 * attention.layer;
            hidden = executor.run(tasks, std::move(inputs), mode);
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
        const auto& weight = dynamic_cast<const ::tyloi::DenseTensor&>(*code_head);
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

    ChunkResult generate_duplex_chunk(
            State& state, Executor& executor, const Executor::Mode& mode,
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
                eos_token < 0 || eos_token >= code_embedding->size(0) ||
                temperature <= 0.0 || repetition_penalty <= 0.0) {
            throw std::runtime_error(
                "MiniCPM-o duplex TTS generation limits are invalid");
        }
        std::vector<::tyloi::DenseTensor> sampled;
        sampled.reserve(static_cast<size_t>(max_new_tokens));
        auto current = condition_embeddings.to(::tyloi::kBFloat16).contiguous();
        bool finished = false;
        for (int64_t step = 0; step < max_new_tokens; ++step) {
            auto hidden = hidden_forward(state, executor, mode, current);
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
                *code_embedding, token.unsqueeze(1));
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
                ::tyloi::TensorOptions().device(::tyloi::kCUDA)
                    .dtype(::tyloi::kInt64));
        }
        return {codes, finished};
    }

    ::tyloi::DenseTensor generate_official(
        State& state, Executor& executor, const Executor::Mode& mode,
        ::tyloi::DenseTensor condition_embeddings, int64_t steps, int64_t eos_token = 6561,
        int64_t minimum_steps = 50, double temperature = 0.8, double top_p = 0.85,
        int64_t top_k = 25, double repetition_penalty = 1.05,
        std::vector<::tyloi::DenseTensor> *logits_trace = nullptr, double min_p = 0.0,
        std::mt19937 *evaluator_rng = nullptr, int64_t evaluator_min_keep = 0) const {
        MfqCudaGuard guard(device);
        const tyloi::models::minicpmo45::TtsSampling config{eos_token,          minimum_steps, top_k,
                                                          evaluator_min_keep, temperature,   top_p,
                                                          repetition_penalty, min_p};
        config.validate(steps, 6562);
        if (condition_embeddings.dim() != 3 || condition_embeddings.size(0) != 1)
            throw std::runtime_error("official MiniCPM-o TTS generation requires batch size one");
        state.reset(1);
        using DenseTensor = ::tyloi::DenseTensor;
        std::vector<DenseTensor> generated;
        generated.reserve(static_cast<size_t>(steps));
        auto current = condition_embeddings;
        MiniCPMO45TtsSamplingOps sampling_ops{evaluator_rng};
        const auto result = tyloi::engine::generate_tokens(
            steps,
            [&](int64_t step) {
                auto hidden = hidden_forward(state, executor, mode, current);
                auto raw_step_logits =
                    logits(hidden.index({Slice(), -1, Slice()})).to(::tyloi::kFloat32);
                if (logits_trace)
                    logits_trace->push_back(raw_step_logits.clone());
                return tyloi::models::minicpmo45::sample_tts(
                    sampling_ops, std::move(raw_step_logits), config,
                    std::span<const DenseTensor>(generated), step, evaluator_rng != nullptr);
            },
            [&](const DenseTensor &token, int64_t) { generated.push_back(token); },
            [&](const DenseTensor &token) { return token.eq(eos_token).all().item<bool>(); },
            [&](const DenseTensor &token) {
                current = minicpmo45_embedding(*code_embedding, token.unsqueeze(1));
            });
        auto sampled = ::tyloi::cuda::stack(generated, 1);
        return sampled.narrow(1, 0, result.tokens - (result.hit_eos ? 1 : 0))
            .unsqueeze(-1)
            .contiguous();
    }
};

} // namespace tyloi::cuda::minicpmo45
