#pragma once

#include "engine/weights/loader.h"

#include "vision.h"
#include <random>
#include "engine/requests/policy.h"
#include "engine/executor/sampling.h"
#include "backends/cuda/engine/models/load.h"
#include "backends/cuda/engine/executor/run.h"
#include "engine/state/kv_cache.h"

#include <array>

using mfq_tensor_backend::indexing::Slice;

struct MiniCPMO45TtsSamplingOps {
    using Tensor = mfq_tensor_backend::Tensor;
    std::mt19937 *evaluator_rng = nullptr;
    static Tensor temperature(Tensor scores, double value) { return scores / value; }
    static Tensor penalties(Tensor scores, std::span<const Tensor> history, double penalty) {
        auto counts = mfq_tensor_backend::zeros_like(scores);
        for (const auto &token : history)
            counts.scatter_add_(1, token.reshape({1, 1}),
                                mfq_tensor_backend::ones({1, 1}, counts.options()));
        auto alpha =
            mfq_tensor_backend::pow(mfq_tensor_backend::full_like(counts, penalty), counts);
        return mfq_tensor_backend::where(scores < 0, scores * alpha, scores / alpha);
    }
    static void mask_eos(Tensor &scores, int64_t token) {
        scores.index_put_({Slice(), token}, -std::numeric_limits<float>::infinity());
    }
    Tensor reference(Tensor scores, int64_t top_k, double top_p, int64_t minimum_keep) const {
        auto host = scores.reshape({-1})
                        .to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kFloat32)
                        .contiguous();
        auto token = mfq::engine::sample_top_k_top_p(
            std::span<const float>(host.data_ptr<float>(), host.numel()), top_k, top_p,
            minimum_keep, *evaluator_rng);
        return mfq_tensor_backend::tensor(std::vector<int64_t>{token},
                                          mfq_tensor_backend::TensorOptions()
                                              .device(mfq_tensor_backend::kCUDA)
                                              .dtype(mfq_tensor_backend::kInt64));
    }
    static Tensor top_k(Tensor scores, int64_t count) {
        if (count >= scores.size(1))
            return scores;
        auto values = std::get<0>(mfq_tensor_backend::topk(scores, count, -1, true, true));
        auto threshold = values.select(1, count - 1).unsqueeze(1);
        return scores.masked_fill(scores < threshold, -std::numeric_limits<float>::infinity());
    }
    static Tensor top_p(Tensor scores, double value) {
        auto sorted = mfq_tensor_backend::sort(scores, -1, true);
        auto logits = std::get<0>(sorted);
        auto indices = std::get<1>(sorted);
        auto cumulative = mfq_tensor_backend::cumsum(mfq_tensor_backend::softmax(logits, -1), -1);
        auto keep = cumulative <= value;
        keep.narrow(1, 1, keep.size(1) - 1).copy_(keep.narrow(1, 0, keep.size(1) - 1).clone());
        keep.index_put_({Slice(), 0}, true);
        logits = logits.masked_fill(keep.logical_not(), -std::numeric_limits<float>::infinity());
        scores = mfq_tensor_backend::full_like(scores, -std::numeric_limits<float>::infinity());
        scores.scatter_(1, indices, logits);
        return scores;
    }
    static Tensor min_p(Tensor scores, double value) {
        auto maximum = std::get<0>(mfq_tensor_backend::max(scores, -1, true));
        return scores.masked_fill(scores < maximum + std::log(value),
                                  -std::numeric_limits<float>::infinity());
    }
    static Tensor sample(Tensor scores) {
        return mfq_tensor_backend::multinomial(mfq_tensor_backend::softmax(scores, -1), 1)
            .reshape({1})
            .to(mfq_tensor_backend::kInt64);
    }
};

namespace mfq::cuda::minicpmo45 {

struct Tts {
    struct Layer {
        FullAttention attention;
        mfq_tensor_backend::Tensor ffn_norm;
        Ffn ffn;
    };

    struct State {
        explicit State(const Tts& model) : kv(model.layers.size()) {}
        std::vector<KVCache> kv;
        int64_t position = 0;

        void reset(int64_t batch) {
            position = 0;
            for (auto& cache : kv)
                if (cache.defined() && cache.batch_size() != batch) cache = {};
        }
    };

    mfq::models::ModelConfig config;
    int device = 0;
    RopeCache rope;
    QuantLinear text_embedding;
    QuantLinear code_embedding;
    MiniCPMO45Linear semantic_projector1;
    MiniCPMO45Linear semantic_projector2;
    MiniCPMO45Linear speaker_projector1;
    MiniCPMO45Linear speaker_projector2;
    std::vector<Layer> layers;
    mfq_tensor_backend::Tensor output_norm;
    mfq_tensor_backend::Tensor code_head;
    static Tts load(const weight_loader::Loader& weights) {
        MfqCudaGuard guard(weights.placement.device);
        Tts result;
        result.device = weights.placement.device;
        result.config = mfq::models::minicpmo45::tts_decoder_config();
        result.rope = RopeCache(
            result.config.max_position_embeddings,
            result.config.rotary_dim,
            result.config.rope_base, 0, -1,
            mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, result.device));
        result.text_embedding = weights.projection("tts.text_embedding.weight");
        result.code_embedding = weights.projection("tts.code_embedding.0.weight");
        result.semantic_projector1 = mfq::cuda::vision::load_biased(weights, "tts.semantic_projector.input");
        result.semantic_projector2 = mfq::cuda::vision::load_biased(weights, "tts.semantic_projector.output");
        result.speaker_projector1 = mfq::cuda::vision::load_biased(weights, "tts.speaker_projector.input");
        result.speaker_projector2 = mfq::cuda::vision::load_biased(weights, "tts.speaker_projector.output");
        result.output_norm = weights.native("tts.output_norm.weight");
        auto head_g = weights.native("tts.code_output.0.weight_norm.magnitude");
        auto head_v = weights.native("tts.code_output.0.weight_norm.direction");
        if (head_g.sizes().vec() != std::vector<int64_t>({6562, 1}) ||
                head_v.sizes().vec() != std::vector<int64_t>({6562, 768}) ||
                result.text_embedding.out() != 152064 ||
                result.text_embedding.neuron_len() != 768 ||
                result.code_embedding.out() != 6562 ||
                result.code_embedding.neuron_len() != 768) {
            throw std::runtime_error(
                "MiniCPM-o TTS tensor shapes disagree with version 4.5");
        }
        auto head_v_f32 = head_v.to(mfq_tensor_backend::kFloat32);
        auto row_norm = mfq_tensor_backend::sqrt(
            mfq_tensor_backend::sum(head_v_f32.square(), -1, true))
            .clamp_min(1e-12);
        result.code_head = (
            head_g.to(mfq_tensor_backend::kFloat32) * head_v_f32 /
            row_norm).to(head_v.scalar_type()).contiguous();
        result.layers.reserve(20);
        TransformerWeightLoader loader{weights};
        for (int index = 0; index < 20; ++index) {
            auto& layer = result.layers.emplace_back();
            const auto prefix = "tts.block." + std::to_string(index) + ".";
            mfq::models::load_attention_mixer(layer.attention, loader, result.config, index,
                                             prefix, 0.0);
            layer.ffn_norm = loader.dense(prefix + "mlp.norm.weight");
            layer.ffn = mfq::models::load_dense_ffn<Ffn>(loader, result.config, prefix + "mlp.");
        }
        return result;
    }

    mfq_tensor_backend::Tensor semantic_projection(const Projector& projector,
                                                  mfq_tensor_backend::Tensor hidden) const {
        auto projected = semantic_projector1.forward(
            projector, hidden);
        projected = mfq_tensor_backend::relu(projected);
        projected = semantic_projector2.forward(
            projector, projected);
        auto norm = mfq_tensor_backend::sqrt(
            mfq_tensor_backend::sum(projected.to(mfq_tensor_backend::kFloat32).square(), -1, true))
            .clamp_min(1e-12);
        return (projected / norm.to(projected.scalar_type())).contiguous();
    }

    mfq_tensor_backend::Tensor speaker_projection(const Projector& projector,
                                                 mfq_tensor_backend::Tensor hidden) const {
        return speaker_projector2.forward(
            projector,
            mfq_tensor_backend::relu(
                speaker_projector1.forward(projector, hidden)));
    }

    mfq_tensor_backend::Tensor condition(
            const Projector& projector,
            mfq_tensor_backend::Tensor text_ids,
            mfq_tensor_backend::Tensor language_hidden) const {
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
        auto semantic = semantic_projection(projector, language_hidden)
            .to(text.scalar_type());
        auto merged = (text + semantic).contiguous();
        auto suffix_ids = mfq_tensor_backend::tensor(
            std::vector<int64_t>{151692, 151687},
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)
                .device(mfq_tensor_backend::kCUDA)).reshape({1, 2});
        auto suffix = minicpmo45_embedding(
            text_embedding, suffix_ids, text.scalar_type());
        return mfq_tensor_backend::cat({merged, suffix}, 1).contiguous();
    }

    mfq_tensor_backend::Tensor duplex_condition(
            const Projector& projector,
            mfq_tensor_backend::Tensor text_ids,
            mfq_tensor_backend::Tensor language_hidden,
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
                audio_bos_token < 0 || audio_bos_token >= text_embedding.out()) {
            throw std::runtime_error(
                "MiniCPM-o duplex TTS condition has invalid geometry or audio BOS");
        }
        mfq_tensor_backend::Tensor condition;
        if (text_ids.size(1) > 0) {
            auto text = minicpmo45_embedding(text_embedding, text_ids);
            auto semantic = semantic_projection(projector, language_hidden)
                .to(text.scalar_type());
            condition = (text + semantic).contiguous();
        }
        auto bos_ids = mfq_tensor_backend::tensor(
            std::vector<int64_t>{audio_bos_token},
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)
                .device(mfq_tensor_backend::kCUDA)).reshape({1, 1});
        auto bos = minicpmo45_embedding(
            text_embedding, bos_ids, mfq_tensor_backend::kBFloat16);
        return condition.defined()
            ? mfq_tensor_backend::cat({condition, bos}, 1).contiguous()
            : bos.contiguous();
    }

    mfq_tensor_backend::Tensor hidden_forward(State& state, Executor& executor,
            const Executor::Mode& mode, mfq_tensor_backend::Tensor input_embeddings) const {
        MfqCudaGuard guard(device);
        if (input_embeddings.dim() != 3 ||
                input_embeddings.size(2) != config.hidden_size) {
            throw std::runtime_error(
                "MiniCPM-o TTS embeddings must be [batch,tokens,768]");
        }
        const int64_t batch = input_embeddings.size(0);
        const int64_t tokens = input_embeddings.size(1);
        TYLOINFER_RUNTIME_CHECK(state.kv.size() == layers.size(), "MiniCPM-o TTS state layer count differs");
        return mfq::models::minicpmo45::tts_forward(
            std::move(input_embeddings), batch, tokens, config.max_position_embeddings,
            state.position, layers,
            [&] { state.reset(batch); },
            [&](auto value) { return value.to(mfq_tensor_backend::kBFloat16).contiguous(); },
            [&] {
                auto positions = mfq_tensor_backend::arange(
                    state.position, state.position + tokens,
                    mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA)
                        .dtype(mfq_tensor_backend::kInt64));
                MfqOptional<mfq_tensor_backend::Tensor> sequence_length = mfq_nullopt;
                if (state.position > 0) {
                    sequence_length = mfq_tensor_backend::full(
                        {batch}, state.position + tokens,
                        mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA)
                            .dtype(mfq_tensor_backend::kInt64));
                }
                return std::make_pair(positions, sequence_length);
            },
            [&](const Layer& layer, auto hidden, const auto& positions) {
                const auto& attention = layer.attention;
                auto& cache = state.kv.at(attention.layer);
                if (!cache.defined()) {
                    const auto capacity = mode.kv_capacity > 0
                        ? std::max(state.position + tokens, mode.kv_capacity)
                        : attention.max_position_embeddings;
                    cache = KVCache(batch, attention.kv_heads, capacity, attention.attention_head_dim,
                        false, hidden.device(), mfq_tensor_backend::kFloat16, attention.kv_quantization);
                }
                cache.quantization = attention.kv_quantization;
                const std::array<AttentionKvView, 1> kv{cache.view()};
                const std::array<Task, 2> tasks{{
                    {task::Attention{0, 1, 2, attention, 0}, attention.layer},
                    {task::FullFfn{1, 2, 0, layer.ffn, layer.ffn_norm, attention}, attention.layer},
                }};
                Inputs inputs;
                inputs.values = {std::move(hidden), {}, {}};
                inputs.kv = kv;
                inputs.attention.positions = positions.first;
                inputs.attention.sequence_lengths = positions.second;
                inputs.attention.cache_position = state.position;
                inputs.rope = &rope;
                inputs.trace = mode.trace;
                inputs.trace_scope = mode.trace_layer;
                inputs.scratch_offset = 2 * attention.layer;
                return executor.run(tasks, std::move(inputs), mode);
            },
            [&](auto hidden) {
                auto flat = hidden.reshape(
                    {batch * tokens, config.hidden_size});
                auto normalized_f32 = flat.to(mfq_tensor_backend::kFloat32);
                normalized_f32 = normalized_f32 * mfq_tensor_backend::rsqrt(
                    mfq_tensor_backend::mean(normalized_f32.square(), -1, true) +
                    config.rms_norm_eps);
                return (
                    normalized_f32.to(flat.scalar_type()) *
                    output_norm.to(flat.scalar_type()))
                    .reshape({batch, tokens, config.hidden_size})
                    .contiguous();
            });
    }

    mfq_tensor_backend::Tensor logits(mfq_tensor_backend::Tensor hidden) const {
        return mfq_tensor_backend::matmul(
            hidden.to(code_head.scalar_type()),
            code_head.transpose(0, 1));
    }

    struct ChunkResult {
        mfq_tensor_backend::Tensor codes;
        bool finished = false;
    };

    ChunkResult generate_duplex_chunk(
            State& state, Executor& executor, const Executor::Mode& mode,
            mfq_tensor_backend::Tensor condition_embeddings,
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
                eos_token < 0 || eos_token >= code_embedding.out() ||
                temperature <= 0.0 || repetition_penalty <= 0.0) {
            throw std::runtime_error(
                "MiniCPM-o duplex TTS generation limits are invalid");
        }
        std::vector<mfq_tensor_backend::Tensor> sampled;
        sampled.reserve(static_cast<size_t>(max_new_tokens));
        auto current = condition_embeddings.to(mfq_tensor_backend::kBFloat16).contiguous();
        bool finished = false;
        for (int64_t step = 0; step < max_new_tokens; ++step) {
            auto hidden = hidden_forward(state, executor, mode, current);
            auto step_logits = logits(
                hidden.index({Slice(), -1, Slice()}))
                .to(mfq_tensor_backend::kFloat32) / temperature;
            if (!sampled.empty() && repetition_penalty != 1.0) {
                const size_t begin = sampled.size() > 16
                    ? sampled.size() - 16 : 0;
                auto counts = mfq_tensor_backend::zeros_like(step_logits);
                for (size_t index = begin; index < sampled.size(); ++index) {
                    counts.scatter_add_(
                        1, sampled[index].reshape({1, 1}),
                        mfq_tensor_backend::ones({1, 1}, counts.options()));
                }
                auto alpha = mfq_tensor_backend::pow(
                    mfq_tensor_backend::full_like(counts, repetition_penalty), counts);
                step_logits = mfq_tensor_backend::where(
                    step_logits < 0,
                    step_logits * alpha,
                    step_logits / alpha);
            }
            if (step < minimum_new_tokens) {
                step_logits.index_put_(
                    {Slice(), eos_token},
                    -std::numeric_limits<float>::infinity());
            }
            auto token = mfq_tensor_backend::multinomial(
                mfq_tensor_backend::softmax(step_logits, -1), 1)
                .reshape({1}).to(mfq_tensor_backend::kInt64);
            sampled.push_back(token);
            finished = token.eq(eos_token).all().item<bool>();
            if (finished) break;
            current = minicpmo45_embedding(
                code_embedding, token.unsqueeze(1));
        }
        const int64_t returned = std::max<int64_t>(
            0, static_cast<int64_t>(sampled.size()) - 1);
        mfq_tensor_backend::Tensor codes;
        if (returned > 0) {
            codes = mfq_tensor_backend::stack(sampled, 1)
                .narrow(1, 0, returned)
                .unsqueeze(-1).contiguous();
        } else {
            codes = mfq_tensor_backend::empty(
                {1, 0, 1},
                mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA)
                    .dtype(mfq_tensor_backend::kInt64));
        }
        return {codes, finished};
    }

    mfq_tensor_backend::Tensor generate_official(
        State& state, Executor& executor, const Executor::Mode& mode,
        mfq_tensor_backend::Tensor condition_embeddings, int64_t steps, int64_t eos_token = 6561,
        int64_t minimum_steps = 50, double temperature = 0.8, double top_p = 0.85,
        int64_t top_k = 25, double repetition_penalty = 1.05,
        std::vector<mfq_tensor_backend::Tensor> *logits_trace = nullptr, double min_p = 0.0,
        std::mt19937 *evaluator_rng = nullptr, int64_t evaluator_min_keep = 0) const {
        MfqCudaGuard guard(device);
        const mfq::models::minicpmo45::TtsSampling config{eos_token,          minimum_steps, top_k,
                                                          evaluator_min_keep, temperature,   top_p,
                                                          repetition_penalty, min_p};
        config.validate(steps, 6562);
        if (condition_embeddings.dim() != 3 || condition_embeddings.size(0) != 1)
            throw std::runtime_error("official MiniCPM-o TTS generation requires batch size one");
        state.reset(1);
        using Tensor = mfq_tensor_backend::Tensor;
        std::vector<Tensor> generated;
        generated.reserve(static_cast<size_t>(steps));
        auto current = condition_embeddings;
        MiniCPMO45TtsSamplingOps sampling_ops{evaluator_rng};
        const auto result = mfq::engine::generate_tokens(
            steps,
            [&](int64_t step) {
                auto hidden = hidden_forward(state, executor, mode, current);
                auto raw_step_logits =
                    logits(hidden.index({Slice(), -1, Slice()})).to(mfq_tensor_backend::kFloat32);
                if (logits_trace)
                    logits_trace->push_back(raw_step_logits.clone());
                return mfq::models::minicpmo45::sample_tts(
                    sampling_ops, std::move(raw_step_logits), config,
                    std::span<const Tensor>(generated), step, evaluator_rng != nullptr);
            },
            [&](const Tensor &token, int64_t) { generated.push_back(token); },
            [&](const Tensor &token) { return token.eq(eos_token).all().item<bool>(); },
            [&](const Tensor &token) {
                current = minicpmo45_embedding(code_embedding, token.unsqueeze(1));
            });
        auto sampled = mfq_tensor_backend::stack(generated, 1);
        return sampled.narrow(1, 0, result.tokens - (result.hit_eos ? 1 : 0))
            .unsqueeze(-1)
            .contiguous();
    }
};

} // namespace mfq::cuda::minicpmo45
