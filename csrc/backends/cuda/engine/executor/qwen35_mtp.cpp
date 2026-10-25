#include "engine/executor/qwen35_mtp.h"

#include "engine/executor/attention.h"
#include "engine/executor/execution.h"
#include "engine/executor/ffn.h"
#include "engine/executor/options.h"
#include "engine/models/load.h"
#include "engine/models/qwen35/weights.h"
#include "engine/weights/load.h"
#include "ops/norm/rms.h"

#include <utility>

namespace mfq::cuda::qwen35 {

namespace tb = mfq_tensor_backend;

std::unique_ptr<Mtp> Mtp::load(const mfq::ModelSource& source,
        const mfq::models::qwen35::Config& config, CudaExecutionContext& execution) {
    const bool fusion_present = has_tensor(source, "predictor.fusion.weight");
    const bool any = fusion_present || has_tensor(source, "predictor.hidden_norm.weight") ||
        has_tensor(source, "predictor.embedding_norm.weight") ||
        has_tensor(source, "predictor.output_norm.weight") ||
        has_tensor(source, "predictor.block.0.attention.query.weight");
    if (!fusion_present) {
        TYLOINFER_RUNTIME_CHECK(!any, "Qwen model source contains an incomplete MTP head");
        return nullptr;
    }
    TYLOINFER_RUNTIME_CHECK(config.mtp_num_hidden_layers > 0 && !config.mtp_use_dedicated_embeddings,
        "Qwen MTP requires declared predictor layers and shared embeddings");
    Predictor model;
    model.config = config;
    TransformerWeightLoader loader{execution, source};
    model.hidden_norm = loader.dense("predictor.hidden_norm.weight");
    model.embedding_norm = loader.dense("predictor.embedding_norm.weight");
    model.output_norm = loader.dense("predictor.output_norm.weight");
    model.fusion = loader.linear("predictor.fusion.weight");
    TYLOINFER_RUNTIME_CHECK(model.fusion.neuron_len() == 2 * config.hidden_size &&
        model.fusion.out() == config.hidden_size &&
        model.hidden_norm.numel() == config.hidden_size &&
        model.embedding_norm.numel() == config.hidden_size &&
        model.output_norm.numel() == config.hidden_size,
        "Qwen MTP component dimensions disagree with the backbone");
    model.layers.reserve(config.mtp_num_hidden_layers);
    for (int index = 0; index < config.mtp_num_hidden_layers; ++index) {
        const auto prefix = "predictor.block." + std::to_string(index) + ".";
        Predictor::Layer layer;
        layer.attention.kv_quantization = cuda_kv_quantization();
        mfq::models::load_attention_mixer(layer.attention, loader, config, index, prefix,
            config.legacy_tensor_layout.norm_weight_offset, config.attention_output_gate);
        layer.ffn_norm = loader.dense(prefix + "mlp.norm.weight");
        layer.ffn = mfq::models::qwen35::load_ffn<Ffn>(loader, config, index, "predictor");
        model.layers.push_back(std::move(layer));
    }
    return std::unique_ptr<Mtp>(new Mtp(std::move(model), execution));
}

Mtp::Mtp(Predictor model, CudaExecutionContext& execution)
    : model_(std::move(model)), scratch_(model_.layers.size()), execution_(execution) {
    state_.kv.resize(model_.layers.size());
    if (!runtime_options::kv_vram_preallocated()) return;
    const int device = execution_.layer_placement.primary_device();
    MfqCudaGuard guard(device);
    for (size_t index = 0; index < model_.layers.size(); ++index) {
        const auto& attention = model_.layers[index].attention;
        state_.kv[index] = KVCache(1, attention.kv_heads, attention.max_position_embeddings,
            attention.attention_head_dim, false, tb::Device(tb::kCUDA, device),
            attention.official_bf16 ? tb::kBFloat16 : tb::kFloat16, attention.kv_quantization);
    }
}

void Mtp::reset(int64_t batch) {
    state_.position = 0;
    for (auto& cache : state_.kv)
        if (!cache.defined() || cache.batch_size() != batch) cache = KVCache();
}

tb::Tensor Mtp::normalize(tb::Tensor input, const tb::Tensor& weight) const {
    const auto& config = model_.config;
    return qwen_rms_norm(input.reshape({-1, config.hidden_size}).to(tb::kFloat32),
        weight, config.rms_norm_eps, config.legacy_tensor_layout.norm_weight_offset)
        .reshape(input.sizes());
}

tb::Tensor Mtp::forward(const MtpTarget& target, tb::Tensor hidden, tb::Tensor next_ids) {
    return evaluate(target, std::move(hidden), std::move(next_ids), {});
}

MtpStep Mtp::step_positioned(const MtpTarget& target, tb::Tensor hidden,
        tb::Tensor next_ids, tb::Tensor positions) {
    auto output = evaluate(target, std::move(hidden), std::move(next_ids), std::move(positions));
    return {output, output};
}

tb::Tensor Mtp::evaluate(const MtpTarget& target, tb::Tensor hidden,
        tb::Tensor next_ids, tb::Tensor positions) {
    const auto& config = model_.config;
    TYLOINFER_RUNTIME_CHECK(hidden.dim() == 3 && next_ids.dim() == 2 &&
        hidden.size(0) == next_ids.size(0) && hidden.size(1) == next_ids.size(1) &&
        hidden.size(2) == config.hidden_size && hidden.size(1) > 0,
        "Qwen MTP inputs must be matching [B,T,H] hidden states and [B,T] next-token IDs");
    const auto batch = hidden.size(0), tokens = hidden.size(1);
    TYLOINFER_RUNTIME_CHECK(state_.position + tokens <= config.max_position_embeddings,
        "Qwen MTP history exceeds context capacity");
    const int device = execution_.layer_placement.primary_device();
    MfqCudaGuard guard(device);
    auto embedded = target.embed(std::move(next_ids)).to(hidden.scalar_type());
    auto e = normalize(std::move(embedded), model_.embedding_norm);
    auto h = normalize(std::move(hidden), model_.hidden_norm);
    trace_gemma_stage(execution_, 0, "mtp.embedding_norm", e);
    trace_gemma_stage(execution_, 0, "mtp.hidden_norm", h);
    auto current = execution_.projector().run(model_.fusion, tb::cat({e, h}, -1));
    trace_gemma_stage(execution_, 0, "mtp.fusion", current);

    const bool explicit_positions = positions.defined();
    auto pos = explicit_positions
        ? execution_.collectives.copy(std::move(positions), device).to(tb::kInt64).contiguous()
        : tb::arange(state_.position, state_.position + tokens,
            tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt64));
    TYLOINFER_RUNTIME_CHECK((pos.dim() == 1 && pos.numel() == tokens) ||
        (pos.dim() == 2 && pos.size(1) == tokens &&
            (pos.size(0) == batch || (pos.size(0) == 3 && !config.mrope_sections.empty()))),
        "Qwen MTP positions must have shape [T], [B,T], or configured grid-MRoPE [3,T]");
    AttentionStep step;
    step.positions = pos;
    step.cache_position = state_.position;
    if (tokens == 1 && state_.position > 0)
        step.sequence_lengths = tb::full({batch}, state_.position + 1, pos.options());
    if (explicit_positions)
        step.cache_positions = tb::arange(state_.position, state_.position + tokens, pos.options());
    for (size_t index = 0; index < model_.layers.size(); ++index) {
        const auto& layer = model_.layers[index];
        auto attention = run_attention(execution_, layer.attention, state_.kv[index],
            scratch_[index], std::move(current), step, *target.rope);
        current = full_ffn(execution_, layer.ffn, layer.ffn_norm, layer.attention,
            std::move(attention[0]), std::move(attention[1]));
    }
    state_.position += tokens;
    auto output = normalize(std::move(current), model_.output_norm);
    trace_gemma_stage(execution_, 0, "mtp.output_norm", output);
    return output;
}

void Mtp::trim_cache_to(int64_t position) {
    TYLOINFER_RUNTIME_CHECK(position >= 0 && position <= state_.position,
        "Qwen MTP cache trim position is invalid");
    // The next pass overwrites the discarded suffix in the position-addressed KV storage.
    state_.position = position;
}

MtpSessionState Mtp::capture_session_state(int64_t target_cache_position,
        const tb::Tensor& last_target_hidden) const {
    const int64_t position = target_cache_position - 1;
    if (position <= 0 || position > state_.position || !last_target_hidden.defined() ||
        last_target_hidden.dim() != 3 || last_target_hidden.size(0) != 1 ||
        last_target_hidden.size(1) != 1 || last_target_hidden.size(2) != model_.config.hidden_size) {
        throw std::runtime_error("Qwen MTP session boundary is unavailable");
    }
    MtpSessionState snapshot;
    snapshot.cache_pos = position;
    snapshot.blocks.reserve(state_.kv.size());
    MfqCudaGuard guard(execution_.layer_placement.primary_device());
    for (const auto& cache : state_.kv)
        snapshot.blocks.push_back(capture_kv(cache, position, snapshot.bytes));
    snapshot.last_target_hidden = last_target_hidden.clone();
    snapshot.bytes += session_tensor_bytes(snapshot.last_target_hidden);
    return snapshot;
}

void Mtp::restore_session_state(const MtpSessionState& snapshot) {
    if (snapshot.cache_pos <= 0 || snapshot.blocks.size() != state_.kv.size() ||
        !snapshot.last_target_hidden.defined() || snapshot.last_target_hidden.dim() != 3 ||
        snapshot.last_target_hidden.size(0) != 1 || snapshot.last_target_hidden.size(1) != 1 ||
        snapshot.last_target_hidden.size(2) != model_.config.hidden_size) {
        throw std::runtime_error("Qwen MTP session state is incompatible");
    }
    MfqCudaGuard guard(execution_.layer_placement.primary_device());
    for (size_t index = 0; index < state_.kv.size(); ++index) {
        const auto& attention = model_.layers[index].attention;
        restore_kv(state_.kv[index], snapshot.blocks[index], attention.attention_head_dim,
            attention.official_bf16 ? tb::kBFloat16 : tb::kFloat16, attention.kv_quantization);
    }
    state_.position = snapshot.cache_pos;
}

} // namespace mfq::cuda::qwen35
