#include "engine/models/qwen35/mtp.h"
#include "engine/models/session.h"

#include "engine/models/qwen35/state.h"
#include "engine/backends/cuda/native/collectives.h"

#include <exception>
#include <optional>

namespace tyloi::cuda::qwen35 {


namespace {

std::vector<MtpState::Layout> layout(const Predictor& model, int device) {
    std::vector<MtpState::Layout> result;
    for (const auto& layer : model.layers) {
        const auto& a = layer.attention;
        result.push_back({device, a.kv_heads, a.attention_head_dim, a.max_position_embeddings,
            a.official_bf16 ? ::tyloi::kBFloat16 : ::tyloi::kFloat16, a.kv_quantization});
    }
    return result;
}

} // namespace

Mtp::Mtp(Predictor model, std::shared_ptr<const ::tyloi::Tensor> embedding,
#ifdef TYLOINFER_NATIVE_CUDA
        tyloi::ops::RopeTable rope,
#else
        RopeCache rope,
#endif
        std::unique_ptr<Executor> executor, Collectives& copies,
        int device, const int64_t& kv_capacity, Trace* const& trace, const int& trace_scope,
        bool preallocate)
    : model_(std::move(model)), embedding_(std::move(embedding)), rope_(std::move(rope)),
      state_(layout(model_, device), model_.config.hidden_size),
      tasks_(model_.tasks()), executor_(std::move(executor)), copies_(copies), device_(device),
      kv_capacity_(kv_capacity), trace_(trace), trace_scope_(trace_scope) {
    if (preallocate) state_.prepare(1, 1);
}

void Mtp::reset(int64_t batch) { state_.reset(batch); }

tyloi::models::MtpStep Mtp::step(const ::tyloi::Tensor& hidden,
        const ::tyloi::Tensor& next_ids, const ::tyloi::Tensor* positions) {
    auto output = std::make_shared<::tyloi::DenseTensor>(evaluate(
        dynamic_cast<const ::tyloi::DenseTensor&>(hidden),
        dynamic_cast<const ::tyloi::DenseTensor&>(next_ids),
        positions && positions->defined() ? dynamic_cast<const ::tyloi::DenseTensor&>(*positions)
                                          : ::tyloi::DenseTensor{}));
    return {output, output};
}

::tyloi::DenseTensor Mtp::evaluate(::tyloi::DenseTensor hidden,
        ::tyloi::DenseTensor next_ids, ::tyloi::DenseTensor positions) {
    const auto& config = model_.config;
    TYLOINFER_RUNTIME_CHECK(hidden.dim() == 3 && next_ids.dim() == 2 &&
        hidden.size(0) == next_ids.size(0) && hidden.size(1) == next_ids.size(1) &&
        hidden.size(2) == config.hidden_size && hidden.size(0) > 0 && hidden.size(1) > 0,
        "Qwen MTP inputs must be matching [B,T,H] hidden states and [B,T] next-token IDs");
    const auto batch = hidden.size(0), tokens = hidden.size(1);
    TYLOINFER_RUNTIME_CHECK(state_.position() <= config.max_position_embeddings - tokens,
        "Qwen MTP history exceeds context capacity");
    const int device = device_;
    MfqCudaGuard guard(device);
    const bool explicit_positions = positions.defined();
    auto pos = explicit_positions
        ? copies_.copy(std::move(positions), device).to(::tyloi::kInt64).contiguous()
        : ::tyloi::cuda::arange(state_.position(), state_.position() + tokens,
            ::tyloi::TensorOptions().device(::tyloi::kCUDA).dtype(::tyloi::kInt64));
    TYLOINFER_RUNTIME_CHECK((pos.dim() == 1 && pos.numel() == tokens) ||
        (pos.dim() == 2 && pos.size(1) == tokens &&
            (pos.size(0) == batch || (pos.size(0) == 3 && !config.mrope_sections.empty()))),
        "Qwen MTP positions must have shape [T], [B,T], or configured grid-MRoPE [3,T]");

    state_.prepare(batch, tokens, kv_capacity_);
    const auto kv = state_.views();
    const Task embedding{task::Embed{0, 0, *embedding_}, -1, {}, "model.embed"};
    Inputs tokens_input;
    tokens_input.values = {copies_.copy(next_ids.to(::tyloi::kInt64), device_).contiguous()};
    auto embedded = executor_->run(std::span<const Task>{&embedding, 1}, std::move(tokens_input), {})
                        .to(::tyloi::kFloat16).contiguous()
                        .to(hidden.scalar_type());
    Inputs inputs;
    inputs.values = {std::move(hidden), std::move(embedded), {}};
    inputs.kv = kv;
#ifdef TYLOINFER_NATIVE_CUDA
    inputs.rope = rope_;
#else
    inputs.rope = &rope_;
#endif
    inputs.attention.positions = pos;
    inputs.attention.cache_position = state_.position();
    if (tokens == 1 && state_.position() > 0)
        inputs.attention.sequence_lengths = ::tyloi::cuda::full({batch}, state_.position() + 1, pos.options());
    if (explicit_positions)
        inputs.attention.cache_positions = ::tyloi::cuda::arange(state_.position(), state_.position() + tokens, pos.options());
    inputs.trace = trace_;
    inputs.trace_scope = trace_scope_;
    auto output = executor_->run(tasks_, std::move(inputs), {});
    state_.advance(tokens);
    return output;
}

void Mtp::trim_cache_to(int64_t position) { state_.trim(position); }

MtpSessionState Mtp::capture_session_state(int64_t position, const ::tyloi::Tensor& hidden) const {
    MfqCudaGuard guard(device_);
    return state_.capture(position, dynamic_cast<const ::tyloi::DenseTensor&>(hidden));
}

void restore_session(State& state, Mtp* mtp, const TextSessionState& snapshot,
        const std::function<void()>& drain) {
    auto target = state.prepare(snapshot);
    std::optional<MtpState::Restore> predictor;
    if (mtp != nullptr) {
        if (!snapshot.mtp || snapshot.mtp->cache_pos != snapshot.cache_pos - 1)
            throw CudaSessionStateError("Qwen MTP session boundary is incompatible");
        predictor.emplace(mtp->state_.prepare(*snapshot.mtp));
    }
    try {
        state.stage(target, snapshot);
        if (mtp != nullptr) mtp->state_.stage(*predictor, *snapshot.mtp);
        drain();
        state.publish(std::move(target));
        if (mtp != nullptr) mtp->state_.publish(std::move(*predictor));
    } catch (...) {
        const auto error = std::current_exception();
        try {
            drain();
        } catch (...) {
        }
        std::rethrow_exception(error);
    }
}

} // namespace tyloi::cuda::qwen35
