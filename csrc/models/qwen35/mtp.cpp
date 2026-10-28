#include "models/qwen35/mtp.h"

#include "native/cuda/collectives.h"

namespace mfq::cuda::qwen35 {

namespace tb = mfq_tensor_backend;

namespace {

std::vector<MtpState::Layout> layout(const Predictor& model, int device) {
    std::vector<MtpState::Layout> result;
    for (const auto& layer : model.layers) {
        const auto& a = layer.attention;
        result.push_back({device, a.kv_heads, a.attention_head_dim, a.max_position_embeddings,
            a.official_bf16 ? tb::kBFloat16 : tb::kFloat16, a.kv_quantization});
    }
    return result;
}

} // namespace

Mtp::Mtp(Predictor model, std::unique_ptr<Executor> executor, Collectives& copies,
        int device, const int64_t& kv_capacity, Trace* const& trace, const int& trace_scope,
        bool preallocate)
    : model_(std::move(model)), state_(layout(model_, device), model_.config.hidden_size),
      tasks_(model_.tasks()), executor_(std::move(executor)), copies_(copies), device_(device),
      kv_capacity_(kv_capacity), trace_(trace), trace_scope_(trace_scope) {
    if (preallocate) state_.prepare(1, 1);
}

void Mtp::reset(int64_t batch) { state_.reset(batch); }

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
    TYLOINFER_RUNTIME_CHECK(state_.position() <= config.max_position_embeddings - tokens,
        "Qwen MTP history exceeds context capacity");
    const int device = device_;
    MfqCudaGuard guard(device);
    const bool explicit_positions = positions.defined();
    auto pos = explicit_positions
        ? copies_.copy(std::move(positions), device).to(tb::kInt64).contiguous()
        : tb::arange(state_.position(), state_.position() + tokens,
            tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt64));
    TYLOINFER_RUNTIME_CHECK((pos.dim() == 1 && pos.numel() == tokens) ||
        (pos.dim() == 2 && pos.size(1) == tokens &&
            (pos.size(0) == batch || (pos.size(0) == 3 && !config.mrope_sections.empty()))),
        "Qwen MTP positions must have shape [T], [B,T], or configured grid-MRoPE [3,T]");

    state_.prepare(batch, tokens, kv_capacity_);
    const auto kv = state_.views();
    auto embedded = target.embed(std::move(next_ids)).to(hidden.scalar_type());
    Inputs inputs;
    inputs.values = {std::move(hidden), std::move(embedded), {}};
    inputs.kv = kv;
    inputs.rope = target.rope;
    inputs.attention.positions = pos;
    inputs.attention.cache_position = state_.position();
    if (tokens == 1 && state_.position() > 0)
        inputs.attention.sequence_lengths = tb::full({batch}, state_.position() + 1, pos.options());
    if (explicit_positions)
        inputs.attention.cache_positions = tb::arange(state_.position(), state_.position() + tokens, pos.options());
    inputs.trace = trace_;
    inputs.trace_scope = trace_scope_;
    auto output = executor_->run(tasks_, std::move(inputs), {});
    state_.advance(tokens);
    return output;
}

void Mtp::trim_cache_to(int64_t position) { state_.trim(position); }

MtpSessionState Mtp::capture_session_state(int64_t position, const tb::Tensor& hidden) const {
    MfqCudaGuard guard(device_);
    return state_.capture(position, hidden);
}

void Mtp::restore_session_state(const MtpSessionState& saved) {
    state_.restore(saved);
}

} // namespace mfq::cuda::qwen35
