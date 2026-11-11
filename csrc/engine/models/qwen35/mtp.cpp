#include "engine/models/qwen35/mtp.h"
#include "engine/models/session/snapshot.h"
#include "engine/models/common/ffn.h"

#include "engine/models/qwen35/state.h"
#ifndef TYLOINFER_NATIVE_CUDA
#include "engine/backends/cuda/native/collectives.h"
#include "engine/backends/cuda/ops/norm/rms.h"
#include "engine/backends/cuda/ops/projection/quant_linear.h"
#endif
#include "engine/ops.h"

#include <array>
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

#ifdef TYLOINFER_NATIVE_CUDA
Mtp::Mtp(Predictor model, ::tyloi::Tensor embedding, tyloi::ops::RopeTable rope,
        int device)
    : model_(std::move(model)), embedding_(std::move(embedding)), rope_(std::move(rope)),
      device_(device) {
    for (const auto& layer : model_.layers)
        if (layer.ffn.is_moe || (layer.ffn.important_neurons && layer.ffn.important_neurons->is_moe))
            throw std::runtime_error("Qwen MTP requires dense predictor projections");
}
#else
Mtp::Mtp(Predictor model, ::tyloi::Tensor embedding,
#ifdef TYLOINFER_NATIVE_CUDA
        tyloi::ops::RopeTable rope,
#else
        RopeCache rope,
#endif
        std::unique_ptr<Executor> executor, Collectives& copies,
        int device, const int64_t& kv_capacity, Trace* const& trace, const int& trace_scope,
        bool preallocate)
    : model_(std::move(model)), embedding_(std::move(embedding)), rope_(std::move(rope)),
      executor_(std::move(executor)), copies_(copies), device_(device),
      kv_capacity_(kv_capacity), trace_(trace), trace_scope_(trace_scope), preallocate_(preallocate) {
    for (const auto& layer : model_.layers)
        TYLOINFER_RUNTIME_CHECK(!layer.ffn.is_moe &&
            (!layer.ffn.important_neurons || !layer.ffn.important_neurons->is_moe),
            "Qwen MTP requires dense predictor projections");
}

#endif

std::unique_ptr<tyloi::models::MtpState> Mtp::make_state() const {
    auto result = std::make_unique<MtpState>(*this, layout(model_, device_), model_.config.hidden_size);
#ifndef TYLOINFER_NATIVE_CUDA
    if (preallocate_) result->prepare(1, 1);
#endif
    return result;
}

void Mtp::reset(tyloi::models::MtpState& value, int64_t batch) const {
    state<MtpState>(value).reset(batch);
}

#ifdef TYLOINFER_NATIVE_CUDA
std::vector<std::pair<Device, std::size_t>> Mtp::growth(
        const tyloi::models::MtpState& value, const tyloi::models::Mtp::Input& input) const {
    const auto* step = std::get_if<tyloi::models::Mtp::Step>(&input);
    if (!step || !step->ids.defined() || step->ids.dim()!=2)
        throw std::invalid_argument("unsupported Qwen MTP growth input");
    return state<MtpState>(value).growth(step->ids.size(0), step->ids.size(1), step->options.kv_capacity);
}
std::unique_ptr<tyloi::models::Restore> Mtp::prepare_restore(
        tyloi::models::MtpState& value, const MtpSessionState& snapshot) const {
    return state<MtpState>(value).prepare_restore(snapshot);
}
tyloi::models::MtpStep Mtp::step(tyloi::models::MtpState& value, const Tensor& hidden,
        const Tensor& next_ids, const Tensor& positions, const tyloi::models::ForwardOptions& options) const {
    auto output = evaluate(state<MtpState>(value), hidden, next_ids, positions, options);
    return {output, output};
}
Tensor Mtp::evaluate(MtpState& state, const Tensor& hidden, const Tensor& next_ids,
                     const Tensor& positions, const tyloi::models::ForwardOptions& options) const {
    const auto& config = model_.config;
    if (hidden.dim()!=3 || next_ids.dim()!=2 || hidden.size(0)!=next_ids.size(0) ||
        hidden.size(1)!=next_ids.size(1) || hidden.size(2)!=config.hidden_size ||
        hidden.size(0)<=0 || hidden.size(1)<=0)
        throw std::runtime_error("Qwen MTP inputs must be matching [B,T,H] hidden states and [B,T] next-token IDs");
    const auto batch=hidden.size(0), tokens=hidden.size(1);
    if (state.position()>config.max_position_embeddings-tokens)
        throw std::runtime_error("Qwen MTP history exceeds context capacity");
    tyloi::ops::FfnScope stage(tyloi::ops::FfnScope::Role::predictor);
    const Device device{kCUDA,device_};
    const bool explicit_positions=positions.defined();
    auto pos=explicit_positions ? ::tyloi::ops::contiguous(::tyloi::ops::cast(::tyloi::ops::copy(positions,device),kInt64))
        : ::tyloi::ops::arange(state.position(),state.position()+tokens,device);
    if (!((pos.dim()==1 && pos.numel()==tokens) ||
          (pos.dim()==2 && pos.size(1)==tokens && (pos.size(0)==batch ||
           (pos.size(0)==3 && !config.mrope_sections.empty())))))
        throw std::runtime_error("Qwen MTP positions must have shape [T], [B,T], or configured grid-MRoPE [3,T]");
    state.prepare(batch,tokens,options.kv_capacity);
    const auto kv=state.views();
    const auto trace_for=[&](int layer) -> ::tyloi::ops::Trace* {
        return options.trace && options.trace_layer==layer ? options.trace : nullptr;
    };
    const auto record=[&](::tyloi::ops::Trace* trace,const char* name,const Tensor& value) {
        if (trace) trace->emplace_back(name,::tyloi::ops::clone(::tyloi::ops::contiguous(::tyloi::ops::cast(value,kFloat32))));
    };
    auto normalized_ids=::tyloi::ops::contiguous(::tyloi::ops::copy(::tyloi::ops::cast(next_ids,kInt64),device));
    auto embedded=::tyloi::ops::measure("model.embed",normalized_ids.device(),[&] {
        return ::tyloi::ops::embedding(normalized_ids,embedding_);
    });
    embedded=::tyloi::ops::cast(::tyloi::ops::contiguous(::tyloi::ops::cast(embedded,kFloat16)),hidden.scalar_type());
    auto embedded_norm=::tyloi::ops::rms_norm(embedded,model_.embedding_norm,config.rms_norm_eps,
                                     config.legacy_tensor_layout.norm_weight_offset);
    auto hidden_norm=::tyloi::ops::rms_norm(hidden,model_.hidden_norm,config.rms_norm_eps,
                                   config.legacy_tensor_layout.norm_weight_offset);
    record(trace_for(0),"mtp.embedding_norm",embedded_norm);
    record(trace_for(0),"mtp.hidden_norm",hidden_norm);
    std::array<Tensor,2> fusion_inputs{embedded_norm,hidden_norm};
    auto current=::tyloi::ops::project(::tyloi::ops::cat(fusion_inputs,-1),model_.fusion);
    record(trace_for(0),"mtp.fusion",current);
    AttentionStep attention;
    attention.positions=pos;
    attention.cache_position=state.position();
    if (tokens==1 && state.position()>0)
        attention.sequence_lengths=::tyloi::ops::full(std::array{batch},state.position()+1,device);
    if (explicit_positions)
        attention.cache_positions=::tyloi::ops::arange(state.position(),state.position()+tokens,device);
    for (std::size_t index=0;index<model_.layers.size();++index) {
        const auto& layer=model_.layers[index];
        const auto& full=layer.attention;
        auto output=full.forward(current,attention,kv[index],rope_,trace_for(full.layer));
        current=tyloi::models::full_ffn(layer.ffn,layer.ffn_norm,output[0],output[1],
            full.rms_norm_eps,full.norm_weight_offset,full.official_bf16,full.attention_output_gate,trace_for(full.layer));
    }
    auto output=::tyloi::ops::rms_norm(current,model_.output_norm,config.rms_norm_eps,
                              config.legacy_tensor_layout.norm_weight_offset);
    record(trace_for(0),"mtp.output_norm",output);
    state.advance(tokens);
    return output;
}
#else
tyloi::models::MtpStep Mtp::step(tyloi::models::MtpState& value, const ::tyloi::Tensor& hidden,
        const ::tyloi::Tensor& next_ids, const ::tyloi::Tensor& positions,
        const tyloi::models::ForwardOptions&) const {
    auto output = std::make_shared<::tyloi::DenseTensor>(evaluate(
        state<MtpState>(value),
        hidden.dense(),
        next_ids.dense(),
        positions.defined() ? positions.dense()
                                          : ::tyloi::DenseTensor{}));
    return {output, output};
}

::tyloi::DenseTensor Mtp::evaluate(MtpState& state, ::tyloi::DenseTensor hidden,
        ::tyloi::DenseTensor next_ids, ::tyloi::DenseTensor positions) const {
    const auto& config = model_.config;
    TYLOINFER_RUNTIME_CHECK(hidden.dim() == 3 && next_ids.dim() == 2 &&
        hidden.size(0) == next_ids.size(0) && hidden.size(1) == next_ids.size(1) &&
        hidden.size(2) == config.hidden_size && hidden.size(0) > 0 && hidden.size(1) > 0,
        "Qwen MTP inputs must be matching [B,T,H] hidden states and [B,T] next-token IDs");
    const auto batch = hidden.size(0), tokens = hidden.size(1);
    TYLOINFER_RUNTIME_CHECK(state.position() <= config.max_position_embeddings - tokens,
        "Qwen MTP history exceeds context capacity");
    const int device = device_;
    MfqCudaGuard guard(device);
    const FfnMode mode{};
#ifdef TYLOINFER_NATIVE_CUDA
    auto executor_scope = executor_->enter(device, mode);
#else
    auto executor_scope = executor_->enter(device);
#endif
    const bool explicit_positions = positions.defined();
    auto pos = explicit_positions
        ? copies_.copy(std::move(positions), device).to(::tyloi::kInt64).contiguous()
        : ::tyloi::cuda::arange(state.position(), state.position() + tokens,
            ::tyloi::TensorOptions().device(::tyloi::kCUDA).dtype(::tyloi::kInt64));
    TYLOINFER_RUNTIME_CHECK((pos.dim() == 1 && pos.numel() == tokens) ||
        (pos.dim() == 2 && pos.size(1) == tokens &&
            (pos.size(0) == batch || (pos.size(0) == 3 && !config.mrope_sections.empty()))),
        "Qwen MTP positions must have shape [T], [B,T], or configured grid-MRoPE [3,T]");

    state.prepare(batch, tokens, kv_capacity_);
    const auto kv = state.views();
    auto trace_for = [&](int scope) -> Trace* {
        return trace_ && trace_scope_ == scope ? trace_ : nullptr;
    };
    auto record = [&](Trace* trace, const char* name, const ::tyloi::DenseTensor& value) {
        if (trace == nullptr) return;
#ifdef TYLOINFER_NATIVE_CUDA
        trace->emplace_back(std::string(name),
            std::make_shared<::tyloi::DenseTensor>(value.to(::tyloi::kFloat32).contiguous().clone()));
#else
        trace->emplace_back(std::string(name), value.to(::tyloi::kFloat32).contiguous().clone());
#endif
    };
#ifdef TYLOINFER_NATIVE_CUDA
    ::tyloi::Tensor current;
    {
        auto normalized_ids = copies_.copy(next_ids.to(::tyloi::kInt64), device_).contiguous();
        auto embedded = executor_->profiler().measure("model.embed", [&] {
            return ::tyloi::ops::embedding(::tyloi::Tensor(normalized_ids), embedding_);
        });
        auto embedded_dense = embedded.dense().to(::tyloi::kFloat16).contiguous()
                                  .to(hidden.scalar_type());
        ::tyloi::Tensor hidden_value(std::move(hidden));
        ::tyloi::Tensor embedded_value(std::move(embedded_dense));
        auto embedded_norm = ::tyloi::ops::rms_norm(embedded_value, model_.embedding_norm,
            config.rms_norm_eps, config.legacy_tensor_layout.norm_weight_offset);
        auto hidden_norm = ::tyloi::ops::rms_norm(hidden_value, model_.hidden_norm,
            config.rms_norm_eps, config.legacy_tensor_layout.norm_weight_offset);
        record(trace_for(0), "mtp.embedding_norm", embedded_norm.dense());
        record(trace_for(0), "mtp.hidden_norm", hidden_norm.dense());
        std::array<::tyloi::Tensor, 2> fusion_inputs{embedded_norm, hidden_norm};
        current = ::tyloi::ops::project(::tyloi::ops::cat(fusion_inputs, -1), model_.fusion);
        record(trace_for(0), "mtp.fusion", current.dense());
    }

    AttentionStep attention;
    attention.positions = pos;
#else
    auto project = executor_->projector(mode);
    ::tyloi::DenseTensor current;
    {
        auto normalized_ids = copies_.copy(next_ids.to(::tyloi::kInt64), device_).contiguous();
        auto embedded = executor_->profiler().measure("model.embed", [&] {
            return quant_embedding_lookup(embedding_, std::move(normalized_ids));
        });
        embedded = embedded.to(::tyloi::kFloat16).contiguous().to(hidden.scalar_type());
        auto embedded_norm = qwen_rms_norm(
            embedded.reshape({-1, embedded.size(-1)}).to(::tyloi::kFloat32),
            model_.embedding_norm, config.rms_norm_eps,
            config.legacy_tensor_layout.norm_weight_offset).reshape_as(embedded);
        auto hidden_norm = qwen_rms_norm(
            hidden.reshape({-1, hidden.size(-1)}).to(::tyloi::kFloat32), model_.hidden_norm,
            config.rms_norm_eps, config.legacy_tensor_layout.norm_weight_offset).reshape_as(hidden);
        record(trace_for(0), "mtp.embedding_norm", embedded_norm);
        record(trace_for(0), "mtp.hidden_norm", hidden_norm);
        current = project.run(model_.fusion,
            ::tyloi::cuda::cat({embedded_norm, hidden_norm}, -1));
        record(trace_for(0), "mtp.fusion", current);
        hidden = {};
    }

    AttentionStep attention;
    attention.positions = pos;
#endif
    attention.cache_position = state.position();
    if (tokens == 1 && state.position() > 0)
        attention.sequence_lengths = ::tyloi::cuda::full({batch}, state.position() + 1, pos.options());
    if (explicit_positions)
        attention.cache_positions = ::tyloi::cuda::arange(state.position(), state.position() + tokens, pos.options());

    for (std::size_t index = 0; index < model_.layers.size(); ++index) {
        const auto& layer = model_.layers[index];
        const auto& full_attention = layer.attention;
        TYLOINFER_RUNTIME_CHECK(!layer.ffn.is_moe &&
            (!layer.ffn.important_neurons || !layer.ffn.important_neurons->is_moe),
            "Qwen MTP requires dense predictor projections");
#ifdef TYLOINFER_NATIVE_CUDA
        auto attention_output = full_attention.forward(current, attention, kv[index], rope_,
            trace_for(full_attention.layer));
        current = ::tyloi::models::full_ffn(
            layer.ffn, layer.ffn_norm, attention_output[0], attention_output[1],
            full_attention.rms_norm_eps, full_attention.norm_weight_offset,
            full_attention.official_bf16, full_attention.attention_output_gate,
            trace_for(full_attention.layer));
#else
        auto attention_output = full_attention.forward(project, executor_->profiler(),
            executor_->attention_options(), std::move(current), attention, kv[index],
            executor_->attention_scratch(index), rope_, trace_for(full_attention.layer));
        current = ::tyloi::models::full_ffn(
            layer.ffn, layer.ffn_norm, std::move(attention_output[0]),
            std::move(attention_output[1]), full_attention.rms_norm_eps,
            full_attention.norm_weight_offset, full_attention.official_bf16,
            full_attention.attention_output_gate, trace_for(full_attention.layer)).dense();
#endif
    }
#ifdef TYLOINFER_NATIVE_CUDA
    auto output = ::tyloi::ops::rms_norm(current, model_.output_norm, config.rms_norm_eps,
        config.legacy_tensor_layout.norm_weight_offset);
    record(trace_for(0), "mtp.output_norm", output.dense());
    auto result = output.dense();
#else
    auto output = qwen_rms_norm(
        current.reshape({-1, current.size(-1)}).to(::tyloi::kFloat32), model_.output_norm,
        config.rms_norm_eps, config.legacy_tensor_layout.norm_weight_offset).reshape_as(current);
    record(trace_for(0), "mtp.output_norm", output);
    auto result = std::move(output);
#endif
    state.advance(tokens);
    return result;
}

#endif

void Mtp::trim_cache_to(tyloi::models::MtpState& value, int64_t position) const {
    state<MtpState>(value).trim(position);
}

MtpSessionState Mtp::capture_session_state(const tyloi::models::MtpState& value,
                                           int64_t position, const ::tyloi::Tensor& hidden) const {
#ifndef TYLOINFER_NATIVE_CUDA
    MfqCudaGuard guard(device_);
#endif
    return state<MtpState>(value).capture(position, hidden.dense());
}

#ifndef TYLOINFER_NATIVE_CUDA
void restore_session(State& state, Mtp* mtp, const TextSessionState& snapshot,
        const std::function<void()>& drain) {
    auto target = state.prepare(snapshot);
    std::optional<MtpState::Restore> predictor;
    MtpState* predictor_state = nullptr;
    if (mtp != nullptr) {
        if (!state.predictor || !snapshot.mtp || snapshot.mtp->cache_pos != snapshot.cache_pos - 1)
            throw CudaSessionStateError("Qwen MTP session boundary is incompatible");
        predictor_state = &mtp->state<MtpState>(*state.predictor);
        predictor.emplace(predictor_state->prepare(*snapshot.mtp));
    }
    try {
        state.stage(target, snapshot);
        if (predictor_state) predictor_state->stage(*predictor, *snapshot.mtp);
        drain();
        state.publish(std::move(target));
        if (predictor_state) predictor_state->publish(std::move(*predictor));
    } catch (...) {
        const auto error = std::current_exception();
        try {
            drain();
        } catch (...) {
        }
        std::rethrow_exception(error);
    }
}
#endif

} // namespace tyloi::cuda::qwen35
