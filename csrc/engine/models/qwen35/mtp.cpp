#include "engine/models/qwen35/mtp.h"
#include "engine/models/session/snapshot.h"
#include "engine/models/common/ffn.h"

#include "engine/models/qwen35/state.h"
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

Mtp::Mtp(Predictor model, ::tyloi::Tensor embedding, tyloi::ops::RopeTable rope,
        int device)
    : model_(std::move(model)), embedding_(std::move(embedding)), rope_(std::move(rope)),
      device_(device) {
    for (const auto& layer : model_.layers)
        if (layer.ffn.is_moe || (layer.ffn.important_neurons && layer.ffn.important_neurons->is_moe))
            throw std::runtime_error("Qwen MTP requires dense predictor projections");
}

std::unique_ptr<tyloi::models::MtpState> Mtp::make_state() const {
    auto result = std::make_unique<MtpState>(*this, layout(model_, device_), model_.config.hidden_size);
    return result;
}

void Mtp::reset(tyloi::models::MtpState& value, int64_t batch) const {
    state<MtpState>(value).reset(batch);
}

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

void Mtp::trim_cache_to(tyloi::models::MtpState& value, int64_t position) const {
    state<MtpState>(value).trim(position);
}

MtpSessionState Mtp::capture_session_state(const tyloi::models::MtpState& value,
                                           int64_t position, const ::tyloi::Tensor& hidden) const {
    return state<MtpState>(value).capture(position, hidden.dense());
}


} // namespace tyloi::cuda::qwen35
