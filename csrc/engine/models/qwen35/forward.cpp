#include "engine/models/qwen35/forward.h"
#include "engine/models/common/ffn.h"


#include <algorithm>
#include <array>
#include <stdexcept>
#include <utility>

namespace tyloi::cuda::qwen35 {

namespace {

void require_forward(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}


} // namespace

namespace {
Tensor normalized_ids(const Tensor& value, int primary) {
    require_forward(value.dim()==1 || value.dim()==2, "token IDs must have shape [tokens] or [batch,tokens]");
    auto ids=value.dim()==1 ? ::tyloi::ops::unsqueeze(value,0) : value;
    require_forward(ids.size(0)>0 && ids.size(1)>0,"token IDs must not be empty");
    return ::tyloi::ops::contiguous(::tyloi::ops::copy(::tyloi::ops::cast(ids,kInt64),Device(kCUDA,primary)));
}
}
Tensor embed(const Model& model,int primary,const Tensor& ids) {
    auto normalized=normalized_ids(ids,primary);
    auto result=::tyloi::ops::measure("model.embed",normalized.device(),[&] { return ::tyloi::ops::embedding(normalized,model.embed); });
    return ::tyloi::ops::contiguous(::tyloi::ops::cast(result,kFloat16));
}
Tensor normalize(const Model& model,const Tensor& hidden,std::int64_t batch,std::int64_t tokens) {
    auto shaped=::tyloi::ops::reshape(hidden,{batch,tokens,model.metadata.hidden_size});
    return ::tyloi::ops::measure("model.output_norm",shaped.device(),[&] {
        return ::tyloi::ops::rms_norm(shaped,model.output_norm,model.metadata.rms_norm_eps,model.metadata.norm_weight_offset);
    });
}
Tensor softcap(const Model& model,const Tensor& logits) {
    return ::tyloi::ops::softcap(logits,model.metadata.final_logit_softcapping);
}
Tensor logits(const Model& model,const Tensor& hidden,bool capped) {
    auto result=::tyloi::ops::measure("model.lm_head",hidden.device(),[&] { return ::tyloi::ops::project(hidden,model.lm_head); });
    return capped ? softcap(model,result) : result;
}
bool supports_speculation(const Model& model) noexcept {
    return !model.layers.empty() && std::all_of(model.layers.begin(),model.layers.end(),[](const Layer& layer) {
        const auto* full=std::get_if<Full>(&layer.weights);
        return !full || !full->attention.sliding;
    });
}
Tensor forward(const Model& model,State& state,tyloi::models::Input input,
               const tyloi::models::ForwardOptions& options) {
    require_forward(state.belongs_to(model),"Qwen3.5 state belongs to another model");
    auto ids=normalized_ids(input.ids,options.primary);
    auto embeddings=input.embeddings ? input.embeddings : embed(model,options.primary,ids);
    ids=normalized_ids(ids,options.primary);
    const auto B=ids.size(0), T=ids.size(1);
    require_forward(embeddings.dim()==3 && embeddings.size(0)==B && embeddings.size(1)==T &&
        embeddings.size(2)==model.metadata.hidden_size,"inputs_embeds shape must match [batch,tokens,hidden_size]");
    require_forward(input.confirmed>=0 && (input.confirmed==0 ||
        (input.confirmed<T && B==1 && state.cache_pos>0 && !input.positions && !input.cache_positions && supports_speculation(model))),
        "unsupported speculative backbone geometry");
    require_forward(state.speculative_start<0,"commit or roll back the pending speculative pass before forwarding");
    require_forward(state.cache_pos+T<=model.metadata.max_position_embeddings,"Qwen3.5 history exceeds context capacity");
    if (state.cache_pos==0) state.reset(B);
    else require_forward(state.batch==B,"Qwen3.5 cache batch does not match input batch");
    if (input.lengths) require_forward(input.lengths.dim()==1 && input.lengths.size(0)==B,
        "sequence lengths must have shape [batch]");
    const Device primary(kCUDA,options.primary);
    auto physical=input.cache_positions ? ::tyloi::ops::copy(::tyloi::ops::cast(input.cache_positions,kInt64),primary)
        : ::tyloi::ops::arange(state.cache_pos,state.cache_pos+T,primary);
    physical=::tyloi::ops::contiguous(physical);
    require_forward((physical.dim()==1 && physical.numel()==T) ||
        (physical.dim()==2 && physical.size(0)==B && physical.size(1)==T),
        "cache_positions must have shape [tokens] or [batch,tokens]");
    auto semantic=input.positions ? ::tyloi::ops::copy(::tyloi::ops::cast(input.positions,kInt64),primary)
        : (state.decode_position_delta==0 ? physical : ::tyloi::ops::add(physical,state.decode_position_delta));
    semantic=::tyloi::ops::contiguous(semantic);
    const auto rank=semantic.dim();
    const bool has_mrope=model.rope.sections && model.rope.sections.numel()>0;
    require_forward(Config::valid_positions(rank,rank>1 ? semantic.size(0) : 0,semantic.size(-1),B,T,has_mrope),
        "position_ids must have shape [tokens], [batch,tokens], or configured grid-MRoPE [3,tokens]");
    Tensor lengths;
    if (input.lengths) lengths=::tyloi::ops::contiguous(::tyloi::ops::copy(::tyloi::ops::cast(input.lengths,kInt64),primary));
    else if (state.cache_pos>0 && T==1) lengths=::tyloi::ops::full(std::array{B},state.cache_pos+1,primary);
    AttentionStep step{semantic,lengths,
        (has_mrope || input.positions || input.cache_positions) ? physical : Tensor{},
        state.cache_pos,input.kv_length,input.attention_parts};
    auto hidden=::tyloi::ops::contiguous(::tyloi::ops::copy(embeddings,primary));
    const auto record=[&](const Tensor& value) {
        if (input.trace) input.trace->push_back(::tyloi::ops::clone(::tyloi::ops::contiguous(::tyloi::ops::cast(::tyloi::ops::copy(value,primary),kFloat32))));
    };
    record(hidden);
    if (input.confirmed>0) { state.speculative_start=state.cache_pos; state.speculative_confirmed=input.confirmed; }
    try {
        for (std::size_t index=0;index<model.layers.size();++index) {
            const auto& layer=model.layers[index];
            const auto* rope=layer.cpu ? &model.cpu_rope : &model.rope;
            if (!layer.cpu && !model.device_ropes.empty()) {
                const auto found=model.device_ropes.find(layer.device);
                if (found!=model.device_ropes.end()) rope=&found->second;
            }
            hidden=forward_layer(layer,state.layers[index],hidden,step,input.confirmed,*rope,options);
            record(hidden);
        }
    } catch (...) { state.abort_speculative(); throw; }
    if (!input.positions || input.advance) state.cache_pos+=T;
    hidden=::tyloi::ops::contiguous(::tyloi::ops::copy(hidden,primary));
    auto finalized=normalize(model,hidden,B,T);
    if (input.raw_hidden) *input.raw_hidden=hidden;
    return finalized;
}
void replay(const Model& model,State& state,std::int64_t accepted) {
    require_forward(state.belongs_to(model),"Qwen3.5 state belongs to another model");
    state.rollback_speculative(accepted);
}

Tensor forward_layer(const Layer& layer, LayerState& state, Tensor input,
                     const AttentionStep& step, std::int64_t confirmed_prefix,
                     const tyloi::ops::RopeTable& rope, const tyloi::models::ForwardOptions& options) {
    const auto target = layer.cpu ? ::tyloi::Device(::tyloi::kCPU) : ::tyloi::Device(::tyloi::kCUDA, layer.device);
    const auto copy = [&](const tyloi::Tensor& value) {
        return value.defined() ? tyloi::ops::contiguous(tyloi::ops::copy(value, target)) : tyloi::Tensor{};
    };
    AttentionStep local_step = step;
    local_step.positions = copy(step.positions);
    local_step.sequence_lengths = copy(step.sequence_lengths);
    local_step.cache_positions = copy(step.cache_positions);
    input = copy(input);
    const auto* full = std::get_if<Full>(&layer.weights);
    auto* trace = full && options.trace_layer >= 0 && options.trace_layer == full->attention.layer ? options.trace : nullptr;
    if (full) {
        auto* cache = std::get_if<tyloi::state::KvView>(&state);
        require_forward(cache != nullptr, "Qwen3.5 full layer state kind mismatch");
        const auto& attention = full->attention;
        const auto capacity = attention.sliding ? attention.attention_window :
            (options.kv_capacity > 0 ? std::max<std::int64_t>(local_step.cache_position + input.size(1), options.kv_capacity)
                                     : attention.max_position_embeddings);
        if (!cache->defined()) *cache = tyloi::state::allocate_kv(input.size(0), attention.kv_heads, capacity,
            attention.attention_head_dim, attention.sliding, input.device(),
            attention.official_bf16 ? ::tyloi::kBFloat16 : ::tyloi::kFloat16, attention.kv_quantization);
        cache->quantization = attention.kv_quantization;
        auto output = attention.forward(input, local_step, *cache, rope, trace);
        return tyloi::models::full_ffn(full->ffn, full->ffn_norm, output[0], output[1],
            attention.rms_norm_eps, attention.norm_weight_offset,
            attention.official_bf16, attention.attention_output_gate, trace);
    }
    const auto* linear = std::get_if<Linear>(&layer.weights);
    auto* request = std::get_if<LinearState>(&state);
    require_forward(linear && request, "Qwen3.5 linear layer state kind mismatch");
    try {
        auto output = linear->attention->forward(request->attention, input, local_step.cache_position, confirmed_prefix);
        auto result = tyloi::models::linear_ffn(linear->ffn, linear->ffn_norm, output[0], output[1],
            linear->geometry.rms_norm_eps, linear->geometry.norm_weight_offset);
        if (confirmed_prefix > 0) { ++request->speculative_projection_batches; ++request->speculative_ffn_batches; }
        return result;
    } catch (...) {
        if (confirmed_prefix > 0 && request->attention.pending()) request->attention.abort();
        throw;
    }
}

} // namespace tyloi::cuda::qwen35
