#include "engine/models/minicpmo45/forward.h"
#include "engine/models/common/ffn.h"
#include "engine/backends/cuda/native/collectives.h"
#include "engine/backends/cuda/ops/norm/rms.h"
#include "engine/backends/cuda/ops/projection/projection.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <utility>

namespace tyloi::cuda::minicpmo45 {


namespace {
Tensor token_input(const Tensor& ids,int primary) {
    if (ids.dim()!=1 && ids.dim()!=2) throw std::runtime_error("token IDs must have shape [tokens] or [batch,tokens]");
    auto value=ids.dim()==1?::tyloi::ops::unsqueeze(ids,0):ids;
    if (value.size(0)<=0 || value.size(1)<=0) throw std::runtime_error("token IDs must not be empty");
    return ::tyloi::ops::contiguous(::tyloi::ops::copy(::tyloi::ops::cast(value,kInt64),Device(kCUDA,primary)));
}
}
Tensor embed(const Model& model,int primary,const Tensor& ids) {
    auto normalized=token_input(ids,primary);
    auto result=::tyloi::ops::measure("model.embed",normalized.device(),[&] {return ::tyloi::ops::embedding(normalized,model.embed);});
    return result;
}
Tensor normalize(const Model& model,const Tensor& hidden,int64_t batch,int64_t tokens) {
    auto shaped=::tyloi::ops::reshape(hidden,{batch,tokens,model.metadata.hidden_size});
    return ::tyloi::ops::measure("model.output_norm",shaped.device(),[&] {
        return ::tyloi::ops::rms_norm(::tyloi::ops::cast(shaped,kFloat32),model.output_norm,model.metadata.rms_norm_eps,1.0);
    });
}
Tensor logits(const Model& model,const Tensor& hidden,bool) {
    auto result=::tyloi::ops::measure("model.lm_head",hidden.device(),[&] {return ::tyloi::ops::project(hidden,model.lm_head);});
    return result;
}
Tensor forward(const Model& model,State& state,tyloi::models::Input input,const tyloi::models::ForwardOptions& options) {
    if (!state.belongs_to(model)) throw std::invalid_argument("MiniCPM request belongs to another model");

    auto ids=token_input(input.ids,options.primary);
    const auto B=ids.size(0),T=ids.size(1);
    if (input.confirmed!=0) throw std::runtime_error("unsupported speculative backbone geometry");
    if (state.layers.size()!=model.layers.size()) throw std::runtime_error("MiniCPM layer state mismatch");
    if (T>model.metadata.max_position_embeddings || state.cache_pos>model.metadata.max_position_embeddings-T)
        throw std::runtime_error("MiniCPM history exceeds context capacity");
    if (state.cache_pos==0) state.reset(B);
    if (state.batch!=B) throw std::runtime_error("MiniCPM cache batch does not match inputs");
    auto hidden=input.embeddings ? input.embeddings : embed(model,options.primary,ids);
    if (hidden.dim()!=3 || hidden.size(0)!=B || hidden.size(1)!=T || hidden.size(2)!=model.metadata.hidden_size) throw std::runtime_error("inputs_embeds must match [batch,tokens,hidden_size]");
    const Device primary(kCUDA,options.primary);
    auto physical=input.cache_positions ? ::tyloi::ops::contiguous(::tyloi::ops::copy(::tyloi::ops::cast(input.cache_positions,kInt64),primary))
        : ::tyloi::ops::arange(state.cache_pos,state.cache_pos+T,primary);
    if (!((physical.dim()==1 && physical.numel()==T) || (physical.dim()==2 && physical.size(0)==B && physical.size(1)==T)))
        throw std::runtime_error("cache_positions must have shape [tokens] or [batch,tokens]");
    auto positions=input.positions ? ::tyloi::ops::contiguous(::tyloi::ops::copy(::tyloi::ops::cast(input.positions,kInt64),primary))
        : (state.decode_position_delta ? ::tyloi::ops::add(physical,state.decode_position_delta) : physical);
    if (!((positions.dim()==1 && positions.numel()==T) || (positions.dim()==2 && positions.size(0)==B && positions.size(1)==T)))
        throw std::runtime_error("position_ids must have shape [tokens] or [batch,tokens]");
    if (input.lengths) {
        if (input.lengths.dim()!=1 || input.lengths.size(0)!=B) throw std::runtime_error("sequence lengths must have shape [batch]");
        input.lengths=::tyloi::ops::contiguous(::tyloi::ops::copy(::tyloi::ops::cast(input.lengths,kInt64),primary));
    } else if (state.cache_pos>0 && T==1) input.lengths=::tyloi::ops::full(std::array{B},state.cache_pos+1,primary);
    hidden=::tyloi::ops::contiguous(::tyloi::ops::copy(hidden,primary));
    const auto trace=[&] { if (input.trace) input.trace->push_back(::tyloi::ops::clone(::tyloi::ops::cast(::tyloi::ops::copy(hidden,primary),kFloat32))); };
    trace();
    for (std::size_t i=0;i<model.layers.size();++i) {
        const auto& layer=model.layers[i]; const auto& attention=layer.attention;
        const Device target(layer.cpu?kCPU:kCUDA,layer.cpu?0:layer.device);
        const auto local=[&](const Tensor& value) {return ::tyloi::ops::contiguous(::tyloi::ops::copy(value,target));};
        hidden=local(hidden);
        AttentionStep step{local(positions),{},(input.cache_positions ? local(physical) : Tensor{}),state.cache_pos,input.kv_length,input.attention_parts};
        if (input.lengths) step.sequence_lengths=local(input.lengths);
        auto& cache=state.layers[i];
        const auto capacity=false ? attention.attention_window : (options.kv_capacity>0 ?
            std::max(state.cache_pos+T,options.kv_capacity) : attention.max_position_embeddings);
        if (!cache.defined()) cache=tyloi::state::allocate_kv(B,attention.kv_heads,capacity,attention.attention_head_dim,
            false,hidden.device(),kFloat16,{});

        const auto* rope=layer.cpu ? &model.cpu_rope : &model.rope;
        if (!layer.cpu) { const auto found=model.device_ropes.find(layer.device); if (found!=model.device_ropes.end()) rope=&found->second; }
        auto* layer_trace=options.trace_layer==attention.layer?options.trace:nullptr;
        auto output=attention.forward(hidden,step,cache,*rope,layer_trace);
        hidden=::tyloi::models::full_ffn(layer.ffn,layer.ffn_norm,output[0],output[1],attention.rms_norm_eps,
            attention.norm_weight_offset,attention.official_bf16,attention.attention_output_gate,layer_trace);
        trace();
    }
    if (!input.positions || input.advance) state.cache_pos+=T;
    hidden=::tyloi::ops::copy(hidden,primary);
    hidden=::tyloi::ops::contiguous(hidden);
    if (input.raw_hidden) *input.raw_hidden=hidden;
    auto finalized=normalize(model,hidden,B,T);
    return finalized;
}

} // namespace tyloi::cuda::minicpmo45
