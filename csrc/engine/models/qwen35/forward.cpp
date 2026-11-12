#include "engine/models/qwen35/forward.h"
#include "engine/models/common/ffn.h"

#ifndef TYLOINFER_NATIVE_CUDA
#include "engine/backends/cuda/native/collectives.h"
#include "engine/backends/cuda/ops/norm/rms.h"
#include "engine/backends/cuda/ops/tensor.h"
#endif

#include <algorithm>
#include <array>
#include <stdexcept>
#include <utility>

namespace tyloi::cuda::qwen35 {

namespace {

void require_forward(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

#ifndef TYLOINFER_NATIVE_CUDA
DenseTensor normalize_ids(Collectives& collectives, int primary, DenseTensor ids) {
    require_forward(ids.dim() == 1 || ids.dim() == 2,
                    "token IDs must have shape [tokens] or [batch,tokens]");
    if (ids.dim() == 1) ids = ids.unsqueeze(0);
    require_forward(ids.size(0) > 0 && ids.size(1) > 0,
                    "token IDs must not be empty");
    return collectives.copy(ids.to(::tyloi::kInt64), primary).contiguous();
}

std::optional<DenseTensor> decode_lengths(const State& state, const DenseTensor& ids, int primary) {
    if (state.cache_pos <= 0 || ids.size(1) != 1) return {};
    return ::tyloi::cuda::full(
        {ids.size(0)}, state.cache_pos + 1,
        ::tyloi::TensorOptions()
            .dtype(::tyloi::kInt64)
            .device(::tyloi::Device(::tyloi::kCUDA, primary)));
}

DenseTensor copy_to_layer(Collectives& collectives, DenseTensor value, const Layer& layer) {
    if (layer.cpu) return value.to(::tyloi::kCPU).contiguous();
    return collectives.copy(value, layer.device).contiguous();
}

AttentionStep layer_step(Collectives& collectives, const Layer& layer,
                         const AttentionStep& step) {
    AttentionStep local = step;
    local.positions = copy_to_layer(collectives, step.positions.dense(), layer);
    if (step.sequence_lengths.defined())
        local.sequence_lengths = copy_to_layer(collectives, step.sequence_lengths.dense(), layer);
    if (step.cache_positions.defined())
        local.cache_positions = copy_to_layer(collectives, step.cache_positions.dense(), layer);
    return local;
}

void prepare_linear_state(const LinearAttention& attention, LinearState& state,
                          const DenseTensor& input, bool transposed) {
    if (state.mixer.defined()) return;
    const auto batch = input.size(0);
    const auto width = 2 * attention.key_heads * attention.key_dim +
                       attention.value_heads * attention.value_dim;
    const auto options = input.options().dtype(::tyloi::kFloat32);
    state.mixer.view() = {
        ::tyloi::cuda::zeros({batch, attention.conv_kernel - 1, width}, options),
        ::tyloi::cuda::zeros(
            {batch, attention.value_heads, attention.value_dim, attention.value_dim}, options),
        input.is_cuda() && transposed};
}

void clear_linear_speculative(LinearState& state) noexcept {
    state.projected = {};
    state.mixer.commit();
}
#endif

} // namespace

#ifdef TYLOINFER_NATIVE_CUDA
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
#else
DenseTensor embed(const Model& model, Executor& executor, Collectives& collectives,
             int primary, DenseTensor ids, const Executor::Mode& mode) {
    auto normalized = normalize_ids(collectives, primary, std::move(ids));
    return ::tyloi::cuda::embed(model.embed, std::move(normalized), collectives,
                                executor.profiler(), primary)
        .to(::tyloi::kFloat16).contiguous();
}

DenseTensor normalize(const Model& model, Executor& executor, DenseTensor hidden,
                 std::int64_t batch, std::int64_t tokens) {
    hidden = hidden.reshape({batch, tokens, model.metadata.hidden_size});
    return executor.profiler().measure("model.output_norm", [&] {
        return qwen_rms_norm(hidden.reshape({-1, hidden.size(-1)}).to(::tyloi::kFloat32),
                             model.output_norm, model.metadata.rms_norm_eps,
                             model.metadata.norm_weight_offset)
            .reshape(hidden.sizes());
    });
}

DenseTensor softcap(const Model& model, Executor& executor, DenseTensor logits) {
    return ::tyloi::cuda::ops::softcap(std::move(logits), model.metadata.final_logit_softcapping);
}

DenseTensor logits(const Model& model, Executor& executor, DenseTensor hidden, bool capped,
              const Executor::Mode& mode) {
    auto project = executor.projector(mode.ffn);
    return ::tyloi::cuda::logits(model.lm_head, std::move(hidden), project,
                                 executor.profiler(),
                                 capped ? model.metadata.final_logit_softcapping : 0.0);
}

namespace {

DenseTensor forward_impl(const Model& model, State& state, Executor& executor,
                    Collectives& collectives, const Executor::Mode& options, DenseTensor ids,
                    DenseTensor embeddings, std::optional<DenseTensor> positions,
                    std::optional<DenseTensor> lengths, tyloi::models::Input::Trace* trace,
                    bool advance, std::optional<DenseTensor> cache_positions,
                    DenseTensor* raw_hidden,
                    std::int64_t confirmed, std::int64_t kv_length,
                    std::int64_t attention_parts) {
    ids = normalize_ids(collectives, options.primary, std::move(ids));
    const auto B = ids.size(0);
    const auto T = ids.size(1);
    require_forward(embeddings.dim() == 3 && embeddings.size(0) == B &&
                        embeddings.size(1) == T &&
                        embeddings.size(2) == model.metadata.hidden_size,
                    "inputs_embeds shape must match [batch,tokens,hidden_size]");
    require_forward(
        confirmed >= 0 &&
            (confirmed == 0 ||
             (confirmed < T && B == 1 && state.cache_pos > 0 && !positions.has_value() &&
              !cache_positions.has_value() &&
              supports_speculation(model))),
        "unsupported speculative backbone geometry");
    require_forward(state.speculative_start < 0,
                    "commit or roll back the pending speculative pass before forwarding");
    require_forward(state.cache_pos + T <= model.metadata.max_position_embeddings,
                    "Qwen3.5 history exceeds context capacity");
    if (state.cache_pos == 0) {
        state.reset(B);
    } else {
        require_forward(state.batch == B, "Qwen3.5 cache batch does not match input batch");
    }
    if (lengths.has_value())
        require_forward(lengths->dim() == 1 && lengths->size(0) == B,
                        "sequence lengths must have shape [batch]");
    auto physical = cache_positions.has_value()
                        ? collectives.copy(DenseTensor(*cache_positions).to(::tyloi::kInt64),
                                           options.primary)
                        : ::tyloi::cuda::arange(
                              state.cache_pos, state.cache_pos + T,
                              ::tyloi::TensorOptions()
                                  .device(::tyloi::Device(
                                      ::tyloi::kCUDA, options.primary))
                                  .dtype(::tyloi::kInt64));
    physical = physical.contiguous();
    require_forward((physical.dim() == 1 && physical.numel() == T) ||
                        (physical.dim() == 2 && physical.size(0) == B &&
                         physical.size(1) == T),
                    "cache_positions must have shape [tokens] or [batch,tokens]");
    auto semantic = positions.has_value()
                        ? collectives.copy(DenseTensor(*positions).to(::tyloi::kInt64),
                                           options.primary)
                        : (state.decode_position_delta == 0
                               ? physical
                               : physical + state.decode_position_delta);
    semantic = semantic.contiguous();
    const auto rank = semantic.dim();
    const bool has_mrope = model.rope.sections.numel() > 0;
    require_forward(
        Config::valid_positions(rank, rank > 1 ? semantic.size(0) : 0,
                                semantic.size(-1), B, T,
                                has_mrope),
        "position_ids must have shape [tokens], [batch,tokens], or configured grid-MRoPE [3,tokens]");
    if (lengths.has_value()) {
        lengths = collectives.copy(DenseTensor(*lengths).to(::tyloi::kInt64), options.primary)
                       .contiguous();
    } else {
        lengths = decode_lengths(state, ids, options.primary);
    }

    std::optional<DenseTensor> physical_for_layers;
    if (has_mrope || positions.has_value() || cache_positions.has_value())
        physical_for_layers = physical;
    AttentionStep step{semantic, lengths ? MfqOptional<DenseTensor>(*lengths) : mfq_nullopt,
                       physical_for_layers, state.cache_pos, kv_length, attention_parts};
    auto hidden = collectives.copy(std::move(embeddings), options.primary).contiguous();
    if (trace)
        trace->push_back(collectives.copy(hidden, options.primary)
                             .to(::tyloi::kFloat32)
                             .contiguous()
                             .clone());
    if (confirmed > 0) {
        state.speculative_start = state.cache_pos;
        state.speculative_confirmed = confirmed;
    }
    try {
        for (std::size_t index = 0; index < model.layers.size(); ++index) {
            const auto& layer = model.layers[index];
            const auto* active_rope = layer.cpu ? &model.cpu_rope : &model.rope;
            if (!layer.cpu && !model.device_ropes.empty()) {
                const auto it = model.device_ropes.find(layer.device);
                if (it != model.device_ropes.end()) active_rope = &it->second;
            }
            hidden = forward_layer(collectives, layer, state.layers[index], executor,
                                   2 * index, std::move(hidden), step, confirmed,
                                   *active_rope, options);
            if (trace)
                trace->push_back(collectives.copy(hidden, options.primary)
                                     .to(::tyloi::kFloat32)
                                     .contiguous()
                                     .clone());
        }
    } catch (...) {
        state.abort_speculative();
        throw;
    }
    if (!positions.has_value() || advance) state.cache_pos += T;
    hidden = collectives.copy(std::move(hidden), options.primary).contiguous();
    auto raw = raw_hidden ? hidden : DenseTensor{};
    auto finalized = normalize(model, executor, std::move(hidden), B, T);
    if (raw_hidden) *raw_hidden = std::move(raw);
    return finalized;
}

} // namespace

DenseTensor forward(const Model& model, State& state, Executor& executor, Collectives& collectives,
               const Executor::Mode& options, tyloi::models::Input input) {
    auto ids = normalize_ids(collectives, options.primary, std::move(input.ids));
    auto embeddings = input.embeddings ? *input.embeddings
        : embed(model, executor, collectives, options.primary, ids, options);
    auto positions = std::move(input.positions);
    auto lengths = std::move(input.lengths);
    auto cache_positions = std::move(input.cache_positions);
    return forward_impl(model, state, executor, collectives, options, std::move(ids),
                        std::move(embeddings), std::move(positions),
                        std::move(lengths), input.trace, input.advance,
                        std::move(cache_positions), input.raw_hidden,
                        input.confirmed, input.kv_length, input.attention_parts);
}

bool supports_speculation(const Model& model) noexcept {
    return !model.layers.empty() &&
           std::all_of(model.layers.begin(), model.layers.end(), [](const Layer& layer) {
               if (const auto* full = std::get_if<Full>(&layer.weights))
                   return !full->attention.sliding;
               return true;
           });
}

void replay(const Model& model, State& state, Executor& executor, std::int64_t accepted) {
    require_forward(state.speculative_start >= 0 && accepted >= 0,
                    "no speculative transaction");
    try {
        const auto keep = state.rewind_suffix(accepted);
        for (std::size_t index = 0; index < state.layers.size(); ++index) {
            auto* linear = std::get_if<LinearState>(&state.layers[index]);
            if (linear == nullptr) continue;
            MfqCudaGuard guard(model.layers[index].device);
            const auto* layer = std::get_if<Linear>(&model.layers[index].weights);
            require_forward(layer != nullptr, "Qwen3.5 recurrent layer weight kind mismatch");
            auto& state_view = linear->mixer.view();
            const LinearView linear_view{
                state_view.convolution.dense(), state_view.recurrent.dense(), state_view.transposed};
            layer->attention.replay(linear_view, linear->projected,
                                    keep - state.speculative_start);
        }
        state.commit_speculative();
        state.cache_pos = keep;
    } catch (...) {
        state.abort_speculative();
        throw;
    }
}

#endif

#ifdef TYLOINFER_NATIVE_CUDA
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
#else
DenseTensor forward_layer(Collectives& collectives, const Layer& layer, LayerState& state,
                     tyloi::cuda::Executor& executor, std::size_t scratch_offset, DenseTensor input,
                     const AttentionStep& step, std::int64_t confirmed_prefix,
                     const RopeCache& rope,
                     const Executor::Mode& options) {
    MfqCudaGuard guard(layer.device);
    auto local_step = layer_step(collectives, layer, step);
    input = layer.cpu ? input.to(::tyloi::kCPU).contiguous()
                      : collectives.copy(input, layer.device).contiguous();
    const auto* full = std::get_if<Full>(&layer.weights);
    auto* trace = full != nullptr && options.trace_layer >= 0 &&
                                options.trace_layer == full->attention.layer
                            ? options.trace
                            : nullptr;
    if (full != nullptr) {
        auto* cache = std::get_if<tyloi::state::KvView>(&state);
        require_forward(cache != nullptr, "Qwen3.5 full layer state kind mismatch");
        const auto& attention = full->attention;
        const auto capacity = attention.sliding
                                  ? attention.attention_window
                                  : (options.kv_capacity > 0
                                         ? std::max<std::int64_t>(
                                               local_step.cache_position + input.size(1),
                                               options.kv_capacity)
                                         : attention.max_position_embeddings);
        if (!cache->defined()) {
            *cache = tyloi::state::allocate_kv(
                input.size(0), attention.kv_heads, capacity, attention.attention_head_dim,
                attention.sliding, input.device(),
                attention.official_bf16 ? ::tyloi::kBFloat16
                                        : ::tyloi::kFloat16,
                attention.kv_quantization);
        }
        cache->quantization = attention.kv_quantization;
        auto output = attention.forward(executor.projector(options.ffn), executor.profiler(),
                                        executor.attention_options(), std::move(input), local_step,
                                        *cache, executor.attention_scratch(scratch_offset), rope,
                                        trace);
        return ::tyloi::models::full_ffn(
            full->ffn, full->ffn_norm, std::move(output[0]), std::move(output[1]),
            attention.rms_norm_eps, attention.norm_weight_offset,
            attention.official_bf16, attention.attention_output_gate, trace).dense();
    }
    const auto* linear = std::get_if<Linear>(&layer.weights);
    auto* linear_state = std::get_if<LinearState>(&state);
    require_forward(linear != nullptr && linear_state != nullptr,
                    "Qwen3.5 linear layer state kind mismatch");
    const auto& attention = linear->attention;
    prepare_linear_state(attention, *linear_state, input, options.transposed);
    if (confirmed_prefix == 0) {
        require_forward(!linear_state->mixer.pending(),
                        "linear attention has a pending speculative pass");
    } else {
        require_forward(input.is_cuda() && !linear_state->mixer.pending() &&
                            linear_state->mixer.defined() && confirmed_prefix > 0 &&
                            confirmed_prefix < input.size(1),
                        "invalid Qwen3.5 speculative linear-attention transaction");
        linear_state->mixer.begin(local_step.cache_position, confirmed_prefix, input.size(1));
    }
    auto& state_view = linear_state->mixer.view();
    const auto captured = confirmed_prefix > 0 ? &linear_state->projected : nullptr;
    const LinearView linear_view{
        state_view.convolution.dense(), state_view.recurrent.dense(), state_view.transposed};
    try {
        auto output = linear->attention.forward(
            executor.projector(options.ffn), executor.profiler(), options.linear,
            std::move(input), linear_view, captured);
        auto result = ::tyloi::models::linear_ffn(
            linear->ffn, linear->ffn_norm, std::move(output[0]), std::move(output[1]),
            attention.rms_norm_eps, attention.norm_weight_offset).dense();
        if (confirmed_prefix > 0) {
            ++linear_state->speculative_projection_batches;
            ++linear_state->speculative_ffn_batches;
        }
        return result;
    } catch (...) {
        if (confirmed_prefix > 0) {
            linear_state->mixer.abort();
            clear_linear_speculative(*linear_state);
        }
        throw;
    }
}
#endif

} // namespace tyloi::cuda::qwen35
