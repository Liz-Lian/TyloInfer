#include "engine/executor/qwen35.h"

#include "backends/cuda/engine/executor/options.h"
#include "backends/cuda/include/mfq_cuda_norm_ops.h"
#include "backends/cuda/include/mfq_cuda_sampling_ops.h"
#include "ops/norm/rms.h"

#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace mfq::cuda::qwen35 {

namespace {

void require_run(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

FfnExec::Mode ffn_mode(const CudaExecutionContext& execution) {
    return {execution.force_moe_materialized_swiglu,
            execution.force_moe_unfused_reduce,
            execution.decode_graph_serial_branches,
            execution.continuous_batch_cache_serial,
            execution.continuous_batch_ffn_tile,
            execution.continuous_batch_ffn_tiles,
            execution.config.moe_ffn.reuse_input};
}

Tensor copy_to_layer(CudaExecutionContext& execution, Tensor value, const Layer& layer) {
    if (layer.cpu) return value.to(mfq_tensor_backend::kCPU).contiguous();
    return execution.collectives.copy(value, layer.device).contiguous();
}

AttentionStep layer_step(CudaExecutionContext& execution, const Layer& layer,
                         const AttentionStep& step) {
    AttentionStep local = step;
    local.positions = copy_to_layer(execution, step.positions, layer);
    if (step.sequence_lengths.has_value())
        local.sequence_lengths = copy_to_layer(execution, *step.sequence_lengths, layer);
    if (step.cache_positions.has_value())
        local.cache_positions = copy_to_layer(execution, *step.cache_positions, layer);
    if (step.attention_mask.has_value())
        local.attention_mask = copy_to_layer(execution, *step.attention_mask, layer);
    return local;
}

void prepare_linear_state(const LinearAttention& attention, LinearState& state,
                          const Tensor& input, bool transposed) {
    if (state.mixer.defined()) return;
    const auto batch = input.size(0);
    const auto width = 2 * attention.key_heads * attention.key_dim +
                       attention.value_heads * attention.value_dim;
    const auto options = input.options().dtype(mfq_tensor_backend::kFloat32);
    state.mixer.view() = {
        mfq_tensor_backend::zeros({batch, attention.conv_kernel - 1, width}, options),
        mfq_tensor_backend::zeros(
            {batch, attention.value_heads, attention.value_dim, attention.value_dim}, options),
        input.is_cuda() && transposed};
}

void clear_linear_speculative(LinearState& state) noexcept {
    state.projected = {};
    state.mixer.commit();
}

} // namespace

Tensor forward_layer(CudaExecutionContext& execution, const Layer& layer, LayerState& state,
                     mfq::cuda::Executor& executor, std::size_t scratch_offset, Tensor input,
                     const AttentionStep& step, std::int64_t confirmed_prefix,
                     const RopeCache& rope) {
    MfqCudaGuard guard(layer.device);
    auto local_step = layer_step(execution, layer, step);
    input = layer.cpu ? input.to(mfq_tensor_backend::kCPU).contiguous()
                      : execution.collectives.copy(input, layer.device).contiguous();
    const auto tasks = layer.tasks();
    const auto trace = execution.gemma_trace_layer >= 0 &&
                               execution.gemma_trace_layer == tasks[0].scope
                           ? execution.gemma_stage_trace
                           : nullptr;
    const auto execute = [&](std::span<const AttentionKvView> kv,
                             std::span<const LinearStateView> linear,
                             LinearAttention::Options linear_options) {
        Inputs inputs{{std::move(input), {}, {}}, 0, kv, local_step, &rope, trace,
                      execution.gemma_trace_layer, linear, linear_options, scratch_offset};
        return executor.run(tasks, std::move(inputs), ffn_mode(execution));
    };
    if (const auto* full = std::get_if<Full>(&layer.weights)) {
        auto* cache = std::get_if<KVCache>(&state);
        require_run(cache != nullptr, "Qwen3.5 full layer state kind mismatch");
        const auto& attention = full->attention;
        const auto kl_capacity = execution.kl_mmq.kv_cache_capacity;
        const auto capacity = attention.sliding
                                  ? attention.attention_window
                                  : (kl_capacity > 0
                                         ? std::max<std::int64_t>(
                                               local_step.cache_position + input.size(1),
                                               kl_capacity)
                                         : attention.max_position_embeddings);
        if (!cache->defined()) {
            *cache = KVCache(
                input.size(0), attention.kv_heads, capacity, attention.attention_head_dim,
                attention.sliding, input.device(),
                attention.official_bf16 ? mfq_tensor_backend::kBFloat16
                                        : mfq_tensor_backend::kFloat16,
                attention.kv_quantization);
        }
        cache->quantization = attention.kv_quantization;
        const std::array<AttentionKvView, 1> kv{cache->view()};
        return execute(kv, std::span<const LinearStateView>{}, {});
    }
    const auto* linear = std::get_if<Linear>(&layer.weights);
    auto* linear_state = std::get_if<LinearState>(&state);
    require_run(linear != nullptr && linear_state != nullptr,
                "Qwen3.5 linear layer state kind mismatch");
    const auto& attention = linear->attention;
    prepare_linear_state(attention, *linear_state, input, execution.config.gdn_transposed_state);
    if (confirmed_prefix == 0) {
        require_run(!linear_state->mixer.pending(),
                    "linear attention has a pending speculative pass");
    } else {
        require_run(input.is_cuda() && !linear_state->mixer.pending() &&
                        linear_state->mixer.defined() && confirmed_prefix > 0 &&
                        confirmed_prefix < input.size(1),
                    "invalid Qwen3.5 speculative linear-attention transaction");
        linear_state->mixer.begin(local_step.cache_position, confirmed_prefix, input.size(1));
    }
    auto& state_view = linear_state->mixer.view();
    const auto captured = confirmed_prefix > 0 ? &linear_state->projected : nullptr;
    const LinearStateView linear_view{
        {state_view.convolution, state_view.recurrent, state_view.transposed}, captured};
    const std::array<LinearStateView, 1> linear_views{linear_view};
    try {
        auto result = execute(std::span<const AttentionKvView>{}, linear_views,
                              {execution.config.tasks.projection.grouped &&
                                   execution.config.tensor_parallel_shared_linear_attention_input,
                               execution.config.linear_conv_prefill_fused});
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

Run::Run(std::shared_ptr<const Model> model, CudaExecutionContext& context)
    : weights([&model] {
          require_run(model != nullptr, "Qwen3.5 Run requires model weights");
          return std::move(model);
      }()),
      execution(&context),
      executor(std::make_unique<mfq::cuda::Executor>(
          context.profiler, context.kl_mmq, *context.tq, context.branches,
          context.collectives, context.routes, context.config.tasks, context.config.ffn,
          model_parallel_primary_device(context), context.decode_graph_serial_branches,
          context.decode_graph_tp_projection_major)),
      state(*weights, context.config.gdn_transposed_state) {}

void Run::preallocate() {
    if (!runtime_options::kv_vram_preallocated()) return;
    for (std::size_t index = 0; index < weights->layers.size(); ++index) {
        const auto& layer = weights->layers[index];
        if (layer.cpu || !std::holds_alternative<Full>(layer.weights)) continue;
        auto& cache = std::get<KVCache>(state.layers[index]);
        if (cache.defined()) continue;
        const auto& attention = std::get<Full>(layer.weights).attention;
        MfqCudaGuard guard(layer.device);
        cache = KVCache(
            1, attention.kv_heads,
            attention.sliding ? attention.attention_window : attention.max_position_embeddings,
            attention.attention_head_dim, attention.sliding,
            mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, layer.device),
            attention.official_bf16 ? mfq_tensor_backend::kBFloat16
                                   : mfq_tensor_backend::kFloat16,
            attention.kv_quantization);
    }
}

std::int64_t Run::vocab_size() const noexcept { return weights->metadata.vocab_size; }
std::int64_t Run::hidden_size() const noexcept { return weights->metadata.hidden_size; }
std::int64_t Run::num_hidden_layers() const noexcept {
    return weights->metadata.num_hidden_layers;
}
std::int64_t Run::num_attention_heads() const noexcept {
    return weights->metadata.num_attention_heads;
}
std::int64_t Run::num_key_value_heads() const noexcept {
    return weights->metadata.num_key_value_heads;
}
std::int64_t Run::head_dim() const noexcept { return weights->metadata.head_dim; }
std::int64_t Run::max_position_embeddings() const noexcept {
    return weights->metadata.max_position_embeddings;
}
std::int64_t Run::rotary_dim() const noexcept { return weights->metadata.rotary_dim; }
double Run::rope_base() const noexcept { return weights->metadata.rope_base; }
double Run::rms_norm_eps() const noexcept { return weights->metadata.rms_norm_eps; }
double Run::norm_weight_offset() const noexcept {
    return weights->metadata.norm_weight_offset;
}
double Run::final_logit_softcapping() const noexcept {
    return weights->metadata.final_logit_softcapping;
}
double Run::embedding_scale() const noexcept { return weights->metadata.embedding_scale; }
std::int64_t Run::hc_mult() const noexcept { return weights->metadata.hc_mult; }
double Run::hc_eps() const noexcept { return weights->metadata.hc_eps; }
bool Run::tie_word_embeddings() const noexcept { return weights->metadata.tie_word_embeddings; }
std::int64_t Run::num_experts() const noexcept { return weights->metadata.num_experts; }
std::string_view Run::model_type() const noexcept { return weights->metadata.model_type; }
std::string_view Run::layer_type(std::int64_t layer) const {
    return weights->metadata.layer_types.at(static_cast<std::size_t>(layer));
}

Tensor Run::normalize_ids(Tensor ids) const {
    require_run(ids.dim() == 1 || ids.dim() == 2,
                "token IDs must have shape [tokens] or [batch,tokens]");
    if (ids.dim() == 1) ids = ids.unsqueeze(0);
    require_run(ids.size(0) > 0 && ids.size(1) > 0, "token IDs must not be empty");
    return execution->collectives
        .copy(ids.to(mfq_tensor_backend::kInt64), execution->layer_placement.primary_device())
        .contiguous();
}

Tensor Run::embed_forward(Tensor ids) const {
    CudaScope scope(*execution);
    auto normalized = normalize_ids(std::move(ids));
    return execution->profiler.measure("model.embed", [&] {
        return quant_embedding_lookup(weights->embed, std::move(normalized))
            .to(mfq_tensor_backend::kFloat16)
            .contiguous();
    });
}

std::optional<Tensor> Run::decode_lengths(const Tensor& ids) const {
    if (state.cache_pos <= 0 || ids.size(1) != 1) return {};
    return mfq_tensor_backend::full(
        {ids.size(0)}, state.cache_pos + 1,
        mfq_tensor_backend::TensorOptions()
            .dtype(mfq_tensor_backend::kInt64)
            .device(mfq_tensor_backend::Device(
                mfq_tensor_backend::kCUDA, execution->layer_placement.primary_device())));
}

Tensor Run::hidden_forward(Tensor ids, std::optional<Tensor> positions,
                           std::optional<Tensor> lengths, std::vector<Tensor>* trace,
                           std::optional<Tensor> cache_positions, Tensor* raw_hidden,
                           std::int64_t confirmed_prefix, Plan plan) {
    CudaScope scope(*execution);
    auto normalized = normalize_ids(std::move(ids));
    return hidden_forward_inputs(
        normalized, embed_forward(normalized), std::move(positions), std::move(lengths), trace,
        {}, false, std::move(cache_positions), raw_hidden, confirmed_prefix, plan);
}

Tensor Run::hidden_forward_inputs(Tensor ids, Tensor embeddings,
                                  std::optional<Tensor> positions,
                                  std::optional<Tensor> lengths,
                                  std::vector<Tensor>* trace,
                                  std::optional<Tensor> attention_mask,
                                  bool advance_cache_with_position_ids,
                                  std::optional<Tensor> cache_positions,
                                  Tensor* raw_hidden,
                                  std::int64_t confirmed_prefix, Plan plan) {
    CudaScope scope(*execution);
    ids = normalize_ids(std::move(ids));
    const auto B = ids.size(0);
    const auto T = ids.size(1);
    require_run(embeddings.dim() == 3 && embeddings.size(0) == B && embeddings.size(1) == T &&
                    embeddings.size(2) == hidden_size(),
                "inputs_embeds shape must match [batch,tokens,hidden_size]");
    require_run(confirmed_prefix >= 0 &&
                    (confirmed_prefix == 0 ||
                     (confirmed_prefix < T && B == 1 && state.cache_pos > 0 &&
                      !positions.has_value() && !cache_positions.has_value() &&
                      !attention_mask.has_value() && supports_speculation())),
                "unsupported speculative backbone geometry");
    require_run(state.speculative_start < 0,
                "commit or roll back the pending speculative pass before forwarding");
    require_run(state.cache_pos + T <= max_position_embeddings(),
                "Qwen3.5 history exceeds context capacity");
    if (state.cache_pos == 0) {
        state.reset(B);
    } else {
        require_run(state.batch == B, "Qwen3.5 cache batch does not match input batch");
    }
    if (lengths.has_value())
        require_run(lengths->dim() == 1 && lengths->size(0) == B,
                    "sequence lengths must have shape [batch]");
    if (attention_mask.has_value())
        require_run(attention_mask->dim() == 2 && attention_mask->size(0) == B &&
                        attention_mask->size(1) >= state.cache_pos + T,
                    "attention_mask must cover [batch,cached_plus_current_tokens]");

    const auto primary = execution->layer_placement.primary_device();
    auto physical = cache_positions.has_value()
                        ? execution->collectives.copy(
                              cache_positions->to(mfq_tensor_backend::kInt64), primary)
                        : mfq_tensor_backend::arange(
                              state.cache_pos, state.cache_pos + T,
                              mfq_tensor_backend::TensorOptions()
                                  .device(mfq_tensor_backend::Device(
                                      mfq_tensor_backend::kCUDA, primary))
                                  .dtype(mfq_tensor_backend::kInt64));
    physical = physical.contiguous();
    require_run((physical.dim() == 1 && physical.numel() == T) ||
                    (physical.dim() == 2 && physical.size(0) == B && physical.size(1) == T),
                "cache_positions must have shape [tokens] or [batch,tokens]");
    auto semantic = positions.has_value()
                        ? execution->collectives.copy(positions->to(mfq_tensor_backend::kInt64),
                                                       primary)
                        : (state.decode_position_delta == 0
                               ? physical
                               : physical + state.decode_position_delta);
    semantic = semantic.contiguous();
    const auto rank = semantic.dim();
    require_run(
        Config::valid_positions(rank, rank > 1 ? semantic.size(0) : 0, semantic.size(-1), B, T,
                                weights->rope.sections.numel() > 0),
        "position_ids must have shape [tokens], [batch,tokens], or configured grid-MRoPE [3,tokens]");
    if (lengths.has_value()) {
        lengths = execution->collectives
                      .copy(lengths->to(mfq_tensor_backend::kInt64), primary)
                      .contiguous();
    } else {
        lengths = decode_lengths(ids);
    }

    std::optional<Tensor> physical_for_layers;
    if (weights->rope.sections.numel() > 0 || positions.has_value() || cache_positions.has_value())
        physical_for_layers = physical;
    AttentionStep step{semantic, lengths, physical_for_layers, {}, state.cache_pos, plan.kv_length,
                       plan.attention_parts};
    auto hidden = execution->collectives.copy(std::move(embeddings), primary).contiguous();
    if (trace)
        trace->push_back(execution->collectives.copy(hidden, primary)
                             .to(mfq_tensor_backend::kFloat32)
                             .contiguous()
                             .clone());
    if (confirmed_prefix > 0) {
        state.speculative_start = state.cache_pos;
        state.speculative_confirmed = confirmed_prefix;
    }
    try {
        for (std::size_t index = 0; index < weights->layers.size(); ++index) {
            const auto& layer = weights->layers[index];
            const RopeCache* active_rope = layer.cpu ? &weights->cpu_rope : &weights->rope;
            if (!layer.cpu && !weights->device_ropes.empty()) {
                const auto it = weights->device_ropes.find(layer.device);
                if (it != weights->device_ropes.end()) active_rope = &it->second;
            }
            hidden = forward_layer(*execution, layer, state.layers[index], *executor, 2 * index,
                                   std::move(hidden), step, confirmed_prefix, *active_rope);
            if (trace)
                trace->push_back(execution->collectives.copy(hidden, primary)
                                     .to(mfq_tensor_backend::kFloat32)
                                     .contiguous()
                                     .clone());
        }
    } catch (...) {
        state.abort_speculative();
        throw;
    }
    if (!positions.has_value() || advance_cache_with_position_ids) state.cache_pos += T;
    hidden = execution->collectives.copy(std::move(hidden), primary).contiguous();
    auto raw = raw_hidden ? hidden : Tensor{};
    auto finalized = finalize_hidden(std::move(hidden), B, T);
    if (raw_hidden) *raw_hidden = std::move(raw);
    return finalized;
}

Tensor Run::finalize_hidden(Tensor hidden, std::int64_t B, std::int64_t T) const {
    return execution->profiler.measure("model.output_norm", [&] {
        auto flat = hidden.reshape({B * T, hidden_size()});
        return qwen_rms_norm(flat.to(mfq_tensor_backend::kFloat32), weights->output_norm,
                             rms_norm_eps(), norm_weight_offset())
            .reshape({B, T, hidden_size()});
    });
}

Tensor Run::apply_final_logit_softcap(Tensor logits) const {
    const auto cap = final_logit_softcapping();
    return cap > 0.0 ? mfq_tensor_backend::tanh(logits / cap) * cap : std::move(logits);
}

Tensor Run::logits_from_hidden(Tensor hidden) const {
    return apply_final_logit_softcap(execution->profiler.measure(
        "model.lm_head", [&] { return execution->projector().run(weights->lm_head, hidden); }));
}

Tensor Run::last_hidden(Tensor hidden) const {
    return hidden.index({mfq_tensor_backend::indexing::Slice(), -1,
                         mfq_tensor_backend::indexing::Slice()});
}

Tensor Run::next_token_from_hidden(Tensor hidden) const {
    auto input = last_hidden(std::move(hidden)).to(mfq_tensor_backend::kFloat16).contiguous();
    auto logits = execution->profiler.measure(
        "model.lm_head", [&] { return execution->projector().run(weights->lm_head, input); });
    return sample_greedy_cuda(logits.contiguous().view({input.size(0), -1}));
}

Tensor Run::forward(Tensor ids) { return logits_from_hidden(hidden_forward(std::move(ids))); }

Tensor Run::forward_inputs(Tensor ids, Tensor embeddings, std::optional<Tensor> positions,
                           std::optional<Tensor> lengths) {
    return logits_from_hidden(hidden_forward_inputs(
        std::move(ids), std::move(embeddings), std::move(positions), std::move(lengths)));
}

Tensor Run::last_logits(Tensor ids) {
    auto normalized = normalize_ids(std::move(ids));
    auto hidden = hidden_forward(normalized, {}, decode_lengths(normalized));
    auto input = last_hidden(std::move(hidden)).to(mfq_tensor_backend::kFloat16).contiguous();
    return apply_final_logit_softcap(execution->profiler.measure(
        "model.lm_head", [&] { return execution->projector().run(weights->lm_head, std::move(input)); }));
}

Tensor Run::next_token(Tensor ids) {
    auto normalized = normalize_ids(std::move(ids));
    return next_token_from_hidden(hidden_forward(normalized, {}, decode_lengths(normalized)));
}

Tensor Run::hidden_forward_static(Tensor ids, Tensor positions, Tensor lengths, Plan plan) {
    return hidden_forward(std::move(ids), positions, lengths, nullptr, positions, nullptr, 0, plan);
}

Tensor Run::last_logits_static(Tensor ids, Tensor positions, Tensor lengths, Plan plan) {
    auto hidden = hidden_forward_static(std::move(ids), std::move(positions), std::move(lengths), plan);
    auto input = last_hidden(std::move(hidden)).to(mfq_tensor_backend::kFloat16).contiguous();
    return apply_final_logit_softcap(execution->profiler.measure(
        "model.lm_head", [&] { return execution->projector().run(weights->lm_head, std::move(input)); }));
}

Tensor Run::next_token_static(Tensor ids, Tensor positions, Tensor lengths, Plan plan) {
    return next_token_from_hidden(
        hidden_forward_static(std::move(ids), std::move(positions), std::move(lengths), plan));
}

bool Run::supports_speculation() const noexcept {
    return !weights->layers.empty() &&
           std::all_of(weights->layers.begin(), weights->layers.end(), [](const Layer& layer) {
               if (const auto* full = std::get_if<Full>(&layer.weights))
                   return !full->attention.sliding;
               return true;
           });
}

std::vector<Tensor> Run::graph_storage() const {
    std::vector<Tensor> result;
    const auto append = [&result](const Tensor& tensor) {
        if (tensor.defined()) result.push_back(tensor);
    };
    for (std::size_t index = 0; index < state.layers.size(); ++index) {
        if (const auto* cache = std::get_if<KVCache>(&state.layers[index])) {
            if (cache->is_paged()) {
                append(cache->k_chunk_ptrs);
                append(cache->v_chunk_ptrs);
                append(cache->page_table);
            } else {
                append(cache->k);
                append(cache->v);
            }
        } else {
            const auto& linear = std::get<LinearState>(state.layers[index]);
            append(linear.mixer.view().convolution);
            append(linear.mixer.view().recurrent);
        }
    }
    for (const auto& attention : executor->scratch()) {
        append(attention.partial_o);
        append(attention.partial_m);
        append(attention.partial_l);
        append(attention.mask);
        append(attention.kv_max);
        append(attention.meta);
    }
    return result;
}

void Run::rollback_speculative(std::int64_t accepted_suffix) {
    require_run(state.speculative_start >= 0 && accepted_suffix >= 0,
                "no speculative transaction");
    try {
        const auto keep = state.rewind(accepted_suffix);
        for (std::size_t index = 0; index < state.layers.size(); ++index) {
            auto* linear = std::get_if<LinearState>(&state.layers[index]);
            if (linear == nullptr) continue;
            MfqCudaGuard guard(weights->layers[index].device);
            auto& view = linear->mixer.view();
            const auto* layer = std::get_if<Linear>(&weights->layers[index].weights);
            require_run(layer != nullptr, "Qwen3.5 recurrent layer weight kind mismatch");
            const auto replay = layer->replay(keep - state.speculative_start);
            const LinearStateView linear_view{
                {view.convolution, view.recurrent, view.transposed}, &linear->projected};
            Inputs inputs;
            inputs.output = std::nullopt;
            inputs.linear = {&linear_view, 1};
            executor->run(std::span<const Task>{&replay, 1}, std::move(inputs));
        }
        state.commit_speculative();
        state.cache_pos = keep;
    } catch (...) {
        state.abort_speculative();
        throw;
    }
}

} // namespace mfq::cuda::qwen35
