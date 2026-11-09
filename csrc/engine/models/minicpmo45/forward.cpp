#include "engine/models/minicpmo45/forward.h"
#include "engine/backends/cuda/native/collectives.h"
#include "engine/backends/cuda/ops/norm/rms.h"
#include "engine/backends/cuda/ops/projection/projection.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace tyloi::cuda::minicpmo45 {

namespace {
DenseTensor token_ids(Collectives& copies, int device, DenseTensor ids) {
    TYLOINFER_RUNTIME_CHECK(ids.dim() == 1 || ids.dim() == 2,
        "token IDs must have shape [tokens] or [batch,tokens]");
    if (ids.dim() == 1) ids = ids.unsqueeze(0);
    TYLOINFER_RUNTIME_CHECK(ids.size(0) > 0 && ids.size(1) > 0, "token IDs must not be empty");
    return copies.copy(ids.to(::tyloi::kInt64), device).contiguous();
}
}

#ifdef TYLOINFER_NATIVE_CUDA
::tyloi::Tensor embed(const Model& model, Executor& executor, Collectives& copies, int primary,
             const ::tyloi::Tensor& ids_handle, const Executor::Mode& mode) {
    auto ids = ids_handle.dense();
#else
DenseTensor embed(const Model& model, Executor& executor, Collectives& copies, int primary,
             DenseTensor ids, const Executor::Mode& mode) {
#endif
    auto normalized = token_ids(copies, primary, std::move(ids));
#ifdef TYLOINFER_NATIVE_CUDA
    auto scope = executor.enter(primary, mode.ffn);
    return ::tyloi::ops::measure("model.embed", normalized.device(), [&] {
        return ::tyloi::ops::embedding(normalized, model.embed);
    });
#else
    return executor.profiler().measure("model.embed", [&] {
        return ::tyloi::cuda::quant_embedding_lookup(model.embed, std::move(normalized));
    });
#endif
}

#ifdef TYLOINFER_NATIVE_CUDA
::tyloi::Tensor normalize(const Model& model, Executor& executor, const ::tyloi::Tensor& hidden_handle, int64_t batch, int64_t tokens) {
    auto hidden = hidden_handle.dense();
#else
DenseTensor normalize(const Model& model, Executor& executor, DenseTensor hidden, int64_t batch, int64_t tokens) {
#endif
    hidden = hidden.reshape({batch, tokens, model.metadata.hidden_size});
#ifdef TYLOINFER_NATIVE_CUDA
    auto scope = executor.enter(hidden.device().index, Ffn::Mode{});
    return ::tyloi::ops::measure("model.output_norm", hidden.device(), [&] {
        return ::tyloi::ops::rms_norm(hidden.to(::tyloi::kFloat32), model.output_norm,
            model.metadata.rms_norm_eps, 1.0);
    });
#else
    return executor.profiler().measure("model.output_norm", [&] {
        return ::tyloi::cuda::qwen_rms_norm(
            hidden.reshape({-1, hidden.size(-1)}).to(::tyloi::kFloat32),
            model.output_norm, model.metadata.rms_norm_eps, 1.0).reshape(hidden.sizes());
    });
#endif
}

#ifdef TYLOINFER_NATIVE_CUDA
::tyloi::Tensor logits(const Model& model, Executor& executor, const ::tyloi::Tensor& hidden_handle, bool,
              const Executor::Mode& mode) {
    auto hidden = hidden_handle.dense();
#else
DenseTensor logits(const Model& model, Executor& executor, DenseTensor hidden, bool,
              const Executor::Mode& mode) {
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    auto scope = executor.enter(hidden.device().index, mode.ffn);
    return ::tyloi::ops::measure("model.lm_head", hidden.device(), [&] {
        return ::tyloi::ops::project(hidden, model.lm_head);
    });
#else
    return ::tyloi::cuda::logits(model.lm_head, std::move(hidden),
        executor.projector(mode.ffn), executor.profiler(), 0.0);
#endif
}

#ifdef TYLOINFER_NATIVE_CUDA
::tyloi::Tensor forward(const Model& model, State& state, Executor& executor, Collectives& copies,
               const Executor::Mode& mode, tyloi::models::Input input) {
#else
DenseTensor forward(const Model& model, State& state, Executor& executor, Collectives& copies,
               const Executor::Mode& mode, tyloi::models::Input input) {
#endif
    auto ids = token_ids(copies, mode.primary, std::move(input.ids).dense());
    const auto batch = ids.size(0), tokens = ids.size(1);
    TYLOINFER_RUNTIME_CHECK(input.confirmed == 0, "MiniCPM TTS has no speculative predictor");
    TYLOINFER_RUNTIME_CHECK(state.layers.size() == model.layers.size(), "MiniCPM TTS layer state mismatch");
    TYLOINFER_RUNTIME_CHECK(state.cache_pos <= model.metadata.max_position_embeddings - tokens,
        "MiniCPM TTS history exceeds context capacity");
    if (state.cache_pos == 0) state.reset(batch);
    TYLOINFER_RUNTIME_CHECK(state.batch == batch, "MiniCPM TTS cache batch does not match inputs");
#ifdef TYLOINFER_NATIVE_CUDA
    auto hidden = input.embeddings ? input.embeddings.dense()
                                  : embed(model, executor, copies, mode.primary, ids, mode).dense();
#else
    auto hidden = input.embeddings ? *input.embeddings
                                  : embed(model, executor, copies, mode.primary, ids, mode);
#endif
    TYLOINFER_RUNTIME_CHECK(hidden.dim() == 3 && hidden.size(0) == batch && hidden.size(1) == tokens &&
        hidden.size(2) == model.metadata.hidden_size, "inputs_embeds must match [batch,tokens,hidden_size]");
    auto physical = input.cache_positions
        ? copies.copy(DenseTensor(*input.cache_positions).to(::tyloi::kInt64), mode.primary).contiguous()
        : ::tyloi::cuda::arange(state.cache_pos, state.cache_pos + tokens,
            ::tyloi::TensorOptions().dtype(::tyloi::kInt64).device(::tyloi::Device(::tyloi::kCUDA, mode.primary)));
    TYLOINFER_RUNTIME_CHECK((physical.dim() == 1 && physical.numel() == tokens) ||
        (physical.dim() == 2 && physical.size(0) == batch && physical.size(1) == tokens),
        "cache_positions must have shape [tokens] or [batch,tokens]");
    auto positions = input.positions
        ? copies.copy(DenseTensor(*input.positions).to(::tyloi::kInt64), mode.primary).contiguous()
        : (state.decode_position_delta ? physical + state.decode_position_delta : physical);
    TYLOINFER_RUNTIME_CHECK((positions.dim() == 1 && positions.numel() == tokens) ||
        (positions.dim() == 2 && positions.size(0) == batch && positions.size(1) == tokens),
        "position_ids must have shape [tokens] or [batch,tokens]");
    if (input.lengths) {
        TYLOINFER_RUNTIME_CHECK(input.lengths->dim() == 1 && input.lengths->size(0) == batch,
            "sequence lengths must have shape [batch]");
        input.lengths = copies.copy(DenseTensor(*input.lengths).to(::tyloi::kInt64), mode.primary).contiguous();
    } else if (state.cache_pos > 0 && tokens == 1) {
        input.lengths = ::tyloi::cuda::full({batch}, state.cache_pos + 1, physical.options());
    }
    hidden = copies.copy(std::move(hidden), mode.primary).contiguous();
    const auto trace = [&] {
        if (input.trace) input.trace->push_back(copies.copy(hidden, mode.primary).to(::tyloi::kFloat32).contiguous().clone());
    };
    trace();
    for (size_t index = 0; index < model.layers.size(); ++index) {
        const auto& layer = model.layers[index];
        const auto& attention = layer.attention;
        MfqCudaGuard guard(layer.device);
        const auto local = [&](DenseTensor value) {
            return layer.cpu ? value.to(::tyloi::kCPU).contiguous() : copies.copy(value, layer.device).contiguous();
        };
        hidden = local(std::move(hidden));
        AttentionStep step{local(positions), {}, {}, state.cache_pos, input.kv_length, input.attention_parts};
        if (input.lengths) step.sequence_lengths = local(DenseTensor(*input.lengths));
        if (input.cache_positions) step.cache_positions = local(physical);
        auto& cache = state.layers[index];
        if (!cache.defined()) {
            const auto capacity = mode.kv_capacity > 0
                ? std::max(state.cache_pos + tokens, mode.kv_capacity) : attention.max_position_embeddings;
            cache = tyloi::state::allocate_kv(
                batch, attention.kv_heads, capacity, attention.attention_head_dim,
                false, hidden.device(), ::tyloi::kFloat16);
        }
        const auto* rope = layer.cpu ? &model.cpu_rope : &model.rope;
        if (!layer.cpu) {
            const auto found = model.device_ropes.find(layer.device);
            if (found != model.device_ropes.end()) rope = &found->second;
        }
        auto* layer_trace = mode.trace_layer == attention.layer ? mode.trace : nullptr;
#ifdef TYLOINFER_NATIVE_CUDA
        auto scope = executor.enter(layer.device, mode.ffn);
        auto output = attention.forward(::tyloi::Tensor(std::move(hidden)), step, cache, *rope, layer_trace);
        hidden = executor.full_ffn(layer.ffn, layer.ffn_norm, attention,
            output[0].dense(), output[1].dense(), mode.ffn, layer_trace);
#else
        auto output = attention.forward(executor.projector(mode.ffn), executor.profiler(),
            executor.attention_options(), std::move(hidden), step, cache,
            executor.attention_scratch(2 * index), *rope, layer_trace);
        hidden = executor.full_ffn(layer.ffn, layer.ffn_norm, attention,
            std::move(output[0]), std::move(output[1]), mode.ffn, layer_trace);
#endif
        trace();
    }
    if (!input.positions || input.advance) state.cache_pos += tokens;
    hidden = copies.copy(std::move(hidden), mode.primary).contiguous();
    if (input.raw_hidden) *input.raw_hidden = hidden;
    return normalize(model, executor, std::move(hidden), batch, tokens);
}

} // namespace tyloi::cuda::minicpmo45
