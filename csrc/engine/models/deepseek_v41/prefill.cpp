#include "engine/weights/cache.h"
#include "engine/backends/cuda/executor/execution.h"
#include "env.h"
#include "ops.h"
#include <cstring>

namespace tyloi::cuda::deepseek_v41 {
namespace {
using DenseTensor = ::tyloi::DenseTensor;
using HashBatch = tyloi::models::deepseek_v41::EngramHashBatch;

DenseTensor append_tail(const DenseTensor& previous, const DenseTensor& current, int64_t window) {
    if (current.size(1) >= window)
        return current.narrow(1, current.size(1) - window, window).clone();
    auto combined = previous.defined() ? ::tyloi::cuda::cat({previous, current}, 1) : current;
    const auto count = std::min(window, combined.size(1));
    return combined.narrow(1, combined.size(1) - count, count).clone();
}

HashBatch append_hash_tail(const HashBatch& previous, const HashBatch& current, int window) {
    HashBatch result;
    result.batch = current.batch;
    result.layers = current.layers;
    result.columns = current.columns;
    result.tokens = std::min(window, previous.tokens + current.tokens);
    result.values.resize(std::size_t(result.layers) * result.batch * result.tokens * result.columns);
    const int start = previous.tokens + current.tokens - result.tokens;
    for (int layer = 0; layer < result.layers; ++layer) for (int batch = 0; batch < result.batch; ++batch) {
        auto* output = result.values.data() +
            ((std::size_t(layer) * result.batch + batch) * result.tokens) * result.columns;
        for (int token = 0; token < result.tokens; ++token) {
            const int position = start + token;
            const auto* input = position < previous.tokens
                ? previous.layer_data(layer) + (std::size_t(batch) * previous.tokens + position) * result.columns
                : current.layer_data(layer) + (std::size_t(batch) * current.tokens + position - previous.tokens) * result.columns;
            std::memcpy(output + std::size_t(token) * result.columns, input,
                        std::size_t(result.columns) * sizeof(std::int64_t));
        }
    }
    return result;
}
}

std::int64_t prefill_span(const Model&, const State&, std::size_t tokens,
                          std::int64_t chunk, std::size_t, std::size_t,
                          bool transformed, bool retained) {
    auto logical_chunk=chunk;
    // Retained small-M continuations must bound decoder replay as well as
    // encoder tiles; merging the suffix would re-enter the large PP pipeline.
    if(!retained && !transformed)logical_chunk=tokens;
    return logical_chunk;
}

::tyloi::Tensor deepseek_v41_prefill_hidden(Model& model, State& state, const ::tyloi::Tensor& input_ids,
                                   int64_t compute_chunk, const std::function<bool()>& stopped, ::tyloi::Tensor* raw_hidden) {
    auto scope = model.execution_scope();
    auto ids = model.device_ids(input_ids.dense());
    const auto batch = ids.size(0), tokens = ids.size(1);
    const auto* profile=tyloi::cuda::env_lookup("TYLOINFER_PROFILE_SSD_EXPERTS");
    const bool trace=profile && profile[0]=='1';
    if(trace)std::cerr<<"runtime_deepseek_v41_prefill phase=prepare tokens="<<tokens<<'\n';
    TYLOINFER_RUNTIME_CHECK(state.shared && batch > 0 && tokens > 0 && compute_chunk > 0 &&
                          state.speculative_start < 0 &&
                          state.cache_pos + tokens <= model.max_position_embeddings(),
                      "invalid DeepSeek-V4.1 encoder prefill request");
    if (state.batch != batch || state.cache_pos == 0) state.reset(batch);
    const auto begin = state.cache_pos, end = begin + tokens;
    const auto& config = model.config;
    const auto split = config.causal_encoder_layers;
    TYLOINFER_RUNTIME_CHECK(split > 0 && split < model.num_hidden_layers(), "invalid CED encoder boundary");
    compute_chunk = std::min<int64_t>(compute_chunk, 8192);
    std::optional<tyloi::weights::Cache::Priority> priority;
    if(tokens > config.sliding_window)priority.emplace(model.execution->weights,0,static_cast<int>(split));
    DenseTensor tail_hidden, tail_pre, tail_ids;
    HashBatch tail_hashes;
    const auto cancelled = [&] {
        if (!stopped || !stopped()) return false;
        state.reset(batch);
        return true;
    };
    const auto* profile_layer=tyloi::cuda::env_lookup("TYLOINFER_PROFILE_DSV41_LAYER");
    const auto forward_layer=[&](int64_t layer,DenseTensor hidden,DenseTensor positions,
                                 DenseTensor host_ids) {
        auto& profiler=model.execution->executor().profiler();
        const bool selected=profile_layer && layer==std::atoi(profile_layer);
        const bool previous=profiler.enabled;
        if(selected) {profiler.reset();profiler.enabled=true;}
        try {
            auto& block=*model.blocks.at(layer);
            MfqCudaGuard guard(block.cuda_device);
            auto local_ids = block.cpu_offloaded
                ? host_ids
                : model.execution->executor().collectives().copy(ids, block.cuda_device);
            auto local_pos = block.cpu_offloaded
                ? positions
                : model.execution->executor().collectives().copy(positions, block.cuda_device);
            block.set_token_ids(*state.shared, local_ids);
            BlockContext context;
            context.token_ids=local_ids;
            context.host_token_ids=host_ids;
            context.positions=local_pos;
            context.full_positions=local_pos;
            context.cache_position=state.cache_pos;
            hidden = block.cpu_offloaded
                ? hidden.to(::tyloi::kCPU).contiguous()
                : model.execution->executor().collectives().copy(hidden, block.cuda_device);
            hidden = block.forward_context(*state.shared, *model.execution, std::move(hidden), context);
            if(selected)profiler.report("deepseek_v41.layer."+std::to_string(layer));
        } catch(...) {profiler.enabled=previous;throw;}
        profiler.enabled=previous;
        return hidden;
    };
    for (int64_t offset = 0; offset < tokens; offset += compute_chunk) {
        if (cancelled()) return {};
        const auto count = std::min(compute_chunk, tokens - offset);
        auto chunk_ids = ids.narrow(1, offset, count).contiguous();
        auto positions = ::tyloi::cuda::arange(state.cache_pos, state.cache_pos + count,
                                               ::tyloi::TensorOptions()
                                                   .device(::tyloi::kCUDA)
                                                   .dtype(::tyloi::kInt64));
        DenseTensor host_ids;
        if (model.execution->config.cache.pipeline || model.execution->config.dense_cpu_layer_count > 0)
            host_ids = chunk_ids.to(::tyloi::kCPU, ::tyloi::kInt64).contiguous();
        auto hidden = model.prepare_hidden(model.embed_tokens(chunk_ids), batch, count);
        state.begin_forward(raw_hidden != nullptr);
        for (int64_t layer = 0; layer < split; ++layer) {
            if (cancelled()) return {};
            if(trace)std::cerr<<"runtime_deepseek_v41_prefill phase=encoder layer="<<layer<<" tokens="<<count<<'\n';
            hidden = forward_layer(layer,std::move(hidden),positions,host_ids);
        }
        // Decoder global KV comes from the encoder boundary for every prompt token.
        // Decoder attention and FFNs run only over the bounded local replay below.
        for (int64_t layer = split; layer < model.num_hidden_layers(); ++layer) {
            auto& block = *model.blocks.at(layer);
            if (!block.kv_source()) continue;
            auto source = deepseek_v41_hc_input_cuda(hidden, state.shared->previous_pre,
#ifdef TYLOINFER_NATIVE_CUDA
                                                     dynamic_cast<const DenseTensor&>(*block.attention_norm), config.rms_eps);
#else
                                                     block.attention_norm, config.rms_eps);
#endif
            tyloi::ops::measure("deepseek_v41.ced.global_kv", [&] {
                block.publish_compressed_source(*state.shared, source, state.cache_pos);
                return source;
            });
            state.shared->layers.at(static_cast<std::size_t>(layer)).prefilled_global_end = state.cache_pos + count;
        }
        tail_hidden = append_tail(tail_hidden, hidden, config.sliding_window);
        tail_pre = append_tail(tail_pre, state.shared->previous_pre, config.sliding_window);
        tail_ids = append_tail(tail_ids, chunk_ids, config.sliding_window);
        tail_hashes = append_hash_tail(tail_hashes, state.shared->engram_hashes,
                                       static_cast<int>(config.sliding_window));
        state.cache_pos += count;
    }
    if(priority)priority->finish();
    const auto replay_tokens = tail_ids.size(1), replay_begin = end - replay_tokens;
    const bool truncated_local = replay_begin > begin;
    state.cache_pos = replay_begin;
    state.shared->batch = batch;
    state.shared->tokens = replay_tokens;
    state.shared->cache_position = replay_begin;
    state.shared->previous_pre = std::move(tail_pre);
    state.shared->engram_hashes = std::move(tail_hashes);
    auto replay_positions = ::tyloi::cuda::arange(replay_begin, replay_begin + replay_tokens,
                                                  ::tyloi::TensorOptions()
                                                      .device(::tyloi::kCUDA)
                                                      .dtype(::tyloi::kInt64));
    DenseTensor replay_host_ids;
    if (model.execution->config.cache.pipeline || model.execution->config.dense_cpu_layer_count > 0)
        replay_host_ids = tail_ids.to(::tyloi::kCPU, ::tyloi::kInt64).contiguous();
    for (int64_t layer = split; layer < model.num_hidden_layers(); ++layer) {
        if (cancelled()) return {};
        if(trace)std::cerr<<"runtime_deepseek_v41_prefill phase=decoder layer="<<layer<<" tokens="<<replay_tokens<<'\n';
        auto& block = *model.blocks.at(layer);
        state.shared->layers.at(static_cast<std::size_t>(layer)).position = replay_begin;
        state.shared->layers.at(static_cast<std::size_t>(layer)).replay_local_window = truncated_local;
        tail_hidden = forward_layer(layer,std::move(tail_hidden),replay_positions,replay_host_ids);
    }
    state.cache_pos = end;
    std::cerr << "runtime_deepseek_v41_ced_prefill tokens=" << tokens << " encoder_layers=" << split
              << " decoder_tokens=" << replay_tokens << " compute_chunk=" << compute_chunk << '\n';
    auto finalized = model.finalize_hidden(state,
        model.execution->executor().collectives().copy(tail_hidden,
            model.execution->config.layer_placement.primary_device()),
        batch, replay_tokens);
    if (raw_hidden)
        *raw_hidden = state.shared->dspark_target_hidden();
    return finalized;
}
} // namespace tyloi::cuda::deepseek_v41
