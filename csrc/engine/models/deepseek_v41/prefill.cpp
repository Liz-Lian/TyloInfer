#include "env.h"
#include "engine/models/common/check.h"
#include "ops.h"
#include <cstring>

namespace tyloi::cuda::deepseek_v41 {
namespace {
using DenseTensor = ::tyloi::DenseTensor;
using HashBatch = tyloi::models::deepseek_v41::EngramHashBatch;

DenseTensor append_tail(const DenseTensor& previous, const DenseTensor& current, int64_t window) {
    if (current.size(1) >= window)
        return tyloi::ops::clone(current.narrow(1,current.size(1)-window,window)).dense();
    auto combined = previous.defined() ? tyloi::ops::cat(std::vector<tyloi::Tensor>{previous,current},1).dense() : current;
    const auto count=std::min(window,combined.size(1));
    return tyloi::ops::clone(combined.narrow(1,combined.size(1)-count,count)).dense();
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
DenseTensor forward_layer(const Model& model, State& state, const DenseTensor& ids, int64_t layer,
                          DenseTensor hidden, DenseTensor positions, DenseTensor host_ids) {
    const auto* profile_layer=tyloi::cuda::env_lookup("TYLOINFER_PROFILE_DSV41_LAYER");
    const bool selected=profile_layer && layer==std::atoi(profile_layer);
    tyloi::ops::ProfileCapture capture(selected);
        auto& block=*model.blocks.at(layer);
        auto local_ids = block.cpu_offloaded
            ? host_ids
            : tyloi::ops::copy(ids, ::tyloi::Device(::tyloi::kCUDA, block.cuda_device)).dense();
        auto local_pos = block.cpu_offloaded
            ? positions
            : tyloi::ops::copy(positions, ::tyloi::Device(::tyloi::kCUDA, block.cuda_device)).dense();
        block.set_token_ids(*state.shared, local_ids);
        hidden = block.cpu_offloaded
            ? hidden.to(::tyloi::kCPU).contiguous()
            : tyloi::ops::copy(hidden, ::tyloi::Device(::tyloi::kCUDA, block.cuda_device)).dense();
        hidden = block.forward(*state.shared,
                                       std::move(hidden), local_pos, state.cache_pos,
                                       mfq_nullopt, mfq_nullopt, 0);
        if(selected)capture.report("deepseek_v41.layer."+std::to_string(layer));
    return hidden;
}
}

tyloi::StepSequence<tyloi::models::Prefill> Model::prefill(
        tyloi::models::Input input, tyloi::models::State& state_value,
        const tyloi::models::ForwardOptions&, tyloi::models::PrefillOptions prefill) const {
    auto compute_chunk = prefill.compute;
    using tyloi::models::Prefill;
    auto* target = dynamic_cast<State*>(&state_value);
    tyloi::models::require_model(target && target->belongs_to(*this),
        "DeepSeek-V4.1 state belongs to a different model");
    if (input.embeddings || input.positions || input.cache_positions || input.lengths || input.image_mask)
        co_return;
    auto& state = *target;
    auto ids = device_ids(input.ids.dense());
    const auto batch = ids.size(0), tokens = ids.size(1);
    const auto* profile=tyloi::cuda::env_lookup("TYLOINFER_PROFILE_SSD_EXPERTS");
    const bool trace=profile && profile[0]=='1';
    if(trace)std::cerr<<"runtime_deepseek_v41_prefill phase=prepare tokens="<<tokens<<'\n';
    TYLOINFER_RUNTIME_CHECK(state.shared && batch > 0 && tokens > 0 && compute_chunk > 0 &&
        state.speculative_start < 0 && state.cache_pos + tokens <= max_position_embeddings(),
        "invalid DeepSeek-V4.1 encoder prefill request");
    const auto split = config.causal_encoder_layers;
    TYLOINFER_RUNTIME_CHECK(split > 0 && split < num_hidden_layers(), "invalid CED encoder boundary");
    co_yield Prefill{Prefill::Phase::encode, 0,
        tokens > config.sliding_window ? static_cast<int>(split) : 0};
    if (state.batch != batch || state.cache_pos == 0) state.reset(batch);
    state.set_image_mask({});
    const auto begin = state.cache_pos, end = begin + tokens;
    compute_chunk = std::min<int64_t>(compute_chunk, 8192);
    DenseTensor tail_hidden, tail_pre, tail_ids;
    HashBatch tail_hashes;
    std::vector<DenseTensor> encoder_targets;
    if (input.raw_hidden) encoder_targets.resize(config.dspark_target_layer_ids.size());
    for (int64_t offset = 0; offset < tokens; offset += compute_chunk) {
        const auto count = std::min(compute_chunk, tokens - offset);
        auto chunk_ids = ids.narrow(1, offset, count).contiguous();
        auto positions = ::tyloi::cuda::arange(state.cache_pos, state.cache_pos + count,
            ::tyloi::TensorOptions().device(::tyloi::Device(::tyloi::kCUDA, primary_device)).dtype(::tyloi::kInt64));
        DenseTensor host_ids;
        if (needs_host_tokens) host_ids = chunk_ids.to(::tyloi::kCPU, ::tyloi::kInt64).contiguous();
        auto hidden = prepare_hidden(embed_tokens(chunk_ids), batch, count);
        state.begin_forward(input.raw_hidden != nullptr);
        for (int64_t layer = 0; layer < split; ++layer) {
            if(trace)std::cerr<<"runtime_deepseek_v41_prefill phase=encoder layer="<<layer<<" tokens="<<count<<'\n';
            hidden = forward_layer(*this,state,chunk_ids,layer,std::move(hidden),positions,host_ids);
            co_yield tyloi::StepState::advanced;
        }
        if (input.raw_hidden) {
            for (std::size_t slot = 0; slot < encoder_targets.size(); ++slot) {
                const auto layer = config.dspark_target_layer_ids[slot];
                if (layer >= split) continue;
                encoder_targets[slot] = append_tail(encoder_targets[slot],
                    state.shared->dspark_target_hiddens[slot], config.sliding_window);
            }
        }
        for (int64_t layer = split; layer < num_hidden_layers(); ++layer) {
            auto& block = *blocks.at(layer);
            if (!block.kv_source()) continue;
            auto& source=state.shared->ced_source;
            source.dense() = tyloi::ops::mhc_input(hidden,state.shared->previous_pre,
                block.attention_norm,config.rms_eps).dense();
            tyloi::ops::measure("deepseek_v41.ced.global_kv",tyloi::Device(tyloi::kCUDA,block.cuda_device), [&] {
                block.attention->publish_compressed(state.shared->layers.at(static_cast<std::size_t>(layer)).attention,
                    state.shared->attention,source,state.cache_pos);
                return source.dense();
            });
            source.dense() = {};
            state.shared->layers.at(static_cast<std::size_t>(layer)).attention.prefilled_global_end = state.cache_pos + count;
            co_yield tyloi::StepState::advanced;
        }
        tail_hidden = append_tail(tail_hidden, hidden, config.sliding_window);
        tail_pre = append_tail(tail_pre, state.shared->previous_pre, config.sliding_window);
        tail_ids = append_tail(tail_ids, chunk_ids, config.sliding_window);
        tail_hashes = append_hash_tail(tail_hashes, state.shared->engram_hashes,
            static_cast<int>(config.sliding_window));
        state.cache_pos += count;
    }
    co_yield Prefill{Prefill::Phase::decode};
    const auto replay_tokens = tail_ids.size(1), replay_begin = end - replay_tokens;
    const bool truncated_local = replay_begin > begin;
    state.cache_pos = replay_begin;
    state.shared->batch = batch;
    state.shared->tokens = replay_tokens;
    state.shared->cache_position = replay_begin;
    state.shared->previous_pre = std::move(tail_pre);
    state.shared->engram_hashes = std::move(tail_hashes);
    if (input.raw_hidden) {
        for (std::size_t slot = 0; slot < encoder_targets.size(); ++slot)
            if (config.dspark_target_layer_ids[slot] < split)
                state.shared->dspark_target_hiddens[slot] = std::move(encoder_targets[slot]);
    }
    auto replay_positions = ::tyloi::cuda::arange(replay_begin, replay_begin + replay_tokens,
        ::tyloi::TensorOptions().device(::tyloi::Device(::tyloi::kCUDA, primary_device)).dtype(::tyloi::kInt64));
    DenseTensor replay_host_ids;
    if (needs_host_tokens) replay_host_ids = tail_ids.to(::tyloi::kCPU, ::tyloi::kInt64).contiguous();
    for (int64_t layer = split; layer < num_hidden_layers(); ++layer) {
        if(trace)std::cerr<<"runtime_deepseek_v41_prefill phase=decoder layer="<<layer<<" tokens="<<replay_tokens<<'\n';
        auto& layer_state = state.shared->layers.at(static_cast<std::size_t>(layer));
        layer_state.attention.position = replay_begin;
        layer_state.attention.replay_local_window = truncated_local;
        tail_hidden = forward_layer(*this,state,tail_ids,layer,std::move(tail_hidden),replay_positions,replay_host_ids);
        co_yield tyloi::StepState::advanced;
    }
    state.cache_pos = end;
    std::cerr << "runtime_deepseek_v41_ced_prefill tokens=" << tokens << " encoder_layers=" << split
              << " decoder_tokens=" << replay_tokens << " compute_chunk=" << compute_chunk << '\n';
    auto finalized = finalize_hidden(state,
        tyloi::ops::copy(tail_hidden, ::tyloi::Device(::tyloi::kCUDA, primary_device)).dense(),
        batch, replay_tokens);
    ::tyloi::Tensor raw;
    if (input.raw_hidden) {
        raw = state.shared->dspark_target_hidden(primary_device);
        *input.raw_hidden = raw;
    }
    co_yield Prefill{Prefill::Phase::complete, 0, 0, std::move(finalized), std::move(raw)};
}
} // namespace tyloi::cuda::deepseek_v41
