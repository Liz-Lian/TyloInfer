#include "env.h"
#include "engine/scheduler/table.h"
#include "runtime_checks.h"
#include <map>

#include "engine/load.h"
#include "engine/config.h"
#include "engine/generation/cuda.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/models/session/cache.h"
#include "engine/models/qwen35/batch.h"
#include "engine/models/qwen4_exp/batch.h"

#include <cuda_profiler_api.h>

#include <chrono>
#include <cmath>
#include <atomic>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

namespace tyloi::cuda::diagnostics {

using namespace tyloi::cuda::internal;

template<class Model, class State>
int run_qwen_continuous_batching_check(Model& model,
        State& state,
        tyloi::cuda::Executor* executor,
        CudaExecutionContext& execution_context,
        const tyloi::models::CausalLmMetadata& metadata,
        const CudaRuntimeConfig& runtime_config, int requests = 2,
        std::shared_ptr<const tyloi::cuda::qwen35::Model> model_owner = {}) {
    using namespace tyloi::engine;
    auto config = runtime_config;
    config.generation.prefill_chunk_size = 64;
    TYLOINFER_RUNTIME_CHECK(requests >= 2 && requests <= 32,"batching check supports 2..32 requests");
    config.continuous_batch.max_sequences = std::max(4,requests);
    MfqSamplingParams sampling;
    sampling.max_tokens = 20; sampling.temperature = 0; sampling.top_k = 1; sampling.enable_mtp = false;
    std::vector<int64_t> first_prompt(193), second_prompt(17);
    for (size_t i = 0; i < first_prompt.size(); ++i) first_prompt[i] = (101 + (i * 37) % 900) % diagnostic_vocab_size(model);
    for (size_t i = 0; i < second_prompt.size(); ++i) second_prompt[i] = (113 + (i * 53) % 880) % diagnostic_vocab_size(model);
    DecodeGraphCache graph(diagnostic_max_position_embeddings(model));
    TextSessionCache cache(config.session_cache, config.prefix_cache);
    tyloi::engine::mtp::State mtp_state;
    auto serial = [&](const std::vector<int64_t>& prompt, MfqPromptCachePlan plan = {}) {
        InferenceRequest request; request.prompt = prompt; request.sampling = sampling; request.cache_plan = std::move(plan);
        InferenceOutput output(request, nullptr, "serial");
        return collect_generation(generate(
            model, state, executor, execution_context.collectives,
            execution_context.forward_mode(), execution_context.profiler,
            execution_context.weights, execution_context.decode_graph_serial_branches,
            execution_context.config.report_cuda_memory,
            execution_context.dsv4_cpu_offload_layers.empty() &&
                execution_context.dense_cpu_layer_count == 0 &&
                model_parallel_cuda_graph_enabled(execution_context),
            [owner = &execution_context] { owner->drain(); },
            metadata, graph, cache, config, request, output, mtp_state)).tokens;
    };
    const auto first_reference = serial(first_prompt), second_reference = serial(second_prompt);
    std::vector<int64_t> growth_prompt(253);
    for (std::size_t i = 0; i < growth_prompt.size(); ++i)
        growth_prompt[i] = (137 + i*41) % diagnostic_vocab_size(model);
    const auto growth_reference = requests > 2 ? serial(growth_prompt) : std::vector<int64_t>{};
    graph.invalidate(); // Physical batch slots replace the serial graph's storage.
    {
        struct Execution {
            Model& model;
            const tyloi::models::CausalLmMetadata& metadata;
            DecodeGraphCache& graph;
            TextSessionCache& cache;
            const CudaRuntimeConfig& config;
            State& state;
            tyloi::engine::mtp::State mtp_state;
            tyloi::cuda::Executor* executor;
            CudaExecutionContext& execution_context;
            std::shared_ptr<const tyloi::cuda::qwen35::Model> model_owner;
            std::unique_ptr<tyloi::engine::Batch> batcher;
            Execution(Model& model, const tyloi::models::CausalLmMetadata& metadata,
                      DecodeGraphCache& graph, TextSessionCache& cache,
                      const CudaRuntimeConfig& config, State& state,
                      tyloi::cuda::Executor* executor,
                      CudaExecutionContext& execution_context,
                      std::shared_ptr<const tyloi::cuda::qwen35::Model> model_owner)
                : model(model), metadata(metadata), graph(graph), cache(cache),
                  config(config), state(state), executor(executor),
                  execution_context(execution_context), model_owner(std::move(model_owner)),
                  batcher(make_batcher(model, execution_context,
                                       config.continuous_batch,
                                       this->model_owner)) {}
            static std::unique_ptr<tyloi::engine::Batch> make_batcher(
                    Model& model, CudaExecutionContext& execution,
                    const CudaContinuousBatchConfig& config,
                    const std::shared_ptr<const tyloi::cuda::qwen35::Model>& owner) {
                if constexpr (std::is_same_v<Model, tyloi::cuda::qwen35::Model>) {
                    auto batch_config = config;
                    batch_config.cuda_graph &= model_parallel_cuda_graph_enabled(execution);
                    return make_batch(
                        owner, nullptr, execution.make_executor(), execution.collectives,
                        execution.forward_mode(), execution.weights.moe,
                        std::move(batch_config), 64,
                        ::tyloi::cuda::options::kv_vram_preallocated(),
                        execution.dense_cpu_layer_count != 0 ||
                            !execution.dsv4_cpu_offload_layers.empty(),
                        execution.config.weights.min_slots);
                } else if constexpr (std::is_same_v<Model, Qwen4CausalLm>) {
                    return make_batch(model, execution, config, 64);
                }
            }
            bool can_batch(const EngineRequest&) const { return true; }
            bool exclusive() const { return false; }
            bool mtp_available() const { return false; }
            void execute(const std::vector<RequestId>& eligible) { batcher->step(eligible); }
            Generation generate(const RequestId& id, ExecutionRequest& request) {
                return internal::generate<Model, State, MtpModule>(
                    model, state, executor, execution_context.collectives,
                    execution_context.forward_mode(), execution_context.profiler,
                    execution_context.weights,
                    execution_context.decode_graph_serial_branches,
                    execution_context.config.report_cuda_memory,
                    execution_context.dsv4_cpu_offload_layers.empty() &&
                        execution_context.dense_cpu_layer_count == 0 &&
                        model_parallel_cuda_graph_enabled(execution_context),
                    [owner = &execution_context] { owner->drain(); },
                    metadata, graph, cache, config, request.input, request.output,
                    mtp_state, nullptr, {}, batcher.get(), id);
            }
        } execution(model, metadata, graph, cache, config, state, executor,
                    execution_context, std::move(model_owner));
        auto& batcher = execution.batcher;
        RequestExecutor executor(config.continuous_batch.max_sequences);
        EngineInfo info;
        info.max_requests = config.continuous_batch.max_sequences;
        info.vocab_size = diagnostic_vocab_size(model);
        info.max_context = diagnostic_max_position_embeddings(model);
        std::map<std::string, std::vector<int64_t>> tokens;
        std::map<std::string, int> terminals, cancellations, prefills;
        const auto admit = [&](const std::string& id, const std::vector<int64_t>& prompt) {
            EngineRequest request;
            request.id = id; request.token_ids = prompt; request.input.sampling = sampling;
            TYLOINFER_RUNTIME_CHECK(executor.admit(std::move(request), nullptr, info, execution) == Admission::accepted,
                              "diagnostic request was not admitted");
        };
        const auto tick = [&](std::vector<RequestId> eligible) {
            auto result = executor.step(eligible, execution);
            for (const auto& event : result.events) {
                if (const auto* error = std::get_if<Failed>(&event.data)) throw std::runtime_error(error->message);
                if (const auto* delta = std::get_if<OutputDelta>(&event.data))
                    tokens[event.id].insert(tokens[event.id].end(), delta->token_ids.begin(), delta->token_ids.end());
                prefills[event.id] += std::holds_alternative<PrefillProgress>(event.data);
                cancellations[event.id] += std::holds_alternative<Cancelled>(event.data);
                terminals[event.id] += terminal(event.data);
            }
        };
        admit("first", first_prompt); admit("second", second_prompt);
        int paused_ticks = 0;
        for (int ticks = 0; ticks < 500 && !executor.empty(); ++ticks) {
            const bool pause = !tokens["first"].empty() && paused_ticks < 4 && !terminals["second"];
            const auto prior = tokens["first"].size();
            tick(pause ? std::vector<std::string>{"second"} : std::vector<std::string>{"first", "second"});
            if (pause) { ++paused_ticks; TYLOINFER_RUNTIME_CHECK(tokens["first"].size() == prior, "paused row advanced"); }
        }
        for (const auto& [id,reference] : std::vector<std::pair<std::string,std::vector<int64_t>>>{
                {"first",first_reference},{"second",second_reference}})
            if (tokens[id] != reference) {
                std::size_t index = 0;
                while (index < std::min(tokens[id].size(),reference.size()) && tokens[id][index] == reference[index]) ++index;
                std::cerr << "continuous_batching_mismatch request=" << id << " index=" << index
                    << " actual_count=" << tokens[id].size() << " reference_count=" << reference.size();
                if (index < tokens[id].size()) std::cerr << " actual=" << tokens[id][index];
                if (index < reference.size()) std::cerr << " reference=" << reference[index];
                std::cerr << '\n';
            }
        TYLOINFER_RUNTIME_CHECK(executor.empty() && tokens["first"] == first_reference &&
            tokens["second"] == second_reference && paused_ticks == 4 &&
            terminals["first"] == 1 && terminals["second"] == 1,
            "batched or resumed output differs from serial oracle");
        admit("cancel", second_prompt);
        for (int ticks = 0; ticks < 100 && tokens["cancel"].empty(); ++ticks) tick({"cancel"});
        TYLOINFER_RUNTIME_CHECK(!tokens["cancel"].empty(), "cancel test never decoded");
        const auto before_cancel = tokens["cancel"].size();
        executor.cancel("cancel"); tick({}); tick({});
        TYLOINFER_RUNTIME_CHECK(executor.empty() && cancellations["cancel"] == 1 && terminals["cancel"] == 1 &&
            tokens["cancel"].size() == before_cancel, "cancelled request published more output");
        // Cancel a partially prefetched request while another row owns a slot.
        admit("survivor", second_prompt);
        for (int ticks = 0; ticks < 100 && tokens["survivor"].empty(); ++ticks) tick({"survivor"});
        admit("partial", first_prompt);
        for (int ticks = 0; ticks < 100 && !prefills["partial"]; ++ticks) tick({"partial"});
        TYLOINFER_RUNTIME_CHECK(prefills["partial"] == 1 && tokens["partial"].empty(), "partial prefill did not yield");
        executor.cancel("partial");
        for (int ticks = 0; ticks < 500 && !executor.empty(); ++ticks) tick({"survivor"});
        TYLOINFER_RUNTIME_CHECK(executor.empty() && cancellations["partial"] == 1 && terminals["partial"] == 1 &&
            tokens["partial"].empty() && terminals["survivor"] == 1 && tokens["survivor"] == second_reference,
            "prefill cancellation corrupted a live slot");
        if (requests > 2) {
            std::vector<RequestId> ids;
            for (int row = 0; row < requests; ++row) {
                ids.push_back("parallel-" + std::to_string(row));
                admit(ids.back(),row%2 ? second_prompt : growth_prompt);
            }
            // Hold completed prefills until every row has joined the batch.
            for (const auto& id : ids) {
                for (int ticks = 0; ticks < 100 && tokens[id].empty(); ++ticks) tick({id});
                TYLOINFER_RUNTIME_CHECK(tokens[id].size() == 1,"stress request never completed prefill");
            }
            for (int ticks = 0; ticks < 500 && !executor.empty(); ++ticks) tick(ids);
            TYLOINFER_RUNTIME_CHECK(executor.empty(),"parallel context growth did not finish");
            for (int row = 0; row < requests; ++row)
                TYLOINFER_RUNTIME_CHECK(tokens[ids[row]] == (row%2 ? second_reference : growth_reference) &&
                    terminals[ids[row]] == 1,"parallel context growth differs from serial oracle");
            std::cout << "continuous_batching_growth_check requests=" << requests << " boundary=256 PASS\n";
        }
        double captures = 0, replays = 0, packed_prefills = 0;
        if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
            std::vector<RequestId> ids;
            for(int row=0;row<requests;++row) {
                ids.push_back("packed"+std::to_string(row));
                admit(ids.back(),row%2?second_prompt:first_prompt);
            }
            for(int ticks=0;ticks<500 && !executor.empty();++ticks)tick(ids);
            TYLOINFER_RUNTIME_CHECK(executor.empty(),"packed prefill requests did not finish");
            for(int row=0;row<requests;++row)
                TYLOINFER_RUNTIME_CHECK(tokens[ids[row]]==(row%2?second_reference:first_reference) &&
                    terminals[ids[row]]==1,"packed prefill differs from serial oracle");
        }
        for (const auto& [key, value] : batcher->metrics()) {
            if (key == "paged_kv_live_pages") TYLOINFER_RUNTIME_CHECK(value == 0, "paged KV leaked");
            if (key == "continuous_batching_active" || key == "continuous_batching_prefilling")
                TYLOINFER_RUNTIME_CHECK(value == 0, "batch retained finished requests");
            if (key == "continuous_batching_cuda_graph_captures") captures = value;
            if (key == "continuous_batching_cuda_graph_replays") replays = value;
            if (key == "continuous_batching_packed_prefill_batches") packed_prefills = value;
        }
        if constexpr (std::is_same_v<Model,tyloi::cuda::qwen35::Model>)
            if (config.continuous_batch.greedy &&
                tyloi::cuda::qwen35::batch_graph_enabled(
                    model, execution_context, config.continuous_batch))
                TYLOINFER_RUNTIME_CHECK(captures > 0 && replays > captures, "batch graph was not replayed");
        std::cout << "continuous_batching_state_check paged=" << config.continuous_batch.paged_kv
            << " captures=" << captures << " replays=" << replays
            << " packed_prefills=" << packed_prefills
            << " prefill_cancel=1 slot_reuse=1\n";
    }
    MfqPromptCachePlan plan{"batch-check", first_prompt.size() - 1};
    cache.clear();
    const auto fresh_session = serial(first_prompt,plan), restored_session = serial(first_prompt,plan);
    // Prefix capture changes workspace/residency. Compare the restore against
    // a fresh request with the same cache plan and operator choices.
    TYLOINFER_RUNTIME_CHECK(restored_session == fresh_session,
                      "session restore differs from serial oracle");
    std::cout << "continuous_batching_check PASS requests=2 paused_ticks=4 cancellation=1 session_reuse=1\n";
    return 0;
}

int run_cuda_continuous_batching_check(
        tyloi::cuda::qwen35::Model& model,
        tyloi::cuda::qwen35::State& state,
        tyloi::cuda::Executor* executor,
        CudaExecutionContext& execution,
        int requests) {
    auto owner = std::make_shared<tyloi::cuda::qwen35::Model>(std::move(model));
    return run_qwen_continuous_batching_check(
        *owner, state, executor, execution, owner->metadata,
        tyloi::engine::resolve_config(), requests, owner);
}
int run_cuda_continuous_batching_check(tyloi::cuda::Qwen4CausalLm& model,int requests) {
    return run_qwen_continuous_batching_check(
        model, model, nullptr, *model.execution, model.metadata,
        tyloi::engine::resolve_config(),requests);
}


} // namespace tyloi::cuda::diagnostics
