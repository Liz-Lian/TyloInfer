#pragma once
#include "env.h"
#include "cuda_activity_trace.h"
#include "engine/requests/inference.h"

#include "engine/backends/cuda/kernels/mfq_cuda_sampling_ops.h"
#include "engine/executor/scope.h"
#include "backends/cuda/engine/executor/execution.h"
#include "backends/cuda/engine/executor/decode_graph.h"
#include "models/qwen35/forward.h"
#include "models/common/block.h"
#include "models/common/full_block.h"
#include "engine/weights/moe/expert_cache.h"
#include "models/qwen4_exp/ops.h"
#include "backend.h"
#include "source/model_source.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <type_traits>
#include <vector>

namespace mfq::cuda::internal {

template <typename Model>
inline constexpr bool is_qwen35_v =
    std::is_same_v<std::remove_cvref_t<Model>, mfq::cuda::qwen35::Model>;

template <typename Model>
const mfq::models::CausalLmMetadata* diagnostic_metadata(const Model& model) {
    return &model.metadata;
}

template <typename Model>
const mfq::ModelSource* diagnostic_source(const Model& model) {
    return model.source.get();
}

template <typename Model>
std::int64_t diagnostic_vocab_size(const Model& model) {
    if constexpr (is_qwen35_v<Model>) return model.metadata.vocab_size;
    else return model.vocab_size();
}

template <typename Model>
std::int64_t diagnostic_max_position_embeddings(const Model& model) {
    if constexpr (is_qwen35_v<Model>) return model.metadata.max_position_embeddings;
    else return model.max_position_embeddings();
}

template <typename Model, typename State>
void diagnostic_reset(Model& model, State& state, std::int64_t batch) {
    if constexpr (is_qwen35_v<Model>) state.reset(batch);
    else model.reset(batch);
}

template <typename Model, typename State>
mfq_tensor_backend::Tensor diagnostic_hidden(
        Model& model, State& state, mfq::cuda::Executor* executor,
        CudaExecutionContext& execution, mfq_tensor_backend::Tensor ids,
        std::optional<mfq_tensor_backend::Tensor> positions = {},
        std::optional<mfq_tensor_backend::Tensor> lengths = {},
        std::vector<mfq_tensor_backend::Tensor>* trace = nullptr,
        std::optional<mfq_tensor_backend::Tensor> cache_positions = {},
        mfq_tensor_backend::Tensor* raw_hidden = nullptr,
        std::int64_t confirmed = 0, std::int64_t kv_length = 0,
        std::int64_t attention_parts = 0) {
    if constexpr (is_qwen35_v<Model>) {
        TYLOINFER_RUNTIME_CHECK(executor != nullptr,
                                "Qwen3.5 diagnostics require an executor");
        CudaScope scope(execution.layer_placement.primary_device(), execution.contexts);
        mfq::cuda::Input input;
        input.ids = std::move(ids);
        input.positions = std::move(positions);
        input.lengths = std::move(lengths);
        input.cache_positions = std::move(cache_positions);
        input.trace = trace;
        input.raw_hidden = raw_hidden;
        input.confirmed = confirmed;
        input.kv_length = kv_length;
        input.attention_parts = attention_parts;
        return mfq::cuda::qwen35::forward(
            model, state, *executor, execution.collectives,
            execution.forward_mode(), std::move(input));
    } else {
        return model.hidden_forward(
            std::move(ids), std::move(positions), std::move(lengths), trace,
            std::move(cache_positions), raw_hidden, confirmed,
            {kv_length, attention_parts});
    }
}

inline mfq_tensor_backend::Tensor diagnostic_last_hidden(
        mfq_tensor_backend::Tensor hidden) {
    return hidden.index({Slice(), -1, Slice()})
        .to(mfq_tensor_backend::kFloat16).contiguous();
}

template <typename Model>
mfq_tensor_backend::Tensor diagnostic_logits(
        Model& model, mfq::cuda::Executor* executor,
        CudaExecutionContext& execution, mfq_tensor_backend::Tensor hidden,
        bool capped = false) {
    if constexpr (is_qwen35_v<Model>) {
        TYLOINFER_RUNTIME_CHECK(executor != nullptr,
                                "Qwen3.5 diagnostics require an executor");
        CudaScope scope(execution.layer_placement.primary_device(), execution.contexts);
        return mfq::cuda::qwen35::logits(
            model, *executor, std::move(hidden), capped);
    } else {
        auto logits = execution.projector().run(model.lm_head, std::move(hidden));
        return capped ? model.apply_final_logit_softcap(std::move(logits))
                      : std::move(logits);
    }
}

template <typename Model, typename State>
mfq_tensor_backend::Tensor diagnostic_next_token(
        Model& model, State& state, mfq::cuda::Executor* executor,
        CudaExecutionContext& execution, mfq_tensor_backend::Tensor ids) {
    if constexpr (is_qwen35_v<Model>) {
        auto hidden = diagnostic_hidden(model, state, executor, execution,
                                        std::move(ids));
        auto logits = diagnostic_logits(
            model, executor, execution,
            diagnostic_last_hidden(std::move(hidden)),
            false);
        return sample_greedy_cuda(
            logits.contiguous().view({logits.size(0), -1}));
    } else {
        return model.next_token(std::move(ids));
    }
}

template <typename Model, typename State>
mfq_tensor_backend::Tensor diagnostic_last_logits(
        Model& model, State& state, mfq::cuda::Executor* executor,
        CudaExecutionContext& execution, mfq_tensor_backend::Tensor ids) {
    if constexpr (is_qwen35_v<Model>) {
        auto hidden = diagnostic_hidden(model, state, executor, execution,
                                        std::move(ids));
        return diagnostic_logits(model, executor, execution,
                                 diagnostic_last_hidden(std::move(hidden)), true);
    } else {
        return model.last_logits(std::move(ids));
    }
}

template <typename Model, typename State>
mfq_tensor_backend::Tensor diagnostic_forward(
        Model& model, State& state, mfq::cuda::Executor* executor,
        CudaExecutionContext& execution, mfq_tensor_backend::Tensor ids) {
    if constexpr (is_qwen35_v<Model>) {
        return diagnostic_logits(
            model, executor, execution,
            diagnostic_hidden(model, state, executor, execution, std::move(ids)),
            true);
    } else {
        return model.forward(std::move(ids));
    }
}

template <typename Model, typename State>
mfq_tensor_backend::Tensor diagnostic_next_token_static(
        Model& model, State& state, mfq::cuda::Executor* executor,
        CudaExecutionContext& execution, mfq_tensor_backend::Tensor ids,
        mfq_tensor_backend::Tensor positions,
        mfq_tensor_backend::Tensor lengths, std::int64_t kv_length,
        std::int64_t attention_parts) {
    if constexpr (is_qwen35_v<Model>) {
        auto hidden = diagnostic_hidden(
            model, state, executor, execution, std::move(ids), positions,
            lengths, nullptr, positions, nullptr, 0, kv_length,
            attention_parts);
        auto logits = diagnostic_logits(
            model, executor, execution,
            diagnostic_last_hidden(std::move(hidden)),
            false);
        return sample_greedy_cuda(
            logits.contiguous().view({logits.size(0), -1}));
    } else {
        return model.next_token_static(
            std::move(ids), std::move(positions), std::move(lengths),
            {kv_length, attention_parts});
    }
}

inline void report_source_file_io(const mfq::ModelSource* source, const char* phase) noexcept {
    const char* report = mfq::cuda::env_lookup("TYLOINFER_REPORT_FILE_IO");
    if (!source || !report || report[0] != '1') return;
    const auto stats = source->file_read_stats();
    try {
        std::cout << "source_file_io phase=" << phase
                  << " mode=" << mfq::file_read_mode_name(stats.mode)
                  << " files=" << stats.files << " calls=" << stats.calls
                  << " logical_bytes=" << stats.logical_bytes << " physical_bytes=" << stats.physical_bytes
                  << " errors=" << stats.errors << " staging_bytes=" << stats.staging_bytes
                  << " staging_peak_sum_bytes=" << stats.staging_peak_bytes
                  << " staging_cached_bytes=" << stats.staging_cached_bytes << " staging_allocations=" << stats.staging_allocations
                  << " read_ms=" << double(stats.read_nanoseconds) / 1e6 << std::endl;
    } catch (const std::exception&) { } // Optional diagnostics cannot abort generation.
}

inline mfq_tensor_backend::Tensor diagnostic_decode_input(
        const mfq_tensor_backend::Tensor& predicted,
        const mfq_tensor_backend::Tensor& forced,int step) {
    return forced.defined()?forced.narrow(0,step,1).view({1,1}):predicted.view({1,1});
}

template <typename Model, typename State>
int generate_diagnostic_tokens(
        CudaExecutionContext& execution,
        Model& model,
        State& state,
        mfq::cuda::Executor* executor,
        mfq_tensor_backend::Tensor ids,
        int gen,
        bool profile,
        std::chrono::steady_clock::time_point t0,
        std::chrono::steady_clock::time_point t1,
        const std::vector<int64_t>& decode_inputs = {},
        const std::unordered_set<int64_t>& stop_token_ids = {},
        bool production_prefill = false) {
        const auto* metadata = diagnostic_metadata(model);
        const auto* source = diagnostic_source(model);
        if(!decode_inputs.empty() && (gen<=1 || decode_inputs.size()!=size_t(gen-1)))
            throw std::invalid_argument("diagnostic decode input count differs from gen-1");
        mfq_tensor_backend::Tensor forced_inputs;
        if(!decode_inputs.empty())
            forced_inputs=mfq_tensor_backend::tensor(decode_inputs).to(mfq_tensor_backend::kCUDA);
        std::cout<<"decode_input_mode="<<(forced_inputs.defined()?"forced":"generated")<<"\n";
        auto& profiler = execution.profiler;
        report_source_file_io(source, "prefill_begin");
        profiler.reset();
        auto next = [&] {
            if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
                if(production_prefill && ids.size(1)>128 && !qwen4_layer_major_prefill_eligible(model,ids.size(1)))
                    throw std::invalid_argument("router audit requires production prefill; enable TYLOINFER_MOE_PIPELINE and TYLOINFER_MOE_PREFILL_LAYER");
                if(production_prefill && qwen4_layer_major_prefill_eligible(model,ids.size(1))) {
                    std::cout<<"diagnostic_prefill path=production_layer_major tokens="<<ids.size(1)<<std::endl;
                    auto hidden=qwen4_layer_major_prefill_hidden(model,ids,1024,true);
                    return model.adapter_next_token(model.lm_head,model.last_hidden(std::move(hidden)));
                }
            }
            return diagnostic_next_token(model, state, executor, execution, ids);
        }();
        mfq_cuda_synchronize();
        auto t2 = std::chrono::steady_clock::now();
        report_cuda_memory(execution.config, "prefill");
        report_source_file_io(source, "prefill");
        const char * empty_cache_env = mfq::cuda::env_lookup("TYLOINFER_EMPTY_CACHE_BEFORE_GRAPH");
        if (empty_cache_env != nullptr && std::atoi(empty_cache_env) != 0) {
            mfq_cuda_empty_cache();
            report_cuda_memory(execution.config, "prefill_empty_cache");
        }
        profiler.report("prefill");
        profiler.reset();
        if (gen == 0) return 0;
        int produced = 1;
        bool hit_eos = !stop_token_ids.empty() && stop_token_ids.contains(next.template item<int64_t>());
        auto generated_cuda = mfq_tensor_backend::empty({gen}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCUDA));
        cudaStream_t stream = mfq_get_current_cuda_stream().stream();
        TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(generated_cuda.template data_ptr<int64_t>(), next.template data_ptr<int64_t>(),
                                   sizeof(int64_t), cudaMemcpyDeviceToDevice, stream));
        const char* graph_env = mfq::cuda::env_lookup("TYLOINFER_CUDA_GRAPH");
        const char * profile_graph_env = mfq::cuda::env_lookup("TYLOINFER_PROFILE_CUDA_GRAPH");
        const bool profile_cuda_graph = profile && profile_graph_env != nullptr &&
            std::atoi(profile_graph_env) != 0;
        bool use_cuda_graph =
            !forced_inputs.defined() &&
            !is_qwen35_v<Model> &&
            (graph_env == nullptr || graph_env[0] != '0') &&
            !metadata->flash_next &&
            mfq_cuda_graph_capture_supported() &&
            execution.dsv4_cpu_offload_layers.empty() &&
            execution.dense_cpu_layer_count == 0 &&
            !execution.weights.moe &&
            model_parallel_cuda_graph_enabled(execution) &&
            (!profile || profile_cuda_graph) && gen > 1;
        const char * cuda_profiler_env = mfq::cuda::env_lookup("TYLOINFER_CUDA_PROFILER_RANGE");
        const bool cuda_profiler_range = cuda_profiler_env != nullptr &&
            std::atoi(cuda_profiler_env) != 0;
        auto decode_replay_t0 = t2;
        auto first_decode_done=t2;
        bool measured_first_decode=false;
        auto halfway_done=t2;
        bool measured_halfway=false;
        auto last_window_done=t2;
        bool measured_last_window=false;
        if (cuda_profiler_range) TYLOINFER_CUDA_CHECK(cudaProfilerStart());
        if (use_cuda_graph && !hit_eos) {
            auto static_input = mfq_tensor_backend::empty({1, 1}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCUDA));
            auto static_pos = mfq_tensor_backend::empty({1}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCUDA));
            auto static_len = mfq_tensor_backend::empty({1}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCUDA));
            auto static_step = mfq_tensor_backend::empty({1}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCUDA));
            auto graph_stream = mfq_get_stream_from_pool(false);
            MfqCudaGuard graph_device_guard(
                graph_stream.device_index());
            const auto& parallel = execution.tensor_parallel.enabled()
                ? execution.tensor_parallel
                : execution.expert_parallel;
            auto graph_compute_streams =
                make_cuda_graph_compute_streams(graph_stream, parallel);
            auto graph_stream_guards =
                activate_cuda_graph_compute_streams(
                    graph_compute_streams);
            cudaStream_t graph_raw_stream = graph_stream.stream();

            int64_t pos_h = state.cache_pos;
            int64_t len_h = pos_h + 1;
            int64_t step_h = 1;
            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(static_input.template data_ptr<int64_t>(), next.template data_ptr<int64_t>(),
                                       sizeof(int64_t), cudaMemcpyDeviceToDevice, graph_raw_stream));
            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(static_pos.template data_ptr<int64_t>(), &pos_h,
                                       sizeof(int64_t), cudaMemcpyHostToDevice, graph_raw_stream));
            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(static_len.template data_ptr<int64_t>(), &len_h,
                                       sizeof(int64_t), cudaMemcpyHostToDevice, graph_raw_stream));
            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(static_step.template data_ptr<int64_t>(), &step_h,
                                       sizeof(int64_t), cudaMemcpyHostToDevice, graph_raw_stream));
            TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(graph_raw_stream));

            MfqCudaGraph graph;
            mfq_tensor_backend::Tensor static_next;
            const int64_t planned_len = state.cache_pos + gen;
            const int64_t attention_parts = decode_graph_attention_parts(
                planned_len, FullBlock::kDecodeAttentionMaxParts);
            {
                DecodeGraphBranchScope branch_scope(
                    execution.decode_graph_serial_branches);
                std::vector<mfq_tensor_backend::Tensor *> graph_warmup_states;
                if constexpr (is_qwen35_v<Model>) {
                    graph_warmup_states = state.graph_state();
                } else {
                    for (const auto &block : model.blocks)
                        for (auto *state : block->graph_warmup_state())
                            graph_warmup_states.push_back(state);
                }
                prepare_decode_graph_memory(
                    graph_warmup_states,
                    state.cache_pos,
                    metadata->decode_graph_double_warmup, graph, [&]() {
                        (void)diagnostic_next_token_static(
                            model, state, executor, execution, static_input,
                            static_pos, static_len, planned_len,
                            attention_parts);
                    }, cuda_graph_participant_streams(
                        graph_compute_streams,
                        execution.collectives));
                profiler.reset();
                profiler.graph_events = profile_cuda_graph;
                graph.capture_begin();
                static_next = profiler.measure("decode.model_total", [&]() {
                    return diagnostic_next_token_static(
                        model, state, executor, execution, static_input,
                        static_pos, static_len, planned_len, attention_parts);
                });
                profiler.measure("decode.commit", [&]() {
                    decode_graph_commit_cuda(
                        static_next, generated_cuda, static_step,
                        static_input, static_pos, static_len);
                    return 0;
                });
                graph.capture_end();
            }
            mfq_debug_dump_cuda_graph(graph);
            report_cuda_memory(execution.config, "graph_captured");

            decode_replay_t0 = std::chrono::steady_clock::now();
            for (int i = 1; i < gen; ++i) {
                graph.replay();
                produced = i + 1;
                if (!stop_token_ids.empty()) {
                    TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(graph_raw_stream));
                    if (stop_token_ids.contains(static_next.template item<int64_t>())) {
                        hit_eos = true;
                        break;
                    }
                }
            }
            TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(graph_raw_stream));
        } else if (!hit_eos) {
            for (int i = 1; i < gen; ++i) {
                next = profiler.measure("decode.eager_model", [&]() {
                    return diagnostic_next_token(
                        model, state, executor, execution,
                        diagnostic_decode_input(next,forced_inputs,i-1));
                });
                profiler.measure("decode.eager_commit", [&]() {
                    TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(
                        generated_cuda.template data_ptr<int64_t>() + i,
                        next.template data_ptr<int64_t>(), sizeof(int64_t),
                        cudaMemcpyDeviceToDevice, stream));
                    return 0;
                });
                produced = i + 1;
                if (!stop_token_ids.empty() && stop_token_ids.contains(next.template item<int64_t>())) {
                    hit_eos = true;
                    break;
                }
                if(i==1 && metadata->flash_next && stop_token_ids.empty()) {
                    mfq_cuda_synchronize();first_decode_done=std::chrono::steady_clock::now();
                    measured_first_decode=true;
                }
                if(gen>=4 && i==gen/2 && metadata->flash_next && stop_token_ids.empty()) {
                    mfq_cuda_synchronize();halfway_done=std::chrono::steady_clock::now();
                    measured_halfway=true;
                }
                if(gen>256 && i==gen-1-128 && metadata->flash_next && stop_token_ids.empty()) {
                    mfq_cuda_synchronize();last_window_done=std::chrono::steady_clock::now();
                    measured_last_window=true;
                }
            }
        }
        mfq_cuda_synchronize();
        if (cuda_profiler_range) TYLOINFER_CUDA_CHECK(cudaProfilerStop());
        auto t3 = std::chrono::steady_clock::now();
        report_cuda_memory(execution.config, "decode_complete");
        profiler.report("decode");
        gen = produced;
        auto generated_tensor = generated_cuda.narrow(0, 0, produced).to(mfq_tensor_backend::kCPU).contiguous();
        auto generated_ptr = generated_tensor.template data_ptr<int64_t>();
        double load_s = std::chrono::duration<double>(t1 - t0).count();
        double prefill_s = std::chrono::duration<double>(t2 - t1).count();
        double decode_s = std::chrono::duration<double>(t3 - t2).count();
        double decode_replay_s = std::chrono::duration<double>(t3 - decode_replay_t0).count();
        std::cout << "load_sec=" << load_s << "\n";
        std::cout << "prefill_sec=" << prefill_s << "\n";
        std::cout << "finish_reason=" << (hit_eos ? "stop" : "length") << "\n";
        std::cout << "decode_tokens=" << std::max(0, gen - 1) << "\n";
        std::cout << "decode_setup_sec=" << (decode_s - decode_replay_s) << "\n";
        std::cout << "decode_replay_sec=" << decode_replay_s << "\n";
        if(measured_first_decode) {
            const double first_sec=std::chrono::duration<double>(first_decode_done-t2).count();
            const double after_first_sec=std::chrono::duration<double>(t3-first_decode_done).count();
            std::cout<<"decode_first_step_sec="<<first_sec<<"\n"
                <<"decode_after_first_tokens="<<std::max(0,gen-2)<<"\n"
                <<"decode_after_first_sec="<<after_first_sec<<"\n";
            if(gen>2)std::cout<<"decode_after_first_tok_per_s="<<double(gen-2)/after_first_sec<<"\n";
        }
        if(measured_halfway) {
            const int head_tokens=gen/2,tail_tokens=gen-1-head_tokens;
            const double head_sec=std::chrono::duration<double>(halfway_done-t2).count();
            const double tail_sec=std::chrono::duration<double>(t3-halfway_done).count();
            std::cout<<"decode_head_tokens="<<head_tokens<<"\n"
                <<"decode_head_sec="<<head_sec<<"\n"
                <<"decode_tail_tokens="<<tail_tokens<<"\n"
                <<"decode_tail_sec="<<tail_sec<<"\n"
                <<"decode_tail_tok_per_s="<<double(tail_tokens)/tail_sec<<"\n";
        }
        if(measured_last_window) {
            const double seconds=std::chrono::duration<double>(t3-last_window_done).count();
            std::cout<<"decode_last_128_tokens=128\n"
                <<"decode_last_128_begin_cache_position="<<ids.size(1)+gen-1-128<<"\n"
                <<"decode_last_128_sec="<<seconds<<"\n"
                <<"decode_last_128_tok_per_s="<<128./seconds<<"\n";
        }
        std::cout << "decode_sec=" << decode_s << "\n";
        if (gen > 1) std::cout << "decode_tok_per_s=" << (double)(gen - 1) / decode_s << "\n";
        if (gen > 1) std::cout << "decode_steady_tok_per_s=" << (double)(gen - 1) / decode_replay_s << "\n";
        std::cout << "generated_ids=";
        for (int64_t i = 0; i < generated_tensor.numel(); ++i) {
            if (i) std::cout << ",";
            std::cout << generated_ptr[i];
        }
        std::cout << "\n";
        report_source_file_io(source, "decode");
        if (execution.weights.moe) {
            print_moe_expert_cache_stats(
                execution.weights.moe, std::cout);
        }
        finish_cuda_activity_trace();
        return 0;
 }

} // namespace mfq::cuda::internal
