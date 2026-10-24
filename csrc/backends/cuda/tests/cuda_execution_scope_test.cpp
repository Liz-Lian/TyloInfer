#include "env.h"
#include "execution/execution.h"
#include "execution/options.h"

#include <atomic>
#include <barrier>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <utility>
#include <vector>

namespace {
void set_option(const char* name, const char* value) {
    if (value) mfq::cuda::env_set(name, value); else mfq::cuda::env_clear(name);
}
}

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
static void check_contexts() {
    using namespace mfq::cuda;
    CudaExecutionContext engines[2];
    std::shared_ptr<Context> contexts[2];
    std::barrier ready(2);
    std::exception_ptr errors[2];
    std::thread workers[2];
    for (int i = 0; i < 2; ++i) workers[i] = std::thread([&, i] {
        try {
            CudaScope scope(engines[i]);
            contexts[i] = default_context(0);
            auto options = TensorOptions().device(kCUDA).dtype(kFloat32);
            auto input = full({32, 32}, double(i + 1), options);
            auto weight = full({32, 32}, 1.0 / 32.0, options);
            auto own_stream = stream_from_pool(false, 0);
            StreamGuard stream(own_stream);
            contexts[i]->stream().synchronize();
            ready.arrive_and_wait();
            for (int repeat = 0; repeat < 8; ++repeat) {
                auto output = matmul(input, weight);
                if (output.sum().item<float>() != 1024.0f * (i + 1))
                    throw std::runtime_error("concurrent matmul changed output");
            }
            Graph graph;
            graph.prepare_memory();
            Tensor output;
            for (int warmup = 0; warmup < 2; ++warmup) {
                output = matmul(input, weight);
                output = {};
            }
            graph.capture_begin();
            output = matmul(input, weight);
            graph.capture_end();
            for (int replay = 0; replay < 8; ++replay) graph.replay();
            if (output.sum().item<float>() != 1024.0f * (i + 1))
                throw std::runtime_error("concurrent graph changed output");
        } catch (...) {
            errors[i] = std::current_exception();
            ready.arrive_and_drop();
        }
    });
    for (auto& worker : workers) worker.join();
    for (auto error : errors) if (error) std::rethrow_exception(error);
    if (contexts[0] == contexts[1] ||
        contexts[0]->stream().get() == contexts[1]->stream().get() ||
        contexts[0]->blas().get() == contexts[1]->blas().get() ||
        (contexts[0]->supports_async_allocations() &&
         contexts[0]->memory_pool() == contexts[1]->memory_pool()))
        throw std::runtime_error("Engine native resources are shared");
    {
        CudaScope first(engines[0]);
        auto stream = stream_from_pool(false, 0);
        StreamGuard active(stream);
        { CudaScope same(engines[0]);
          if (current_stream().stream() != stream.stream())
              throw std::runtime_error("same Engine lost its stream override"); }
        { CudaScope second(engines[1]);
          if (current_stream().stream() == stream.stream())
              throw std::runtime_error("nested Engine inherited a stream override"); }
        if (current_stream().stream() != stream.stream())
            throw std::runtime_error("Engine scope did not restore its stream");
    }
    contexts[0].reset();
    engines[0].native_contexts.clear();
    CudaScope survivor(engines[1]);
    auto value = full({4}, 3.0, TensorOptions().device(kCUDA));
    if (value.sum().item<float>() != 12.0f)
        throw std::runtime_error("peer Engine teardown changed survivor");
}
#endif

int main() try {
    using namespace mfq::cuda::runtime_options;
    // Presence, first-character and exact-value controls have distinct meanings.
    set_option("TYLOINFER_NINT_ROUTE_ROW_WORKSPACE", nullptr);
    if (nint_route_row_workspace()) return 1;
    set_option("TYLOINFER_NINT_ROUTE_ROW_WORKSPACE", "0");
    if (nint_route_row_workspace().value_or(true)) return 1;
    set_option("TYLOINFER_NINT_ROUTE_ROW_WORKSPACE", "10");
    if (!nint_route_row_workspace().value_or(false)) return 1;
    set_option("TYLOINFER_NINT_ROUTE_ROW_WORKSPACE", nullptr);
    set_option("TYLOINFER_MFE_NINT_LATE_SCALE", "10");
    if (mfe_nint_late_scale().value_or(true)) return 1;
    set_option("TYLOINFER_MFE_NINT_LATE_SCALE", "1");
    if (!mfe_nint_late_scale().value_or(false)) return 1;
    set_option("TYLOINFER_MFE_NINT_LATE_SCALE", nullptr);
    set_option("TYLOINFER_EMBEDDING_MAPPED", nullptr);
    if (embedding_mapped()) return 1;
    set_option("TYLOINFER_EMBEDDING_MAPPED", "2");
    if (embedding_mapped().value_or(0) != 2) return 1;
    set_option("TYLOINFER_EMBEDDING_MAPPED", nullptr);
    set_option("TYLOINFER_MOE_PREFILL_EXPERT_BATCH_TOKENS", "8192");
    if (moe_prefill_expert_batch_tokens().value_or(0) != 8192) return 1;
    set_option("TYLOINFER_MOE_PREFILL_EXPERT_BATCH_TOKENS", nullptr);
    set_option("TYLOINFER_NVQ_DENSE_REFERENCE", "0");
    if (!nvq_dense_reference()) return 1;
    set_option("TYLOINFER_NVQ_DENSE_REFERENCE", nullptr);
    if (nvq_dense_reference()) return 1;
    // Ablation switches must remain live between calls; ordinary generation
    // must honor the same explicit residency override as benchmark modes.
    set_option("TYLOINFER_MFE_RESIDENT_PLAN_OVERLAP", "1");
    if (!resident_plan_overlap()) return 1;
    set_option("TYLOINFER_MFE_RESIDENT_PLAN_OVERLAP", "0");
    if (resident_plan_overlap()) return 1;
    set_option("TYLOINFER_MFE_RESIDENT_PLAN_OVERLAP", nullptr);
    set_option("TYLOINFER_MFE_EARLY_GU", "0");
    if (early_gate_up()) return 1;
    set_option("TYLOINFER_MFE_EARLY_GU", nullptr);
    if (!early_gate_up()) return 1;
    set_option("TYLOINFER_MOE_RESIDENCY_ADAPT", "0");
    if (load_cuda_execution_config().moe_residency_adapt) return 1;
    set_option("TYLOINFER_MOE_RESIDENCY_ADAPT", nullptr);
    if (!load_cuda_execution_config().moe_residency_adapt) return 1;
    mfq::cuda::env_set("TYLOINFER_DISABLE_NVQ_FUSION", "1");
    const auto disabled_config = load_cuda_execution_config();
    mfq::cuda::env_clear("TYLOINFER_DISABLE_NVQ_FUSION");
    const auto enabled_config = load_cuda_execution_config();
    if (disabled_config.nvq_fusion || !enabled_config.nvq_fusion) return 1;

    CudaExecutionContext first;
    CudaExecutionContext second;
    first.tensor_parallel.devices = {0, 1};
    second.expert_parallel.devices = {2, 3};
    first.drop_file_cache = true;
    first.config.nvq_fusion = false;
    second.decode_graph_serial_branches = true;

    std::atomic<int> ready{0};
    std::atomic<bool> isolated{true};
    auto run = [&](CudaExecutionContext& context,
                   bool first_context) {
        context.kl_mmq.dense_calls = first_context ? 11 : 29;
        context.continuous_batch_cache_serial = first_context;
        context.profiler.stats["engine"].calls = first_context ? 3 : 7;
        ready.fetch_add(1, std::memory_order_release);
        while (ready.load(std::memory_order_acquire) != 2) {
            std::this_thread::yield();
        }
        const bool valid = first_context
            ? model_parallel_enabled(context) &&
                context.tensor_parallel.primary_device() == 0 &&
                !context.expert_parallel.enabled() &&
                context.kl_mmq.dense_calls == 11 &&
                context.profiler.stats.at("engine").calls == 3 &&
                !context.config.nvq_fusion
            : model_parallel_enabled(context) &&
                !context.tensor_parallel.enabled() &&
                moe_parallel_config(context).primary_device() == 2 &&
                context.kl_mmq.dense_calls == 29 &&
                context.profiler.stats.at("engine").calls == 7 &&
                context.config.nvq_fusion;
        if (!valid) {
            isolated.store(false, std::memory_order_relaxed);
        }
    };
    std::thread first_thread(
        run, std::ref(first), true);
    std::thread second_thread(
        run, std::ref(second), false);
    first_thread.join();
    second_thread.join();

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    int devices = 0;
    if (cudaGetDeviceCount(&devices) == cudaSuccess && devices) check_contexts();
#endif
    return isolated.load(std::memory_order_relaxed) &&
            first.drop_file_cache && !second.drop_file_cache &&
            !first.decode_graph_serial_branches &&
            second.decode_graph_serial_branches &&
            first.continuous_batch_cache_serial &&
            !second.continuous_batch_cache_serial
        ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
