#include "engine/backends/cuda/executor/options.h"
#include "engine/backends/cuda/native/scope.h"
#include "engine/backends/cuda/native/tensor/backend.h"

#include <barrier>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <utility>
#include <vector>

namespace {
void set_option(const char* name, const char* value) {
    if (value) tyloi::cuda::env_set(name, value); else tyloi::cuda::env_clear(name);
}
}

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
// A reset/drain must wait for work issued on pool streams, not only the
// active one: launch a slow host callback and retire the context set.
static void check_drain() {
    using namespace tyloi::cuda;
    using namespace tyloi;
    ContextSet contexts;
    ContextGuard guard(contexts);
    auto auxiliary = stream_from_pool(false, 0);
    StreamGuard active(auxiliary);
    Event finished;
    TYLOINFER_CUDA_CHECK(cudaLaunchHostFunc(auxiliary.stream(), [](void*) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }, nullptr));
    finished.record(auxiliary.stream());
    contexts.clear();
    if (!finished.ready()) throw std::runtime_error("reset missed an auxiliary consumer");
}

// Each thread guards its own ContextSet: contexts, streams, BLAS handles and
// graph captures must stay isolated, and nested guards must not leak streams.
static void check_contexts() {
    using namespace tyloi::cuda;
    using namespace tyloi;
    ContextSet context_sets[2];
    std::shared_ptr<Context> contexts[2];
    std::barrier ready(2);
    std::exception_ptr errors[2];
    std::thread workers[2];
    for (int i = 0; i < 2; ++i) workers[i] = std::thread([&, i] {
        try {
            CudaScope scope(0, context_sets[i]);
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
            DenseTensor output;
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
        CudaScope first(0, context_sets[0]);
        auto stream = stream_from_pool(false, 0);
        StreamGuard active(stream);
        { CudaScope same(0, context_sets[0]);
          if (current_stream().stream() != stream.stream())
            throw std::runtime_error("same Engine lost its stream override"); }
        { CudaScope second(0, context_sets[1]);
          if (current_stream().stream() == stream.stream())
            throw std::runtime_error("nested Engine inherited a stream override"); }
        if (current_stream().stream() != stream.stream())
            throw std::runtime_error("Engine scope did not restore its stream");
    }
    contexts[0].reset();
    context_sets[0].clear();
    CudaScope survivor(0, context_sets[1]);
    auto value = full({4}, 3.0, TensorOptions().device(kCUDA));
    if (value.sum().item<float>() != 12.0f)
        throw std::runtime_error("peer Engine teardown changed survivor");
}
#endif

int main() try {
    using namespace tyloi::cuda::runtime_options;
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
    set_option("TYLOINFER_MOE_PREFILL_EXPERT_BATCH_TOKENS", "8192");
    if (moe_prefill_expert_batch_tokens().value_or(0) != 8192) return 1;
    set_option("TYLOINFER_MOE_PREFILL_EXPERT_BATCH_TOKENS", nullptr);
    set_option("TYLOINFER_NVQ_DENSE_REFERENCE", "0");
    if (!nvq_dense_reference()) return 1;
    set_option("TYLOINFER_NVQ_DENSE_REFERENCE", nullptr);
    if (nvq_dense_reference()) return 1;
    // Ablation switches must remain live between calls.
    set_option("TYLOINFER_MFE_RESIDENT_PLAN_OVERLAP", "1");
    if (!resident_plan_overlap()) return 1;
    set_option("TYLOINFER_MFE_RESIDENT_PLAN_OVERLAP", "0");
    if (resident_plan_overlap()) return 1;
    set_option("TYLOINFER_MFE_RESIDENT_PLAN_OVERLAP", nullptr);
    set_option("TYLOINFER_MFE_EARLY_GU", "0");
    if (early_gate_up()) return 1;
    set_option("TYLOINFER_MFE_EARLY_GU", nullptr);
    if (!early_gate_up()) return 1;

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    int devices = 0;
    if (cudaGetDeviceCount(&devices) == cudaSuccess && devices) {
        check_contexts();
        check_drain();
    }
#endif
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
