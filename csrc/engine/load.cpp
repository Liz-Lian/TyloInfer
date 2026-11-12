#include "engine/load.h"
#include "engine/config.h"
#include "engine/ops.h"
#include "engine/models/load.h"
#ifndef TYLOINFER_NATIVE_CUDA
#include "engine/models/assemble.h"
#include "engine/models/minicpmo45/forward.h"
#include "engine/models/minicpmo45/text/forward.h"
#include "engine/models/qwen35/forward.h"
#include "engine/models/deepseek_v41/vision.h"
#include "engine/models/glm_dsa/forward.h"
#include "engine/models/deepseek_v4/forward.h"
#include "engine/models/glm5_next/forward.h"
#include "engine/models/nemotron_h/forward.h"
#endif
#include "engine/models/source/model_graph.h"
#ifndef TYLOINFER_NATIVE_CUDA
#include "engine/instance_impl.h"
#endif
#include "engine/drain.h"
#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/program.h"
#include "engine/models/causal_model.h"
#include "engine/models/source/model_config.h"
#include "engine/models/session/cache.h"
#include "engine/models/input.h"
#include "engine/requests/processor.h"
#include "engine/state/session/sessions.h"
#endif

#include "engine/backends/cuda/native/env.h"
#include "engine/backends/cuda/native/collectives.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/weights/cache.h"
#include "engine/state/kv_memory.h"
#include "engine/backends/cuda/native/profile.h"
#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/weights/moe/profile.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>
#include <variant>
#include <vector>

#ifndef TYLOINFER_NATIVE_CUDA
namespace tyloi::cuda::engine_detail {

using State = std::variant<
    std::unique_ptr<Text<minicpmo45::text::Model, minicpmo45::text::State, std::unique_ptr<minicpmo45::Components>>>,
#ifndef TYLOINFER_NATIVE_CUDA
    std::unique_ptr<Text<qwen35::Model, qwen35::State, qwen35::Components>>,
    std::unique_ptr<Text<glm_dsa::Model, glm_dsa::State, std::monostate>>,
    std::unique_ptr<Text<glm5_next::Model, glm5_next::State, std::unique_ptr<tyloi::models::Mtp>>>,
    std::unique_ptr<Text<nemotron_h::Model, nemotron_h::State, nemotron_h::Components>>,
    std::unique_ptr<Text<deepseek_v4::Model, deepseek_v4::State, std::monostate>>,
    std::unique_ptr<Text<deepseek_v41::Model, deepseek_v41::State, std::unique_ptr<tyloi::models::Mtp>>>,
    std::unique_ptr<Text<qwen4_exp::Model, qwen4_exp::State, qwen4_exp::Components>>,
#endif
    std::unique_ptr<Text<minicpmo45::Model, minicpmo45::State, std::monostate>>>;

} // namespace tyloi::cuda::engine_detail
#endif

namespace tyloi::cuda {

namespace {

#ifdef TYLOINFER_NATIVE_CUDA
std::unique_ptr<engine::Engine> load_native_engine(CudaEngineOptions options, engine::Engine* target,
    std::shared_ptr<const ModelSource> source);
#endif

} // namespace

std::unique_ptr<engine::Engine> load_cuda_engine(CudaEngineOptions options) {
#ifdef TYLOINFER_NATIVE_CUDA
    auto source = tyloi::open_model_source(options.model_path);
    return load_native_engine(std::move(options), nullptr, std::move(source));
#else
    return std::make_unique<engine_detail::EngineInstance<
        engine_detail::State, with_model<>>>(std::move(options));
#endif
}

void reload_cuda_engine(engine::Engine& engine, CudaEngineOptions options) {
#ifndef TYLOINFER_NATIVE_CUDA
    if (auto* target = dynamic_cast<engine_detail::EngineInstance<engine_detail::State, with_model<>>*>(&engine)) {
        target->options = std::move(options);
        target->load();
        return;
    }
#else
    if (typeid(engine) == typeid(tyloi::engine::Engine)) {
        auto source = tyloi::open_model_source(options.model_path);
        load_native_engine(std::move(options), &engine, std::move(source));
        return;
    }
#endif
    throw std::invalid_argument("unsupported CUDA engine for reload");
}

#ifdef TYLOINFER_NATIVE_CUDA
namespace {
std::unique_ptr<engine::Engine> load_native_engine(CudaEngineOptions options, engine::Engine* target,
        std::shared_ptr<const ModelSource> source) {
    std::shared_ptr<tyloi::weights::Memory> ledger;
    auto weights = std::make_unique<tyloi::weights::Cache>();
    std::shared_ptr<KvMemoryManager> kv;
    std::shared_ptr<KvOffloadStore> offload;
    CudaLoadConfig load;
    tyloi::engine::Config config;
    std::unique_ptr<Executor> root;
    std::unique_ptr<models::CausalModel> model;
    std::unique_ptr<models::State> state;
    std::unique_ptr<tyloi::state::Sessions> sessions;
    std::unique_ptr<tyloi::Executor::Graph> graph;
    try {
        const bool dump_env = env_flag("TYLOINFER_CUDA_ENV_DUMP", false);
        env_freeze();
        if (dump_env) std::cerr << "cuda knobs at load:\n" << env_dump();
        if (!options.context_size) options.context_size = 32768;
        auto [execution_config, numeric_options] = load_cuda_execution_config();
        load = std::move(execution_config);
        root = std::make_unique<Executor>(std::move(numeric_options), 0);
        {
            auto outer_scope = root->enter(load.layer_placement.primary_device());
            internal::setup_cuda_load(options, load, *root, *weights, kv, offload);
            root->set_primary(load.layer_placement.primary_device());
            root->profiler().enabled = false;
            ledger = std::make_shared<tyloi::weights::Memory>();
            tyloi::weights::LoadScope memory_load(*ledger);
            config = engine::resolve_config(options.prefill_chunk_size, options.continuous_batching);
            config.profile_decode = load.profile_decode;
            config.profile_mtp = load.profile_mtp;
            config.trace_moe = load.trace_moe;
            config.prefetch = load.prefetch;
            if (config.continuous_batch.max_sequences) {
                config.workspace_reserve = ::tyloi::cuda::options::workspace_reserve_bytes();
                config.continuous_batch.packed_prefill = ::tyloi::cuda::options::batch_packed_prefill();
                config.continuous_batch.packed_tokens = ::tyloi::cuda::options::batch_packed_prefill_token_budget();
                config.continuous_batch.preallocate = ::tyloi::cuda::options::kv_vram_preallocated();
                config.continuous_batch.cpu_offload = !load.dsv4_cpu_offload_layers.empty();
                config.continuous_batch.cached_slots = load.weights.min_slots;
            }
            internal::configure_model_kv(*root, kv, offload);
            {
                auto scope = root->enter(load.layer_placement.primary_device());
                model = internal::load_model(options, load, *root, *weights, source, true);
                if (offload && !model->info().streamed_kv)
                    throw std::invalid_argument("CUDA streamed KV requires a QSA model");
                state = model->make_state();
                if (::tyloi::cuda::options::kv_vram_preallocated()) state->preallocate();
                Executor::prepare(model->info().vocab_size);
                finalize_kv(kv.get(), *weights, offload);
                finish_load(kv.get(), *weights, load.report_cuda_memory);
                root->configure_kv(kv, offload);
                if (offload && !model->info().streamed_kv)
                    throw std::invalid_argument("CUDA streamed KV requires a QSA model");
            }
            graph = root->graph();
            sessions = internal::make_sessions(*source, config.session_cache, config.prefix_cache,
                model->info().max_context, state->supports_text_session_state(),
                state->supports_paged_text_session_state());
        }
        const auto mode = root->forward_mode(load.layer_placement.primary_device());
        config.devices = cuda_devices(load);
        config.online_budget = load.layer_placement.devices.size() <= 1 &&
            !load.tensor_parallel.enabled() && !load.expert_parallel.enabled();
        const auto model_info = model->info();
        const auto graph_weights = model_info.graph_weights == models::GraphWeights::resident
            ? tyloi::weights::Cache::Graph::legacy : tyloi::weights::Cache::Graph::native;
        const bool graph_allowed = model_info.decode_graph && !root->attention_options().kernels.decode_aten &&
            model_parallel_cuda_graph_enabled(load, *root, *weights) &&
            weights->graph_safe(graph_weights, moe_parallel_config(load).enabled());
        const auto vocab_size = static_cast<std::int32_t>(model_info.vocab_size);
        auto text = engine::TextProcessor::load(*source, options.tokenizer_model, vocab_size, model_info.model_type);
        config.stop_token_ids = models::stop_token_ids(models::config_json(*source, options.config_path), vocab_size);
        config.resident_budget_bytes = ::tyloi::cuda::options::resident_budget_bytes();
        config.decode_graph.enabled &= graph_allowed;
        config.continuous_batch.cuda_graph &= graph_allowed;
        models::ForwardOptions forward{mode.kv_capacity, mode.trace, mode.trace_layer, mode.primary};
        if (target) {
            target->rebind(std::move(model), std::move(root), std::move(config), std::move(source),
                std::move(ledger), std::move(weights), std::move(state), std::move(sessions),
                std::move(text), forward, std::move(graph));
            return {};
        }
        return std::make_unique<engine::Engine>(std::move(model), std::move(root), std::move(config),
            std::move(source), std::move(ledger), std::move(weights), std::move(state),
            std::move(sessions), std::move(text), forward, std::move(graph));
    } catch (...) {
        const auto failure = std::current_exception();
        if (root) {
            try { if (weights) drain_cuda(*root, *weights, offload.get(), load.layer_placement, load.tensor_parallel, load.expert_parallel); }
            catch (const std::exception& error) { std::cerr << "runtime_load_cleanup error=" << error.what() << '\n'; }
            catch (...) { std::cerr << "runtime_load_cleanup error=unknown\n"; }
            std::unique_ptr<tyloi::Executor::Scope> scope;
            try {
                scope = root->enter(load.layer_placement.primary_device());
            } catch (const std::exception& error) {
                std::cerr << "runtime_load_cleanup error=" << error.what() << '\n';
            } catch (...) {
                std::cerr << "runtime_load_cleanup error=unknown\n";
            }
            graph.reset();
            state.reset(); model.reset(); sessions.reset();
            try { root->clear(); }
            catch (const std::exception& error) { std::cerr << "runtime_load_cleanup error=" << error.what() << '\n'; }
            catch (...) { std::cerr << "runtime_load_cleanup error=unknown\n"; }
            root->configure_kv({}, {});
            offload.reset(); kv.reset(); weights.reset();
            scope.reset();
            root.reset();
        }
        std::rethrow_exception(failure);
    }
}
} // namespace
#endif

} // namespace tyloi::cuda

namespace tyloi::cuda::internal {
void setup_cuda_load(
        const tyloi::cuda::CudaLoadOptions& options,
        CudaLoadConfig& config, Executor& executor, tyloi::weights::Cache& weights,
        std::shared_ptr<KvMemoryManager>& kv, std::shared_ptr<KvOffloadStore>& offload) {
    drain_cuda(executor, weights, offload.get(),
               config.layer_placement, config.tensor_parallel, config.expert_parallel);
    executor.reset();
    config.tensor_parallel = {};
    config.expert_parallel = {};
    config.layer_placement = {};
    config.dsv4_cpu_offload_layers.clear();
    config.dsv4_cpu_offload_host_bytes = 0;
    config.n_gpu_layers = -1;
    config.dense_cpu_layer_count = 0;
    config.drop_file_cache = false;
    config.weights.min_slots = 8;
    offload.reset();
    kv.reset();
    weights.begin_load();
    if (options.n_gpu_layers_set) config.n_gpu_layers = options.n_gpu_layers;
    if (options.cpu_threads_set && options.cpu_threads <= 0) {
        throw std::runtime_error("--threads must be positive");
    }
    if (options.cpu_threads > 0) {
        mfq_set_num_threads(options.cpu_threads);
    }
    config.yarn_context_extension=options.yarn_context_extension;
    config.tensor_parallel = options.tensor_parallel;
    config.expert_parallel = options.expert_parallel;
    config.layer_placement = options.layer_placement;
    configure_placement(config, executor.collectives());
#ifdef TYLOINFER_NATIVE_CUDA
    executor.set_primary(model_parallel_primary_device(config));
#endif
    if (!options.cpu_offload_layers.empty()) {
        config.dsv4_cpu_offload_layers = options.cpu_offload_layers;
        std::vector<int> ordered(
            config.dsv4_cpu_offload_layers.begin(),
            config.dsv4_cpu_offload_layers.end());
        std::sort(ordered.begin(), ordered.end());
        if (ordered.front() < 0)
            throw std::runtime_error("--cpu-offload-layers must be non-negative");
        std::cerr << "cpu_offload_layers=";
        for (size_t index = 0; index < ordered.size(); ++index) {
            if (index) std::cerr << ',';
            std::cerr << ordered[index];
        }
        std::cerr << std::endl;
    }
    if (options.n_gpu_layers_set && config.n_gpu_layers < 0) {
        throw std::runtime_error("--n-gpu-layers must be non-negative");
    }
    if (options.n_gpu_layers_set && !options.cpu_offload_layers.empty()) {
        throw std::runtime_error(
            "--n-gpu-layers cannot be combined with --cpu-offload-layers");
    }
    if (options.moe_gpu_cache_gb < 0.0 ||
            !std::isfinite(options.moe_gpu_cache_gb)) {
        throw std::runtime_error(
            "--moe-gpu-cache-gb must be finite and non-negative");
    }
    if (options.moe_gpu_cache_gb > 0.0 &&
            !options.cpu_offload_layers.empty()) {
        throw std::runtime_error(
            "--moe-gpu-cache-gb cannot be combined with "
            "--cpu-offload-layers");
    }
    if (options.moe_gpu_cache_gb > 0.0 &&
            (config.tensor_parallel.enabled() ||
             config.expert_parallel.enabled())) {
        throw std::runtime_error(
            "--moe-gpu-cache-gb cannot be combined with "
            "tensor/expert parallelism");
    }
    const auto* auto_profile=tyloi::cuda::env_lookup("TYLOINFER_MOE_AUTO_CACHE_PROFILE");
    const auto profile_path=tyloi::resolve_moe_cache_profile_path(options.moe_cache_profile_path,
        options.moe_gpu_cache_gb>0.0?options.model_path:std::string{},!auto_profile || auto_profile[0]!='0');
    if (!profile_path.empty() &&
            options.moe_gpu_cache_gb <= 0.0) {
        throw std::runtime_error(
            "--moe-cache-profile requires --moe-gpu-cache-gb");
    }
    if (options.moe_gpu_cache_gb > 0.0) {
        constexpr double gib =
            1024.0 * 1024.0 * 1024.0;
        const double bytes = options.moe_gpu_cache_gb * gib;
        if (bytes < 1.0 ||
                bytes >
                    static_cast<double>(
                        std::numeric_limits<int64_t>::max())) {
            throw std::runtime_error(
                "--moe-gpu-cache-gb is outside the supported range");
        }
        weights.configure(static_cast<std::size_t>(bytes), config.cache, config.weights.preload, profile_path);
        if (!profile_path.empty() && !weights.empty()) {
            std::cerr<<"moe_cache_profile source="<<(options.moe_cache_profile_path.empty()?"sidecar":"explicit")
                <<" path="<<profile_path<<std::endl;
        }
        config.drop_file_cache = true;
    }
}

} // namespace tyloi::cuda::internal

namespace tyloi::cuda {

void finalize_kv(KvMemoryManager* kv, tyloi::weights::Cache& weights,
        std::shared_ptr<KvOffloadStore>& offload) {
    if (kv) {
        kv->finalize(weights, weights.minimum());
        weights.commit_kv();
        if (auto store = kv->offload()) offload = std::move(store);
    }
}

std::chrono::steady_clock::time_point finish_load(
        KvMemoryManager* kv, tyloi::weights::Cache& weights, bool report_memory) {
    if (weights.needs_finalize()) {
        weights.finalize(
            kv ? kv->growth_reserve() : 0,
            kv ? kv->expert_budget_charge() : 0);
    }
    mfq_cuda_synchronize();
    const auto loaded = std::chrono::steady_clock::now();
    report_cuda_memory(report_memory, "loaded");
    return loaded;
}

} // namespace tyloi::cuda
