#include "engine/program.h"

#include "engine/backends/cuda/executor/config.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/backends/cuda/native/env.h"
#include "engine/backends/cuda/native/profile.h"
#include "engine/drain.h"
#include "engine/models/causal_model.h"
#include "engine/models/common/load.h"
#include "engine/models/layers.h"
#include "engine/models/groups.h"
#include "engine/models/load.h"
#include "engine/models/session/cache.h"
#include "engine/models/source.h"
#include "engine/models/source/model_config.h"
#include "engine/models/source/model_graph.h"
#include "engine/state/kv_memory.h"
#include "engine/state/kv_offload.h"
#include "engine/weights/cache.h"
#include "engine/weights/memory.h"

#include <algorithm>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace tyloi::cuda {
namespace {
struct Assembly {
    std::string definition;
    weights::Loader loader;
    models::LoadOptions policy;
};

/** Apply the owner's placement and cache policy to every tool and model loader. */
weights::Loader loader(CudaLoadConfig& config, weights::Cache& weights, const ModelSource& source) {
    return weights.loader(source, config.weights,
        {config.layer_placement.primary_device(), false, config.drop_file_cache,
         config.tensor_parallel, moe_parallel_config(config)});
}

/** Resolve the shared definition, weight loader, and native model policy. */
Assembly assembly(const CudaLoadOptions& options, CudaLoadConfig& config,
        Executor& executor, weights::Cache& weights, const ModelSource& source,
        bool components) {
    auto definition = models::config_json(source, options.config_path);
    definition = configure_yarn(config.yarn, config.yarn_context_extension,
        std::move(definition), models::graph(source).backbone, options.context_size);
    if (!config.dsv4_cpu_offload_layers.empty()) config.drop_file_cache = true;
    auto weights_loader = loader(config, weights, source);
    return {std::move(definition), std::move(weights_loader),
        models::LoadOptions{.context = options.context_size, .gpu_layers = config.n_gpu_layers,
            .expert_parallel = config.expert_parallel.enabled(), .yarn = config.yarn,
            .cpu_offload_layers = config.dsv4_cpu_offload_layers,
            .engram_cache_rows = config.deepseek_v41_engram_cache_rows,
            .host_tokens = config.cache.pipeline,
            .mirror_scalars = config.tensor_parallel_mirror_linear_attention_scalars,
            .optional_components = components,
            .transfer_auto = weights.transfer() == ::tyloi::weights::Cache::Transfer::automatic,
            .transposed = executor.mode().transposed}};
}
} // namespace

namespace internal {
void configure_model_kv(Executor& executor, std::shared_ptr<KvMemoryManager>& kv,
        std::shared_ptr<KvOffloadStore>& offload) {
    const auto options = cuda::options::qsa_kv_offload();
    kv = std::make_shared<KvMemoryManager>();
    if (options.gpu_budget_bytes) {
        kv.reset();
        KvOffloadConfig config;
        config.gpu_budget_bytes = options.gpu_budget_bytes;
        config.ram_budget_bytes = cuda_has_separate_memory() ? options.ram_budget_bytes : 0;
        config.buffer_bytes = std::min<std::size_t>(config.buffer_bytes, config.gpu_budget_bytes / 4);
        config.directory = options.directory;
        offload = std::make_shared<KvOffloadStore>(std::move(config));
    }
    executor.configure_kv(kv, offload);
}

std::unique_ptr<models::CausalModel> load_model(const CudaLoadOptions& options,
        CudaLoadConfig& config, Executor& executor, weights::Cache& weights,
        const std::shared_ptr<const ModelSource>& source, bool components) {
    auto input = assembly(options, config, executor, weights, *source, components);
    return models::load(source, input.definition, input.loader, config.layer_placement, input.policy);
}
} // namespace internal

struct Program::Impl {
    CudaLoadOptions options;
    CudaLoadConfig config;
    std::unique_ptr<Executor> executor;
    std::unique_ptr<::tyloi::weights::Cache> weights;
    std::shared_ptr<KvMemoryManager> kv;
    std::shared_ptr<KvOffloadStore> offload;
    std::shared_ptr<const ModelSource> source;
    std::shared_ptr<::tyloi::weights::Memory> ledger;
    std::unique_ptr<models::CausalModel> model;
    std::unique_ptr<models::State> state;
    std::unique_ptr<models::Layers> layers;
    std::unique_ptr<models::Groups> groups;
    bool attempted = false;

    /** Open the source only for a model-loading operation, after execution setup. */
    void open_source() {
        if (options.model_path.empty()) throw std::logic_error("program has no model source");
        if (!source) source = open_model_source(options.model_path);
    }

    /** Drain before destroying execution-dependent resources, including on partial setup. */
    void cleanup() noexcept {
        if (!executor) return;
        try { if (weights) drain_cuda(*executor, *weights, offload.get(),
            config.layer_placement, config.tensor_parallel, config.expert_parallel); }
        catch (const std::exception& error) { std::cerr << "program_cleanup error=" << error.what() << '\n'; }
        catch (...) { std::cerr << "program_cleanup error=unknown\n"; }
        std::unique_ptr<tyloi::Executor::Scope> scope;
        try { scope = executor->enter(config.layer_placement.primary_device()); }
        catch (const std::exception& error) { std::cerr << "program_cleanup error=" << error.what() << '\n'; }
        catch (...) { std::cerr << "program_cleanup error=unknown\n"; }
        state.reset();
        layers.reset();
        groups.reset();
        model.reset();
        try { executor->clear(); }
        catch (const std::exception& error) { std::cerr << "program_cleanup error=" << error.what() << '\n'; }
        catch (...) { std::cerr << "program_cleanup error=unknown\n"; }
        try { executor->configure_kv({}, {}); }
        catch (const std::exception& error) { std::cerr << "program_cleanup error=" << error.what() << '\n'; }
        catch (...) { std::cerr << "program_cleanup error=unknown\n"; }
        offload.reset(); kv.reset(); weights.reset(); ledger.reset(); source.reset();
        scope.reset();
        executor.reset();
    }
};

Program::Program() : impl_(std::make_unique<Impl>()) {}
Program::~Program() noexcept { impl_->cleanup(); }

std::unique_ptr<Program> Program::bare() {
    auto program = std::unique_ptr<Program>(new Program());
    auto& p = *program->impl_;
    const bool dump_env = env_flag("TYLOINFER_CUDA_ENV_DUMP", false);
    if (dump_env) std::cerr << "cuda knobs at load:\n" << env_dump();
    auto [config, numeric] = load_cuda_execution_config();
    p.config = std::move(config);
    p.weights = std::make_unique<::tyloi::weights::Cache>();
    p.executor = std::make_unique<Executor>(std::move(numeric), 0);
    p.executor->profiler().enabled = false;
    p.ledger = std::make_shared<::tyloi::weights::Memory>();
    return program;
}

std::unique_ptr<Program> Program::open(CudaLoadOptions options, bool adapt) {
    auto program = bare();
    auto& p = *program->impl_;
    p.options = std::move(options);
    if (!adapt) p.config.cache.adapt = false;
    auto scope = p.executor->enter(p.config.layer_placement.primary_device());
    internal::setup_cuda_load(p.options, p.config, *p.executor, *p.weights, p.kv, p.offload);
    p.executor->set_primary(p.config.layer_placement.primary_device());
    internal::configure_model_kv(*p.executor, p.kv, p.offload);
    return program;
}

void Program::load(bool components) {
    auto& p = *impl_;
    if (p.attempted || !p.executor) throw std::logic_error("program load already attempted");
    p.attempted = true;
    p.open_source();
    {
        auto scope = p.executor->enter(p.config.layer_placement.primary_device());
        ::tyloi::weights::LoadScope memory(*p.ledger);
        p.model = internal::load_model(p.options, p.config, *p.executor, *p.weights, p.source, components);
        if (p.offload && !p.model->info().streamed_kv)
            throw std::invalid_argument("CUDA streamed KV requires a QSA model");
        p.state = p.model->make_state();
        if (cuda::options::kv_vram_preallocated()) p.state->preallocate();
        Executor::prepare(p.model->info().vocab_size);
        finalize_kv(p.kv.get(), *p.weights, p.offload);
        finish_load(p.kv.get(), *p.weights, p.config.report_cuda_memory);
        p.executor->configure_kv(p.kv, p.offload);
        if (p.offload && !p.model->info().streamed_kv)
            throw std::invalid_argument("CUDA streamed KV requires a QSA model");
    }
}

void Program::load_layers() {
    auto& p = *impl_;
    if (p.attempted || !p.executor) throw std::logic_error("program load already attempted");
    p.attempted = true;
    p.open_source();
    {
        auto scope = p.executor->enter(p.config.layer_placement.primary_device());
        ::tyloi::weights::LoadScope memory(*p.ledger);
        if (p.offload) throw std::invalid_argument("layer-wise execution does not support streamed KV");
        auto input = assembly(p.options, p.config, *p.executor, *p.weights, *p.source, false);
        p.layers = models::load_layers(p.source, input.definition, input.loader,
            p.config.layer_placement, input.policy);
        finish_load(p.kv.get(), *p.weights, p.config.report_cuda_memory);
    }
}

void Program::load_groups(std::int64_t context) {
    auto& p = *impl_;
    if (p.attempted || !p.executor) throw std::logic_error("program load already attempted");
    if (context < 1) throw std::invalid_argument("grouped execution requires a positive context");
    p.attempted = true;
    p.open_source();
    auto scope = p.executor->enter(p.config.layer_placement.primary_device());
    ::tyloi::weights::LoadScope memory(*p.ledger);
    if (p.offload) throw std::invalid_argument("grouped execution does not support streamed KV");
    auto options = p.options;
    options.context_size = context;
    auto input = assembly(options, p.config, *p.executor, *p.weights, *p.source, false);
    p.groups = models::load_groups(p.source, input.definition, input.loader,
        p.config.layer_placement, input.policy);
    p.groups->prepare();
    finish_load(p.kv.get(), *p.weights, p.config.report_cuda_memory);
}

Executor& Program::executor() { if (!impl_->executor) throw std::logic_error("program closed"); return *impl_->executor; }
::tyloi::weights::Cache& Program::weights() { if (!impl_->weights) throw std::logic_error("program closed"); return *impl_->weights; }
::tyloi::weights::Loader Program::loader() {
    auto& p = *impl_;
    if (!p.executor || !p.weights) throw std::logic_error("program closed");
    p.open_source();
    return ::tyloi::cuda::loader(p.config, *p.weights, *p.source);
}
CudaLoadConfig& Program::config() { return impl_->config; }
models::CausalModel& Program::model() { if (!impl_->model) throw std::logic_error("model not loaded"); return *impl_->model; }
models::State& Program::state() { if (!impl_->state) throw std::logic_error("state not loaded"); return *impl_->state; }
models::Layers& Program::layers() { if (!impl_->layers) throw std::logic_error("layers not loaded"); return *impl_->layers; }
models::Groups& Program::groups() { if (!impl_->groups) throw std::logic_error("groups not loaded"); return *impl_->groups; }
void Program::drain() {
    auto& p = *impl_;
    if (!p.executor || !p.weights) throw std::logic_error("program closed");
    drain_cuda(*p.executor, *p.weights, p.offload.get(), p.config.layer_placement,
        p.config.tensor_parallel, p.config.expert_parallel);
}
} // namespace tyloi::cuda
