#include "engine/program.h"

#include "engine/backends/cuda/executor/config.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/backends/cuda/native/env.h"
#include "engine/backends/cuda/native/profile.h"
#include "engine/drain.h"
#include "engine/budget.h"
#include "engine/config.h"
#include "engine/generation/decode.h"
#include "engine/generation/batch_driver.h"
#include "engine/generation/batch.h"
#include "engine/generation/batch_ops.h"
#include "engine/generation/prefill_budget.h"
#include "engine/generation/prefill.h"
#include "engine/generation/run.h"
#include "engine/generation/mtp/state.h"
#include "engine/models/causal_model.h"
#include "engine/models/common/load.h"
#include "engine/models/common/mtp.h"
#include "engine/models/layers.h"
#include "engine/models/groups.h"
#include "engine/models/ffn.h"
#include "engine/models/load.h"
#include "engine/models/session/cache.h"
#include "engine/models/source.h"
#include "engine/models/source/model_config.h"
#include "engine/models/source/model_graph.h"
#include "engine/state/kv_memory.h"
#include "engine/state/kv_offload.h"
#include "engine/state/session/sessions.h"
#include "engine/weights/cache.h"
#include "engine/weights/memory.h"

#include <algorithm>
#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <utility>

namespace tyloi::cuda {
namespace {
/** Drain before retiring a request's coroutine and release its exclusive owner flag. */
struct RequestWork {
    Program& program;
    engine::Generation job;
    bool& running;
    bool complete = false;
    void drain() noexcept {
        try { program.drain(); }
        catch (const std::exception& error) { std::cerr << "request_cleanup error=" << error.what() << '\n'; }
        catch (...) { std::cerr << "request_cleanup error=unknown\n"; }
    }
    ~RequestWork() noexcept {
        if (!complete) drain();
        job = {};
        if (!complete) drain();
        running = false;
    }
};

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
void configure_generation(engine::Config& runtime, const CudaLoadConfig& load,
        Executor& executor, const weights::Cache& weights, const models::Info& info) {
    runtime.profile_decode = load.profile_decode;
    runtime.profile_mtp = load.profile_mtp;
    runtime.trace_moe = load.trace_moe;
    runtime.prefetch = load.prefetch;
    runtime.devices = cuda_devices(load);
    runtime.online_budget = load.layer_placement.devices.size() <= 1 &&
        !load.tensor_parallel.enabled() && !load.expert_parallel.enabled();
    const auto graph_weights = info.graph_weights == models::GraphWeights::resident
        ? ::tyloi::weights::Cache::Graph::legacy : ::tyloi::weights::Cache::Graph::native;
    const bool graph_allowed = info.decode_graph && !executor.attention_options().kernels.decode_aten &&
        model_parallel_cuda_graph_enabled(load, executor, weights) &&
        weights.graph_safe(graph_weights, moe_parallel_config(load).enabled());
    runtime.decode_graph.enabled &= graph_allowed;
    runtime.continuous_batch.cuda_graph &= graph_allowed;
}

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
    std::unique_ptr<models::Ffn> ffn;
    engine::Config run_config;
    models::Info run_info{};
    models::ForwardOptions forward{};
    std::unique_ptr<tyloi::state::Sessions> sessions;
    std::optional<engine::Decode> graph;
    std::optional<engine::Budget> budget;
    std::unique_ptr<engine::BatchDriver> batch_driver;
    engine::mtp::State mtp_policy;
    bool attempted = false;
    bool run_attempted = false, run_ready = false, running = false;

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
        batch_driver.reset();
        budget.reset();
        graph.reset();
        sessions.reset();
        state.reset();
        layers.reset();
        groups.reset();
        ffn.reset();
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

void Program::load_ffn(int layer) {
    auto& p = *impl_;
    if (p.attempted || !p.executor) throw std::logic_error("program load already attempted");
    p.attempted = true;
    p.open_source();
    auto scope = p.executor->enter(p.config.layer_placement.primary_device());
    ::tyloi::weights::LoadScope memory(*p.ledger);
    if (p.offload) throw std::invalid_argument("FFN execution does not support streamed KV");
    auto input = assembly(p.options, p.config, *p.executor, *p.weights, *p.source, false);
    p.ffn = models::load_ffn(p.source, input.definition, input.loader,
        p.config.layer_placement, input.policy, layer);
    p.ffn->prepare();
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
models::Ffn& Program::ffn() { if (!impl_->ffn) throw std::logic_error("FFN not loaded"); return *impl_->ffn; }

void Program::configure(engine::Config config) {
    auto& p = *impl_;
    if (!p.model || !p.state) throw std::logic_error("model not loaded");
    if (p.running) throw std::logic_error("program request is already running");
    if (p.run_attempted && !p.run_ready) throw std::logic_error("generation configuration failed");
    p.run_attempted = true;
    p.run_ready = false;
    drain();
    auto scope = p.executor->enter(p.config.layer_placement.primary_device());
    p.state->release_graphs();
    p.budget.reset();
    p.graph.reset();
    p.sessions.reset();
    ::tyloi::weights::LoadScope memory(*p.ledger);
    p.run_config = std::move(config);
    auto& runtime = p.run_config;
    p.run_info = p.model->info();
    if (auto* predictor = p.model->mtp(); predictor &&
        (!p.state->predictor || !p.state->predictor->belongs_to(*predictor)))
        throw std::invalid_argument("predictor state is unavailable or belongs to another model");
    internal::configure_generation(runtime, p.config, *p.executor, *p.weights, p.run_info);
    const auto mode = p.executor->forward_mode(p.config.layer_placement.primary_device());
    p.forward = {mode.kv_capacity, mode.trace, mode.trace_layer, mode.primary};
    p.sessions = internal::make_sessions(*p.source, runtime.session_cache, runtime.prefix_cache,
        p.run_info.max_context, p.state->supports_text_session_state(), p.state->supports_paged_text_session_state());
    p.graph.emplace(p.run_info.max_context, p.executor->device(), p.executor->graph());
    p.budget.emplace(*p.ledger, *p.weights, *p.sessions, *p.executor);
    p.budget->bind(p.model->mtp() != nullptr);
    if (runtime.resident_budget_bytes) {
        if (!runtime.online_budget)
            throw std::invalid_argument("online memory budgeting requires a single-device CUDA worker");
        engine::SessionCommand command;
        command.kind = engine::SessionCommand::Kind::memory_budget;
        command.bytes = runtime.resident_budget_bytes;
        auto change = p.budget->change(command);
        p.weights->settle();
        p.graph->invalidate();
        p.state->release_graphs();
        auto* predictor = p.model->mtp();
        if (predictor && (!p.state->predictor || !p.state->predictor->belongs_to(*predictor)))
            throw std::invalid_argument("predictor state is unavailable or belongs to another model");
        for (const auto& [device, bytes] : p.state->reset_growth(1))
            if (bytes) p.budget->reserve_kv(device, engine::prefill_budget_add(bytes, std::size_t(64) << 20));
        p.state->reset(1);
        if (predictor) {
            for (const auto& [device, bytes] : predictor->reset_growth(*p.state->predictor, 1))
                if (bytes) p.budget->reserve_kv(device, engine::prefill_budget_add(bytes, std::size_t(64) << 20));
            predictor->reset(*p.state->predictor, 1);
        }
        (void)change.apply();
    }
    p.run_ready = true;
}

Program::BatchLease::BatchLease(Program& owner) : owner_(owner) {}

Program::BatchLease::~BatchLease() noexcept {
    try { owner_.drain(); }
    catch (const std::exception& error) { std::cerr << "batch_cleanup error=" << error.what() << '\n'; }
    catch (...) { std::cerr << "batch_cleanup error=unknown\n"; }
    std::unique_ptr<tyloi::Executor::Scope> scope;
    try { scope = owner_.executor().enter(owner_.config().layer_placement.primary_device()); }
    catch (const std::exception& error) { std::cerr << "batch_cleanup error=" << error.what() << '\n'; }
    catch (...) { std::cerr << "batch_cleanup error=unknown\n"; }
    batch_.reset();
    owner_.impl_->running = false;
}

engine::Batch& Program::BatchLease::batch() {
    if (!batch_) throw std::logic_error("batch is not available");
    return *batch_;
}

std::unique_ptr<Program::BatchLease> Program::make_batch() {
    auto& p = *impl_;
    if (p.running) throw std::logic_error("program request is already running");
    if (!p.run_ready) configure(engine::resolve_config());
    auto lease = std::unique_ptr<BatchLease>(new BatchLease(*this));
    p.running = true;
    auto scope = p.executor->enter(p.config.layer_placement.primary_device());
    p.graph->invalidate();
    p.state->release_graphs();
    lease->batch_ = engine::make_batch(*p.model, p.state.get(), *p.executor,
        *p.weights, *p.budget, p.run_config);
    return lease;
}

engine::BatchDriver& Program::begin_batch(models::BatchConfig config) {
    auto& p=*impl_;
    if(p.running)throw std::logic_error("program request is already running");
    if(!p.run_ready)configure(engine::resolve_config());
    drain();
    p.running=true;
    try {
        auto scope=p.executor->enter(p.config.layer_placement.primary_device());
        p.graph->invalidate();
        p.state->release_graphs();
        p.batch_driver=std::make_unique<engine::BatchDriver>(
            p.model->batch(config,p.state.get()),*p.executor,*p.weights,*p.budget);
        p.batch_driver->initialize();
        return *p.batch_driver;
    } catch(...) {
        auto failure=std::current_exception();
        try { drain(); } catch(...) {}
        p.batch_driver.reset(); p.running=false;
        std::rethrow_exception(failure);
    }
}

void Program::end_batch() {
    auto& p=*impl_;
    if(!p.running || !p.batch_driver)throw std::logic_error("numerical batch is not active");
    drain();
    auto scope=p.executor->enter(p.config.layer_placement.primary_device());
    p.batch_driver->close(); p.batch_driver.reset(); p.running=false;
}

models::Prefill Program::prefill(models::Input input, std::int64_t compute, engine::InferenceOutput& output) {
    auto& p = *impl_;
    if (p.running) throw std::logic_error("program request is already running");
    if (!p.run_ready) configure(engine::resolve_config());
    if (compute <= 0) throw std::invalid_argument("prefill compute chunk must be positive");
    p.running = true;
    models::Prefill result;
    try {
        result = engine::prefill(*p.model, *p.state, *p.executor, *p.weights,
            *p.budget, p.forward, input, compute, output);
        p.running = false;
        return result;
    } catch (...) {
        auto failure = std::current_exception();
        try { drain(); }
        catch (const std::exception& error) { std::cerr << "prefill_cleanup error=" << error.what() << '\n'; }
        catch (...) { std::cerr << "prefill_cleanup error=unknown\n"; }
        p.running = false;
        std::rethrow_exception(failure);
    }
}

StepSequence<engine::EventData> Program::generate(engine::InferenceRequest& request,
        engine::InferenceOutput& output, bool predictor, std::string id) {
    auto& p = *impl_;
    if (p.running) throw std::logic_error("program request is already running");
    if (!p.run_ready) configure(engine::resolve_config());
    p.running = true;
    RequestWork work{*this, {}, p.running};
    work.job = engine::generate({*p.model, *p.state, *p.executor, *p.budget, *p.graph, *p.weights,
        *p.sessions, p.run_info, p.run_config, p.forward, p.source.get(), p.mtp_policy,
        predictor ? p.model->mtp() : nullptr}, request, output, std::move(id));
    while (auto step = work.job.next()) co_yield std::move(step);
    work.complete = true;
}

Program::Stats Program::stats() const {
    const auto& p = *impl_;
    return {p.mtp_policy.stats, p.mtp_policy.cycles, p.mtp_policy.accepted, p.mtp_policy.rejected,
        p.graph ? p.graph->captures : 0, p.graph ? p.graph->reuses : 0};
}

void Program::drain() {
    auto& p = *impl_;
    if (!p.executor || !p.weights) throw std::logic_error("program closed");
    drain_cuda(*p.executor, *p.weights, p.offload.get(), p.config.layer_placement,
        p.config.tensor_parallel, p.config.expert_parallel);
}
} // namespace tyloi::cuda
