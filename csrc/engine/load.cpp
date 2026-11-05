#include "engine/load.h"
#include "engine/ops.h"
#include "engine/models/load.h"
#include "engine/models/deepseek_v41/vision.h"
#include "engine/models/batch.h"
#include "engine/models/qwen35/forward.h"
#include "engine/models/minicpmo45/forward.h"
#include "engine/models/minicpmo45/text/forward.h"
#include "engine/models/gemma4/forward.h"
#include "engine/models/glm_dsa/forward.h"
#include "engine/models/deepseek_v4/forward.h"
#include "engine/models/glm5_next/forward.h"
#include "engine/models/nemotron_h/forward.h"
#include "engine/models/common/model_graph.h"
#include "engine/engine_impl.h"
#include "engine/instance_impl.h"
#include "engine/drain.h"
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/impl.h"
#include "engine/models/common/model_config.h"
#include "engine/models/gemma4/load.h"
#endif

#include "engine/backends/cuda/native/env.h"
#include "engine/backends/cuda/native/collectives.h"
#include "engine/weights/moe/expert_cache.h"
#include "engine/state/kv_memory.h"
#include "engine/backends/cuda/native/profile.h"
#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/weights/profile.h"
#include "engine/weights/profile_path.h"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace tyloi::cuda::engine_detail {

using State = std::variant<
    std::unique_ptr<Text<minicpmo45::text::Model, minicpmo45::text::State>>,
    std::unique_ptr<Text<minicpmo45::Model, minicpmo45::State>>,
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
    std::unique_ptr<Text<gemma4::Model, gemma4::State>>,
#endif
    std::unique_ptr<Text<glm_dsa::Model, glm_dsa::State>>,
    std::unique_ptr<Text<glm5_next::Model, glm5_next::State>>,
    std::unique_ptr<Text<qwen4_exp::Model, qwen4_exp::State>>,
    std::unique_ptr<Text<deepseek_v4::Model, deepseek_v4::State>>,
    std::unique_ptr<Text<deepseek_v41::Model, deepseek_v41::State>>,
    std::unique_ptr<Text<nemotron_h::Model, nemotron_h::State>>>;

} // namespace tyloi::cuda::engine_detail

namespace tyloi::cuda {

bool model_parallel_enabled(const CudaLoadConfig& config) {
    return config.tensor_parallel.enabled() || config.expert_parallel.enabled();
}

const ParallelConfig& model_parallel_config(const CudaLoadConfig& config) {
    return config.tensor_parallel.enabled() ? config.tensor_parallel : config.expert_parallel;
}

const ParallelConfig& moe_parallel_config(const CudaLoadConfig& config) {
    return config.expert_parallel.enabled() ? config.expert_parallel : config.tensor_parallel;
}

int model_parallel_primary_device(const CudaLoadConfig& config) {
    if (!config.tensor_parallel.devices.empty()) return config.tensor_parallel.primary_device();
    if (!config.expert_parallel.devices.empty()) return config.expert_parallel.primary_device();
    return 0;
}

bool model_parallel_cuda_graph_enabled(const CudaLoadConfig& config,
        const Executor& executor, const RoutedCache& weights) {
    if (weights.tq && moe_parallel_config(config).enabled()) return false;
    if (!config.tensor_parallel.enabled() && !config.expert_parallel.enabled()) return true;
    const bool enabled = config.expert_parallel.enabled() && !config.tensor_parallel.enabled()
        ? config.expert_parallel_cuda_graph : config.tensor_parallel_cuda_graph;
    return executor.collectives().enabled() && enabled;
}

namespace {

auto load_qwen(
    std::shared_ptr<const ModelSource> source,
    const weight_loader::Loader& weights,
    LayerPlacementConfig& placement,
    tyloi::ops::YarnScaling& yarn,
    int& cpu_layers,
    const CudaEngineOptions& options,
    int gpu_layers,
    bool extend_yarn,
    bool expert_parallel,
    bool mirror_scalars) {
    return std::make_shared<qwen35::Model>(qwen35::load(
        source, weights, placement, yarn, cpu_layers,
        qwen35::LoadOptions{
            .config_path = options.config_path,
            .context_size = options.context_size,
            .gpu_layers = gpu_layers,
            .extend_yarn = extend_yarn,
            .expert_parallel = expert_parallel,
            .mirror_scalars = mirror_scalars}));
}

} // namespace

std::unique_ptr<engine::Engine> load_cuda_engine(CudaEngineOptions options) {
    auto source = tyloi::open_model_source(options.model_path);
    const auto backbone = cuda_backbone(tyloi::models::graph(*source).backbone);
    if (backbone == CudaBackbone::generic_qwen) {
        return std::make_unique<CudaEngine<
            qwen35::Model, qwen35::State, qwen35::Components,
            load_qwen>>(std::move(options), std::move(source));
    }
    source.reset();
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    if (backbone == CudaBackbone::gemma4) {
        auto target = std::unique_ptr<engine::Engine>(
            new engine::Engine(internal::build_engine_parts(std::move(options))));
        target->install();
        target->start();
        return target;
    }
#endif
    return std::make_unique<engine_detail::EngineInstance<
        engine_detail::State, with_model<false>>>(std::move(options));
}

void reload_cuda_engine(engine::Engine& engine, CudaEngineOptions options) {
    if (auto* target = dynamic_cast<CudaEngine<qwen35::Model, qwen35::State, qwen35::Components, load_qwen>*>(&engine)) {
        target->options_ = std::move(options);
        target->source_ = tyloi::open_model_source(target->options_.model_path);
        target->load();
        return;
    }
    if (auto* target = dynamic_cast<engine_detail::EngineInstance<engine_detail::State, with_model<false>>*>(&engine)) {
        target->options = std::move(options);
        target->load();
        return;
    }
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    if (typeid(engine) == typeid(tyloi::engine::Engine)) {
        engine.rebuild(internal::build_engine_parts(std::move(options)));
        return;
    }
#endif
    throw std::invalid_argument("unsupported CUDA engine for reload");
}

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
engine::EngineParts internal::build_engine_parts(CudaEngineOptions options) {
    std::shared_ptr<const ModelSource> source;
    std::shared_ptr<WeightMemoryLedger> ledger;
    auto weights = std::make_unique<RoutedCache>();
    std::shared_ptr<KvMemoryManager> kv;
    std::shared_ptr<KvOffloadStore> offload;
    CudaLoadConfig load;
    CudaRuntimeConfig runtime;
    std::unique_ptr<Executor> root;
    std::shared_ptr<models::CausalModel> model;
    std::unique_ptr<models::State> state;
    std::unique_ptr<Executor> forward;
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
            root->profiler().enabled = false;
            ledger = std::make_shared<WeightMemoryLedger>();
            WeightMemoryLoadScope memory_load(*ledger);
            runtime = internal::resolve_cuda_runtime_config(options);
            const auto kv_options = runtime_options::qsa_kv_offload();
            kv = std::make_shared<KvMemoryManager>();
            if (kv_options.gpu_budget_bytes) {
                kv.reset();
                KvOffloadConfig config;
                config.gpu_budget_bytes = kv_options.gpu_budget_bytes;
                config.ram_budget_bytes = internal::cuda_has_separate_memory() ? kv_options.ram_budget_bytes : 0;
                config.buffer_bytes = std::min<std::size_t>(config.buffer_bytes, config.gpu_budget_bytes / 4);
                config.directory = kv_options.directory;
                offload = std::make_shared<KvOffloadStore>(std::move(config));
            }
            {
                auto scope = root->enter(load.layer_placement.primary_device());
                source = tyloi::open_model_source(options.model_path);
                auto definition = models::config_json(*source, options.config_path);
                definition = configure_yarn(load.yarn, load.yarn_context_extension,
                    std::move(definition), models::graph(*source).backbone, options.context_size);
                auto owner = std::make_shared<gemma4::Model>(gemma4::read(source, definition, options.context_size));
                model = owner;
                weight_loader::Loader loader{*source, load.weights,
                    {load.layer_placement.primary_device(), false, load.drop_file_cache,
                     load.tensor_parallel, moe_parallel_config(load)}, *weights, root->tq()};
                load.dense_cpu_layer_count = 0;
                gemma4::load(*owner, loader, load.layer_placement, load.n_gpu_layers, load.expert_parallel.enabled());
                state = model->make_state();
                if (runtime_options::kv_vram_preallocated()) state->preallocate();
                forward = root->fork(load.layer_placement.primary_device());
                if (owner->plan.vision != CudaVisionAdapter::none || owner->plan.predictor != CudaPredictorAdapter::none)
                    throw std::runtime_error("CUDA component adapter is unsupported for this model");
                finalize_kv(kv.get(), *weights, offload);
                finish_load(kv.get(), *weights, 0, load.report_cuda_memory);
                if (offload) throw std::invalid_argument("CUDA streamed KV requires a QSA model");
            }
        }
        auto impl = std::make_unique<engine::Engine::Impl>();
        const auto mode = root->forward_mode(load.layer_placement.primary_device());
        impl->forward_options_ = {mode.kv_capacity, mode.trace, mode.trace_layer, mode.primary};
        impl->graph_allowed_ = model_parallel_cuda_graph_enabled(load, *root, *weights);
        impl->options_ = std::move(options);
        impl->load_ = std::move(load);
        impl->runtime_ = std::move(runtime);
        impl->source_ = std::move(source);
        impl->ledger_ = std::move(ledger);
        impl->weights_ = std::move(weights);
        impl->kv_ = std::move(kv);
        impl->offload_ = std::move(offload);
        impl->idle_state_ = std::move(state);
        impl->forward_ = std::move(forward);
        return engine::EngineParts{std::move(model), std::move(root), std::move(impl)};
    } catch (...) {
        const auto failure = std::current_exception();
        if (root) {
            try { drain_cuda(*root, *weights, offload.get(), load.layer_placement, load.tensor_parallel, load.expert_parallel); }
            catch (const std::exception& error) { std::cerr << "runtime_load_cleanup error=" << error.what() << '\n'; }
            catch (...) { std::cerr << "runtime_load_cleanup error=unknown\n"; }
            const auto release = [&] {
                forward.reset(); state.reset(); model.reset(); root->pipelines().clear();
                offload.reset(); kv.reset(); weights.reset();
            };
            try {
                auto scope = root->enter(load.layer_placement.primary_device());
                release();
            } catch (const std::exception& error) {
                std::cerr << "runtime_load_cleanup error=" << error.what() << '\n';
                release();
            } catch (...) {
                std::cerr << "runtime_load_cleanup error=unknown\n";
                release();
            }
            root.reset();
        }
        std::rethrow_exception(failure);
    }
}
#endif

} // namespace tyloi::cuda

namespace tyloi::cuda::internal {
using tyloi::engine::environment_uint64;
using tyloi::engine::environment_enabled;
namespace {

int strict_int(const std::string& text, const char* option) {
    int result = 0;
    const auto [end, error] = std::from_chars(
        text.data(), text.data() + text.size(), result);
    if (error != std::errc{} || end != text.data() + text.size()) {
        throw std::runtime_error(std::string(option) + " requires integers");
    }
    return result;
}

double strict_double(const std::string& text, const char* option) {
    double result = 0.0;
    const auto [end, error] = std::from_chars(
        text.data(), text.data() + text.size(), result,
        std::chars_format::general);
    if (error != std::errc{} || end != text.data() + text.size() ||
            !std::isfinite(result)) {
        throw std::runtime_error(
            std::string(option) + " requires finite numbers");
    }
    return result;
}

std::int32_t graph_minimum(const char* name) {
    const auto value = std::max<std::uint64_t>(
        2, environment_uint64(name, 16));
    if (value > static_cast<std::uint64_t>(
            std::numeric_limits<std::int32_t>::max())) {
        throw std::runtime_error(std::string(name) + " exceeds int32");
    }
    return static_cast<std::int32_t>(value);
}

} // namespace

CudaRuntimeConfig resolve_cuda_runtime_config(
        const CudaEngineOptions& options) {
    CudaRuntimeConfig config;
    if(options.prefill_chunk_size < 0 || options.prefill_chunk_size > 8192)
        throw std::invalid_argument("prefill chunk size must be Auto (0) or in [1, 8192]");
    config.prefill_chunk_auto = options.prefill_chunk_size == 0;
    const auto chunk_limit = config.prefill_chunk_auto ? int64_t(8192) : options.prefill_chunk_size;
    static_cast<tyloi::engine::RuntimeConfig&>(config) =
        tyloi::engine::resolve_runtime_config(chunk_limit);
    static_cast<tyloi::engine::ContinuousBatchConfig&>(config.continuous_batch) =
        tyloi::engine::resolve_batch_config(options.continuous_batching, chunk_limit);
    config.decode_graph.enabled = environment_enabled(
        "TYLOINFER_RUNTIME_CUDA_GRAPH", true);
    config.decode_graph.trace = environment_enabled(
        "TYLOINFER_RUNTIME_TRACE_CUDA_GRAPH", false);
    config.decode_graph.minimum_generation_tokens = graph_minimum(
        "TYLOINFER_RUNTIME_CUDA_GRAPH_MIN_TOKENS");

    config.continuous_batch.greedy = environment_enabled(
        "TYLOINFER_CONTINUOUS_BATCH_GREEDY", true);
    config.continuous_batch.cuda_graph =
        config.decode_graph.enabled && environment_enabled(
            "TYLOINFER_CONTINUOUS_BATCH_CUDA_GRAPH", true);
    config.continuous_batch.paged_kv = environment_enabled(
        "TYLOINFER_CONTINUOUS_PAGED_KV", true);
    config.continuous_batch.cuda_graph_minimum_tokens = graph_minimum(
        "TYLOINFER_CONTINUOUS_BATCH_CUDA_GRAPH_MIN_TOKENS");
    return config;
}

static std::vector<std::string> split_csv_values(
        const std::string & value,
        const char * option) {
    std::vector<std::string> result;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        const auto first = item.find_first_not_of(" \t\r\n");
        const auto last = item.find_last_not_of(" \t\r\n");
        if (first == std::string::npos) {
            throw std::runtime_error(
                std::string(option) + " contains an empty item");
        }
        result.push_back(item.substr(first, last - first + 1));
    }
    if (result.empty()) {
        throw std::runtime_error(
            std::string(option) +
            " requires at least one value");
    }
    if (value.ends_with(',')) {
        throw std::runtime_error(
            std::string(option) + " contains an empty item");
    }
    return result;
}

static ParallelConfig parse_parallel_config(
        const std::string & devices_arg,
        const std::string & split_arg,
        const char * devices_option,
        const char * split_option,
        int available_devices,
        bool allow_duplicate_devices) {
    ParallelConfig config;
    if (devices_arg.empty()) {
        if (!split_arg.empty()) {
            throw std::runtime_error(
                std::string(split_option) + " requires " + devices_option);
        }
        return config;
    }

    const auto values = split_csv_values(
        devices_arg, devices_option);
    if (values.size() == 1 &&
            devices_arg.find(',') == std::string::npos) {
        const int count = strict_int(values.front(), devices_option);
        if (count < 2) {
            throw std::runtime_error(
                std::string(devices_option) +
                " device count must be at least two");
        }
        config.devices.resize(static_cast<size_t>(count));
        std::iota(config.devices.begin(), config.devices.end(), 0);
    } else {
        for (const auto & item : values) {
            config.devices.push_back(strict_int(item, devices_option));
        }
        if (config.devices.size() < 2) {
            throw std::runtime_error(
                std::string(devices_option) +
                " requires at least two devices");
        }
    }
    config.allow_duplicate_devices = allow_duplicate_devices;

    std::unordered_set<int> unique_devices;
    for (const int device : config.devices) {
        if (device < 0 || device >= available_devices) {
            throw std::runtime_error(
                std::string(devices_option) +
                " CUDA device is unavailable: " +
                std::to_string(device));
        }
        const bool inserted = unique_devices.insert(device).second;
        if (!allow_duplicate_devices && !inserted) {
            throw std::runtime_error(
                std::string(devices_option) +
                " CUDA devices must be unique");
        }
    }

    if (!split_arg.empty()) {
        for (const auto & item :
             split_csv_values(split_arg, split_option)) {
            const double weight = strict_double(item, split_option);
            if (!std::isfinite(weight) || weight <= 0.0) {
                throw std::runtime_error(
                    std::string(split_option) +
                    " values must be finite and positive");
            }
            config.split.push_back(weight);
        }
        if (config.split.size() != config.devices.size()) {
            throw std::runtime_error(
                std::string(split_option) + " count must match " +
                devices_option + " devices");
        }
    }
    return config;
}

static void print_parallel_config(
        const char * label,
        const ParallelConfig & config) {
    if (!config.enabled()) return;
    std::cerr << label << " devices=";
    for (size_t index = 0; index < config.devices.size(); ++index) {
        if (index) std::cerr << ',';
        std::cerr << config.devices[index];
    }
    if (!config.split.empty()) {
        std::cerr << " split=";
        for (size_t index = 0; index < config.split.size(); ++index) {
            if (index) std::cerr << ',';
            std::cerr << config.split[index];
        }
    }
    std::cerr << '\n';
}

static void configure_model_parallel(
        CudaLoadConfig& config, Collectives& collectives,
        const std::string & tensor_devices_arg,
        const std::string & tensor_split_arg,
        const std::string & expert_devices_arg,
        const std::string & expert_split_arg,
        bool allow_duplicate_devices = false) {
    collectives.reset();
    config.tensor_parallel = {};
    config.expert_parallel = {};

    int available = 0;
    TYLOINFER_CUDA_CHECK(cudaGetDeviceCount(&available));
    config.tensor_parallel = parse_parallel_config(
        tensor_devices_arg, tensor_split_arg,
        "--tensor-parallel", "--tensor-split",
        available, allow_duplicate_devices);
    config.expert_parallel = parse_parallel_config(
        expert_devices_arg, expert_split_arg,
        "--expert-parallel", "--expert-split",
        available, allow_duplicate_devices);

    if (config.tensor_parallel.enabled() &&
            config.expert_parallel.enabled() &&
            config.tensor_parallel.devices !=
                config.expert_parallel.devices) {
        throw std::runtime_error(
            "combined tensor and expert parallelism requires the same "
            "ordered CUDA device group");
    }
    if (!config.tensor_parallel.enabled() &&
            !config.expert_parallel.enabled()) {
        TYLOINFER_CUDA_CHECK(cudaSetDevice(0));
        return;
    }

    const auto& group = config.tensor_parallel.enabled()
        ? config.tensor_parallel : config.expert_parallel;
    Collectives::enable_peer_access(group.devices);
    collectives.configure(
        group.devices, allow_duplicate_devices);
    TYLOINFER_CUDA_CHECK(cudaSetDevice(group.primary_device()));
    print_parallel_config(
        "tensor_parallel", config.tensor_parallel);
    print_parallel_config(
        "expert_parallel", config.expert_parallel);
    std::cerr << "model_parallel ranks=" << group.devices.size()
              << " collective_backend="
              << (collectives.enabled()
                  ? "nccl" : "serial")
              << '\n';
}

static void configure_layer_placement(
        CudaLoadConfig& config,
        const std::string & devices_arg,
        const std::string & split_arg) {
    auto& placement = config.layer_placement;
    placement = {};
    if (devices_arg.empty()) {
        if (!split_arg.empty()) {
            throw std::runtime_error(
                "--layer-split requires --layer-parallel");
        }
        placement.devices = {
            model_parallel_primary_device(config)};
        return;
    }
    if (config.tensor_parallel.enabled() ||
            config.expert_parallel.enabled()) {
        throw std::runtime_error(
            "--layer-parallel cannot be combined with tensor/expert parallelism");
    }

    int available = 0;
    TYLOINFER_CUDA_CHECK(cudaGetDeviceCount(&available));
    auto parsed = parse_parallel_config(devices_arg, split_arg,
        "--layer-parallel", "--layer-split", available, false);
    TYLOINFER_CUDA_CHECK(cudaSetDevice(parsed.primary_device()));
    print_parallel_config("layer_parallel", parsed);
    placement.devices = std::move(parsed.devices);
    placement.split = std::move(parsed.split);
}

static std::unordered_set<int> parse_layer_ranges(
        const std::string & value) {
    std::unordered_set<int> result;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        item.erase(
            std::remove_if(
                item.begin(), item.end(),
                [](unsigned char ch) { return std::isspace(ch) != 0; }),
            item.end());
        if (item.empty()) {
            throw std::runtime_error(
                "--cpu-offload-layers contains an empty item");
        }
        const size_t dash = item.find('-');
        int first = 0;
        int last = 0;
        if (dash == std::string::npos) {
            first = last = strict_int(item, "--cpu-offload-layers");
        } else {
            if (dash == 0 || dash + 1 >= item.size() ||
                item.find('-', dash + 1) != std::string::npos) {
                throw std::runtime_error(
                    "invalid --cpu-offload-layers range: " + item);
            }
            first = strict_int(
                item.substr(0, dash), "--cpu-offload-layers");
            last = strict_int(
                item.substr(dash + 1), "--cpu-offload-layers");
        }
        if (first < 0 || last < first) {
            throw std::runtime_error(
                "invalid --cpu-offload-layers range: " + item);
        }
        for (int layer = first; layer <= last; ++layer) {
            result.insert(layer);
        }
    }
    if (result.empty()) {
        throw std::runtime_error(
            "--cpu-offload-layers requires at least one layer");
    }
    return result;
}

void setup_cuda_load(
        const tyloi::cuda::CudaLoadOptions& options,
        CudaLoadConfig& config, Executor& executor, RoutedCache& weights,
        std::shared_ptr<KvMemoryManager>& kv, std::shared_ptr<KvOffloadStore>& offload) {
    drain_cuda(executor, weights, offload.get(),
               config.layer_placement, config.tensor_parallel, config.expert_parallel);
    executor.pipelines().clear();
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
    executor.attention_scratch().clear();
#endif
    executor.branches().clear();
    executor.profiler().reset();
    executor.collectives().reset();
    config.tensor_parallel = {};
    config.expert_parallel = {};
    config.layer_placement = {};
    config.dsv4_cpu_offload_layers.clear();
    config.dsv4_cpu_offload_host_bytes = 0;
    config.n_gpu_layers = -1;
    config.dense_cpu_layer_count = 0;
    config.drop_file_cache = false;
    executor.mode().ffn.serial = false;
    executor.mode().ffn.projection_major = false;
    executor.mode().ffn.batch_serial = false;
    executor.routes().clear();
    executor.mmq().reset();
    executor.mode().ffn.unfused = false;
    executor.mode().ffn.materialized = false;
    config.weights.min_slots = 8;
    executor.mode().trace_layer = -1;
    executor.mode().trace = nullptr;
    offload.reset();
    kv.reset();
    weights.reset();
    weights.host_bytes = env_size("TYLOINFER_MOE_HOST_CACHE_BYTES", 0);
    weights.host_limit = env_size("TYLOINFER_MOE_HOST_PHYSICAL_BYTES", 0);
    if (options.n_gpu_layers_set) config.n_gpu_layers = options.n_gpu_layers;
    if (options.cpu_threads_set && options.cpu_threads <= 0) {
        throw std::runtime_error("--threads must be positive");
    }
    if (options.cpu_threads > 0) {
        mfq_set_num_threads(options.cpu_threads);
    }
    config.yarn_context_extension=options.yarn_context_extension;
    configure_model_parallel(
        config, executor.collectives(), options.tensor_parallel_arg, options.tensor_split_arg,
        options.expert_parallel_arg, options.expert_split_arg,
        options.parallel_test_duplicates);
    configure_layer_placement(
        config, options.layer_parallel_arg, options.layer_split_arg);
    if (!options.cpu_offload_layers_arg.empty()) {
        config.dsv4_cpu_offload_layers =
            parse_layer_ranges(options.cpu_offload_layers_arg);
        std::vector<int> ordered(
            config.dsv4_cpu_offload_layers.begin(),
            config.dsv4_cpu_offload_layers.end());
        std::sort(ordered.begin(), ordered.end());
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
    if (options.n_gpu_layers_set && !options.cpu_offload_layers_arg.empty()) {
        throw std::runtime_error(
            "--n-gpu-layers cannot be combined with --cpu-offload-layers");
    }
    if (options.moe_gpu_cache_gb < 0.0 ||
            !std::isfinite(options.moe_gpu_cache_gb)) {
        throw std::runtime_error(
            "--moe-gpu-cache-gb must be finite and non-negative");
    }
    if (options.moe_gpu_cache_gb > 0.0 &&
            !options.cpu_offload_layers_arg.empty()) {
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
        weights.device_bytes = static_cast<std::size_t>(bytes);
        weights.moe =
            make_moe_expert_cache(
                static_cast<int64_t>(bytes), MoeCacheOptions{
                .host_bytes = weights.host_bytes,
                .host_limit = weights.host_limit,
                .mapped = config.moe_mapped_gather,
                .copy_blocks = config.moe_mapped_copy_blocks,
                .io_workers = config.moe_ssd_io_workers,
                .overlap = config.moe_ssd_overlap,
                .pipeline = config.moe_pipeline,
                .hybrid_cpu = config.moe_hybrid_cpu,
                .preload = config.weights.preload,
                .assert_resident = config.moe_assert_resident,
                .ram_pcie = config.moe_ram_pcie,
                .direct_ram = config.moe_direct_ram,
                .adapt = config.moe_residency_adapt,
                .warm = config.moe_residency_warm,
                .projection_heat = config.moe_residency_projection_heat,
                .ram_fraction = config.moe_ram_pcie_fraction,
                .two_stage = executor.options().moe_ffn.two_stage,
                .heterogeneous = executor.options().projection.moe.heterogeneous});
        if (!profile_path.empty() && weights.moe) {
            set_moe_expert_cache_profile(*weights.moe,tyloi::load_moe_cache_profile(profile_path));
            std::cerr<<"moe_cache_profile source="<<(options.moe_cache_profile_path.empty()?"sidecar":"explicit")
                <<" path="<<profile_path<<std::endl;
        }
        config.drop_file_cache = true;
    }
}

} // namespace tyloi::cuda::internal

namespace tyloi::cuda {

void finalize_kv(KvMemoryManager* kv, RoutedCache& weights,
        std::shared_ptr<KvOffloadStore>& offload) {
    if (kv) {
        kv->finalize(moe_expert_cache_minimum_bytes(weights.moe));
        set_moe_kv_expert_loan(weights.moe, kv->expert_loan());
        if (auto store = kv->offload()) offload = std::move(store);
    }
}

std::chrono::steady_clock::time_point finish_load(
        KvMemoryManager* kv, RoutedCache& weights, std::size_t graph_reserve,
        bool report_memory) {
    if (moe_expert_cache_has_sources(weights.moe) &&
            !moe_expert_cache_finalized(weights.moe)) {
        finalize_moe_expert_cache(weights.moe,
            (kv ? kv->growth_reserve() : 0) + graph_reserve,
            kv ? kv->expert_budget_charge() : 0);
    }
    mfq_cuda_synchronize();
    const auto loaded = std::chrono::steady_clock::now();
    report_cuda_memory(report_memory, "loaded");
    return loaded;
}

} // namespace tyloi::cuda
