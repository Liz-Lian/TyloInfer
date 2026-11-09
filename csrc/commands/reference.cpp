#include "cli.h"
#include "run.h"
#include "engine/backends/cuda/native/scope.h"
#include "engine/backends/cuda/executor/execution.h"
#include "engine/models/common/components.h"
#include "engine/models/common/load.h"
#include "engine/models/source/model_graph.h"
#include "engine/weights/loader.h"
#include "engine/weights/moe/expert_cache.h"
#include "engine/models/qwen35/load.h"
#include "engine/models/qwen35/forward.h"
#include "engine/models/gemma4/load.h"
#include "engine/models/gemma4/forward.h"
#include "engine/generation/ops.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace tyloi::cuda::commands {
namespace {

template <typename Model>
Model load_reference_model(CudaExecutionContext& execution,
                           const CudaLoadOptions& options,
                           const std::shared_ptr<const tyloi::ModelSource>& source) {
    if constexpr (std::is_same_v<Model, tyloi::cuda::qwen35::Model>) {
        CudaScope scope(execution.config.layer_placement.primary_device(), execution.executor().contexts());
        tyloi::cuda::weight_loader::Loader weights{*source, execution.config.weights,
            {execution.config.layer_placement.primary_device(), false, execution.config.drop_file_cache,
             execution.config.tensor_parallel, moe_parallel_config(execution.config)},
            execution.weights, execution.executor().tq()};
        auto model = tyloi::cuda::qwen35::load(
            source, weights, execution.config.layer_placement, execution.config.yarn,
            execution.config.dense_cpu_layer_count,
            tyloi::cuda::qwen35::LoadOptions{
                .config_path = options.config_path,
                .context_size = options.context_size,
                .gpu_layers = execution.config.n_gpu_layers,
                .extend_yarn = execution.config.yarn_context_extension,
                .expert_parallel = execution.config.expert_parallel.enabled(),
                .mirror_scalars = execution.config.tensor_parallel_mirror_linear_attention_scalars,
                .load_layers = false,
                .load_input_output = false});
        if (moe_expert_cache_has_sources(execution.weights.moe) &&
            !moe_expert_cache_finalized(execution.weights.moe))
            finalize_moe_expert_cache(execution.weights.moe);
        return model;
    } else {
        CudaScope scope(execution.config.layer_placement.primary_device(), execution.executor().contexts());
        auto definition = tyloi::models::config_json(*source, options.config_path);
        definition = configure_yarn(execution.config.yarn, execution.config.yarn_context_extension,
            std::move(definition), tyloi::models::graph(*source).backbone, options.context_size);
        auto model = tyloi::cuda::gemma4::read(source, definition, options.context_size);
        tyloi::cuda::weight_loader::Loader weights{*source, execution.config.weights,
            {execution.config.layer_placement.primary_device(), false, execution.config.drop_file_cache,
             execution.config.tensor_parallel, moe_parallel_config(execution.config)},
            execution.weights, execution.executor().tq()};
        execution.config.dense_cpu_layer_count = 0;
        tyloi::cuda::gemma4::load(model, weights, execution.config.layer_placement,
            execution.config.n_gpu_layers, execution.config.expert_parallel.enabled(), false, false);
        if (moe_expert_cache_has_sources(execution.weights.moe) &&
            !moe_expert_cache_finalized(execution.weights.moe))
            finalize_moe_expert_cache(execution.weights.moe);
        return model;
    }
}

template <typename Model>
int reference_num_layers(const Model& model) {
    return static_cast<int>(model.config.num_hidden_layers);
}

template <typename Model>
std::string_view reference_layer_type(const Model& model, int index) {
    return model.config.layer_types.at(static_cast<std::size_t>(index));
}

template <typename Model> struct ReferenceLayers {
    Model& model;
    CudaExecutionContext& execution;
    const tyloi::cuda::weight_loader::Loader& weights;
    bool resident = false;
    using Layer = typename std::remove_reference_t<decltype(model.layers)>::value_type;
    using LayerState = std::conditional_t<std::is_same_v<Model, tyloi::cuda::qwen35::Model>,
                                          tyloi::cuda::qwen35::LayerState,
#ifdef TYLOINFER_NATIVE_CUDA
                                          tyloi::state::KvView>;
#else
                                          KVCache>;
#endif
    std::optional<Layer> streamed;
    LayerState state;
    std::unique_ptr<tyloi::cuda::Executor> executor;

    Layer load_layer(int index) {
        const int device = execution.config.layer_placement.device_for_layer(index);
        const bool cpu = index < execution.config.dense_cpu_layer_count;
        auto placed = weights.for_layer(device, cpu);
        if constexpr (std::is_same_v<Model, tyloi::cuda::qwen35::Model>)
            return tyloi::cuda::qwen35::load_layer(placed, model.config, index,
                std::string(reference_layer_type(model, index)),
                execution.config.tensor_parallel_mirror_linear_attention_scalars);
        else
            return tyloi::cuda::gemma4::load_layer(placed, model.config, index,
                std::string(reference_layer_type(model, index)));
    }

    bool load_resident() {
        model.layers.reserve(static_cast<std::size_t>(reference_num_layers(model)));
        for (int index = 0; index < reference_num_layers(model); ++index)
            model.layers.push_back(load_layer(index));
        resident = true;
        return true;
    }

    void load_streamed(int index) {
        streamed = load_layer(index);
    }

    void reset(int index) {
        if constexpr (std::is_same_v<Model, tyloi::cuda::qwen35::Model>) {
            const auto& layer = resident ? model.layers.at(static_cast<std::size_t>(index))
                                         : *streamed;
            if (std::holds_alternative<tyloi::cuda::qwen35::Full>(layer.weights))
                state.template emplace<KVCache>();
            else
                state.template emplace<tyloi::cuda::qwen35::LinearState>();
        } else {
            state = {};
        }
        if (!executor) executor = execution.executor().fork(model_parallel_primary_device(execution.config));
    }

    ::tyloi::DenseTensor forward(int index, ::tyloi::DenseTensor input, const ::tyloi::DenseTensor& positions) {
#ifdef TYLOINFER_NATIVE_CUDA
        auto scope = executor->enter();
#endif
        if constexpr (std::is_same_v<Model, tyloi::cuda::qwen35::Model>) {
            const auto& layer = resident ? model.layers.at(static_cast<std::size_t>(index))
                                         : *streamed;
            const auto* rope = &model.rope;
            if (layer.cpu) rope = &model.cpu_rope;
            else if (!model.device_ropes.empty()) {
                const auto it = model.device_ropes.find(layer.device);
                if (it != model.device_ropes.end()) rope = &it->second;
            }
            const tyloi::cuda::AttentionStep step{positions, mfq_nullopt, mfq_nullopt,
                                                mfq_nullopt, 0, 0, 0};
            return tyloi::cuda::qwen35::forward_layer(
                execution.executor().collectives(), layer, state, *executor, 0, std::move(input), step,
                0, *rope, execution.executor().forward_mode(execution.config.layer_placement.primary_device()));
        } else {
            const auto& layer = resident ? model.layers.at(static_cast<std::size_t>(index))
                                         : *streamed;
            MfqCudaGuard guard(layer.device);
            const tyloi::cuda::AttentionStep step{
                execution.executor().collectives().copy(positions, layer.device), mfq_nullopt, mfq_nullopt,
                mfq_nullopt, 0, 0, 0};
#ifdef TYLOINFER_NATIVE_CUDA
            auto hidden = tyloi::cuda::gemma4::forward_layer(layer, state,
                execution.executor().collectives().copy(std::move(input), layer.device), step,
                tyloi::cuda::forward_options(execution.executor().forward_mode(execution.config.layer_placement.primary_device())));
            return execution.executor().collectives().copy(
                std::move(hidden), execution.config.layer_placement.primary_device());
#else
            auto hidden = tyloi::cuda::gemma4::forward_layer(layer, state, *executor, 0,
                execution.executor().collectives().copy(std::move(input), layer.device), step, execution.executor().forward_mode(execution.config.layer_placement.primary_device()));
            return execution.executor().collectives().copy(std::move(hidden), execution.config.layer_placement.primary_device());
#endif
        }
    }

    void release() {
        model.layers.clear();
        streamed.reset();
        if constexpr (std::is_same_v<Model, tyloi::cuda::qwen35::Model>)
            state.template emplace<KVCache>();
        else
            state = {};
    }

    ::tyloi::DenseTensor finalize(::tyloi::DenseTensor hidden, std::int64_t batch, std::int64_t tokens) {
#ifdef TYLOINFER_NATIVE_CUDA
        auto scope = executor->enter();
#endif
        if constexpr (std::is_same_v<Model, tyloi::cuda::qwen35::Model>)
            return tyloi::cuda::qwen35::normalize(model, *executor,
                                                std::move(hidden), batch, tokens);
        else
#ifdef TYLOINFER_NATIVE_CUDA
            return ::tyloi::DenseTensor(model.normalize(std::move(hidden), batch, tokens));
#else
            return tyloi::cuda::gemma4::normalize(model, *executor, std::move(hidden), batch, tokens);
#endif
    }

    ::tyloi::DenseTensor softcap(::tyloi::DenseTensor logits) {
#ifdef TYLOINFER_NATIVE_CUDA
        auto scope = executor->enter();
#endif
        if constexpr (std::is_same_v<Model, tyloi::cuda::qwen35::Model>)
            return tyloi::cuda::qwen35::softcap(model, *executor, std::move(logits));
        else
#ifdef TYLOINFER_NATIVE_CUDA
            return ::tyloi::DenseTensor(model.softcap(std::move(logits)));
#else
            return tyloi::cuda::gemma4::softcap(model, *executor, std::move(logits));
#endif
    }
};

::tyloi::DenseTensor rows(const tyloi::ModelSource& source, const std::string& name,
                std::int64_t begin, std::int64_t count) {
    const auto* record = source.find_tensor(name);
    if (!record || !record->shape || record->shape->size() != 2)
        throw std::runtime_error("reference requires a matrix: " + name);
    const auto& shape = *record->shape;
    if (begin < 0 || count < 1 || begin > shape[0] - count)
        throw std::runtime_error("reference matrix row range is invalid");
    ::tyloi::ScalarType dtype;
    std::size_t bytes;
    if (record->dtype == "F16") { dtype = ::tyloi::kFloat16; bytes = 2; }
    else if (record->dtype == "BF16") { dtype = ::tyloi::kBFloat16; bytes = 2; }
    else if (record->dtype == "F32") { dtype = ::tyloi::kFloat32; bytes = 4; }
    else throw std::runtime_error("reference embedding/output matrix requires a floating source: " + name);
    auto result = ::tyloi::cuda::empty({count, shape[1]}, ::tyloi::TensorOptions().device(::tyloi::kCPU).dtype(dtype));
    source.read_range_into(name, sizeof(std::uint32_t) + 2 * sizeof(std::int64_t) + begin * shape[1] * bytes,
                          reinterpret_cast<std::byte*>(result.data_ptr()), count * shape[1] * bytes);
    return result;
}

template<class Model>
int generate(CudaExecutionContext& execution, const CudaLoadOptions& options,
             const std::filesystem::path& input, const std::filesystem::path& output,
             bool resident) {
    std::ifstream tokens_file(input, std::ios::binary);
    std::uint32_t geometry[2]{};
    tokens_file.read(reinterpret_cast<char*>(geometry), sizeof(geometry));
    const auto chunks = geometry[0], context = geometry[1];
    if (!tokens_file || !chunks || context < 32 || context != options.context_size ||
        std::uint64_t(chunks) * context > std::uint64_t(std::numeric_limits<std::int32_t>::max()))
        throw std::runtime_error("invalid native reference token geometry");
    std::vector<std::int32_t> tokens(std::size_t(chunks) * context);
    tokens_file.read(reinterpret_cast<char*>(tokens.data()), tokens.size() * sizeof(tokens[0]));
    if (!tokens_file || tokens_file.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("invalid native reference token payload");
    if (std::filesystem::exists(output)) throw std::runtime_error("reference scratch output already exists");
    auto source = tyloi::open_model_source(options.model_path);
    auto model = load_reference_model<Model>(execution, options, source);
    const auto& source_data = *source;
    const auto width = model.config.hidden_size;
    const auto vocab = model.config.vocab_size;
    for (auto token : tokens) if (token < 0 || token >= vocab)
        throw std::runtime_error("reference token is outside the source vocabulary");
    const auto hidden_path = output.string() + ".hidden";
    std::fstream hidden(hidden_path, std::ios::in | std::ios::out | std::ios::binary | std::ios::trunc);
    if (!hidden) throw std::runtime_error("cannot create reference hidden-state scratch");
    const auto hidden_bytes = std::uint64_t(context) * width * sizeof(float);
    std::vector<float> buffer(std::size_t(context) * width);
    auto store = [&](std::uint32_t chunk, ::tyloi::DenseTensor value) {
        auto cpu = value.to(::tyloi::kCPU, ::tyloi::kFloat32).contiguous();
        hidden.seekp(chunk * hidden_bytes);
        hidden.write(reinterpret_cast<const char*>(cpu.template data_ptr<float>()), hidden_bytes);
        hidden.flush();
        if (!hidden) throw std::runtime_error("cannot write reference hidden-state scratch");
    };
    auto restore = [&](std::uint32_t chunk) {
        hidden.seekg(chunk * hidden_bytes);
        hidden.read(reinterpret_cast<char*>(buffer.data()), hidden_bytes);
        if (!hidden) throw std::runtime_error("cannot read reference hidden-state scratch");
        return ::tyloi::cuda::from_blob(buffer.data(), {1, context, width}, ::tyloi::TensorOptions().device(::tyloi::kCPU).dtype(::tyloi::kFloat32))
            .to(::tyloi::kCUDA, ::tyloi::kFloat16).contiguous();
    };
    for (std::uint32_t chunk = 0; chunk < chunks; ++chunk) {
        auto embedded = ::tyloi::cuda::empty({1, context, width}, ::tyloi::TensorOptions().device(::tyloi::kCPU).dtype(::tyloi::kFloat16));
        for (std::uint32_t position = 0; position < context; ++position) {
            auto row = rows(source_data, "model.token_embedding.weight",
                            tokens[std::size_t(chunk) * context + position], 1)
                           .to(::tyloi::kFloat16);
            embedded.narrow(1, position, 1).copy_(row.reshape({1, 1, width}));
        }
        if constexpr (std::is_same_v<Model, tyloi::cuda::qwen35::Model>)
            store(chunk, embedded.to(::tyloi::kCUDA, ::tyloi::kFloat16).contiguous());
        else
            store(chunk, embedded.to(::tyloi::kCUDA) * model.embed_scale);
    }
    auto positions = ::tyloi::cuda::arange(0, context, ::tyloi::TensorOptions().device(::tyloi::kCUDA).dtype(::tyloi::kInt64));
    tyloi::cuda::weight_loader::Loader weights{source_data, execution.config.weights,
        {execution.config.layer_placement.primary_device(), false, execution.config.drop_file_cache,
         execution.config.tensor_parallel, moe_parallel_config(execution.config)},
        execution.weights, execution.executor().tq()};
    ReferenceLayers<Model> layers{model, execution, weights};
    if (resident) {
        try {
            resident = layers.load_resident();
        } catch (const std::exception& error) {
            const std::string message = error.what();
            if (message.find("out of memory") == std::string::npos &&
                message.find("memory allocation") == std::string::npos) throw;
            layers.release();
            mfq_cuda_empty_cache();
            resident = false;
            std::cout << "{\"event\":\"source_loading\",\"layerwise\":true,\"requested_layerwise\":false,"
                         "\"fallback_reason\":\"insufficient_memory\"}" << std::endl;
        }
    }
    for (int index = 0; index < reference_num_layers(model); ++index) {
        if (!resident) layers.load_streamed(index);
        for (std::uint32_t chunk = 0; chunk < chunks; ++chunk) {
            layers.reset(index);
            auto value = layers.forward(index, restore(chunk), positions);
            store(chunk, value);
        }
        mfq_cuda_synchronize();
        if (!resident) layers.release();
        if (!resident) mfq_cuda_empty_cache();
        std::cout << "{\"event\":\"wt2_layer\",\"layer\":" << index + 1
                  << ",\"layers\":" << reference_num_layers(model) << "}" << std::endl;
    }
    layers.release();
    mfq_cuda_empty_cache();
    const auto head = !model.config.tie_word_embeddings && source_data.find_tensor("model.output.weight")
        ? "model.output.weight" : "model.token_embedding.weight";
    std::ofstream logits(output, std::ios::binary);
    if (!logits) throw std::runtime_error("cannot create reference logits scratch");
    const std::int64_t first = context / 2, scored = context - first - 1;
    for (std::uint32_t chunk = 0; chunk < chunks; ++chunk) {
        auto normalized = layers.finalize(restore(chunk), 1, context);
        for (std::int64_t begin = 0; begin < scored; begin += 32) {
            const auto count = std::min<std::int64_t>(32, scored - begin);
            auto value = normalized.narrow(1, first + begin, count).reshape({count, width}).to(::tyloi::kFloat16);
            std::vector<float> row_logits(count * vocab);
            for (std::int64_t row = 0; row < vocab; row += 4096) {
                const auto rows_count = std::min<std::int64_t>(4096, vocab - row);
                auto weight = rows(source_data, head, row, rows_count).to(::tyloi::kCUDA, ::tyloi::kFloat16);
                auto result = layers.softcap(::tyloi::cuda::matmul(value, weight.transpose(0, 1)))
                    .to(::tyloi::kCPU, ::tyloi::kFloat32).contiguous();
                for (std::int64_t token = 0; token < count; ++token)
                    std::memcpy(row_logits.data() + token * vocab + row,
                                result.template data_ptr<float>() + token * rows_count, rows_count * sizeof(float));
            }
            logits.write(reinterpret_cast<const char*>(row_logits.data()), row_logits.size() * sizeof(float));
            if (!logits) throw std::runtime_error("cannot write reference logits scratch");
        }
        std::cout << "{\"event\":\"wt2_head\",\"chunk\":" << chunk + 1
                  << ",\"chunks\":" << chunks << "}" << std::endl;
    }
    logits.close();
    if (!logits) throw std::runtime_error("cannot finalize reference logits scratch");
    hidden.close();
    std::filesystem::remove(hidden_path);
    return 0;
}
}

int run_reference(int argc, char** argv) {
    return internal::with_command_errors([&] {
        ArgCursor args(argc, argv, 2);
        CudaLoadOptions options;
        std::filesystem::path input, output;
        bool resident = false;
        while (!args.empty()) {
            const auto option = args.next();
            if (parse_cuda_load_option(option, args, options)) continue;
            if (option == "--reference-input") input = args.value(option);
            else if (option == "--reference-output") output = args.value(option);
            else if (option == "--resident") resident = true;
            else usage_error("unknown native reference option: " + std::string(option));
        }
        if (options.model_path.empty() || input.empty() || output.empty() || options.context_size < 32)
            usage_error("native reference requires --model, --ctx-size, --reference-input and --reference-output");
        if (!std::filesystem::is_directory(options.model_path))
            usage_error("native reference requires an original HF model directory");
        std::shared_ptr<tyloi::cuda::KvMemoryManager> kv;
        tyloi::cuda::RoutedCache weights;
        std::shared_ptr<tyloi::cuda::KvOffloadStore> offload;
        auto [execution_config, numeric_options] = tyloi::cuda::load_cuda_execution_config();
        auto executor = std::make_unique<tyloi::cuda::Executor>(std::move(numeric_options), 0);
        CudaExecutionContext execution(*executor, std::move(execution_config), weights, kv, offload);
        CudaScope scope(execution.config.layer_placement.primary_device(), execution.executor().contexts());
        internal::setup_cuda_load(options, execution.config, *executor, weights, kv, offload);
        ::tyloi::cuda::NoGradGuard guard;
        auto source = tyloi::open_model_source(options.model_path);
        const auto backbone = tyloi::models::graph(*source).backbone;
        if (backbone == "qwen3_5" || backbone == "generic_qwen")
            return generate<tyloi::cuda::qwen35::Model>(execution, options, input, output, resident);
        if (backbone == "gemma4") return generate<tyloi::cuda::gemma4::Model>(execution, options, input, output, resident);
        throw std::runtime_error("native reference architecture is unsupported: " + backbone);
    });
}
} // namespace tyloi::cuda::commands
