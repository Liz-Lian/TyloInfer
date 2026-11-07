#include "engine/models/source.h"
#include "engine/backends/cuda/executor/execution.h"
#include "engine/backends/cuda/native/scope.h"
#include "engine/models/common/load.h"
#include "engine/models/common/model_config.h"
#include "engine/models/common/model_graph.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/weights/loader.h"
#include "engine/weights/cache_internal.h"
#include "engine/models/common/components.h"
#include "engine/models/qwen4_exp/predictor.h"
#include "engine/models/glm5_next/mtp.h"
#include "engine/models/glm5_next/forward.h"
#include "engine/models/nemotron_h/mtp.h"
#include "engine/models/deepseek_v41/dspark.h"
#include "engine/models/deepseek_v41/ops.h"
#include "engine/models/deepseek_v41/vision.h"
#include "engine/models/common/check.h"

#include "engine/weights/moe/expert_cache.h"

#include <algorithm>
#include <iostream>
#include <utility>

namespace {

// Allocation, packing, placement and device lifetime are native. The shared
// causal model supplies weight relationships and layer traversal.
template <class Model> struct ModelLoader {
    Model &model;
    const tyloi::weights::Loader& weights;
    auto embedding(const std::string& name) {
        return weights.embedding(name,
                              model.has_independent_output_weights(*this));
    }
    auto output(const std::string &name) {
        return weights.weight(name, TensorParallelAxis::Output);
    }
    auto tied_output(const std::shared_ptr<const ::tyloi::Tensor> &embedding, const std::string &name) {
        return weights.placement.tensor.enabled() ? output(name) : embedding;
    }
    bool has_weight(const std::string &name) { return weights.has(name); }
    void final_state(::tyloi::DenseTensor &norm) {
        model.adapter_load_final_state(weights, norm);
    }
    void prepare_blocks() { model.adapter_prepare_blocks(weights); }
    auto block(int layer, const std::string &type) {
        auto &execution = *model.execution;
        const int device = execution.config.layer_placement.device_for_layer(layer);
        const bool cpu_offloaded = layer < execution.config.dense_cpu_layer_count;
        MfqCudaGuard layer_guard(device);
        std::cerr << "loading layer " << layer << ' ' << type << ' '
                  << (cpu_offloaded ? "CPU" : "CUDA") << std::endl;
        auto placed = weights.for_layer(device, cpu_offloaded);
        auto block = model.adapter_load_block(placed, layer, device, type);
        block->cuda_device = device;
        block->cpu_offloaded = cpu_offloaded;
        return block;
    }
};

} // namespace

template <typename Model>
RuntimeComponents<Model> load_runtime_components(
        Model& model,
        const tyloi::weights::Loader& weights,
        bool load_optional_components) {
    RuntimeComponents<Model> result{model.graph, model.plan};
    if (load_optional_components &&
            (result.plan.vision != tyloi::cuda::CudaVisionAdapter::none ||
             result.plan.predictor != tyloi::cuda::CudaPredictorAdapter::none)) {
        throw std::runtime_error(
            "CUDA component adapter is unsupported for this model");
    }
    return result;
}

template <>
RuntimeComponents<tyloi::cuda::qwen4_exp::Model> load_runtime_components(
        tyloi::cuda::qwen4_exp::Model& model,
        const tyloi::weights::Loader& weights,
        bool load_optional_components) {
    RuntimeComponents<tyloi::cuda::qwen4_exp::Model> result{model.graph, model.plan};
    if (!load_optional_components) return result;
    if (result.plan.vision == tyloi::cuda::CudaVisionAdapter::grid_vit) {
        const auto document = nlohmann::json::parse(
            tyloi::models::config_json(*model.source));
        const auto& vision = document.at("vision_config");
        if (vision.value("hidden_act", std::string("gelu_pytorch_tanh")) != "gelu_pytorch_tanh" ||
                !vision.value("deepstack_visual_indexes", std::vector<int>{}).empty())
            throw std::runtime_error("unsupported Qwen4 grid-ViT activation or deepstack");
        tyloi::GridVisionConfig config;
        config.hidden_size = vision.at("hidden_size").get<int64_t>();
        config.intermediate_size = vision.at("intermediate_size").get<int64_t>();
        config.depth = vision.at("depth").get<int64_t>();
        config.num_heads = vision.at("num_heads").get<int64_t>();
        config.in_channels = vision.value("in_channels", int64_t{3});
        config.patch_size = vision.at("patch_size").get<int64_t>();
        config.temporal_patch_size = vision.at("temporal_patch_size").get<int64_t>();
        config.spatial_merge_size = vision.value("spatial_merge_size", int64_t{2});
        config.out_hidden_size = vision.at("out_hidden_size").get<int64_t>();
        config.num_position_embeddings = vision.at("num_position_embeddings").get<int64_t>();
        config.rope_theta = vision.value("rope_theta", 10000.0);
        if (vision.contains("rope_parameters"))
            config.rope_theta = vision.at("rope_parameters").value("rope_theta", config.rope_theta);
        config.layer_norm_eps = vision.value("layer_norm_eps", 1e-6);
        config.validate();
        const auto image = document.at("image_token_id").get<int64_t>();
        const auto video = document.at("video_token_id").get<int64_t>();
        if (config.out_hidden_size != model.config.hidden || model.config.sections.size() != 3 ||
                image < 0 || video < 0 || image >= model.config.vocab || video >= model.config.vocab || image == video)
            throw std::runtime_error("invalid Qwen4 grid-ViT text geometry or placeholder IDs");
        const auto* component = result.graph.component("vision");
        result.grid_vision.emplace(
            tyloi::cuda::grid_vision_runtime::CudaGridVisionPromptComponent::load(
                weights, model.execution->weights, config, image, video,
                component->input_contract, component->position_policy));
        result.vision_available = true;
    } else if (result.plan.vision != tyloi::cuda::CudaVisionAdapter::none) {
        throw std::runtime_error("unsupported Qwen4 CUDA vision adapter");
    }
    if (result.plan.predictor != tyloi::cuda::CudaPredictorAdapter::none &&
             result.plan.predictor !=
                 tyloi::cuda::CudaPredictorAdapter::flash_next) {
        throw std::runtime_error(
            "unsupported Qwen4 CUDA component adapter");
    }
    if (result.plan.predictor ==
            tyloi::cuda::CudaPredictorAdapter::none) return result;
    TYLOINFER_RUNTIME_CHECK(
        model.supports_speculation(),
        "invalid Flash-Next predictor model");
    auto predictor = tyloi::cuda::qwen4_exp::Predictor::load_if_present(
        *model.execution, weights, model.config, model.embed, model.lm_head,
        model.metadata.final_logit_softcapping);
    if (predictor) {
        result.mtp =
            std::make_unique<tyloi::cuda::qwen4_exp::Predictor>(
                std::move(*predictor));
    }
    result.mtp_available = static_cast<bool>(result.mtp);
    return result;
}

RuntimeComponents<tyloi::cuda::glm5_next::Model> load_runtime_components(
        tyloi::cuda::glm5_next::Model& model,
        const tyloi::weights::Loader& weights,
        bool load_optional_components, tyloi::cuda::Projector projector,
        tyloi::cuda::Collectives& copies, CudaProfiler& profiler) {
    RuntimeComponents<tyloi::cuda::glm5_next::Model> result{model.graph, model.plan};
    if (!load_optional_components) return result;
    if (result.plan.vision != tyloi::cuda::CudaVisionAdapter::none ||
            (result.plan.predictor != tyloi::cuda::CudaPredictorAdapter::none &&
             result.plan.predictor !=
                 tyloi::cuda::CudaPredictorAdapter::flash_next)) {
        throw std::runtime_error(
            "unsupported GLM5 CUDA component adapter");
    }
    if (result.plan.predictor ==
            tyloi::cuda::CudaPredictorAdapter::none) return result;
    TYLOINFER_RUNTIME_CHECK(
        tyloi::cuda::glm5_next::supports_speculation(model),
        "invalid Flash-Next predictor model");
    auto predictor = tyloi::cuda::glm5_next::Mtp::load_if_present(
        weights, std::move(projector), copies, profiler, model.embed, model.config);
    if (predictor) {
        result.mtp =
            std::make_unique<tyloi::cuda::glm5_next::Mtp>(
                std::move(*predictor));
    }
    result.mtp_available = static_cast<bool>(result.mtp);
    return result;
}

namespace tyloi::cuda::deepseek_v41 {

using tyloi::models::require_model;

void read(Model& model, std::shared_ptr<const tyloi::ModelSource> model_source,
          std::string_view definition, std::int64_t context_size) {
    model.source = std::move(model_source);
    const auto& source = *model.source;
    validate_model_source(source);
    model.graph = tyloi::models::graph(source);
    model.plan = cuda_model_plan(model.graph);
    require_model(Model::accepts_backbone(model.graph.backbone),
                  "model backbone does not match the requested causal model");
    model.load_config(definition);
    require_model(model.num_hidden_layers() == model.graph.topology.text_layers,
                  "model graph/config text-layer topology mismatch");
    if (context_size > 0) {
        require_model(context_size <= model.max_position_embeddings(),
                      "--ctx-size exceeds max_position_embeddings");
        model.config.max_position_embeddings = context_size;
        model.metadata.max_position_embeddings = context_size;
    }
}

void load(Model& model, CudaExecutionContext& execution, int64_t context_size,
          bool defer_moe_cache_finalize, bool load_blocks, bool load_optional_components) {
    auto definition = tyloi::models::config_json(*model.source, {});
    definition = configure_yarn(execution.config.yarn, execution.config.yarn_context_extension,
        std::move(definition), model.graph.backbone, context_size);
    read(model, model.source, definition, context_size);
    model.execution = &execution;
    model.adapter_validate_load_options();
    if (execution.config.expert_parallel.enabled() && model.metadata.num_experts <= 0)
        throw std::runtime_error("--expert-parallel requires a model with routed experts");
    execution.config.layer_placement.prepare(model.num_hidden_layers());
    execution.config.dense_cpu_layer_count = 0;
    if (execution.config.n_gpu_layers >= 0 && execution.config.n_gpu_layers < model.num_hidden_layers())
        throw std::runtime_error("--n-gpu-layers currently supports dense Qwen-style blocks only");
    CudaScope model_guard(execution.config.layer_placement.primary_device(), execution.executor().contexts());
    auto tensors = execution.weights.loader(*model.source, execution.config.weights,
        {execution.config.layer_placement.primary_device(), false, execution.config.drop_file_cache,
         execution.config.tensor_parallel, moe_parallel_config(execution.config)});
    ModelLoader<Model> weights{model, tensors};
    model.load_weights(weights, load_blocks);
    if (moe_expert_cache_has_sources(::tyloi::weights::detail::cache(execution.weights).moe) &&
        !moe_expert_cache_finalized(::tyloi::weights::detail::cache(execution.weights).moe) && !defer_moe_cache_finalize)
        finalize_moe_expert_cache(::tyloi::weights::detail::cache(execution.weights).moe);
    if (load_optional_components)
        load_runtime_components(model, tensors, true);
}

} // namespace tyloi::cuda::deepseek_v41

namespace tyloi::cuda::qwen4_exp {

using tyloi::models::require_model;

void read(Model& model, std::shared_ptr<const tyloi::ModelSource> model_source,
          std::string_view definition, std::int64_t context_size) {
    model.source = std::move(model_source);
    const auto& source = *model.source;
    validate_model_source(source);
    model.graph = tyloi::models::graph(source);
    model.plan = cuda_model_plan(model.graph);
    require_model(Model::accepts_backbone(model.graph.backbone),
                  "model backbone does not match the requested causal model");
    model.load_config(definition);
    require_model(model.num_hidden_layers() == model.graph.topology.text_layers,
                  "model graph/config text-layer topology mismatch");
    if (context_size > 0) {
        require_model(context_size <= model.max_position_embeddings(),
                      "--ctx-size exceeds max_position_embeddings");
        model.config.maximum = context_size;
        model.metadata.max_position_embeddings = context_size;
    }
}

void load(Model& model, CudaExecutionContext& execution, std::int64_t context_size,
          bool defer_moe_cache_finalize, bool load_blocks) {
    auto definition = tyloi::models::config_json(*model.source, {});
    definition = configure_yarn(execution.config.yarn, execution.config.yarn_context_extension,
        std::move(definition), model.graph.backbone, context_size);
    read(model, model.source, definition, context_size);
    model.execution = &execution;
    model.adapter_validate_load_options();
    if (execution.config.expert_parallel.enabled() && model.metadata.num_experts <= 0)
        throw std::runtime_error("--expert-parallel requires a model with routed experts");
    execution.config.layer_placement.prepare(model.num_hidden_layers());
    execution.config.dense_cpu_layer_count = 0;
    if (execution.config.n_gpu_layers >= 0) {
        execution.config.dense_cpu_layer_count = static_cast<int>(
            std::max<std::int64_t>(model.num_hidden_layers() - execution.config.n_gpu_layers, 0));
        if (execution.config.dense_cpu_layer_count > 0) {
            if (model_parallel_enabled(execution.config) || execution.config.layer_placement.enabled())
                throw std::runtime_error(
                    "--n-gpu-layers cannot be combined with tensor/expert/layer parallelism");
            std::cerr << "dense_layer_placement cpu=0-" << (execution.config.dense_cpu_layer_count - 1)
                      << " gpu=" << execution.config.dense_cpu_layer_count << '-'
                      << (model.num_hidden_layers() - 1) << " cpu_threads=" << mfq_get_num_threads()
                      << std::endl;
        }
    }
    CudaScope model_guard(execution.config.layer_placement.primary_device(), execution.executor().contexts());
    if (model.plan.vision == CudaVisionAdapter::grid_vit && execution.config.dense_cpu_layer_count > 0)
        throw std::runtime_error("CUDA grid-Vision currently requires GPU-resident text layers");
    if (model.adapter_uses_common_rope() && !model.metadata.no_rope) {
        auto make_rope = [&](::tyloi::Device device) {
            RopeCache rope(model.max_position_embeddings(), model.metadata.rotary_dim,
                           model.metadata.rope_base, 0, -1, device, model.metadata.rope_interleaved,
                           execution.config.yarn);
            model.adapter_configure_rope(rope, device);
            return rope;
        };
        const auto primary = ::tyloi::Device(::tyloi::kCUDA,
            execution.config.layer_placement.primary_device());
        model.rope = make_rope(primary);
        if (execution.config.dense_cpu_layer_count > 0)
            model.cpu_rope = make_rope(::tyloi::Device(::tyloi::kCPU));
        if (execution.config.layer_placement.enabled()) {
            for (int device : execution.config.layer_placement.devices) {
                MfqCudaGuard rope_guard(device);
                model.device_ropes.emplace(device, make_rope(::tyloi::Device(::tyloi::kCUDA, device)));
            }
        }
    }
    auto tensors = execution.weights.loader(*model.source, execution.config.weights,
        {execution.config.layer_placement.primary_device(), false, execution.config.drop_file_cache,
         execution.config.tensor_parallel, moe_parallel_config(execution.config)});
    ModelLoader<Model> weights{model, tensors};
    model.load_weights(weights, load_blocks);
    if (moe_expert_cache_has_sources(::tyloi::weights::detail::cache(execution.weights).moe) &&
        !moe_expert_cache_finalized(::tyloi::weights::detail::cache(execution.weights).moe) && !defer_moe_cache_finalize)
        finalize_moe_expert_cache(::tyloi::weights::detail::cache(execution.weights).moe);
}

} // namespace tyloi::cuda::qwen4_exp

RuntimeComponents<tyloi::cuda::deepseek_v41::Model> load_runtime_components(
        tyloi::cuda::deepseek_v41::Model& model,
        const tyloi::weights::Loader& weights,
        bool load_optional_components) {
    using tyloi::cuda::deepseek_v41::Model;
    RuntimeComponents<Model> result{model.graph, model.plan};
    if (!load_optional_components) return result;
    if ((result.plan.vision != tyloi::cuda::CudaVisionAdapter::none &&
         result.plan.vision != tyloi::cuda::CudaVisionAdapter::deepseek_v41) ||
            (result.plan.predictor != tyloi::cuda::CudaPredictorAdapter::none &&
             result.plan.predictor !=
                 tyloi::cuda::CudaPredictorAdapter::deepseek_v41_dspark)) {
        throw std::runtime_error(
            "unsupported DeepSeek-V4.1 CUDA component adapter");
    }
    if(result.plan.vision==tyloi::cuda::CudaVisionAdapter::deepseek_v41) {
        model.vision = std::make_unique<tyloi::cuda::deepseek_v41_runtime::Vision>(
            tyloi::cuda::deepseek_v41_runtime::Vision::load(model.source, weights, model.execution->weights, model.config));
        result.vision_available=true;
    }
    if (result.plan.predictor ==
            tyloi::cuda::CudaPredictorAdapter::none) return result;
    TYLOINFER_RUNTIME_CHECK(
        model.supports_suffix_speculation() && model.shared,
        "invalid DeepSeek-V4.1 DSpark model");
    result.mtp =
        tyloi::cuda::deepseek_v41_runtime::load_dspark_if_present(
            *model.execution, weights, model.shared->config, model.embed, model.lm_head,
            model.metadata.final_logit_softcapping);
    result.mtp_available = static_cast<bool>(result.mtp);
    return result;
}

RuntimeComponents<tyloi::cuda::nemotron_h::Model> load_runtime_components(
        tyloi::cuda::nemotron_h::Model& model,
        const tyloi::weights::Loader& weights,
        bool load_optional_components,
        CudaExecutionContext& execution) {
    RuntimeComponents<tyloi::cuda::nemotron_h::Model> result{model.graph, model.plan};
    if (!load_optional_components) return result;
    if ((result.plan.vision != tyloi::cuda::CudaVisionAdapter::none &&
         result.plan.vision != tyloi::cuda::CudaVisionAdapter::radio) ||
            (result.plan.predictor != tyloi::cuda::CudaPredictorAdapter::none &&
             result.plan.predictor != tyloi::cuda::CudaPredictorAdapter::nemotron_h)) {
        throw std::runtime_error(
            "unsupported Nemotron-H CUDA component adapter");
    }
    if (result.plan.vision == tyloi::cuda::CudaVisionAdapter::radio) {
        const auto* component = result.graph.component("vision");
        TYLOINFER_RUNTIME_CHECK(component != nullptr,
            "Nemotron-H graph declares a RADIO adapter without a vision component");
        const auto vision = tyloi::models::nemotron_h::RadioVisionConfig::from_json(
            tyloi::models::config_json(*model.source));
        result.radio_vision.emplace(
            tyloi::cuda::nemotron_h::CudaRadioPromptComponent::load(
                weights, vision, component->input_contract,
                component->position_policy));
        result.vision_available = true;
    }
    if (result.plan.predictor ==
            tyloi::cuda::CudaPredictorAdapter::none) return result;
    TYLOINFER_RUNTIME_CHECK(tyloi::cuda::nemotron_h::supports_speculation(model),
        "invalid Nemotron-H predictor model");
    auto predictor = tyloi::cuda::nemotron_h::Mtp::load_if_present(
        execution, weights, model.config, model.embed);
    // The graph declares the component, so missing weights are a broken
    // container rather than a text-only model.
    TYLOINFER_RUNTIME_CHECK(predictor,
        "Nemotron-H graph declares the predictor component but the source has no MTP weights");
    result.mtp = std::make_unique<tyloi::cuda::nemotron_h::Mtp>(
        std::move(*predictor));
    result.mtp_available = true;
    return result;
}

RuntimeComponents<tyloi::cuda::minicpmo45::text::Model>
load_runtime_components(
        tyloi::cuda::minicpmo45::text::Model& model,
        tyloi::cuda::minicpmo45::text::State& state,
        tyloi::cuda::Executor& executor,
        const tyloi::weights::Loader& weights,
        bool load_optional_components, CudaExecutionContext& execution) {
    RuntimeComponents<tyloi::cuda::minicpmo45::text::Model> result{model.graph, model.plan};
    if (!load_optional_components ||
            result.plan.vision == tyloi::cuda::CudaVisionAdapter::none) {
        return result;
    }
    if (result.plan.vision !=
            tyloi::cuda::CudaVisionAdapter::minicpmo45) {
        throw std::runtime_error(
            "unsupported MiniCPM-o CUDA vision adapter");
    }

    result.composite = std::make_unique<tyloi::cuda::minicpmo45::Components>(
        model, state, executor, weights, execution);
    result.vision_available = true;
    return result;
}

template <>
RuntimeComponents<tyloi::cuda::minicpmo45::Model>
load_runtime_components(
        tyloi::cuda::minicpmo45::Model& model,
        const tyloi::weights::Loader& weights,
        bool) {
    RuntimeComponents<tyloi::cuda::minicpmo45::Model> result{model.graph, model.plan};
    return result;
}

#define TYLOINFER_INSTANTIATE_COMPONENTS(TYPE)                                  \
    template RuntimeComponents<TYPE> load_runtime_components(TYPE&, const tyloi::weights::Loader&, bool)

#ifndef TYLOINFER_NATIVE_CUDA
TYLOINFER_INSTANTIATE_COMPONENTS(tyloi::cuda::gemma4::Model);
#endif
TYLOINFER_INSTANTIATE_COMPONENTS(tyloi::cuda::glm_dsa::Model);
TYLOINFER_INSTANTIATE_COMPONENTS(tyloi::cuda::deepseek_v4::Model);

#undef TYLOINFER_INSTANTIATE_COMPONENTS
