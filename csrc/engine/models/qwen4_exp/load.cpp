#include "engine/models/qwen4_exp/ops.h"

#include "engine/backends/cuda/executor/execution.h"
#include "engine/backends/cuda/native/scope.h"
#include "engine/models/common/check.h"
#include "engine/models/common/load.h"
#include "engine/models/common/loader.h"
#include "engine/models/common/model_config.h"
#include "engine/models/common/model_graph.h"
#include "engine/models/source.h"
#include "engine/weights/cache.h"

#include <algorithm>
#include <iostream>
#include <utility>

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
    auto tensors = execution.weights.loader(*model.source, execution.config.weights,
        {execution.config.layer_placement.primary_device(), false, execution.config.drop_file_cache,
         execution.config.tensor_parallel, moe_parallel_config(execution.config)});
    detail::ModelLoader<Model> weights{model, tensors};
    model.load_weights(weights, load_blocks);
    if (!defer_moe_cache_finalize)
        execution.weights.finalize();
}

} // namespace tyloi::cuda::qwen4_exp
