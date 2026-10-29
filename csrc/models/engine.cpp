#include "models/load.h"
#include "models/batch.h"
#include "models/qwen35/forward.h"
#include "models/minicpmo45/forward.h"
#include "engine/executor/cuda_engine.h"
#include "engine/executor/backend.h"
#include "engine/executor/instance.h"

#include <memory>
#include <utility>
#include <variant>

namespace mfq::cuda::engine_detail {

using State = std::variant<
    std::unique_ptr<Text<Loaded<MiniCPMO45CausalLm>>>,
    std::unique_ptr<Text<Loaded<minicpmo45::Model, minicpmo45::State>>>,
    std::unique_ptr<Text<Loaded<Gemma4CausalLm>>>,
    std::unique_ptr<Text<Loaded<GlmDsaCausalLm>>>,
    std::unique_ptr<Text<Loaded<Glm5CausalLm>>>,
    std::unique_ptr<Text<Loaded<Qwen4CausalLm>>>,
    std::unique_ptr<Text<Loaded<DeepseekV4CausalLm>>>,
    std::unique_ptr<Text<Loaded<DeepseekV41CausalLm>>>,
    std::unique_ptr<Text<Loaded<NemotronHCausalLm>>> >;

} // namespace mfq::cuda::engine_detail

namespace mfq::cuda {

namespace {

auto load_qwen(
    std::shared_ptr<const ModelSource> source,
    const weight_loader::Loader& weights,
    LayerPlacementConfig& placement,
    mfq::models::YarnScaling& yarn,
    int& cpu_layers,
    const CudaEngineOptions& options,
    int gpu_layers,
    bool extend_yarn,
    bool expert_parallel,
    bool mirror_scalars,
    std::unique_ptr<Executor> executor,
    Collectives& copies,
    const bool& transposed,
    const int64_t& kv_capacity,
    Trace* const& trace,
    const int& trace_scope,
    bool cpu_offload,
    bool preallocate) {
    return qwen35::assemble(
        source, weights, placement, yarn, cpu_layers,
        qwen35::LoadOptions{
            .config_path = options.config_path,
            .context_size = options.context_size,
            .gpu_layers = gpu_layers,
            .extend_yarn = extend_yarn,
            .expert_parallel = expert_parallel,
            .mirror_scalars = mirror_scalars},
        std::move(executor), copies, transposed, kv_capacity, trace,
        trace_scope, true, cpu_offload, preallocate);
}

} // namespace

std::unique_ptr<engine::Engine> load_cuda_engine(CudaEngineOptions options) {
    auto source = mfq::open_model_source(options.model_path);
    if (cuda_backbone(source->resolved_model_graph().backbone) ==
            CudaBackbone::generic_qwen) {
        return std::make_unique<CudaEngine<
            Loaded<qwen35::Model, qwen35::State, qwen35::Components>,
            load_qwen>>(std::move(options), std::move(source));
    }
    source.reset();
    return std::make_unique<engine::EngineInstance<
        engine_detail::Backend<engine_detail::State, with_model<false>>>>(std::move(options));
}

} // namespace mfq::cuda
