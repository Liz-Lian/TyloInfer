#include "models/qwen35/runtime.h"

#include "engine/requests/inference.h"
#include "mfq_cuda_kv_quantization.h"
#include "models/batch.h"
#include "models/qwen35/forward.h"

#include <stdexcept>
#include <utility>

namespace mfq::cuda::qwen35 {

Runtime::Runtime(Model loaded, const weight_loader::Loader& weights,
                 std::unique_ptr<Executor> loaded_executor,
                 Collectives& copies, const bool& transposed,
                 const int64_t& kv_capacity, Trace* const& trace,
                 const int& trace_scope, bool optional, bool mtp_allowed,
                 bool preallocate)
    : model(std::make_shared<Model>(std::move(loaded))) {
    owned_state_ = std::make_unique<State>(*model, transposed);
    if (preallocate) owned_state_->preallocate();
    executor = std::move(loaded_executor);
    components = load_components(
        *model, weights, *executor, copies, kv_capacity, trace, trace_scope,
        optional, mtp_allowed, preallocate);
}

std::unique_ptr<mfq::engine::Batch> Runtime::batch(
    std::unique_ptr<Executor> batch_executor, Collectives& copies,
    Executor::Mode mode, std::shared_ptr<MoeExpertCache> cache,
    CudaContinuousBatchConfig config, std::int64_t chunk_size, bool preallocate,
    bool cpu_offload, int cached_slots) {
    return mfq::cuda::make_batch(
        model, &state(), std::move(batch_executor), copies, mode, std::move(cache),
        std::move(config), chunk_size, preallocate, cpu_offload, cached_slots);
}

mfq::StepSequence<PreparedPrompt> Runtime::prepare(
    mfq::engine::InferenceRequest& input, Projector projector,
    std::shared_ptr<MoeExpertCache> cache, Collectives& copies, int primary) {
    if (components.grid_vision)
        return components.grid_vision->prepare(
            *model, std::move(projector), std::move(cache), copies, primary,
            executor.get(), input.prompt, *input.vision);
    throw std::invalid_argument("model has no multimodal component");
}

std::optional<KvQuantization> Runtime::kv_quantization() const {
    return cuda_kv_quantization();
}

} // namespace mfq::cuda::qwen35
