#include "models/qwen35/runtime.h"

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

} // namespace mfq::cuda::qwen35
