#pragma once

#include "models/components.h"
#include "models/qwen35/model.h"
#include "native/cuda/profile.h"

#include <cstdint>

namespace mfq::cuda::weight_loader {
struct Loader;
}

namespace mfq::cuda {
class Collectives;
class Executor;
}

namespace mfq::cuda::qwen35 {

RuntimeComponents<Model> load_components(
    const Model& model,
    const weight_loader::Loader& weights,
    const Executor& executor,
    Collectives& copies,
    const int64_t& kv_capacity,
    Trace* const& trace,
    const int& trace_scope,
    bool optional,
    bool mtp_allowed,
    bool preallocate);

} // namespace mfq::cuda::qwen35
