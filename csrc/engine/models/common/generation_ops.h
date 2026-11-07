#pragma once

#include "engine/models/common/prompt.h"
#include "engine/requests/policy.h"
#include "engine/requests/request.h"
#include "engine/weights/workspace.h"
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/backends/cuda/executor/executor.h"
#include "engine/models/options.h"
#endif

#include <cstddef>
#include <cstdint>
#include <optional>
#include <tuple>

struct DecodeGraphCache;

namespace tyloi::cuda {

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
inline tyloi::models::ForwardOptions forward_options(const Executor::Mode& mode) {
    return {mode.kv_capacity, mode.trace, mode.trace_layer, mode.primary};
}
#endif

template<class Model, class Ops, class Finish>
std::tuple<::tyloi::DenseTensor, bool, bool> generation_prefill(
        Model&, Ops&, const tyloi::engine::PrefillChunk&, ::tyloi::DenseTensor&, const Finish&) {
    return {{}, false, false};
}

template<class Model>
bool generation_workspace(Model&, tyloi::weights::Loan&, int64_t) { return false; }

template<class Model, class Ops>
std::optional<int64_t> generation_decode(Model&, Ops&) { return std::nullopt; }

template<class Model, class Ops>
void generation_request(Model&, Ops&, const tyloi::engine::InferenceRequest&) {}

template<class Model>
bool generation_borrow(Model&, std::size_t, const PreparedPrompt*) { return false; }

template<class Model>
bool generation_prefill_begin(Model&, DecodeGraphCache&, bool) { return false; }

} // namespace tyloi::cuda
