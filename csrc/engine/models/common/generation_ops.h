#pragma once

#include "engine/models/common/prompt.h"
#include "engine/requests/policy.h"
#include "engine/requests/request.h"
#include "engine/weights/workspace.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <tuple>

struct DecodeGraphCache;

namespace mfq::cuda {

template<class Model, class Ops, class Finish>
std::tuple<mfq_tensor_backend::Tensor, bool, bool> generation_prefill(
        Model&, Ops&, const mfq::engine::PrefillChunk&, mfq_tensor_backend::Tensor&, const Finish&) {
    return {{}, false, false};
}

template<class Model>
bool generation_workspace(Model&, MoePrefillWorkspaceLoan&, int64_t) { return false; }

template<class Model, class Ops>
std::optional<int64_t> generation_decode(Model&, Ops&) { return std::nullopt; }

template<class Model, class Ops>
void generation_request(Model&, Ops&, const mfq::engine::InferenceRequest&) {}

template<class Model>
bool generation_borrow(Model&, std::size_t, const PreparedPrompt*) { return false; }

template<class Model>
void generation_prefill_begin(Model&, DecodeGraphCache&, bool) {}

template<class Model, class Ops>
void generation_prefill_end(Model&, Ops&, bool, bool) {}

} // namespace mfq::cuda
