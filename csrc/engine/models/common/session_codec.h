#pragma once

#include "engine/models/common/causal_model_ops.h"
#include "engine/models/session.h"

namespace tyloi::cuda {

template <typename Model>
TextSessionStateKind CudaSessionCodec<Model>::kind(const CausalLm<Model> &) {
    return TextSessionStateKind::Unsupported;
}

template <typename Model> bool CudaSessionCodec<Model>::supports_paged(const CausalLm<Model> &) {
    return false;
}

template <typename Model>
TextSessionState CudaSessionCodec<Model>::capture(
    const CausalLm<Model> &, const std::vector<int64_t> &) {
    throw std::runtime_error("text session state is unsupported by this model");
}

template <typename Model>
void CudaSessionCodec<Model>::restore(CausalLm<Model> &, const TextSessionState &) {
    throw CudaSessionStateError("text session state is unsupported by this model");
}

} // namespace tyloi::cuda
