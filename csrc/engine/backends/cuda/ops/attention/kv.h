#pragma once

#include "attention.h"

#include <cstdint>
#include <utility>

namespace tyloi::cuda {

#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
std::pair<::tyloi::Tensor, ::tyloi::Tensor> append_kv(
    AttentionKvView cache, bool aten_write,
    ::tyloi::Tensor key, ::tyloi::Tensor value,
    ::tyloi::Tensor positions, int64_t start_position, int64_t end_position,
    bool contiguous_prefill_prefix = false);
#else
namespace ops {
std::pair<tyloi::Tensor, tyloi::Tensor> append_kv(tyloi::state::KvView, tyloi::Tensor, tyloi::Tensor,
                                               tyloi::Tensor, int64_t, bool);
}
#endif

} // namespace tyloi::cuda
