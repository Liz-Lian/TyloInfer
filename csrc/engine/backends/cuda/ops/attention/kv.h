#pragma once

#include "attention.h"

#include <cstdint>
#include <utility>

namespace tyloi::cuda {

#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
std::pair<::tyloi::DenseTensor, ::tyloi::DenseTensor> append_kv(
    AttentionKvView cache, bool aten_write,
    ::tyloi::DenseTensor key, ::tyloi::DenseTensor value,
    ::tyloi::DenseTensor positions, int64_t start_position, int64_t end_position,
    bool contiguous_prefill_prefix = false);
#else
namespace ops {
std::pair<tyloi::DenseTensor, tyloi::DenseTensor> append_kv(tyloi::state::KvView, tyloi::DenseTensor, tyloi::DenseTensor,
                                               tyloi::DenseTensor, int64_t, bool);
}
#endif

} // namespace tyloi::cuda
