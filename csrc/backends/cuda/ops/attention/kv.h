#pragma once

#include "attention.h"

#include <cstdint>
#include <utility>

namespace mfq::cuda {

std::pair<mfq_tensor_backend::Tensor, mfq_tensor_backend::Tensor> append_kv(
    AttentionKvView cache, bool aten_write,
    mfq_tensor_backend::Tensor key, mfq_tensor_backend::Tensor value,
    mfq_tensor_backend::Tensor positions, int64_t start_position, int64_t end_position,
    bool contiguous_prefill_prefix = false);

} // namespace mfq::cuda
