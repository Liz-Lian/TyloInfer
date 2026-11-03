#pragma once

#include "attention.h"

#include <cstdint>
#include <utility>

namespace mfq::cuda {

#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
std::pair<mfq_tensor_backend::Tensor, mfq_tensor_backend::Tensor> append_kv(
    AttentionKvView cache, bool aten_write,
    mfq_tensor_backend::Tensor key, mfq_tensor_backend::Tensor value,
    mfq_tensor_backend::Tensor positions, int64_t start_position, int64_t end_position,
    bool contiguous_prefill_prefix = false);
#else
namespace ops {
std::pair<mfq::Tensor, mfq::Tensor> append_kv(mfq::state::KvView, mfq::Tensor, mfq::Tensor,
                                               mfq::Tensor, int64_t, bool);
}
#endif

} // namespace mfq::cuda
