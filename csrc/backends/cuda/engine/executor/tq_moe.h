#pragma once

#include "backends/cuda/engine/weights/tq/routed.h"
#include "backends/cuda/ops/tq/include/tq.h"
#include "backend.h"

#include <cstdint>
#include <memory>

struct CudaExecutionContext;
struct CudaProfiler;
class TqExec;

bool tq_graph_supports_tokens(std::int64_t tokens) noexcept;

// Three ordinary [experts, rows, columns] TQ tensors. Expert ID selects a row
// range; each row retains its own bit width, scale and encoded data.
struct TqMoeBanks {
    std::shared_ptr<TqRoutedWeight> gate, up, down;

    mfq_tensor_backend::Tensor forward(
        TqExec&, mfq_tensor_backend::Tensor input,
        mfq_tensor_backend::Tensor ids, mfq_tensor_backend::Tensor weights,
        float swiglu_limit = 0, CudaProfiler* profiler = nullptr,
        std::int64_t expert_begin = 0, std::int64_t expert_end = -1) const;

    mfq_tensor_backend::Tensor forward_parallel(
        CudaExecutionContext&, mfq_tensor_backend::Tensor input,
        mfq_tensor_backend::Tensor ids, mfq_tensor_backend::Tensor weights,
        float swiglu_limit = 0) const;
};
