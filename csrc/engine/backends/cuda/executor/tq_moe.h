#pragma once

#include "engine/weights/tq/routed.h"
#include "engine/backends/cuda/ops/projection/tq.h"
#include "backend.h"
#include "engine/weights/tensor_parallel.h"

#include <cstdint>
#include <memory>

struct CudaProfiler;
class TqExec;

bool tq_graph_supports_tokens(std::int64_t tokens) noexcept;

// Three ordinary [experts, rows, columns] TQ tensors. Expert ID selects a row
// range; each row retains its own bit width, scale and encoded data.
struct TqMoeBanks {
    std::shared_ptr<const tyloi::TqBank> gate, up, down;
    std::vector<tyloi::TensorParallelSlice> slices;

    ::tyloi::DenseTensor forward(
        TqExec&, ::tyloi::DenseTensor input,
        ::tyloi::DenseTensor ids, ::tyloi::DenseTensor weights,
        float swiglu_limit = 0, CudaProfiler* profiler = nullptr,
        std::int64_t expert_begin = 0, std::int64_t expert_end = -1) const;
};

TqMoeBanks tq_banks(::tyloi::Tensor gate,
                    ::tyloi::Tensor up,
                    ::tyloi::Tensor down);
