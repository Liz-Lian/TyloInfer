#pragma once

#include "engine/executor.h"
#include "engine/weights/encoding/format_features.h"
#if TYLOINFER_ENABLE_MFQ_COMPAT
#include "engine/weights/nint.h"
#endif

namespace tyloi::cuda {
class Executor;

/** Create the execution-owned per-job decoded projection working set. */
std::unique_ptr<tyloi::Executor::Work> make_working_set(Executor& executor);

/** Return whether decoded projections are reused in the active layer. */
bool decoded_work_active() noexcept;
#if TYLOINFER_ENABLE_MFQ_COMPAT
/** Find a decoded projection in the active layer, or return an empty tensor. */
DenseTensor decoded_work_find(const NintTensor& weight);
/** Retain a decoded projection in the active layer when reuse is enabled. */
void decoded_work_store(const NintTensor& weight, const DenseTensor& value);
#endif
}
