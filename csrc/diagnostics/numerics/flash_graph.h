#pragma once

#include "case.h"

namespace tyloi::diagnostics::numerics {
/** Execute a fixed Flash-Next changed-input CUDA graph case when recognized. */
bool run_flash_graph(Case& value);
} // namespace tyloi::diagnostics::numerics
