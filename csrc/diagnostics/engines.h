#pragma once

#include "engine/load.h"

namespace tyloi::cuda::diagnostics {
/** Compare serial and concurrent generation from two independently loaded engines. */
int run_cuda_engine_isolation_check(CudaEngineOptions options);
}
