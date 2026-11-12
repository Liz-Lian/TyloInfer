#pragma once

#include "engine/load.h"

namespace tyloi::cuda::diagnostics {
/** Compare controlled numerical batching and real scheduler request behavior. */
int run_continuous_batching_check(CudaLoadOptions options, int requests);
}
