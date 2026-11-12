#pragma once

#include <cstdint>
#include <vector>

namespace tyloi::cuda { class Program; }

namespace tyloi::cuda::diagnostics {
/** Benchmark prefill sizes and validate requested chunk and kernel comparisons. */
int run_prefill_sweep(tyloi::cuda::Program& program,
    const std::vector<std::int64_t>& sizes, int repeats, bool profile);
}
