#pragma once

#include "engine/tensor.h"
#include <chrono>
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace tyloi::cuda { class Program; }
namespace tyloi::cuda::diagnostics {
/** Run and report the original prefill warmups with profiling disabled. */
void warmup_prefill(Program& program, const Tensor& ids, bool profile, int rounds = -1);
/** Generate the diagnostic token sequence while preserving its timing boundaries. */
int generate_diagnostic_tokens(Program& program, Tensor ids, int gen, bool profile,
    std::chrono::steady_clock::time_point t0, std::chrono::steady_clock::time_point t1,
    const std::vector<std::int64_t>& forced = {},
    const std::unordered_set<std::int64_t>& stops = {}, bool production_prefill = false,
    int warmup = -1);
}
