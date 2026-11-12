#pragma once

#include <cstdint>
#include <vector>

namespace tyloi::cuda { class Program; }
namespace tyloi::models::gemma4 { struct Config; }

/** Run the original fixed-route Gemma expert benchmark. */
int run_gemma_moe_check(tyloi::cuda::Program& program,
    const tyloi::models::gemma4::Config& config, int layer,
    const std::vector<std::int64_t>& token_sizes, int reps);
