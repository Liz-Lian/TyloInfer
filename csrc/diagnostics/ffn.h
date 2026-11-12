#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace tyloi::cuda { class Program; }
/** Run the routed FFN benchmark through the loaded model view. */
int run_moe_check(tyloi::cuda::Program& program, const std::string& config_path, int layer,
    const std::vector<std::int64_t>& token_sizes, int reps);
