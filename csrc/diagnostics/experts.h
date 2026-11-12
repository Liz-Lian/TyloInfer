#pragma once

#include <string>

namespace tyloi::cuda { class Program; }

/** Compare unsharded and expert-parallel routed projections. */
int run_expert_parallel_moe_check(tyloi::cuda::Program&, const std::string& name, int tokens, int routes);
/** Measure and compare a routed MFE tensor using the original diagnostic paths. */
int run_mfe_tensor_check(tyloi::cuda::Program&, const std::string& name, int tokens, int routes,
    int repetitions, int split_width, bool routed_input, bool benchmark_only);
