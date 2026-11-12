#pragma once

#include "engine/tensor.h"
#include <chrono>
#include <cstdint>
#include <vector>

namespace tyloi::cuda { class Program; }
namespace tyloi::cuda::diagnostics {
/** Select one live production execution policy for an off/on/off comparison. */
enum class Policy {
    route_poll, window_flush, window_input_views, route_spin, host_pin,
    shared_gate_fusion, router_topk, warp_multi_sum,
    mhc_shared_norm, mhc_dense_fusion, mhc_packed_groups, mhc_fixed_formats, mhc_async_normalized,
    gdn_fixed_geometry, gdn_inplace_state, qsa_fusion,
    rotary_fusion, attention_grouping, mhc_fusion
};
/** Return the existing output label for a policy comparison. */
const char* policy_name(Policy policy);
/** Configure a fixed-input policy comparison and its existing timing protocol. */
struct PolicyOptions {
    Policy policy;
    int rounds = 1, warmup = -1, gen = 0;
    bool phase_timings = false;
    std::vector<std::int64_t> forced;
    std::chrono::steady_clock::time_point load_start, load_end;
};
/** Run a live policy comparison using the program's loaded model and execution owner. */
int run_policy(Program& program, Tensor ids, const PolicyOptions& options);
}
