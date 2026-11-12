#pragma once
#include <cstdint>
#include <vector>

namespace tyloi::cuda { class Program; }
namespace tyloi::cuda::diagnostics {
/** Compare Qwen4 generation and short-batch kernel variants on fixed prompt IDs. */
int run_qwen4_generation_variants(Program& program, const std::vector<std::int64_t>& prompt);
}
