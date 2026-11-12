#pragma once

#include <cstdint>
#include <vector>

namespace tyloi::cuda {
class Program;
namespace diagnostics {
/** Check Qwen4 decode graphs against eager hidden states and transactions. */
int run_qwen4_decode_hidden_check(Program& program);
/** Profile Qwen4 eager and graph verification, including configured variants. */
int run_qwen4_mtp_profile(Program& program, const std::vector<std::int64_t>& prompt);
}
}
