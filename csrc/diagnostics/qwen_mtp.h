#pragma once

namespace tyloi::cuda {
class Program;
namespace diagnostics {
/** Run the original Qwen3.5 speculative decoding numerical gate. */
int run_qwen35_mtp_check(Program& program);
}
}
