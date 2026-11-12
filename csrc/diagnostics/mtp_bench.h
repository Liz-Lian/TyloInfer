#pragma once

namespace tyloi::cuda { class Program; }

namespace tyloi::cuda::diagnostics {

/** Benchmark serial and speculative generation against the serial oracle. */
int run_qwen35_mtp_bench(Program& program, bool enable_mtp,
    int generated_tokens, int repetitions);

}  // namespace tyloi::cuda::diagnostics
