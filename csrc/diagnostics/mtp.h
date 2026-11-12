#pragma once

namespace tyloi::cuda { class Program; }

namespace tyloi::cuda::diagnostics {

/** Check Nemotron-H predictor state and generation against the existing gate. */
int run_nemotron_h_mtp_check(Program& program);

}  // namespace tyloi::cuda::diagnostics
