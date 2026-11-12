#pragma once

namespace tyloi::cuda { class Program; }

/** Run the existing Flash-Next text diagnostic after the caller loads its model. */
int run_flash_next_check(tyloi::cuda::Program& program);
/** Run the existing Nemotron-H text diagnostic after the caller loads its model. */
int run_nemotron_h_check(tyloi::cuda::Program& program);
