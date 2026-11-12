#pragma once

#include <cstdint>

namespace tyloi::cuda { class Program; }

/** Compare eager and captured BF16 addition and activation operations. */
int run_backend_bf16_add_check(tyloi::cuda::Program& program, std::int64_t elements, int repetitions);
/** Check native argmax and greedy sampling against the original cases. */
int run_backend_argmax_check(tyloi::cuda::Program& program, std::int64_t elements, int repetitions);
