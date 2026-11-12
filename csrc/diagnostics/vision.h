#pragma once

namespace tyloi::cuda { class Program; }

/** Run the existing Nemotron-H RADIO prompt diagnostic on a loaded program. */
int run_nemotron_h_vision_check(tyloi::cuda::Program& program);
