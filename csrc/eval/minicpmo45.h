#pragma once

#include <cstdint>

namespace tyloi::cuda { class Program; }

namespace tyloi::eval {

/** Run the MiniCPM-o JSONL batch evaluation protocol using one native program. */
int minicpmo45_batch(cuda::Program& program, std::int64_t context_size, std::int64_t vision_batch_size);

} // namespace tyloi::eval
