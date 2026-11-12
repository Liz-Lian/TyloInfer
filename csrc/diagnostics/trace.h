#pragma once

#include "engine/load.h"
#include "engine/tensor.h"

#include <cstdint>
#include <string>

namespace tyloi::cuda { class Program; }

/** Compare block traces against a separately loaded execution reference. */
int run_block_trace_compare(tyloi::cuda::Program& test,
    tyloi::cuda::CudaLoadOptions reference_options, tyloi::Tensor ids);
/** Dump block and terminal traces for the selected token range. */
int run_block_trace_dump(tyloi::cuda::Program& program, const std::string& output_dir,
    tyloi::Tensor ids, std::int64_t token_start, std::int64_t token_count);
