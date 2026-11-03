#pragma once
#include "gguf.h"
namespace mfq::cuda::gguf {
// A homogeneous Q6 group can spend otherwise idle row parallelism on K.
// False leaves the existing mixed-format group path responsible for launch.
bool try_underfilled_q6_group(const DenseProjection*,std::size_t count,std::int64_t columns,
    const void* input,std::int64_t tokens,bool half_output,cudaStream_t);
}
