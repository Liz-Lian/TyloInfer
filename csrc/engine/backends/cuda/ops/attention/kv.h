#pragma once

#include "attention.h"

#include <cstdint>
#include <utility>

namespace tyloi::cuda {

namespace ops {
std::pair<tyloi::DenseTensor, tyloi::DenseTensor> append_kv(tyloi::state::KvView, tyloi::DenseTensor, tyloi::DenseTensor,
                                               tyloi::DenseTensor, int64_t, bool);
}

} // namespace tyloi::cuda
