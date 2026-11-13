#pragma once

#include "engine/backends/cuda/executor/executor.h"
#include "engine/models/input.h"
#include "engine/models/glm5_next/model.h"

namespace tyloi::cuda::glm5_next {
bool supports_speculation(const Model&) noexcept;
} // namespace tyloi::cuda::glm5_next
