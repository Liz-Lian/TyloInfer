#pragma once
#include "case.h"

namespace tyloi::diagnostics::numerics {
/** Execute supported fixed-input basic kernels and publish their observations. */
bool run_basic(Case& value);
} // namespace tyloi::diagnostics::numerics
