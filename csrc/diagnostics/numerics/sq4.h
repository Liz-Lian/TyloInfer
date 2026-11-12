#pragma once

#include "case.h"

namespace tyloi::diagnostics::numerics {
/** Execute the native SQ4 decode, projection, and routed observations. */
bool run_sq4(Case& value);
} // namespace tyloi::diagnostics::numerics
