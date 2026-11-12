#pragma once

#include "case.h"

namespace tyloi::diagnostics::numerics {
/** Execute a fixed Flash-Next case, or return false for another operation. */
bool run_flash(Case &value);
} // namespace tyloi::diagnostics::numerics
