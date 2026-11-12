#pragma once

#include "case.h"

namespace tyloi::diagnostics::numerics {
/** Execute a supported fixed MFQ forward case, or return false for another operation. */
bool run_mfq(Case &value);
} // namespace tyloi::diagnostics::numerics
