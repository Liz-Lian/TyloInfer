#pragma once
#include "case.h"
namespace tyloi::diagnostics::numerics {
/** Execute one fixed TQ case and publish the raw native observations. */
void run_tq(Case &value);
} // namespace tyloi::diagnostics::numerics
