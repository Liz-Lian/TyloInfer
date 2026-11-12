#pragma once
#include "case.h"

namespace tyloi::diagnostics::numerics {
/** Execute fixed-input CUDA gated DeltaNet observations. */
bool run_gdn(Case &value);
}
