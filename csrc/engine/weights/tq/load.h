#pragma once

#include "engine/weights/source/source.h"
#include "engine/weights/encoding/tq.h"
#include "engine/weights/tq/routed.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

/** Load a TQ projection into the requested device representation. */
::tyloi::encoding::TqTensor load_tq_weight(
    const tyloi::source::Source&, const std::string&, int device, bool cuda = true);
