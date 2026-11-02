#pragma once

#include "engine/ops.h"

namespace mfq::models {

struct ForwardOptions {
    ops::FfnMode ffn;
    std::int64_t kv_capacity = 0;
    ops::Trace* trace = nullptr;
    int trace_layer = -1;
    int primary = 0;
    bool fused_norms = true;
};

} // namespace mfq::models
