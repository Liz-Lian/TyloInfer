#pragma once

#include "engine/tensor.h"
#include "engine/requests/media.h"
#include "engine/requests/result.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace tyloi::models {

struct ForwardOptions;

/** Own one duplex session while borrowing its model and language state. */
class Duplex {
public:
    virtual ~Duplex() = default;
    /** Plan persistent allocations for starting a session. */
    virtual std::vector<std::pair<Device, std::size_t>> growth(const MfqDuplexSessionParams&, const ForwardOptions&) const = 0;
    /** Plan persistent allocations for the next session step. */
    virtual std::vector<std::pair<Device, std::size_t>> growth(const MfqDuplexStepInput&, const ForwardOptions&) const = 0;
    /** Start the session after its allocations are admitted. */
    virtual void start(const MfqDuplexSessionParams&, const ForwardOptions&) = 0;
    /** Process the next audio or text input after admission. */
    virtual MfqDuplexStepResult step(const MfqDuplexStepInput&, const ForwardOptions&) = 0;
    /** Retire audio, speech and session state. */
    virtual void stop() = 0;
};

} // namespace tyloi::models
