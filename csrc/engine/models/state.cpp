#include "engine/models/state.h"
#include "engine/models/session/snapshot.h"

#include <stdexcept>

namespace tyloi::models {

TextSessionState State::capture_text_session_state(const std::vector<std::int64_t>&) const {
    throw std::runtime_error("text session state is unsupported by this model");
}

void State::restore_text_session_state(const TextSessionState&) {
    throw std::runtime_error("text session state is unsupported by this model");
}

} // namespace tyloi::models
