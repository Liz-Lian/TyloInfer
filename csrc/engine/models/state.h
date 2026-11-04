#pragma once

#include "engine/tensor.h"

#include <cstdint>
#include <vector>

namespace tyloi::models {

class State {
public:
    virtual ~State() = default;
    virtual void preallocate() = 0;
    virtual void reset(std::int64_t batch) = 0;
    virtual std::int64_t position() const noexcept = 0;
    virtual void advance(std::int64_t tokens) noexcept = 0;
    virtual void set_position_delta(std::int64_t delta) noexcept = 0;
    virtual std::vector<tyloi::Tensor> storage() const = 0;
    virtual std::vector<tyloi::Tensor*> graph_state() = 0;
    virtual bool supports_text_session_state() const = 0;
    virtual bool supports_paged_text_session_state() const = 0;
};

} // namespace tyloi::models
