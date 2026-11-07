#pragma once

#include "engine/requests/result.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace tyloi::state {

class Sessions {
public:
    virtual ~Sessions() = default;
    virtual engine::SessionResult control(const engine::SessionCommand& command) = 0;
    virtual std::vector<std::pair<std::string, double>> metrics() const = 0;
    virtual std::map<int, std::size_t> device_hot_bytes() const = 0;
    virtual std::uint64_t trim_hot(std::uint64_t bytes) = 0;
    virtual void reserve_ram(std::uint64_t bytes) = 0;
};

} // namespace tyloi::state
