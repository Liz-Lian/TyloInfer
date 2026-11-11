#pragma once

#include "engine/tensor.h"

#include <string>
#include <string_view>
#include <utility>

namespace tyloi::cache {

// Preserve the existing single-image cache capacity.
class Media {
public:
    ::tyloi::Tensor find(std::string_view key) const {
        return key_ == key ? value_ : ::tyloi::Tensor{};
    }

    void store(std::string key, ::tyloi::Tensor value) {
        key_ = std::move(key);
        value_ = std::move(value);
    }

    void clear() {
        key_.clear();
        value_ = ::tyloi::Tensor{};
    }

private:
    std::string key_;
    ::tyloi::Tensor value_;
};

} // namespace tyloi::cache
