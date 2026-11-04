#pragma once

#include <string>
#include <string_view>
#include <utility>

namespace tyloi::cache {

// Preserve the existing single-image cache capacity.
template <class Tensor>
class Media {
public:
    Tensor find(std::string_view key) const {
        return key_ == key && value_.defined() ? value_ : Tensor{};
    }

    void store(std::string key, Tensor value) {
        key_ = std::move(key);
        value_ = std::move(value);
    }

    void clear() {
        key_.clear();
        value_ = Tensor{};
    }

private:
    std::string key_;
    Tensor value_;
};

} // namespace tyloi::cache
