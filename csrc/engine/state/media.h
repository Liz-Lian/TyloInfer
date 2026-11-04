#pragma once

#include <string>
#include <string_view>
#include <utility>

namespace tyloi::cache {

// Preserve the existing single-image cache capacity.
template <class DenseTensor>
class Media {
public:
    DenseTensor find(std::string_view key) const {
        return key_ == key && value_.defined() ? value_ : DenseTensor{};
    }

    void store(std::string key, DenseTensor value) {
        key_ = std::move(key);
        value_ = std::move(value);
    }

    void clear() {
        key_.clear();
        value_ = DenseTensor{};
    }

private:
    std::string key_;
    DenseTensor value_;
};

} // namespace tyloi::cache
