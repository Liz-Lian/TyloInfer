#pragma once

#include <string>
#include <string_view>
#include <utility>

namespace tyloi::cache {

// Preserve the existing single-image cache capacity.
template <class Value>
class Media {
public:
    Value find(std::string_view key) const {
        return key_ == key ? value_ : Value{};
    }

    void store(std::string key, Value value) {
        key_ = std::move(key);
        value_ = std::move(value);
    }

    void clear() {
        key_.clear();
        value_ = Value{};
    }

private:
    std::string key_;
    Value value_;
};

} // namespace tyloi::cache
