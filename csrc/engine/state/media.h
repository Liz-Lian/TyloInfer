#pragma once

#include "engine/tensor.h"

#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace tyloi::cache {

// Preserve the existing single-image cache capacity.
class Media {
public:
    std::shared_ptr<::tyloi::Tensor> find(std::string_view key) const {
        return key_ == key ? value_ : std::shared_ptr<::tyloi::Tensor>{};
    }

    void store(std::string key, std::shared_ptr<::tyloi::Tensor> value) {
        key_ = std::move(key);
        value_ = std::move(value);
    }

    void clear() {
        key_.clear();
        value_ = std::shared_ptr<::tyloi::Tensor>{};
    }

private:
    std::string key_;
    std::shared_ptr<::tyloi::Tensor> value_;
};

} // namespace tyloi::cache
