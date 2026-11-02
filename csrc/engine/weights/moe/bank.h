#pragma once

#include <cstdint>

namespace mfq::weights {

class ExpertBank {
public:
    virtual ~ExpertBank() = default;
    virtual std::int64_t count() const = 0;
    virtual std::int64_t rows() const = 0;
    virtual std::int64_t cols() const = 0;
};

} // namespace mfq::weights
