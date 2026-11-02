#pragma once

#include <cstdint>

namespace mfq::weights {

class Matrix {
public:
    virtual ~Matrix() = default;
    virtual std::int64_t rows() const = 0;
    virtual std::int64_t cols() const = 0;
};

} // namespace mfq::weights
