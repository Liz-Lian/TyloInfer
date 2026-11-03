#pragma once

#include <cstdint>
#include <cstddef>

namespace mfq::weights {

class Matrix {
public:
    virtual ~Matrix() = default;
    virtual std::int64_t rows() const = 0;
    virtual std::int64_t cols() const = 0;
};

class MatrixGroup {
public:
    virtual ~MatrixGroup() = default;
    virtual std::size_t size() const = 0;
};

} // namespace mfq::weights
