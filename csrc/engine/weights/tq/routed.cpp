#include "engine/weights/tq/routed.h"

#include <limits>
#include <stdexcept>

std::int64_t tyloi::TqBank::numel() const {
    TensorView view;
    view.rank = 3;
    view.sizes[0] = experts;
    view.sizes[1] = out;
    view.sizes[2] = input;
    return view.numel();
}

std::size_t tyloi::TqBank::nbytes() const {
    const auto elements = numel();
    if (static_cast<std::uint64_t>(elements) > std::numeric_limits<std::size_t>::max() / 2)
        throw std::overflow_error("tensor byte count overflow");
    return static_cast<std::size_t>(elements) * 2;
}

std::int64_t tyloi::TqBank::size(std::int64_t dimension) const {
    if (dimension < 0) dimension += 3;
    switch (dimension) {
        case 0: return experts;
        case 1: return out;
        case 2: return input;
        default: throw std::out_of_range("tensor dimension is out of range");
    }
}

std::shared_ptr<const tyloi::Tensor> tyloi::TqBank::relocated(Device) const {
    throw std::invalid_argument("TQ source relocation is unsupported");
}
