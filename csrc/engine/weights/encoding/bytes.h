#pragma once

#include <cstddef>
#include <type_traits>
#include <vector>

namespace tyloi::encoding {

template <typename T>
void append_little(std::vector<std::byte>& destination, T value) {
    using Unsigned = std::make_unsigned_t<T>;
    const auto bits = static_cast<Unsigned>(value);
    for (std::size_t index = 0; index < sizeof(T); ++index) {
        destination.push_back(static_cast<std::byte>(bits >> (index * 8)));
    }
}

} // namespace tyloi::encoding
