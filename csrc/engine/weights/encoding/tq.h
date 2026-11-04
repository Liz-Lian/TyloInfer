#pragma once

#include <cstddef>
#include <cstdint>

namespace tyloi::encoding {

struct Tq {
    std::size_t words = 0, words_bytes = 0;
    std::size_t scales = 0, scales_bytes = 0;
    std::size_t relative = 0, bytes = 0;
    std::int64_t stored_columns = 0;
    int gslog = 0;
    bool columns = false;
};

} // namespace tyloi::encoding
