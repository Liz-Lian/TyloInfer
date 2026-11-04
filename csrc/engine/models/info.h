#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace tyloi::models {

struct Info {
    std::int64_t vocab_size;
    std::int64_t max_context;
    std::string model_type;
    std::size_t prefill_row_bytes;
    bool decode_graph;
    bool double_warmup;
};

} // namespace tyloi::models
