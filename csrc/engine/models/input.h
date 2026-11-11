#pragma once

#include "engine/tensor.h"
#include "engine/ops.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tyloi::models {

struct Input {
    using Trace = std::vector<Tensor>;

    Tensor ids, embeddings;
    /** Optional semantic positions, per-sequence lengths and physical cache write positions. */
    Tensor positions, lengths, cache_positions;
    /** Optional observation output owned by the synchronous caller. */
    Trace* trace = nullptr;
    /** Optional output handle for hidden states before the final normalization. */
    Tensor* raw_hidden = nullptr;
    /** Commit completed input positions even when explicit positions are supplied. */
    bool advance = false;
    /** Confirmed prefix of a speculative verification window, or zero for ordinary execution. */
    std::int64_t confirmed = 0;
    /** Planned KV length used by static execution; zero selects the model's ordinary path. */
    std::int64_t kv_length = 0;
    /** Optional decode attention partition count selected for the planned KV length. */
    std::int64_t attention_parts = 0;
};

struct ForwardOptions {
    std::int64_t kv_capacity = 0;
    ops::Trace* trace = nullptr;
    int trace_layer = -1;
    int primary = 0;
};

struct Prepared {
    std::vector<std::int64_t> token_ids;
    Tensor embeddings;
    Tensor positions;
    int decode_position_delta = 0;
    std::string cache_key;
    Tensor image_mask;

    /** Return whether prepared embeddings or positions change ordinary text prefill. */
    bool transformed() const noexcept {
        return (embeddings && embeddings.defined()) ||
            (positions && positions.defined()) ||
            decode_position_delta != 0;
    }
};

} // namespace tyloi::models
