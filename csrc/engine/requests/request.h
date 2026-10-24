#pragma once

#include <cstddef>
#include <string>

// Describes the portion of a rendered prompt whose KV state is stable across
// requests. A runtime may retain this exact token prefix after generation and
// reuse it only when the next request starts with the same token sequence.
struct MfqPromptCachePlan {
    std::string session_id;
    size_t stable_prefix_tokens = 0;
    bool cache_output_tokens = false;
};
