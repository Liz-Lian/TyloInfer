#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace tyloi {

// The small set of legacy numeric-layout facts still needed by model loaders.
// Name aliases remain private to each source implementation.
struct TensorLayout {
    double norm_weight_offset = 1.0;
    bool linear_attention_a_is_log = true;
    bool qwen_gdn_gguf_layout = false;
};

} // namespace tyloi

namespace tyloi::models {

// Numeric/layout differences that cannot be represented by a no-copy name
// alias. New schema-v1 artifacts always use canonical semantics; these flags
// exist only while pre-schema files remain readable.
struct Names {
    std::unordered_map<std::string, std::string> canonical_to_stored;
    TensorLayout layout;
};

// The sole backend-neutral C++ name compatibility boundary. Runtime loaders
// ask only for canonical names; this view aliases them to records from older
// MFQ files without copying tensor blobs.
Names legacy_names(
    std::string_view artifact_architecture,
    std::string_view model_config_json,
    const std::vector<std::string>& stored_names);

} // namespace tyloi::models
