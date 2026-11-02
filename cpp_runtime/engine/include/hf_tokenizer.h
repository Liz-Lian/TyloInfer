#pragma once

#include <cstdint>
#include <vector>

namespace mfq { class ModelSource; }

namespace mfq::engine {
std::vector<std::uint8_t> hf_tokenizer_gguf(const ModelSource& source);
}
