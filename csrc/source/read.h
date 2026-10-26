#pragma once

#include "model_source.h"

namespace mfq {

const TensorMetadata& require_tensor(const ModelSource& source, std::string_view name);
bool has_tensor(const ModelSource& source, std::string_view name) noexcept;
bool has_tensor_prefix(const ModelSource& source, std::string_view prefix);
std::vector<std::uint8_t> read_tensor(const ModelSource& source, std::string_view name);
std::vector<std::uint8_t> read_asset(const ModelSource& source, std::string_view name);
std::string read_asset_text(const ModelSource& source, std::string_view name);

} // namespace mfq
