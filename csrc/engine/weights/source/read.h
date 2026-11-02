#pragma once

#include "source.h"

namespace mfq {

const TensorMetadata& require_tensor(const source::Source& source, std::string_view name);
bool has_tensor(const source::Source& source, std::string_view name) noexcept;
bool has_tensor_prefix(const source::Source& source, std::string_view prefix);
std::vector<std::uint8_t> read_tensor(const source::Source& source, std::string_view name);
std::vector<std::uint8_t> read_asset(const source::Source& source, std::string_view name);
std::string read_asset_text(const source::Source& source, std::string_view name);

} // namespace mfq
