#pragma once

#include "formats/tq_table.h"
#include "model_source.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace mfq {

std::vector<std::int64_t> tq_shape(
    const ModelSource& source, std::string_view name);

std::shared_ptr<TqTable> tq_table(
    const ModelSource& source, std::string_view name);

bool is_tq_table(
    const ModelSource& source, std::string_view name);

std::array<std::int64_t, 2> tq_projection_shape(
    const ModelSource& source, std::string_view name);

} // namespace mfq
