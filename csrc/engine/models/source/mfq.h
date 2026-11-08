#pragma once

#include "engine/weights/source/mfq.h"

#include <filesystem>
#include <vector>

namespace tyloi::models {

void engram_shards(archive::Shards& parts);
void check_engram(
    const std::filesystem::path& path,
    const std::vector<archive::Record>& records);

} // namespace tyloi::models
