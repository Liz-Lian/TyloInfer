#pragma once

#include <cstdint>
#include <filesystem>

namespace tyloi::cuda { class Program; }

namespace tyloi::eval {

/** Export the original WT2 selected F32 logits through the native layer-wise model interface. */
int reference(cuda::Program& program, const std::filesystem::path& input,
    const std::filesystem::path& output, std::int64_t context, bool resident,
    const std::filesystem::path& capture = {});

} // namespace tyloi::eval
