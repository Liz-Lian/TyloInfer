#pragma once

#include <mlx/fast.h>

namespace mfq::metal {

mlx::core::fast::CustomKernelFunction cached_metal_kernel(
    const std::string& name, const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs, const std::string& source,
    const std::string& header = "", bool contiguous = true, bool atomic = false,
    const mlx::core::CompileOptions& options = {});

}
