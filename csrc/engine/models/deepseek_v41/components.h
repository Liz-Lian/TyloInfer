#pragma once

#include <memory>

namespace tyloi::models { struct Mtp; }
namespace tyloi::weights { struct Loader; }

namespace tyloi::cuda::deepseek_v41 {

struct Model;

std::unique_ptr<tyloi::models::Mtp> load_components(
    Model& model, const tyloi::weights::Loader& weights, bool load_optional_components);

} // namespace tyloi::cuda::deepseek_v41
