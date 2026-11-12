#pragma once

#include <memory>
#include <string_view>

struct LayerPlacementConfig;
namespace tyloi { class ModelSource; }
namespace tyloi::weights { struct Loader; }
namespace tyloi::models { class Groups; struct LoadOptions; }

namespace tyloi::cuda::deepseek_v4 {

/** Assemble DeepSeek V4 shared weights with deferred transformer layers. */
std::unique_ptr<tyloi::models::Groups> load_groups(
    std::shared_ptr<const tyloi::ModelSource>, std::string_view,
    const tyloi::weights::Loader&, LayerPlacementConfig&, const tyloi::models::LoadOptions&);

} // namespace tyloi::cuda::deepseek_v4
