#pragma once

#include "mlx_server_components.h"

namespace mfq::metal {

bool mlx_dflash2_sampling_supported(const MfqSamplingParams& sampling,
    const MfqTokenConstraintPtr& constraint) noexcept;
void configure_mlx_dflash2(MlxEngineComponents& components,
    std::shared_ptr<std::optional<MlxQwen4CausalLm>> runtime,
    const std::filesystem::path& directory, const MlxSamplingParams& sampling);

}
