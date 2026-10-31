#pragma once

namespace mfq::cuda::gguf {
// Strata iq_kernels.cu uses GRP_NC=4: bound the per-expert register tile,
// independently of the total number of tokens in a verification window.
inline constexpr int route_group_capacity(int tokens) noexcept {
    return tokens < 4 ? tokens : 4;
}
}
