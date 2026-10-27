#pragma once

#include "mlx_ssd_expert_cache.h"

namespace mfq::metal {
class MlxPackedExpertCache {
public:
    virtual ~MlxPackedExpertCache() = default;
    virtual mlx::core::array apply_ffn(std::size_t layer, const mlx::core::array& input,
        const mlx::core::array& ids, const mlx::core::array& weights,
        const mlx::core::array* parallel = nullptr) = 0;
    virtual void validate_layer(std::size_t layer, int experts, int input, int intermediate) const = 0;
    virtual void prepare_layer(std::size_t layer, int top_k) const = 0;
    virtual bool preserves_activation_dtype() const noexcept { return false; }
    virtual std::size_t limit_bytes() const noexcept = 0;
    virtual std::size_t backing_bytes() const noexcept = 0;
    virtual std::size_t resident_bytes() const noexcept = 0;
    virtual std::size_t set_limit(std::size_t bytes) = 0;
    std::size_t trim(std::size_t bytes) { return set_limit(bytes); }
    virtual MlxSsdExpertCacheStats stats() const = 0;
    virtual void clear() = 0;
};
} // namespace mfq::metal
