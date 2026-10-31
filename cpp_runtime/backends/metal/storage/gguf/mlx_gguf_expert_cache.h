#pragma once

#include "mlx_gguf.h"
#include "mlx_packed_expert_cache.h"

#include <memory>
#include <string>
#include <vector>

namespace mfq::metal {
class MfqContainer;

class MlxGgufExpertCache final : public MlxPackedExpertCache {
public:
    MlxGgufExpertCache(const MfqContainer& model, const std::vector<std::string>& prefixes,
        std::size_t budget, int workers = 8);
    mlx::core::array apply_ffn(std::size_t layer, const mlx::core::array& input,
        const mlx::core::array& ids, const mlx::core::array& weights,
        const mlx::core::array* parallel = nullptr) override;
    mlx::core::array apply_ffn(std::size_t layer, const mlx::core::array& input,
        const mlx::core::array& ids, const mlx::core::array& weights,
        const mlx::core::array* parallel, float swiglu_limit);
    void validate_layer(std::size_t layer, int experts, int input, int intermediate) const override;
    void prepare_layer(std::size_t layer, int top_k) const override;
    void prepare_layer(std::size_t layer, int top_k, float swiglu_limit) const;
    void set_prefill_next_rows(std::size_t layer, std::size_t rows);
    std::size_t limit_bytes() const noexcept override;
    std::size_t backing_bytes() const noexcept override;
    std::size_t resident_bytes() const noexcept override;
    std::size_t set_limit(std::size_t bytes) override;
    MlxSsdExpertCacheStats stats() const override;
    void clear() override;
private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};
} // namespace mfq::metal
