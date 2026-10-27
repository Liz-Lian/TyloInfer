#pragma once

#include "mlx_ssd_expert_cache.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mfq::metal {
class MfqContainer;
class MlxWeightResidency;

struct MlxExpertSegment {
    std::string tensor;
    std::uint64_t offset = 0;
    std::size_t expert_bytes = 0;
};
struct MlxExpertSegments {
    int experts = 0;
    std::vector<MlxExpertSegment> fields;
};

class MlxExpertSegmentCache {
public:
    MlxExpertSegmentCache(const MfqContainer& model, std::vector<MlxExpertSegments> layers,
        std::size_t budget, int workers = 8, bool prefill_read_ahead = false);
    mlx::core::array apply(std::size_t layer, const mlx::core::array& ids,
        std::size_t first_stage_fields, const std::function<mlx::core::array()>& first_stage,
        const std::function<mlx::core::array(const mlx::core::array&, bool)>& second_stage,
        const mlx::core::array* parallel = nullptr);
    const std::vector<mlx::core::array>& address_tables(std::size_t layer) const;
    const std::shared_ptr<MlxWeightResidency>& residency() const noexcept;
    std::size_t limit_bytes() const noexcept;
    std::size_t backing_bytes() const noexcept;
    std::size_t resident_bytes() const noexcept;
    std::size_t set_limit(std::size_t bytes);
    MlxSsdExpertCacheStats stats() const;
    void clear();
private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};
} // namespace mfq::metal
