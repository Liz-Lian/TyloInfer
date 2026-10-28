#pragma once

#include "backend.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

namespace mfq {
class ModelSource;

namespace cuda {
class KvExpertLoan;
struct RoutedCache;

// Methods run inside the owner's CUDA scope; the owner drains consumers before destruction, and this class receives no Context or changes the current CUDA device.
class ExpertSpace {
public:
    struct Metrics {
        std::size_t decode_capacity = 0;
        std::size_t decode_used = 0;
        std::size_t prefill_capacity = 0;
        std::size_t prefill_used = 0;
        std::uint64_t prefill_reclaimed_used = 0;
    };

    ExpertSpace(RoutedCache& weights, int device,
        std::shared_ptr<KvExpertLoan> loan);
    ~ExpertSpace() noexcept;

    ExpertSpace(const ExpertSpace&) = delete;
    ExpertSpace& operator=(const ExpertSpace&) = delete;
    ExpertSpace(ExpertSpace&&) = delete;
    ExpertSpace& operator=(ExpertSpace&&) = delete;

    void release(std::size_t required_available =
        std::numeric_limits<std::size_t>::max());
    bool empty() const noexcept;
    bool has_decode() const noexcept { return decode_region_.has_value(); }
    std::size_t lend_decode(std::size_t maximum_minus_active);
    void restore(std::size_t reserve_bytes, const ModelSource* source);
    Metrics metrics() const;

private:
    struct PrefillRegion {
        mfq_tensor_backend::Tensor storage;
        std::size_t region = 0;
    };

    std::size_t prefill_capacity() const noexcept {
        std::size_t bytes = 0;
        for (const auto& part : prefill_regions_) bytes += part.storage.nbytes();
        return bytes;
    }

    RoutedCache& weights_;
    int device_;
    std::shared_ptr<KvExpertLoan> loan_;
    mfq_tensor_backend::Tensor decode_storage_;
    std::optional<std::size_t> decode_region_;
    std::vector<PrefillRegion> prefill_regions_;
    std::uint64_t prefill_reclaimed_used_bytes_ = 0;
};

} // namespace cuda
} // namespace mfq
