#pragma once

#include "engine/tensor.h"
#include "engine/weights/tensor_parallel.h"

#include <cstdint>
#include <memory>
#include <vector>

class TqCache;

namespace tyloi {

// The source table is CPU-backed; selected device images belong to the cache.
class TqBank final : public Tensor {
public:
    std::shared_ptr<TqCache> cache;
    std::uint64_t source_id = 0;
    std::int64_t experts = 0, out = 0, input = 0;
    std::vector<TensorParallelSlice> slices;

    bool defined() const noexcept override { return cache && source_id && experts > 0 && out > 0 && input > 0; }
    std::int64_t dim() const noexcept override { return 3; }
    std::int64_t numel() const override;
    std::size_t nbytes() const override;
    std::size_t backing_nbytes() const override { return 0; }
    std::int64_t size(std::int64_t dimension) const override;
    ScalarType scalar_type() const noexcept override { return kFloat16; }
    const Device& device() const noexcept override { static const Device source{}; return source; }
    bool is_contiguous() const noexcept override { return false; }
    bool is_encoded() const noexcept override { return true; }
    std::vector<const DenseTensor*> buffers() const override { return {}; }
    void record_stream(std::uintptr_t) const override {}
    std::shared_ptr<const Tensor> relocated(Device) const override;
};

} // namespace tyloi
