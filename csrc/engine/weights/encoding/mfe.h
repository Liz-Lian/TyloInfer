#pragma once

#include "engine/tensor.h"
#include "engine/weights/moe/pool.h"

#include <memory>
#include <vector>

struct MixedNvqDispatch;
struct MixedNintDispatch;
class MoeCachedSource;

namespace tyloi {

class MfeTensor final : public TensorImpl {
public:
    struct Shard {
        int device = 0;
        std::int64_t expert_begin = 0, expert_end = 0;
        std::shared_ptr<const MfeTensor> weight;
    };
    int n_experts = 0;
    int out_per_expert = 0;
    int neuron_len = 0;
    bool partial_experts = false;
    // Streamed prompt groups remain prefill even when a rare expert has few rows.
    bool prefill_only = false;
    std::vector<MixedMoePool> pools;
    std::shared_ptr<MixedNvqDispatch> nvq_dispatch;
    std::shared_ptr<MixedNintDispatch> nint_dispatch;
    std::shared_ptr<const std::vector<MfePoolWeight>> compact;
    std::shared_ptr<MoeCachedSource> cached_source;
    std::vector<Shard> shards;
    Device source_device;

    bool nint_only() const;
    bool defined() const noexcept override;
    std::int64_t dim() const noexcept override { return 3; }
    std::int64_t numel() const override;
    std::size_t nbytes() const override;
    std::size_t backing_nbytes() const override;
    std::int64_t size(std::int64_t dimension) const override;
    ScalarType scalar_type() const noexcept override { return kFloat16; }
    const Device& device() const noexcept override;
    bool is_contiguous() const noexcept override { return false; }
    bool is_encoded() const noexcept override { return true; }
    std::vector<const DenseTensor*> buffers() const override;
    void record_stream(std::uintptr_t stream) const override;
    std::shared_ptr<TensorImpl> relocated(Device target) const override;
};

} // namespace tyloi
