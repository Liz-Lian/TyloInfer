#pragma once

#include "engine/state/kv_cache.h"
#include "engine/state/session/state.h"

#include <vector>

namespace mfq::cuda::qwen35 {

class MtpState {
public:
    struct Restore {
        std::vector<KVCache> kv;
        int64_t position;
    };

    struct Layout {
        int device;
        int64_t heads, head_dim, capacity;
        mfq_tensor_backend::ScalarType dtype;
        KvQuantization quantization;
    };

    MtpState(std::vector<Layout> layout, int64_t hidden);
    int64_t position() const noexcept { return position_; }
    void prepare(int64_t batch, int64_t tokens, int64_t kl_capacity = 0);
    void reset(int64_t batch);
    void advance(int64_t tokens);
    void trim(int64_t position);
    std::vector<AttentionKvView> views() const;
    MtpSessionState capture(int64_t target_position,
        const mfq_tensor_backend::Tensor& hidden) const;
    Restore prepare(const MtpSessionState& saved) const;
    void stage(Restore& restore, const MtpSessionState& saved) const;
    void publish(Restore&& restore) noexcept;

private:
    std::vector<Layout> layout_;
    std::vector<KVCache> kv_;
    int64_t hidden_;
    int64_t position_ = 0;
};

} // namespace mfq::cuda::qwen35
