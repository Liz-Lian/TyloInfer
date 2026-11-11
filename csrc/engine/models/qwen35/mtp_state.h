#pragma once

#include "engine/state/kv_view.h"
#include "engine/models/session/snapshot.h"
#include "engine/models/state.h"

#include <vector>

namespace tyloi::cuda::qwen35 {

class MtpState final : public tyloi::models::MtpState {
public:
    struct Restore {
        std::vector<tyloi::state::KvView> kv;
        int64_t position;
    };

    struct Layout {
        int device;
        int64_t heads, head_dim, capacity;
        ::tyloi::ScalarType dtype;
        tyloi::state::KvQuantization quantization;
    };

    MtpState(const tyloi::models::Mtp& owner, std::vector<Layout> layout, int64_t hidden);
    int64_t position() const noexcept override { return position_; }
#ifdef TYLOINFER_NATIVE_CUDA
    /** Plan new KV backing for the next predictor operation. */
    std::vector<std::pair<tyloi::Device, std::size_t>> growth(
        int64_t batch, int64_t tokens, int64_t capacity) const;
    /** Validate and retain an independent predictor restore candidate. */
    std::unique_ptr<tyloi::models::Restore> prepare_restore(const MtpSessionState& saved);
#endif
    void prepare(int64_t batch, int64_t tokens, int64_t kl_capacity = 0);
    void reset(int64_t batch);
    void advance(int64_t tokens);
    void trim(int64_t position);
    std::vector<tyloi::state::KvView> views() const;
    MtpSessionState capture(int64_t target_position,
        const ::tyloi::DenseTensor& hidden) const;
    Restore prepare(const MtpSessionState& saved) const;
    void stage(Restore& restore, const MtpSessionState& saved) const;
    void publish(Restore&& restore) noexcept;

private:
    std::vector<Layout> layout_;
    std::vector<tyloi::state::KvView> kv_;
    int64_t hidden_;
    int64_t position_ = 0;
};

} // namespace tyloi::cuda::qwen35
