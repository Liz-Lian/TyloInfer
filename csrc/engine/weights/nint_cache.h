#pragma once

#include "engine/weights/encoding/format_features.h"
#include "engine/weights/nint.h"

#include <functional>
#include <memory>

struct NintCacheState;

// Reuse the existing temporary dense projection within one prefill layer.
// No decoded weights survive the scope or affect decode/MTP dispatch.
class NintCache {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    std::unique_ptr<NintCacheState> state_;
#endif
public:
#if TYLOINFER_ENABLE_MFQ_COMPAT
    explicit NintCache(bool enabled);
    ~NintCache();
    void clear();
    static bool active() noexcept;
    static ::tyloi::Tensor get(
        const NintWeight& weight,
        const std::function<::tyloi::Tensor()>& decode);
#else
    explicit NintCache(bool) {}
    void clear() {}
#endif
    NintCache(const NintCache&)=delete;
    NintCache& operator=(const NintCache&)=delete;
};
