#pragma once
#include <memory>
#include "tyloinfer/format_features.h"

struct NintPrefillDecodeCacheState;
// Reuse the existing temporary dense projection within one prefill layer.
// No decoded weights survive the scope or affect decode/MTP dispatch.
class NintPrefillDecodeCacheScope {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    std::unique_ptr<NintPrefillDecodeCacheState> state_;
#endif
public:
#if TYLOINFER_ENABLE_MFQ_COMPAT
    explicit NintPrefillDecodeCacheScope(bool enabled);
    ~NintPrefillDecodeCacheScope();
    void clear();
#else
    explicit NintPrefillDecodeCacheScope(bool) {}
    void clear() {}
#endif
    NintPrefillDecodeCacheScope(const NintPrefillDecodeCacheScope&)=delete;
    NintPrefillDecodeCacheScope& operator=(const NintPrefillDecodeCacheScope&)=delete;
};
