#pragma once

#include "engine/tensor.h"

#include <cstdint>
#include <memory>

namespace tyloi::ops {

/** Execute conditional compact top-k sampling within captured decode windows. */
class Compact {
public:
    /** Allocate control storage in the active backend scope. */
    Compact();
    /** Retire control storage after all captured consumers have finished. */
    ~Compact();
    Compact(Compact&&) noexcept;
    Compact& operator=(Compact&&) noexcept;
    Compact(const Compact&) = delete;
    Compact& operator=(const Compact&) = delete;
    /** Publish the next replay's sampling condition on the active stream. */
    void set(bool enabled, const char* role, std::int64_t rows);
    /** Select 64 compact candidates for each logits row under the published condition. */
    Tensor forward(const Tensor& logits) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tyloi::ops
