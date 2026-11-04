#pragma once
#include <cstddef>

namespace tyloi::cpu {
// Copy non-overlapping host fields and publish their stores when the batch ends.
class HostMemoryCopyBatch {
    using Copy=void(*)(void*,const void*,std::size_t) noexcept;
    using Finish=void(*)() noexcept;
    Copy copy_;
    Finish finish_=nullptr;
public:
    explicit HostMemoryCopyBatch(bool streaming=true) noexcept;
    ~HostMemoryCopyBatch() noexcept {if(finish_)finish_();}
    HostMemoryCopyBatch(const HostMemoryCopyBatch&)=delete;
    HostMemoryCopyBatch& operator=(const HostMemoryCopyBatch&)=delete;
    void copy(void* destination,const void* source,std::size_t bytes) const noexcept {
        copy_(destination,source,bytes);
    }
};
} // namespace tyloi::cpu
