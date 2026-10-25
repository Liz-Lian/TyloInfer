#pragma once

#include "backend.h"

#include <cuda_runtime_api.h>
#ifdef TYLOINFER_HAVE_NCCL
#include <nccl.h>
#endif

#include <vector>

namespace mfq::cuda {

class Collectives {
public:
    using Stream = decltype(mfq_get_stream_from_pool(false));
    Collectives() = default;
    ~Collectives();
    Collectives(const Collectives&) = delete;
    Collectives& operator=(const Collectives&) = delete;

    void configure(const std::vector<int>& devices, bool allow_duplicates);
    void drain() const;
    void reset() noexcept;
    bool enabled() const noexcept { return collectives_enabled_; }
    const std::vector<Stream>& streams() const noexcept { return streams_; }

    mfq_tensor_backend::Tensor copy(mfq_tensor_backend::Tensor value, int device,
                                    mfq_tensor_backend::Tensor reusable = {});
    mfq_tensor_backend::Tensor sum(std::vector<mfq_tensor_backend::Tensor> outputs,
                                  int primary, bool reduce_to_primary, bool allow_fp16);

private:
    std::vector<int> devices_;
    std::vector<Stream> streams_;
    std::vector<cudaEvent_t> ready_, completed_;
    std::vector<mfq_tensor_backend::Tensor> reduction_buffers_;
#ifdef TYLOINFER_HAVE_NCCL
    std::vector<ncclComm_t> communicators_;
#endif
    bool collectives_enabled_ = false;
};

} // namespace mfq::cuda
