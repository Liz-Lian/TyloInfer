#pragma once
#include "mfq_tensor_backend.h"
#include <cstdint>
#include <vector>

// A prepared table-row transfer. Encodings stay in their own loader; graph
// consumers use stable output storage and never allocate inside upload/decode.
class RowStage {
public:
    virtual ~RowStage()=default;
    virtual mfq_tensor_backend::Tensor output()const=0;
    virtual bool mapped_rows()const=0;
    virtual void upload(const std::vector<std::int64_t>& ids,const std::vector<std::int64_t>& shape)=0;
    virtual void decode()=0;
    // The owning decode window has completed earlier consumers before this
    // upload; this window's consumer remains behind its publication flag.
    // Providers needing DMA retain their normal upload synchronization.
    virtual void upload_graph(const std::vector<std::int64_t>& ids,const std::vector<std::int64_t>& shape) {
        upload(ids,shape);
    }
};
