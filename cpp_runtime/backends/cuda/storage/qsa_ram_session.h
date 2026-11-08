#pragma once
#include "storage/session_state.h"
#include "runtime/qsa_tq_session.h"
#include "mfq_paged_prefix_cache.h"
#include <cstring>

namespace mfq::cuda::internal {
struct QsaPrefixRangeReader {
    std::shared_ptr<const cache::PagedPrefixReadLease> lease;
    std::vector<uint8_t> operator()(size_t offset,size_t count)const{return lease->read_range(offset,count);}
    void operator()(size_t offset,uint8_t* destination,size_t count)const {
        if(count && !destination)throw std::invalid_argument("QSA prefix destination is null");
        if(lease->read_range_into)lease->read_range_into(offset,destination,count);
        else {
            const auto data=lease->read_range(offset,count);
            if(data.size()!=count)throw std::runtime_error("QSA prefix read length changed");
            std::memcpy(destination,data.data(),count);
        }
    }
};
// Reuse validated immutable readers from the same live file version. The
// second pass releases snapshot RAM only when its encoding matches that file.
bool reuse_ram_qsa_page(QsaTqSession&,size_t start,size_t count,
    const std::shared_ptr<const cache::PagedPrefixReadLease>&,bool release_readers);
void write_ram_qsa_page(CudaPagedWriter&,const QsaTqSession&,size_t,size_t,bool);
// A file lease keeps only the referenced extent alive. Without one, the
// standalone codec owns the compressed bytes copied from this single page.
void read_ram_qsa_page(CudaPagedReader&,const CudaPagedPayload&,
    const std::shared_ptr<const cache::PagedPrefixReadLease>&,
    size_t,size_t,bool,std::shared_ptr<QsaTqSession>&,bool bind_existing=false);
} // namespace mfq::cuda::internal
