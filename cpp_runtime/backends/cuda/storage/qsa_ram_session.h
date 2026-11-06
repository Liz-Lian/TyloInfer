#pragma once
#include "storage/session_state.h"
#include "runtime/qsa_tq_session.h"
#include "mfq_paged_prefix_cache.h"

namespace mfq::cuda::internal {
void write_ram_qsa_page(CudaPagedWriter&,const QsaTqSession&,size_t,size_t,bool);
// A file lease keeps only the referenced extent alive. Without one, the
// standalone codec owns the compressed bytes copied from this single page.
void read_ram_qsa_page(CudaPagedReader&,const CudaPagedPayload&,
    const std::shared_ptr<const cache::PagedPrefixReadLease>&,
    size_t,size_t,bool,std::shared_ptr<QsaTqSession>&,bool bind_existing=false);
} // namespace mfq::cuda::internal
