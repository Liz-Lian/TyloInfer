#pragma once

#include "engine/models/common/session/paged.h"

struct TextSessionState;

namespace tyloi::models::deepseek_v41 {

CudaPagedPayload encode_block(const TextSessionState&, std::size_t, std::size_t);
TextSessionState decode_session(const std::vector<CudaPagedPayload>&,
                                const std::vector<int64_t>&, std::size_t);
bool exact(const CudaPagedPayload&);
bool checkpoint_needs_update(const CudaPagedPayload&, const TextSessionState&);

} // namespace tyloi::models::deepseek_v41
