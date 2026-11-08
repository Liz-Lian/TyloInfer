#pragma once

#include "engine/models/session/paged.h"

struct TextSessionState;

namespace tyloi::models::qwen4_exp {

CudaPagedPayload encode_block(const TextSessionState&, std::size_t, std::size_t);
TextSessionState decode_session(const std::vector<CudaPagedPayload>&,
                                const std::vector<int64_t>&, std::size_t);
bool exact(const CudaPagedPayload&);

} // namespace tyloi::models::qwen4_exp
