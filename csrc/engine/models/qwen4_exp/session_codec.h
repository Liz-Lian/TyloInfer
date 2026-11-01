#pragma once

#include "engine/models/common/session/paged.h"

struct TextSessionState;

namespace mfq::models::qwen4_exp {

CudaPagedPayload encode_block(const TextSessionState&, std::size_t, std::size_t);
TextSessionState decode_session(const std::vector<CudaPagedPayload>&,
                                const std::vector<int64_t>&, std::size_t);
bool exact(const CudaPagedPayload&);

} // namespace mfq::models::qwen4_exp
