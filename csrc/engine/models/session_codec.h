#pragma once

#include "engine/models/common/session/paged.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mfq { class ModelSource; }
namespace mfq::cuda::internal {
std::string cuda_prefix_cache_compatibility_key(const mfq::ModelSource&, int64_t);
}

size_t cuda_paged_decodable_blocks(const std::vector<CudaPagedPayload>&);
size_t cuda_paged_payload_tokens(const CudaPagedPayload&);
