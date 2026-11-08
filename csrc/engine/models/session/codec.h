#pragma once

#include "engine/models/session/paged.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tyloi { class ModelSource; }
namespace tyloi::cuda::internal {
std::string cuda_prefix_cache_compatibility_key(const tyloi::ModelSource&, int64_t);
}

size_t cuda_paged_decodable_blocks(const std::vector<CudaPagedPayload>&);
size_t cuda_paged_payload_tokens(const CudaPagedPayload&);
