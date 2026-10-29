#pragma once
#include "backend.h"
#include <nlohmann/json.hpp>
#include <vector>
namespace mfq::cuda::tests {
std::vector<Tensor> qwen4_session_check(const nlohmann::json&);
}
