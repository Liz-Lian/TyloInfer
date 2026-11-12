#pragma once

#include "engine/tensor.h"

#include <cstdint>
#include <string>

namespace tyloi::eval {

/** Read an MFQTNSR1 diagnostic tensor, or return an undefined tensor when optional and absent. */
DenseTensor read_tensor(const std::string& path, bool required);
/** Write a defined tensor to an MFQTNSR1 diagnostic file. */
void write_tensor(const DenseTensor& value, const std::string& path);
/** Read an optional scalar diagnostic tensor as int64. */
std::int64_t read_scalar(const std::string& path, std::int64_t fallback);

} // namespace tyloi::eval
