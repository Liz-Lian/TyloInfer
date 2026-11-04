#pragma once

#include "engine/weights/source/gguf.h"

#include <string>
#include <vector>

namespace tyloi::gguf {

Record prepare(const Tensor& tensor, std::string name,
               std::vector<std::int64_t> shape);
void normalize(Record& record, const Tensor& tensor, bool decay_log);
Record join(std::string name, const Record& query, const Record& key);
Record integers(std::string name, const std::vector<std::int64_t>& values);

} // namespace tyloi::gguf
