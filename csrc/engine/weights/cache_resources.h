#pragma once

#include "engine/weights/cache.h"

#include <optional>

class MoeExpertCache;
class TqCache;
class TqExec;
namespace tyloi::weights {
struct PrefillWorkspace;
struct KvLoanState;
struct Cache::Resources {
    Resources();
    ~Resources();
    std::shared_ptr<KvLoanState> kv;
    std::shared_ptr<MoeExpertCache> moe;
    std::shared_ptr<TqCache> tq;
    std::optional<std::size_t> device_bytes;
    std::size_t host_bytes = 0;
    std::size_t host_limit = 0;
    std::shared_ptr<TqExec> exec;
    std::unique_ptr<PrefillWorkspace> workspace;
};

} // namespace tyloi::weights
