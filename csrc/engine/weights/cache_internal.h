#pragma once

#include "engine/weights/cache.h"

#include <optional>

class MoeExpertCache;
class TqCache;
class TqExec;
namespace tyloi::weights::detail {
struct Workspace;
struct KvOwner;
struct CacheData {
    CacheData();
    ~CacheData();
    std::shared_ptr<KvOwner> kv;
    std::shared_ptr<MoeExpertCache> moe;
    std::shared_ptr<TqCache> tq;
    std::optional<std::size_t> device_bytes;
    std::size_t host_bytes = 0;
    std::size_t host_limit = 0;
    std::shared_ptr<TqExec> exec;
    std::unique_ptr<Workspace> workspace;
    std::size_t decode_lendable_bytes = 0;
    int decode_graph_depth = 0;
};

} // namespace tyloi::weights::detail
