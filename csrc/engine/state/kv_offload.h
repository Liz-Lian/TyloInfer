#pragma once

#include "backend.h"
#include "engine/state/qsa_blocks.h"
#include <memory>
#include <string>
#include <vector>
#include <utility>

namespace tyloi::cuda {
class Budget;
class KvExpertLoan;
struct KvOffloadConfig {
    std::size_t gpu_budget_bytes = 0;
    std::size_t ram_budget_bytes = 0;
    std::size_t buffer_bytes = 64ULL << 20;
    std::string directory;
    bool preallocate = false;
    bool ram_preallocated = true;
    bool ssd_enabled = true;
};

// Immutable microblocks have one native GPU hot copy. Eviction transfers to
// the shared RAM LRU; only RAM eviction writes the session-owned SSD file.
class KvOffloadStore {
public:
    struct Block;
    using BlockPtr = std::shared_ptr<Block>;
    struct GatherWorkspace {
        ::tyloi::DenseTensor rows, upload, host, descriptors, device_descriptors;
        int64_t heads=1,key_width=0;
    };
    std::shared_ptr<GatherWorkspace> gather_workspace(int64_t rows,int64_t width,::tyloi::ScalarType dtype,int64_t heads=1,int64_t key_width=0);
    std::size_t graph_workspace_bytes() const;
    explicit KvOffloadStore(KvOffloadConfig config);
    ~KvOffloadStore();
    BlockPtr write(const ::tyloi::DenseTensor& rows);
    ::tyloi::DenseTensor read(const BlockPtr& block);
    const KvOffloadConfig& config() const;
    std::vector<std::pair<std::string, double>> metrics() const;
    void flush();
    void bind_budget(Budget* budget);
    void set_expert_loan(std::shared_ptr<KvExpertLoan> loan);
    std::size_t expert_loan_bytes() const;
private:
    friend class KvOffloadSequence;
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

class KvOffloadSequence {
public:
    KvOffloadSequence(std::shared_ptr<KvOffloadStore> store, int64_t width, int64_t block_rows,
        ::tyloi::ScalarType dtype = ::tyloi::kFloat16);
    int64_t position() const { return position_; }
    void reset();
    void truncate(int64_t position);
    void append(const ::tyloi::DenseTensor& rows);
    ::tyloi::DenseTensor gather(const std::vector<int64_t>& rows) const;
    // Shared staging is reused between completed decode graph segments. The
    // prior consumer must finish before the next call overwrites these buffers.
    void gather_into(const std::vector<int64_t>& rows,const KvOffloadStore::GatherWorkspace& workspace) const;
    std::shared_ptr<KvOffloadStore::GatherWorkspace> gather_workspace(int64_t rows,int64_t heads=1,int64_t key_width=0) const {
        return store_->gather_workspace(rows,width_,dtype_,heads,key_width);
    }
    ::tyloi::DenseTensor range(int64_t start, int64_t count) const;
private:
    std::shared_ptr<KvOffloadStore> store_;
    int64_t width_, block_rows_, position_ = 0;
    ::tyloi::ScalarType dtype_;
    std::vector<KvOffloadStore::BlockPtr> blocks_;
};
}
