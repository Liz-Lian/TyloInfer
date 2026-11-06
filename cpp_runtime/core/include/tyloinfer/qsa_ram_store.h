#pragma once
#include "tyloinfer/qsa_adaptive_cache.h"
#include "tyloinfer/qsa_ram_policy.h"
#include <functional>
#include <memory>
#include <vector>

namespace tyloinfer::qsa {
// Storage is supplied by the backend. A CUDA backend can own one mapped,
// pinned image without retaining an additional pageable copy of its bytes.
struct RamKvImage {
    std::shared_ptr<const void> owner;
    const std::uint8_t* key=nullptr;
    const std::uint8_t* value=nullptr;
    std::size_t key_bytes=0,value_bytes=0;
    std::uint8_t key_bits=0,value_bits=0,key_state_bits=16,value_state_bits=16;
    std::size_t bytes()const{return key_bytes+value_bytes;}
};
using RamKvImagePtr=std::shared_ptr<const RamKvImage>;
struct RamKvBacking {
    // Keeping both references retains a borrowed prefix's file and extent.
    // A session spill never overwrites or takes ownership of a prefix file.
    std::shared_ptr<mfq::cache::QsaKvStore> store;
    mfq::cache::QsaKvStore::BlockPtr block;
    // An existing paged prefix supplies a retained byte-range reader. Its
    // callback owns the file lease; no session SSD copy is needed.
    std::function<std::vector<std::uint8_t>()> read;
    std::size_t key_bytes=0,value_bytes=0;
    std::uint8_t key_bits=0,value_bits=0,key_state_bits=16,value_state_bits=16;
    RamKvTier tier=RamKvTier::Tq4;
};
struct RamKvChange {
    std::size_t id=0;
    std::uint64_t generation=0;
    RamKvImagePtr image,previous;
    RamKvTier tier=RamKvTier::Fp16;
};
struct RamKvStoreConfig {
    RamPressureConfig pressure;
    bool prefix_enabled=false;
    std::shared_ptr<mfq::cache::QsaKvStore> session_ssd;
};
struct RamKvStoreStats {
    std::size_t blocks=0,ram_bytes=0;
    std::array<std::uint64_t,4> tokens{};
    std::uint64_t quantized_blocks=0,spilled_blocks=0,promoted_blocks=0;
    std::uint64_t ssd_written_bytes=0,ssd_read_bytes=0,prefix_reused_bytes=0;
};
struct RamKvSnapshotBlock {
    RamKvImagePtr image;
    std::shared_ptr<const RamKvBacking> backing;
    std::array<std::size_t,3> bytes{};
    std::uint64_t generation=0,access=0;
    RamKvTier tier=RamKvTier::Fp16;
    MicroblockImage read()const;
};
struct RamKvSnapshot {
    std::vector<RamKvSnapshotBlock> blocks;
    // Binding is an explicit opt-in from an enabled durable prefix cache.
    // It releases snapshot RAM readers and lends that file to matching live
    // KV versions too. An expired source store is harmless.
    std::function<void(std::size_t,std::uint64_t,std::shared_ptr<const RamKvBacking>)> bind_live;
    void bind_prefix(std::size_t id,std::shared_ptr<const RamKvBacking> backing);
};
class RamKvStore {
public:
    using Allocate=std::function<RamKvImagePtr(const MicroblockImage&)>;
    using Encode=std::function<std::vector<RamKvImagePtr>(const std::vector<RamKvImagePtr>&,RamKvTier)>;
    // Queue descriptors in stream order and retain each previous owner until
    // preceding GPU readers finish. Publication must have a strong exception
    // guarantee: throwing leaves its descriptor queue unchanged.
    using Publish=std::function<void(const std::vector<RamKvChange>&)>;
    RamKvStore(RamKvStoreConfig config,Allocate allocate,Encode encode,Publish publish);
    ~RamKvStore();
    RamKvStore(const RamKvStore&)=delete;
    RamKvStore& operator=(const RamKvStore&)=delete;
    std::size_t append(RamKvImagePtr image,std::array<std::size_t,3> tier_bytes,
        std::uint64_t epoch,std::shared_ptr<const RamKvBacking> prefix={});
    // A newly decoded microblock may enter at a higher tier while a full
    // background quantization batch accumulates. Its actual bytes are charged.
    std::size_t append_at_tier(RamKvImagePtr image,std::array<std::size_t,3> tier_bytes,
        RamKvTier tier,std::uint64_t epoch,std::shared_ptr<const RamKvBacking> prefix={});
    void touch(const std::vector<std::size_t>& ids,std::uint64_t epoch);
    // Touch and retain selected resident versions under one lock. SSD entries
    // are omitted; retained RAM images cannot be evicted during admission.
    std::vector<RamKvChange> lease(const std::vector<std::size_t>& ids,std::uint64_t epoch);
    // Optional format maintenance may quantize full batches without spilling
    // data to SSD merely to meet its artificial byte target.
    RamPressurePlan maintain(std::size_t budget,bool allow_ssd=true);
    // Fetch an SSD miss even if it is too cold to displace RAM. Admission
    // replaces only older TQ4 blocks; higher-precision floors stay protected.
    RamKvImagePtr read(std::size_t id,std::size_t budget,bool admit=true);
    RamKvImagePtr resident(std::size_t id)const;
    std::uint64_t generation(std::size_t id)const;
    std::shared_ptr<const RamKvBacking> backing(std::size_t id)const;
    void truncate(std::size_t blocks);
    RamKvStoreStats stats()const;
    std::shared_ptr<RamKvSnapshot> snapshot(std::size_t blocks)const;
    void restore(const RamKvSnapshot& state,std::size_t budget,bool borrow_prefix,
                 bool reuse_resident=false);
private:
    struct Impl;std::shared_ptr<Impl> impl_;
};
} // namespace tyloinfer::qsa
