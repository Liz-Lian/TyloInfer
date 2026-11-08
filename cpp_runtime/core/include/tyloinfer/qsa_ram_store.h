#pragma once
#include "tyloinfer/qsa_adaptive_cache.h"
#include "tyloinfer/qsa_ram_policy.h"
#include "tyloinfer/read_span.h"
#include "tyloinfer/file_range_reader.h"
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
// One retained file reader serves many immutable block slices. Keep callbacks
// with that source instead of embedding two std::functions in every block.
struct RamKvReadSource {
    std::size_t bytes=0;
    std::function<std::vector<std::uint8_t>(std::size_t,std::size_t)> read_range;
    std::function<void(std::size_t,std::uint8_t*,std::size_t)> read_range_into;
    bool valid()const{return bool(read_range) || bool(read_range_into) || bool(file_reader);}
    void validate(std::size_t offset,std::size_t count)const;
    std::vector<std::uint8_t> read(std::size_t offset,std::size_t count)const;
    void read_into(std::size_t offset,std::uint8_t* destination,std::size_t count)const;
    void read_into_batch(const std::vector<mfq::ReadSpan>& spans)const;
    std::function<void(const std::vector<mfq::ReadSpan>&)> read_ranges_into;
    std::shared_ptr<const mfq::FileRangeReader> file_reader;
    std::uint64_t file_offset=0;
};
struct RamKvBacking {
    // Keeping both references retains a borrowed prefix's file and extent.
    // A session spill never overwrites or takes ownership of a prefix file.
    std::shared_ptr<mfq::cache::QsaKvStore> store;
    mfq::cache::QsaKvStore::BlockPtr block;
    // An existing paged prefix supplies a retained byte-range reader. Its
    // callback owns the file lease; no session SSD copy is needed.
    std::shared_ptr<const RamKvReadSource> source;
    std::size_t offset=0;
    std::size_t key_bytes=0,value_bytes=0;
    std::uint8_t key_bits=0,value_bits=0,key_state_bits=16,value_state_bits=16;
    RamKvTier tier=RamKvTier::Tq4;
};
struct RamKvChange {
    std::size_t id=0;
    std::uint64_t generation=0;
    RamKvImagePtr image,previous;
    RamKvTier tier=RamKvTier::Fp16;
    // SSD descriptors have null payload pointers, but GPU cache hits still
    // need this exact version's byte lengths and precision flags.
    std::shared_ptr<const RamKvBacking> backing;
};
struct RamKvStoreConfig {
    RamPressureConfig pressure;
    bool prefix_enabled=false;
    std::shared_ptr<mfq::cache::QsaKvStore> session_ssd;
};
struct RamKvStoreStats {
    std::size_t blocks=0,ram_bytes=0;
    // Capacity for this context's nominal precision mix, before SSD eviction.
    std::size_t resident_target_bytes=0;
    std::size_t session_disk_bytes=0;
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
struct RamKvReadInto {
    const RamKvSnapshotBlock* source=nullptr;
    std::uint8_t* destination=nullptr;
    std::size_t capacity=0;
};
struct RamKvRelocation {
    std::size_t index=0;
    RamKvImagePtr image;
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
    // Synchronously copy a borrowed encoded image directly into backend
    // storage, without constructing temporary owning K/V arrays.
    using AllocateView=std::function<RamKvImagePtr(const RamKvImage&)>;
    // Remaining images of this K/V byte extent in the current policy batch,
    // including the current one. File-only blocks are excluded from the hint.
    using RestoreAllocate=std::function<RamKvImagePtr(const MicroblockImage&,std::size_t)>;
    using Encode=std::function<std::vector<RamKvImagePtr>(const std::vector<RamKvImagePtr>&,RamKvTier)>;
    // Queue descriptors in stream order and retain each previous owner until
    // preceding GPU readers finish. Publication must have a strong exception
    // guarantee: throwing leaves its descriptor queue unchanged.
    using Publish=std::function<void(const std::vector<RamKvChange>&)>;
    RamKvStore(RamKvStoreConfig config,Allocate allocate,Encode encode,Publish publish);
    RamKvStore(RamKvStoreConfig config,Allocate allocate,Encode encode,Publish publish,AllocateView allocate_view);
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
    // Replace immutable physical images without changing precision, LRU,
    // generation or backing. Publication retains every previous reader.
    using Relocate=std::function<std::vector<RamKvRelocation>(const std::vector<RamKvImagePtr>&)>;
    std::size_t relocate(const Relocate& relocate);
    // Fetch an SSD miss even if it is too cold to displace RAM. Admission
    // retains its actual encoded precision and replaces only older TQ4
    // blocks; higher-precision floors stay protected. Recall never fits KV.
    RamKvImagePtr read(std::size_t id,std::size_t budget,bool admit=true);
    // Immutable CPU reads do not call the backend allocator or encoder. They
    // are safe while a device graph waits for its host I/O task.
    RamKvSnapshotBlock source(std::size_t id)const;
    MicroblockImage read_cpu(const RamKvSnapshotBlock& source);
    // Retain the encoded backing without splitting K/V into new arrays.
    // retained_bytes is the temporary allocation capacity, or zero when
    // the resident storage is already included in the backend's accounting.
    RamKvImage read_cpu_view(const RamKvSnapshotBlock& source,std::size_t* retained_bytes=nullptr);
    // K starts at destination and V at align16(key_bytes). The caller owns
    // this separate buffer and retains it until all readers have finished.
    void read_cpu_into(const RamKvSnapshotBlock& source,std::uint8_t* destination,std::size_t capacity);
    void read_cpu_into_batch(const std::vector<RamKvReadInto>& requests);
    // Upper bound for sequential file plans and their bounded per-worker windows.
    static std::size_t sequential_read_workspace(std::size_t requests);
    // Random recall retains exact byte ranges and skips sequential readahead.
    void read_cpu_into_batch(const std::vector<RamKvReadInto>& requests,bool sequential);
    // Read tightly packed K/V pairs for bulk transfer without image padding.
    void read_packed_cpu_into_batch(const std::vector<RamKvReadInto>& requests,bool sequential=true);
    void admit_read(std::size_t id,std::uint64_t generation,const MicroblockImage& image,std::size_t budget);
    // A borrowed view is consumed synchronously; rejected admissions do not copy it.
    void admit_read(std::size_t id,std::uint64_t generation,const RamKvImage& image,std::size_t budget);
    void admit_read(std::size_t id,std::uint64_t generation,RamKvImagePtr image,std::size_t budget);
    bool graph_reads_supported()const;
    RamKvImagePtr resident(std::size_t id)const;
    std::uint64_t generation(std::size_t id)const;
    std::shared_ptr<const RamKvBacking> backing(std::size_t id)const;
    void truncate(std::size_t blocks);
    // Commit restoration metadata before sharing the remaining payload budget
    // across caches. Live entries, their generations and readers are preserved.
    void reserve_metadata(std::size_t blocks);
    // Reset invalidates entries but may leave a large vector capacity. An idle
    // request boundary can return it without touching retained snapshot data.
    std::size_t release_idle_metadata();
    RamKvStoreStats stats()const;
    std::shared_ptr<RamKvSnapshot> snapshot(std::size_t blocks)const;
    void restore(const RamKvSnapshot& state,std::size_t budget,bool borrow_prefix,
                 bool reuse_resident=false);
    // Runs in block order after each source is read (or retained on SSD),
    // outside the entry mutex. Companion archives can share sequential I/O.
    void restore(const RamKvSnapshot& state,std::size_t budget,bool borrow_prefix,
                 bool reuse_resident,const std::function<void(std::size_t)>& block_read,
                 const RestoreAllocate& allocate_batch={});
    // Prefer spare space for imminent appends, while retaining the real budget
    // if that preference cannot accommodate the protected precision floors.
    void restore(const RamKvSnapshot& state,std::size_t budget,bool borrow_prefix,
                 bool reuse_resident,const std::function<void(std::size_t)>& block_read,
                 const RestoreAllocate& allocate_batch,std::size_t append_bytes);
private:
    void read_cpu_image(const RamKvSnapshotBlock& source,std::uint8_t* destination,std::size_t capacity,bool padded);
    void read_cpu_batch(const std::vector<RamKvReadInto>& requests,bool sequential,bool padded);
    struct Impl;std::shared_ptr<Impl> impl_;
};
} // namespace tyloinfer::qsa
