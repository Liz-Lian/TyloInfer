#pragma once
#include "tyloinfer/qsa_kv_store.h"
#include "tyloinfer/qsa_precision_policy.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tyloinfer::qsa {

inline constexpr std::uint8_t affine8_tag=254;
enum class RowPrecisionAllocation {Energy,Uniform};
struct MicroblockImage {
    std::vector<std::uint8_t> key, value;
    // 254: FP16 scale + INT8 zero per 256-element H4 mode, then INT8 codes.
    // 1..8: uniform TQ; 255: row precision allocated from H4 mode energy.
    std::uint8_t key_bits = affine8_tag, value_bits = affine8_tag;
    // Trellis address width, independent of the payload's bits per value.
    // Canonical Affine8 has no trellis; original TQ8 images use 16-bit states.
    std::uint8_t key_state_bits = 16, value_state_bits = 16;
};
struct BlockReplacement {
    std::size_t block = 0;
    std::uint8_t tier = 5;
    std::shared_ptr<const MicroblockImage> image;
};
struct AdaptiveCacheConfig {
    PrecisionPolicyConfig policy;
    std::size_t rows_per_kind = 0;
    std::uint8_t canonical_tag = affine8_tag;
    RowPrecisionAllocation row_allocation = RowPrecisionAllocation::Energy;
    std::uint8_t tq_state_bits = 16;
    std::size_t io_buffer_bytes = 8ULL<<20;
    // Without originals on SSD, retain only the current compressed image and
    // permit downward transitions. Raising a bit count cannot recover data.
    bool ssd_cache = false;
    std::string directory;
};
struct AdaptiveCacheStats {
    std::size_t blocks=0,resident_bytes=0;
    std::array<std::size_t,6> blocks_per_tier{};
    std::uint64_t promotions=0,demotions=0,updates=0;
    std::uint64_t canonical_read_bytes=0;
    double last_update_ms=0;
    mfq::cache::QsaKvStoreStats canonical;
};

class AdaptiveCache {
public:
    using Reencode=std::function<MicroblockImage(const MicroblockImage&,Precision)>;
    using ReencodeBatch=std::function<std::vector<MicroblockImage>(
        const std::vector<MicroblockImage>&,const std::vector<Precision>&)>;
    // A publisher must retain source images through its asynchronous copies
    // and publish descriptors only after those copies and prior readers finish.
    // A throwing publisher must leave device descriptors unchanged.
    using Publish=std::function<void(const std::vector<BlockReplacement>&)>;
    AdaptiveCache(AdaptiveCacheConfig config,Reencode encode,Publish publish);
    AdaptiveCache(AdaptiveCacheConfig config,ReencodeBatch encode,Publish publish);
    ~AdaptiveCache();
    AdaptiveCache(const AdaptiveCache&)=delete;
    AdaptiveCache& operator=(const AdaptiveCache&)=delete;
    std::size_t append(MicroblockImage canonical,std::uint64_t token_begin);
    // Publish contiguous microblocks together. Returns their first block ID;
    // an empty batch leaves the cache unchanged.
    std::size_t append_batch(std::vector<MicroblockImage> canonical,std::uint64_t token_begin);
    // Observations are consumed once. A pending request coalesces observations
    // instead of discarding them when another re-quantization is in progress.
    void update(std::uint64_t tokens,std::size_t budget,
        std::vector<double> hits,std::vector<double> attention_mass);
    // Finish pending quantization and release retired host images.
    void wait();
    void truncate(std::size_t blocks);
    // Immutable original, including after the resident image was demoted.
    MicroblockImage canonical_image(std::size_t block) const;
    MicroblockImage reconstruction_image(std::size_t block) const;
    AdaptiveCacheStats stats() const;
    std::array<std::size_t,6> tier_bytes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace tyloinfer::qsa
