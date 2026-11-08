#pragma once
#include "qsa_tq.h"
#include <cstddef>

namespace tyloinfer::qsa {
inline constexpr int ram_cache_ways=8;
// A live append may change on every graph replay. Stage it for this window
// without retaining a persistent copy under an immutable block generation.
inline constexpr std::uint64_t ram_uncached_generation=~std::uint64_t(0);
__host__ __device__ constexpr unsigned ram_cache_set(unsigned id,unsigned sets) {
    // Mix high bits before indexing: regular token strides must not all
    // compete for the same small subset of cache sets.
    id^=id>>16;id*=0x7feb352du;id^=id>>15;id*=0x846ca68bu;id^=id>>16;
    return id%sets;
}

// Payload addresses must be GPU-readable: mapped, pinned RAM or device
// memory. Byte lengths preserve FP16, Affine8 and variable-precision TQ.
struct RamBlockView {
    TqBlockView image;
    std::uint32_t key_bytes=0,value_bytes=0;
    // Increment whenever an immutable logical block is replaced, including
    // precision changes and rollback. The GPU cache rejects older versions.
    std::uint64_t generation=0;
};

struct RamCacheSlot {
    // An image can occupy a power-of-two aligned group of base slots. Every
    // occupied way repeats its owner/version; overlapping evictions clear
    // that entire owner before any part of its payload can be reused.
    std::int32_t id=-1;
    std::uint64_t generation=0,last_frame=0;
};
struct RamStageCache {
    // Set-associative LRU. A selected slot is protected through the
    // current attention window; excess misses use transient staging slots.
    RamCacheSlot* slots=nullptr;
    // Temporary selection-list heads. One warp owns each set, with no
    // inter-warp spinlock. Retain this slot in the descriptor's ABI.
    std::uint32_t* locks=nullptr;
    std::uint64_t* frame=nullptr;
    std::uint8_t* payload=nullptr;
    std::size_t payload_bytes=0,slot_bytes=0;
    int sets=0;
};
// Resolve selected records from an already uploaded layer; copy descriptors
// only. Payloads remain shared across all prefill query windows.


int tq_reset_ram_stage_cache(const RamStageCache& cache,cudaStream_t stream);

// Serialized publication follows prior attention reads on the same stream.
int tq_publish_ram_sources(RamBlockView* sources,int capacity,
    const std::int32_t* ids,const RamBlockView* images,int count,cudaStream_t stream,TqBlockView* live=nullptr);
// Bridge newly generated device images while their RAM archive is pending.
// The live table contains FP16 H4 modes or canonical Affine8. Position is
// either a device graph scalar or the supplied host position.
int tq_publish_ram_append(RamBlockView* sources,int capacity,const TqBlockView* live,
    const std::int64_t* position,std::int64_t host_position,int tokens,
    std::uint32_t fp16_bytes,std::uint32_t affine8_bytes,cudaStream_t stream);
// Find the SSD fetch set on device. A packed VRAM cache hit needs no RAM
// restoration. The subsequent staging call protects every selected hit from
// replacement, including hits whose warp has not visited its slot yet.
int tq_collect_ram_missing(const RamBlockView* sources,int capacity,const SelectedBlock* selected,
    int queries,int count,std::uint32_t* first_use,std::int32_t* missing,int* missing_count,
    const RamStageCache& cache,cudaStream_t stream);
// A captured collector reads the current cache layout from a stable device
// descriptor; RAM/SSD transitions can change its slot size between replays.
int tq_collect_ram_missing(const RamBlockView* sources,int capacity,const SelectedBlock* selected,
    int queries,int count,std::uint32_t* first_use,std::int32_t* missing,int* missing_count,
    const RamStageCache* device_cache,cudaStream_t stream);
// Control data only: write the selected IDs and miss list to mapped pinned
// memory. The caller joins the containing graph before reading either list.
int tq_export_ram_selection(const std::int32_t* missing,const SelectedBlock* selected,
    std::int32_t* mapped_missing,SelectedBlock* mapped_selected,int queries,int count,cudaStream_t stream);
enum RamGraphControl {RamGraphRequest=0,RamGraphReady=1,RamGraphImages=2,RamGraphError=3};
// Temporarily publish CPU-read packed images for one selected window. Backup
// count is device-owned, so cancellation cannot prevent its later rollback.
int tq_publish_ram_graph(RamBlockView* sources,int capacity,const std::int32_t* mapped_ids,
    const RamBlockView* mapped_images,std::uint32_t* control,RamBlockView* backup,
    std::uint32_t* backup_count,int maximum,cudaStream_t stream);
int tq_cancel_ram_graph_selection(SelectedBlock* selected,int count,const std::uint32_t* backup_count,cudaStream_t stream);
int tq_restore_ram_graph(RamBlockView* sources,const std::int32_t* mapped_ids,const RamBlockView* backup,
    const std::uint32_t* backup_count,int maximum,cudaStream_t stream);

// Deduplicate the complete query window on device and copy every selected
// logical block once. No selected IDs or payloads pass through CPU staging.
// first_use has capacity entries; selected_views has queries*count entries.
// arena has queries*count slots, each slot_bytes bytes and aligned to 16.
// Each source K image plus its 16-byte padding and V image must fit a slot.
// The returned views are indexed by selection position, as required by
// tq_sparse_attention(selected_views=true). Invalid/empty selections have
// null views. Preserve source images and scratch until prior readers finish.
// All storage is caller-owned, allowing reuse in a captured CUDA graph.
int tq_stage_ram_selected(const RamBlockView* sources,int capacity,
    const SelectedBlock* selected,int queries,int count,
    std::uint32_t* first_use,TqBlockView* selected_views,
    void* arena,std::size_t arena_bytes,std::size_t slot_bytes,
    std::uint64_t* copied_bytes,cudaStream_t stream);

// The cache and source table belong to one serialized execution stream.
// Store only original packed images, without persistent FP16 expansion.
int tq_stage_ram_selected_cached(const RamBlockView* sources,int capacity,
    const SelectedBlock* selected,int queries,int count,
    std::uint32_t* first_use,TqBlockView* selected_views,
    void* arena,std::size_t arena_bytes,std::size_t slot_bytes,
    const RamStageCache& cache,std::uint64_t* copied_bytes,
    std::uint64_t* hit_blocks,cudaStream_t stream);
// The envelope bounds allocated cache sets and the launch grid. Replays read
// the current layout from device_cache, which may shrink its slots as RAM KV
// is quantized without rebuilding the graph or changing its payload budget.
int tq_stage_ram_selected_cached(const RamBlockView* sources,int capacity,
    const SelectedBlock* selected,int queries,int count,
    std::uint32_t* first_use,TqBlockView* selected_views,
    void* arena,std::size_t arena_bytes,std::size_t slot_bytes,
    const RamStageCache& envelope,std::uint64_t* copied_bytes,
    std::uint64_t* hit_blocks,cudaStream_t stream,const RamStageCache* device_cache);

enum class RamStageStep {Available,Hydrated};
// Available copies resident images and packed cache hits, leaving null views
// for actual SSD misses. Hydrated fills only those null views after source
// publication and admits the fetched images. Both stages protect selected
// cache hits and allocations still used by this attention window.
// Keep the same selection, scratch and cache between the steps;
// selected resident images must remain leased until both copies have retired.
// This lets SSD reads overlap the first GPU copy without copying any image
// twice, advancing the LRU twice, or displacing a selected cache hit.
int tq_stage_ram_selected_step(const RamBlockView* sources,int capacity,
    const SelectedBlock* selected,int queries,int count,
    std::uint32_t* first_use,TqBlockView* selected_views,
    void* arena,std::size_t arena_bytes,std::size_t slot_bytes,
    const RamStageCache& cache,RamStageStep step,std::uint64_t* copied_bytes,
    std::uint64_t* hit_blocks,cudaStream_t stream);

} // namespace tyloinfer::qsa
