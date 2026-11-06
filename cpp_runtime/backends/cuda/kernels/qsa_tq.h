#pragma once
#include <cstddef>
#include <cuda_runtime_api.h>
#include <cstdint>

namespace tyloinfer::qsa {

struct TqBlockView {
    const std::uint8_t* key = nullptr;
    const std::uint8_t* value = nullptr;
    // Zero denotes unpacked FP16 H4 modes (an incomplete or pending block).
    // Low byte: 1..8 uniform TQ; 254 Affine8; 255 row-dynamic TQ.
    // Next byte: TQ state width, zero preserves the original 16-bit profile.
    std::uint32_t key_bits = 8, value_bits = 8;
};

// The payload bit widths and trellis address width are independent. Keeping
// this in the descriptor preserves the canonical SSD image and compact rows.
__host__ __device__ constexpr std::uint32_t tq_precision_flags(std::uint8_t tag,std::uint8_t state_bits=16) {
    return std::uint32_t(tag)|(state_bits==16?0u:std::uint32_t(state_bits)<<8);
}

struct SelectedBlock {
    std::int32_t id = -1;
    std::uint32_t token_mask = 0;
};

// Published records and their payloads are immutable until prior consumers
// retire. Snapshot selected records before attention so every Q head sees
// the same image even when a background stream promotes a block concurrently.
int tq_publish_blocks(std::uint64_t* table,const std::int32_t* ids,
    const std::uint64_t* records,std::uint64_t* previous,int count,cudaStream_t stream);
int tq_snapshot_blocks(const std::uint64_t* table,const SelectedBlock* selected,
    TqBlockView* output,int queries,int selected_blocks,cudaStream_t stream);

// Input is contiguous [tokens, kv_heads, logical_width]. Missing lanes in the
// final four-token block and padded channel dimensions are zero-filled.
// Optional rotated outputs are [ceil(tokens/4)*4,heads,padded_width], allowing
// subsequent single-token appends to reuse rotations of the incomplete tail.
int tq_prepare_modes(const void* keys,const void* values,bool fp16,
    const float* key_signs,const float* value_signs,float* key_modes,float* value_modes,
    std::uint16_t* pending_keys,std::uint16_t* pending_values,
    int tokens,int heads,int width,int padded_width,cudaStream_t stream,
    float* rotated_keys=nullptr,float* rotated_values=nullptr);

// Canonical images are independent per four-token block and per K/V kind.
// R=heads*4*(padded_width/256); bytes per image=align4(3*R)+256*R.
// Layout: FP16 scales[R], INT8 zero points[R], alignment, INT8 codes[R,256].
int tq_prepare_affine8(const void* keys,const void* values,bool fp16,
    const float* key_signs,const float* value_signs,std::uint8_t* key_images,
    std::uint8_t* value_images,int tokens,int heads,int width,int padded_width,cudaStream_t stream);

// Unpack H4/RHT modes to [blocks, heads, 4, padded_width] for each K/V kind.
int tq_decode_modes(const TqBlockView* blocks,float* keys,float* values,
    const std::uint16_t* relative,int count,int heads,int padded_width,cudaStream_t stream);
int tq_restore_modes(const float* keys,const float* values,const float* key_signs,
    const float* value_signs,float* restored_keys,float* restored_values,
    int blocks,int heads,int width,int padded_width,cudaStream_t stream);
// The native QSA selector returns contiguous four-token groups plus a tail.
int tq_group_selected(const std::int32_t* tokens,SelectedBlock* blocks,
    int queries,int columns,cudaStream_t stream);
// Stage a strided FP16/FP32 append together with its previous incomplete block.
int tq_stage_append(const void* keys,const void* values,const void* index,bool key_fp16,bool value_fp16,bool index_fp16,
    const float* previous_keys,const float* previous_values,const float* previous_index,
    float* staged_keys,float* staged_values,float* staged_index,int tokens,int prior,int heads,int width,int index_width,
    std::int64_t key_token_stride,std::int64_t key_head_stride,std::int64_t value_token_stride,
    std::int64_t value_head_stride,std::int64_t index_token_stride,cudaStream_t stream);
int tq_commit_tail(const float* keys,const float* values,const float* index,float* tail_keys,float* tail_values,float* tail_index,
    int tokens,int remainder,int heads,int width,int index_width,cudaStream_t stream);
// One contiguous D2H image contains K, V and FP16 raw index rows for rollback.
int tq_pack_archive(const std::uint8_t* keys,const std::uint8_t* values,const float* raw_index,
    std::uint8_t* archive,int blocks,int image_bytes,int index_width,cudaStream_t stream);
// Copy the exact pooled INT16 Affine index beside its immutable raw archive.
int tq_archive_indexer(const std::int16_t* codes,const std::uint16_t* scales,
    const std::int16_t* zero,std::uint8_t* archive,std::size_t offset,int blocks,int width,
    std::int64_t start,const std::int64_t* position,cudaStream_t stream);
// Single-token incomplete block: append raw tails, H4/RHT, and publish FP16
// modes on device. The logical position remains live across graph replays.
// Optional query rotation shares the append launch. The result uses the
// cache's key-sign RHT basis and contiguous [heads,padded_width] storage.
struct PendingQuery {
    const void* input;
    float* output;
    int heads;
    std::int64_t head_stride;
    bool fp16;
};
int tq_append_pending(const void* key,const void* value,const void* index,bool key_fp16,bool value_fp16,bool index_fp16,
    const std::int64_t* position,float* tail_keys,float* tail_values,float* tail_index,
    float* rotated_tail_keys,float* rotated_tail_values,
    const float* key_signs,const float* value_signs,std::uint16_t* pending_keys,std::uint16_t* pending_values,
    TqBlockView* views,int heads,int width,int padded_width,int index_width,
    std::int64_t key_head_stride,std::int64_t value_head_stride,cudaStream_t stream,
    float* key_modes=nullptr,float* value_modes=nullptr,const PendingQuery* query=nullptr);
// Complete the fourth token in a captured graph. Destinations are indirect so
// each replay can own a different immutable image without recapturing.
int tq_finalize_pending(const float* key_modes,const float* value_modes,const float* raw_index,
    const TqBlockView* destination,TqBlockView* views,const std::int64_t* position,
    std::uint8_t* archive,int rows,int index_width,int destination_slots,cudaStream_t stream);
int tq_initialize_destinations(std::uint8_t* payload,TqBlockView* destinations,int image_bytes,int slots,cudaStream_t stream);
int tq_initialize_fp16_destinations(std::uint8_t* payload,TqBlockView* destinations,int image_bytes,int slots,cudaStream_t stream);

int indexer_affine16_encode(const float* input, std::int16_t* output,
    std::uint16_t* scales, std::int16_t* zero, int rows, int width, cudaStream_t stream);
int indexer_affine16_append(const float* input,std::int16_t* output,std::uint16_t* scales,
    std::int16_t* zero,const std::int64_t* position,int width,cudaStream_t stream);
int indexer_affine16_scores(const float* query, const std::int16_t* keys,
    const std::uint16_t* scales, const std::int16_t* zero,
    const std::int64_t* positions, float* output,
    int queries, int heads, int width, int blocks, cudaStream_t stream);

// Online TQ8: full 16-bit serialized states, sparse candidate transitions.
// candidates: [bins, 256 destination symbols, 8 predecessor symbols].
// trace: rows*256*256 bytes; payload: rows*256 bytes, reverse ring order.
int tq8_fit_online(const float* rows, const std::uint16_t* codebook,
    const std::uint8_t* candidates, int bins, float bin_min, float bin_step,
    float book_rms, float relative, std::uint16_t* anchors,
    std::uint8_t* payload, std::uint8_t* trace, float* sse,
    int count, cudaStream_t stream);

// Workspace sizes in floats: partial=queries*heads*parts*padded_width,
// partial_stats=queries*heads*parts*2, block_log_mass=queries*heads*selected,
// log_norm=queries*heads, rotated_query=queries*heads*padded_width.
// Choose bounded partial storage from the actual kernel occupancy. Both
// graph preparation and eager execution must use this same plan.
int tq_attention_partitions(int queries,int heads,int kv_heads,int padded,int count,
    bool query_fp16,bool allow_fp16_math,bool snapshots=false);

int tq_sparse_attention(const void* query, const float* key_signs, const float* value_signs,
    const std::uint16_t* relative, const TqBlockView* blocks, const SelectedBlock* selected,
    const std::int64_t* positions, void* output, float* rotated_query,
    float* partial, float* partial_stats, float* block_log_mass, float* log_norm,
    float* interval_hits, float* interval_mass,
    int queries, int heads, int kv_heads, int width, int padded_width,
    int selected_blocks, int parts, std::int64_t query_token_stride,
    std::int64_t query_head_stride, cudaStream_t stream,bool selected_views=false,
    const void* gate=nullptr,bool gate_fp16=false,bool output_fp16=false,
    std::int64_t gate_token_stride=0,std::int64_t gate_head_stride=0,bool query_fp16=false,
    bool allow_fp16_math=false,bool query_pre_rotated=false);

// Transient FP16 token tiles retain the exact operands of the streaming MMA
// kernel. Compressed resident/archive images remain unchanged.
bool tq_prefill_staging_available(int heads,int kv_heads,int padded);
int tq_stage_attention_blocks(const TqBlockView* blocks,const std::uint16_t* relative,
    std::uint16_t* keys,std::uint16_t* values,int count,int heads,int padded,cudaStream_t stream,
    const SelectedBlock* selected,int queries,int selected_count,std::uint32_t* requested);
int tq_sparse_attention_staged(const std::uint16_t* staged_keys, const std::uint16_t* staged_values, const void* query, const float* key_signs, const float* value_signs,
    const std::uint16_t* relative, const TqBlockView* blocks, const SelectedBlock* selected,
    const std::int64_t* positions, void* output, float* rotated_query,
    float* partial, float* partial_stats, float* block_log_mass, float* log_norm,
    float* interval_hits, float* interval_mass,
    int queries, int heads, int kv_heads, int width, int padded_width,
    int selected_blocks, int parts, std::int64_t query_token_stride,
    std::int64_t query_head_stride, cudaStream_t stream,bool selected_views=false,
    const void* gate=nullptr,bool gate_fp16=false,bool output_fp16=false,
    std::int64_t gate_token_stride=0,std::int64_t gate_head_stride=0,bool query_fp16=false,
    bool allow_fp16_math=false,bool query_pre_rotated=false);

} // namespace tyloinfer::qsa
