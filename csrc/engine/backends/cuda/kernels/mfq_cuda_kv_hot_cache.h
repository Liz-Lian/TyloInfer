#pragma once
#include "backend.h"
#include <cstdint>

namespace tyloi::cuda {
// Addresses stay fixed across CUDA Graph replays. A negative mapping reads
// the canonical mapped-RAM row; resident mappings read the GPU copy.
struct HotKvView {
    const void* keys = nullptr;
    const void* values = nullptr;
    const int32_t* mapping = nullptr;
    int64_t cells = 0;
    const int32_t* cold_mapping = nullptr;
    int64_t cold_cells = 0;
};
struct HotKvCache {
    ::tyloi::DenseTensor keys, values, mapping, slot_ids, requested, epochs, queue, control;
    // In exclusive mode a logical row lives in exactly one tier. The cold
    // tensors contain only maximum-cells rows of mapped RAM.
    ::tyloi::DenseTensor cold_keys, cold_values, cold_mapping, free_cold, free_gpu, cold_control;
    static HotKvCache allocate(int64_t heads, int64_t maximum, int64_t cells,
        int64_t key_width, int64_t value_width, const ::tyloi::TensorOptions& options);
    bool defined() const { return mapping.defined(); }
    bool exclusive() const { return cold_mapping.defined(); }
    void set_cold_storage(const ::tyloi::DenseTensor& keys, const ::tyloi::DenseTensor& values);
    void append(const ::tyloi::DenseTensor& keys, const ::tyloi::DenseTensor& values,
        const ::tyloi::DenseTensor& positions);
    // Short contiguous appends: reserve/evict/write in one CTA, optionally
    // publishing the indexer rows in the same launch. False leaves state alone.
    // begin_prepare initializes the next prepare(..., skip_small_prefix=true,
    // epoch_ready=true); both calls must use the same positions and stream.
    bool append_decode(const ::tyloi::DenseTensor& keys, const ::tyloi::DenseTensor& values,
        const ::tyloi::DenseTensor& positions,
        const ::tyloi::DenseTensor& index_rows = {}, const ::tyloi::DenseTensor& index_storage = {},
        bool begin_prepare = false);
    std::pair<::tyloi::DenseTensor,::tyloi::DenseTensor> gather(int64_t count) const;
    HotKvView view() const;
    std::size_t bytes() const;
    void reset();
    void invalidate(const ::tyloi::DenseTensor& positions);
    // Logical append positions are contiguous and increasing. Publish the
    // newest resident window directly; rewritten positions discard old tags.
    void publish(const ::tyloi::DenseTensor& source_keys, const ::tyloi::DenseTensor& source_values,
        const ::tyloi::DenseTensor& positions);
    void prepare(const ::tyloi::DenseTensor& source_keys, const ::tyloi::DenseTensor& source_values,
        const ::tyloi::DenseTensor& selected, const ::tyloi::DenseTensor& positions = {}, int64_t columns = 0,
        bool skip_small_prefix = false, bool epoch_ready = false);
};
}
