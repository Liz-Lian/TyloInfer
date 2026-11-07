#pragma once

#include "backend.h"
#include "engine/ops.h"

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
namespace tyloi::cuda::ops {
tyloi::ops::RopeTable rope_table(int64_t positions, int64_t dimension, double base, Device device,
                                 int64_t frequency_dim, int64_t active_pairs);
tyloi::DenseTensor rope(tyloi::DenseTensor input, tyloi::DenseTensor positions, const tyloi::ops::RopeTable& table,
                 bool bf16, bool grid_mrope_positions);
}
#endif

struct RopeCache {
    ::tyloi::DenseTensor cos;
    ::tyloi::DenseTensor sin;
    ::tyloi::DenseTensor sections;
    ::tyloi::DenseTensor empty_sections;
    ::tyloi::DenseTensor interleaved_order;
    ::tyloi::DenseTensor interleaved_inverse;
    int64_t rotary_dim = 0;

    RopeCache() = default;
    RopeCache(
        int64_t max_positions,
        int64_t dim,
        double base,
        int64_t frequency_dim = 0,
        int64_t active_pairs = -1,
        ::tyloi::Device device =
            ::tyloi::Device(::tyloi::kCUDA),
        bool official_reciprocal_frequencies = false,
        tyloi::ops::YarnScaling yarn = {});

    void configure_mrope(
        const std::vector<int64_t>& configured_sections,
        bool interleaved,
        int64_t configured_rotary_dim,
        ::tyloi::Device device);
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    tyloi::ops::RopeTable view() const;
#endif
    ::tyloi::DenseTensor apply(
        ::tyloi::DenseTensor x,
        ::tyloi::DenseTensor pos,
        bool grid_mrope_positions = false) const;
    ::tyloi::DenseTensor apply_bf16(
        bool fused,
        ::tyloi::DenseTensor x,
        ::tyloi::DenseTensor pos) const;
};

namespace tyloi::cuda {

// Computes rotate-half frequencies for logical [axis,batch,token] positions.
// Uses FP32 for F32 inputs and FP16 otherwise, without a resident table.
class RotaryEmbedding {
public:
    RotaryEmbedding(int64_t dimension, int64_t maximum, double base,
                    std::vector<int64_t> sections, bool interleaved,
                    bool fused, tyloi::ops::YarnScaling yarn = {});
    ::tyloi::DenseTensor forward(const ::tyloi::DenseTensor& value,
                                       const ::tyloi::DenseTensor& positions) const;
    // Switch only after execution and captured graphs using this module finish.
    void set_fused(bool enabled) { fused_ = enabled && !yarn_.enabled(); }
    bool fused() const { return fused_; }
    bool pool_normalized_into(const ::tyloi::DenseTensor& index,
        const ::tyloi::DenseTensor& weight,const ::tyloi::DenseTensor& positions,
        const ::tyloi::DenseTensor& cache_positions,const ::tyloi::DenseTensor& output,
        int64_t pool,int64_t maximum,double eps) const;
    ::tyloi::DenseTensor forward_normalized(
        const ::tyloi::DenseTensor& value, const ::tyloi::DenseTensor& weight,
        const ::tyloi::DenseTensor& positions, double eps,
        const ::tyloi::DenseTensor& key_cache = {},
        const ::tyloi::DenseTensor& cache_positions = {},
        const ::tyloi::DenseTensor& projected_value = {},
        const ::tyloi::DenseTensor& value_cache = {}) const;
    std::vector<::tyloi::DenseTensor> forward_normalized_grouped(
        const std::vector<::tyloi::DenseTensor>& values,
        const std::vector<::tyloi::DenseTensor>& weights,
        const ::tyloi::DenseTensor& positions,double eps,int cache_entry=1,
        const ::tyloi::DenseTensor& key_cache={},
        const ::tyloi::DenseTensor& cache_positions={},
        const ::tyloi::DenseTensor& projected_value={},
        const ::tyloi::DenseTensor& value_cache={}) const;
private:
    int64_t dimension_, maximum_;
    double base_;
    std::vector<int64_t> sections_;
    bool interleaved_;
    bool fused_;
    tyloi::ops::YarnScaling yarn_;
    struct Prepared { ::tyloi::DenseTensor frequencies, axes; };
    Prepared prepare_parameters(const ::tyloi::DenseTensor& value) const;
    mutable std::mutex prepared_mutex_;
    mutable std::unordered_map<int,Prepared> prepared_;
};

} // namespace tyloi::cuda
