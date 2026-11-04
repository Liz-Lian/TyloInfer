#pragma once

#include "backend.h"
#include "engine/ops.h"

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
namespace tyloi::cuda::ops {
tyloi::Tensor rope(tyloi::Tensor input, tyloi::Tensor positions, const tyloi::ops::RopeTable& table,
                 bool bf16, bool grid_mrope_positions);
}
#endif

struct RopeCache {
    ::tyloi::Tensor cos;
    ::tyloi::Tensor sin;
    ::tyloi::Tensor sections;
    ::tyloi::Tensor empty_sections;
    ::tyloi::Tensor interleaved_order;
    ::tyloi::Tensor interleaved_inverse;
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
    ::tyloi::Tensor apply(
        ::tyloi::Tensor x,
        ::tyloi::Tensor pos,
        bool grid_mrope_positions = false) const;
    ::tyloi::Tensor apply_bf16(
        bool fused,
        ::tyloi::Tensor x,
        ::tyloi::Tensor pos) const;
};

namespace tyloi::cuda {

// Computes rotate-half frequencies for logical [axis,batch,token] positions.
// Uses FP32 for F32 inputs and FP16 otherwise, without a resident table.
class RotaryEmbedding {
public:
    RotaryEmbedding(int64_t dimension, int64_t maximum, double base,
                    std::vector<int64_t> sections, bool interleaved,
                    bool fused, tyloi::ops::YarnScaling yarn = {});
    ::tyloi::Tensor forward(const ::tyloi::Tensor& value,
                                       const ::tyloi::Tensor& positions) const;
    // Switch only after execution and captured graphs using this module finish.
    void set_fused(bool enabled) { fused_ = enabled && !yarn_.enabled(); }
    bool fused() const { return fused_; }
    bool pool_normalized_into(const ::tyloi::Tensor& index,
        const ::tyloi::Tensor& weight,const ::tyloi::Tensor& positions,
        const ::tyloi::Tensor& cache_positions,const ::tyloi::Tensor& output,
        int64_t pool,int64_t maximum,double eps) const;
    ::tyloi::Tensor forward_normalized(
        const ::tyloi::Tensor& value, const ::tyloi::Tensor& weight,
        const ::tyloi::Tensor& positions, double eps,
        const ::tyloi::Tensor& key_cache = {},
        const ::tyloi::Tensor& cache_positions = {},
        const ::tyloi::Tensor& projected_value = {},
        const ::tyloi::Tensor& value_cache = {}) const;
    std::vector<::tyloi::Tensor> forward_normalized_grouped(
        const std::vector<::tyloi::Tensor>& values,
        const std::vector<::tyloi::Tensor>& weights,
        const ::tyloi::Tensor& positions,double eps,int cache_entry=1,
        const ::tyloi::Tensor& key_cache={},
        const ::tyloi::Tensor& cache_positions={},
        const ::tyloi::Tensor& projected_value={},
        const ::tyloi::Tensor& value_cache={}) const;
private:
    int64_t dimension_, maximum_;
    double base_;
    std::vector<int64_t> sections_;
    bool interleaved_;
    bool fused_;
    tyloi::ops::YarnScaling yarn_;
    struct Prepared { ::tyloi::Tensor frequencies, axes; };
    Prepared prepare_parameters(const ::tyloi::Tensor& value) const;
    mutable std::mutex prepared_mutex_;
    mutable std::unordered_map<int,Prepared> prepared_;
};

} // namespace tyloi::cuda
