#pragma once

#include "mfq_container.h"
#include "tyloinfer/mlx_affine.h"
#include <mlx/mlx.h>
#include <array>

namespace mfq::metal {

class MlxAffineRowBatch {
public:
    void append(const mfq::MlxAffineLayout& layout, std::span<const std::uint8_t> values,
        std::span<const std::uint8_t> scales, std::span<const std::uint8_t> biases);
    void append_batch(const MlxAffineRowBatch& batch);
    mlx::core::array decode(mlx::core::Dtype dtype = mlx::core::float16) const;
    std::size_t source_bytes_read() const noexcept { return values_.size() + scales_.size() + biases_.size(); }
private:
    mfq::MlxAffineLayout layout_;
    std::vector<std::uint8_t> values_, scales_, biases_;
};

class MlxMappedAffineRows {
public:
    MlxMappedAffineRows(const MfqContainer& model, const std::string& name);
    void append_row(std::int64_t row, MlxAffineRowBatch& batch) const;
    std::int64_t rows() const noexcept { return layout_.shape[0]; }
    int width() const noexcept { return int(layout_.shape[1]); }
    std::size_t source_bytes() const noexcept { return values_.size() + scales_.size() + biases_.size(); }
    std::array<MfqMappedBytes::FileIdentity, 3> file_identities() const noexcept {
        return {values_.file_identity(), scales_.file_identity(), biases_.file_identity()};
    }
private:
    mfq::MlxAffineLayout layout_;
    MfqMappedBytes values_, scales_, biases_;
    std::size_t value_stride_, scale_stride_;
};

} // namespace mfq::metal
