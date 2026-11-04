#pragma once

#include "engine/tensor.h"

#include <cstdint>
#include <vector>

namespace tyloi {

// Shared plumbing for encoded weight representations: logical [rows, columns]
// shape, the declared logical dtype, and the buffer list that keeps every
// backing alive. Format fields and their validation stay in each representation.
class EncodedTensor : public Tensor {
public:
    bool defined() const noexcept override { return primary().defined(); }
    std::int64_t dim() const noexcept override { return defined() ? 2 : 0; }
    std::int64_t numel() const override { return logical_rows() * logical_columns(); }
    // Logical expansion bytes; the physical payload comes from backing_nbytes().
    std::size_t nbytes() const override {
        return static_cast<std::size_t>(numel()) * scalar_size(logical_type());
    }
    std::size_t backing_nbytes() const override { return backing_bytes(buffers()); }
    std::int64_t size(std::int64_t dimension) const override {
        if (dimension == 0) return logical_rows();
        if (dimension == 1) return logical_columns();
        throw std::out_of_range("tensor dimension is out of range");
    }
    ScalarType scalar_type() const noexcept override { return logical_type(); }
    const Device& device() const noexcept override { return primary().device(); }
    bool is_contiguous() const noexcept override { return false; }
    bool is_encoded() const noexcept override { return true; }
    void record_stream(std::uintptr_t stream) const override {
        for (const auto* buffer : buffers())
            if (buffer && buffer->defined()) buffer->record_stream(stream);
    }

protected:
    // Payload buffer: it carries the device and the stored code domain.
    virtual const DenseTensor& primary() const = 0;
    virtual std::int64_t logical_rows() const = 0;
    virtual std::int64_t logical_columns() const = 0;
    // Encoded payloads store codes, not the numeric type the projection math
    // produces; the representation declares it. Codes stay one byte wide.
    virtual ScalarType logical_type() const { return primary().scalar_type(); }
};

} // namespace tyloi
