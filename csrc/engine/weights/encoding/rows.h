#pragma once

#include "engine/tensor.h"
#include <memory>

namespace tyloi { class NintRows; class TqTable; }
namespace tyloi::weights {
/** Own immutable selected-row readers without materializing the table. */
class Rows final : public TensorImpl {
public:
    /** Retain a streamed TQ table and its segment readers. */
    explicit Rows(std::shared_ptr<TqTable> table);
    /** Retain a streamed NINT table and its row selectors. */
    explicit Rows(std::shared_ptr<NintRows> table);
    bool defined() const noexcept override;
    std::int64_t dim() const noexcept override { return 2; }
    std::int64_t numel() const override;
    std::size_t nbytes() const override;
    std::size_t backing_nbytes() const override;
    std::int64_t size(std::int64_t dimension) const override;
    ScalarType scalar_type() const noexcept override { return kFloat16; }
    const Device& device() const noexcept override { return device_; }
    bool is_contiguous() const noexcept override { return false; }
    bool is_encoded() const noexcept override { return true; }
    std::vector<const DenseTensor*> buffers() const override { return {}; }
    void record_stream(std::uintptr_t) const override {}
    /** Retain the source on CPU; reject relocating an unread table to a device. */
    std::shared_ptr<TensorImpl> relocated(Device device) const override;
    /** Borrow the TQ reader, if this table uses TQ encoding. */
    const std::shared_ptr<TqTable>& tq() const noexcept { return tq_; }
    /** Borrow the NINT reader, if this table uses NINT encoding. */
    const std::shared_ptr<NintRows>& nint() const noexcept { return nint_; }
private:
    std::shared_ptr<TqTable> tq_;
    std::shared_ptr<NintRows> nint_;
    Device device_{kCPU, 0};
};
}
