#include "engine/weights/encoding/rows.h"
#include "engine/weights/encoding/tq_table.h"
#if TYLOINFER_ENABLE_MFQ_COMPAT
#include "engine/weights/encoding/quant/nint_rows.h"
#endif
#include <limits>
#include <stdexcept>

namespace tyloi::weights {
Rows::Rows(std::shared_ptr<TqTable> table) : tq_(std::move(table)) {
    if (tq_ && tq_->rows() > std::uint64_t(std::numeric_limits<std::int64_t>::max()))
        throw std::overflow_error("row table size overflow");
}
Rows::Rows(std::shared_ptr<NintRows> table) : nint_(std::move(table)) {}
bool Rows::defined() const noexcept { return bool(tq_) || bool(nint_); }
std::int64_t Rows::size(std::int64_t dimension) const {
    if (dimension < 0) dimension += 2;
    if (dimension < 0 || dimension >= 2 || !defined()) throw std::out_of_range("invalid row table dimension");
    if (tq_) return dimension == 0 ? static_cast<std::int64_t>(tq_->rows()) : tq_->width();
#if TYLOINFER_ENABLE_MFQ_COMPAT
    return dimension == 0 ? nint_->rows() : nint_->width();
#else
    throw std::logic_error("NINT rows require compatibility support");
#endif
}
std::int64_t Rows::numel() const {
    const auto rows = size(0), width = size(1);
    if (width && rows > std::numeric_limits<std::int64_t>::max() / width)
        throw std::overflow_error("row table size overflow");
    return rows * width;
}
std::size_t Rows::nbytes() const {
    const auto count = static_cast<std::size_t>(numel());
    if (count > std::numeric_limits<std::size_t>::max() / 2) throw std::overflow_error("row table size overflow");
    return count * 2;
}
std::size_t Rows::backing_nbytes() const {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    return nint_ ? nint_->index_nbytes() : 0;
#else
    return 0;
#endif
}
std::shared_ptr<TensorImpl> Rows::relocated(Device device) const {
    if (!device.is_cpu()) throw std::invalid_argument("streamed rows cannot be relocated");
    return std::make_shared<Rows>(*this);
}
}
