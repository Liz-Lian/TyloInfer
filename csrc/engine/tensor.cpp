// DenseTensor view metadata (pure host logic) and the numeric convenience members,
// which forward to the public ops. Device kernels live in the backends.
#include "engine/tensor.h"
#include "engine/ops.h"

#include <algorithm>
#include <cstdlib>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tyloi {
namespace {

DenseTensor dense_result(const Tensor& result) {
    if (!result.defined()) throw std::invalid_argument("operation returned an undefined tensor");
    return result.dense();
}

std::size_t normalize_dimension(std::int64_t dimension, std::size_t rank, bool allow_end = false) {
    const auto upper = static_cast<std::int64_t>(rank) + (allow_end ? 1 : 0);
    auto normalized = dimension;
    if (normalized < 0) {
        normalized += upper;
    }
    if (normalized < 0 || normalized >= upper) {
        throw std::out_of_range("tensor dimension is out of range");
    }
    return static_cast<std::size_t>(normalized);
}

}  // namespace

const TensorImpl& Tensor::reference() const noexcept {
    if (impl_) return *impl_;
    static const DenseTensor undefined;
    return undefined;
}

bool Tensor::same_view(const Tensor& other) const {
    if (!defined() || !other.defined()) return false;
    if (impl_ == other.impl_) return true;
    const auto* left = representation<DenseTensor>();
    const auto* right = other.representation<DenseTensor>();
    if (!left || !right || left->device() != right->device() ||
        left->scalar_type() != right->scalar_type() || left->data_ptr() != right->data_ptr() ||
        left->dim() != right->dim()) return false;
    for (std::int64_t axis = 0; axis < left->dim(); ++axis)
        if (left->size(axis) != right->size(axis) || left->stride(axis) != right->stride(axis)) return false;
    return true;
}

const DenseTensor& Tensor::dense() const {
    const auto* value = dynamic_cast<const DenseTensor*>(impl_.get());
    if (!value) throw std::invalid_argument("operation requires a dense tensor");
    return *value;
}

DenseTensor& Tensor::dense() {
    auto* value = dynamic_cast<DenseTensor*>(impl_.get());
    if (!value) throw std::invalid_argument("operation requires a dense tensor");
    return *value;
}

std::vector<std::shared_ptr<TensorStorage>> TensorImpl::backings() const {
    std::vector<std::shared_ptr<TensorStorage>> result;
    for (const auto* buffer : buffers())
        if (buffer && buffer->defined()) result.push_back(buffer->storage());
    return result;
}


std::shared_ptr<TensorImpl> DenseTensor::relocated(Device target) const {
    return defined() && device() != target
        ? std::make_shared<DenseTensor>(to(target))
        : std::make_shared<DenseTensor>(*this);
}

std::size_t backing_bytes(std::span<const DenseTensor* const> buffers) {
    std::size_t total = 0;
    std::unordered_set<const void*> seen;
    for (const auto* buffer : buffers) {
        if (!buffer || !buffer->defined()) continue;
        const auto& storage = buffer->storage();
        const void* identity = storage->owner ? storage->owner.get() : storage->base;
        if (!identity || !seen.insert(identity).second) continue;
        total += storage->bytes;
    }
    return total;
}

DenseTensor reshape(const DenseTensor& source, std::span<const std::int64_t> shape) {
    const auto& view = source.view_descriptor();
    if (!source.defined()) {
        throw std::invalid_argument("cannot reshape an undefined tensor");
    }
    if (!source.is_contiguous()) {
        // shortcut: getenv replaces the frozen native env snapshot; fine while
        // the switch is only read before serving starts.
        const auto* option = std::getenv("TYLOINFER_TENSOR_VIEW_RESHAPE");
        if (option && option[0] == '0') return reshape(dense_result(ops::contiguous(source)), shape);
        // Validate/infer the requested dimensions without copying storage.
        auto packed=view;packed.rank=1;packed.sizes[0]=source.numel();packed.strides[0]=1;
        auto result=reshape_view(packed,shape);
        if(source.numel()==0)return source.alias(result,source.byte_offset());
        // Each contiguous source chunk can be split or joined independently.
        // A row gap (grouped projection), transpose or broadcast stays a gap;
        // a reshape crossing it still requires the original copy.
        int output_axis=int(result.rank)-1;
        int64_t source_count=1,output_count=1,chunk_stride=view.strides[view.rank-1];
        bool compatible=true;
        for(int axis=int(view.rank)-1;axis>=0;--axis) {
            source_count*=view.sizes[axis];
            const bool boundary=axis==0 || (view.sizes[axis-1]!=1 &&
                view.strides[axis-1]!=source_count*chunk_stride);
            if(!boundary)continue;
            while(output_axis>=0 && (output_count<source_count || result.sizes[output_axis]==1)) {
                result.strides[output_axis]=output_count*chunk_stride;
                output_count*=result.sizes[output_axis];--output_axis;
            }
            if(output_count!=source_count){compatible=false;break;}
            if(axis>0){chunk_stride=view.strides[axis-1];source_count=output_count=1;}
        }
        if(compatible && output_axis==-1)return source.alias(result,source.byte_offset());
        return reshape(dense_result(ops::contiguous(source)), shape);
    }
    return source.alias(reshape_view(view, shape), source.byte_offset());
}

DenseTensor transpose(const DenseTensor& source, std::int64_t first, std::int64_t second) {
    const auto& view = source.view_descriptor();
    if (!source.defined()) {
        throw std::invalid_argument("cannot transpose an undefined tensor");
    }
    return source.alias(transpose_view(view, first, second), source.byte_offset());
}

DenseTensor permute(const DenseTensor& source, std::span<const std::int64_t> order) {
    const auto& view = source.view_descriptor();
    if (!source.defined() || order.size() != view.rank) {
        throw std::invalid_argument("permute order must contain every tensor dimension");
    }
    TensorView result = view;
    std::unordered_set<std::size_t> seen;
    for (std::size_t output = 0; output < order.size(); ++output) {
        const auto input = normalize_dimension(order[output], view.rank);
        if (!seen.insert(input).second) {
            throw std::invalid_argument("permute order contains a duplicate dimension");
        }
        result.sizes[output] = view.sizes[input];
        result.strides[output] = view.strides[input];
    }
    return source.alias(result, source.byte_offset());
}

DenseTensor unsqueeze(const DenseTensor& source, std::int64_t dimension) {
    const auto& view = source.view_descriptor();
    if (!source.defined() || view.rank == kMaximumTensorRank) {
        throw std::invalid_argument("cannot unsqueeze this tensor");
    }
    const auto inserted = normalize_dimension(dimension, view.rank, true);
    TensorView result = view;
    for (std::size_t index = result.rank; index > inserted; --index) {
        result.sizes[index] = result.sizes[index - 1];
        result.strides[index] = result.strides[index - 1];
    }
    result.sizes[inserted] = 1;
    result.strides[inserted] = inserted < result.rank
        ? result.strides[inserted + 1] * result.sizes[inserted + 1]
        : 1;
    ++result.rank;
    return source.alias(result, source.byte_offset());
}

DenseTensor squeeze(const DenseTensor& source, std::int64_t dimension) {
    const auto& view = source.view_descriptor();
    if (!source.defined()) {
        throw std::invalid_argument("cannot squeeze an undefined tensor");
    }
    const auto removed = normalize_dimension(dimension, view.rank);
    if (view.sizes[removed] != 1) {
        return source;
    }
    TensorView result = view;
    for (std::size_t index = removed; index + 1 < result.rank; ++index) {
        result.sizes[index] = result.sizes[index + 1];
        result.strides[index] = result.strides[index + 1];
    }
    --result.rank;
    return source.alias(result, source.byte_offset());
}

DenseTensor squeeze(const DenseTensor& source) {
    source.require_dense();
    if (!source.defined()) {
        throw std::invalid_argument("cannot squeeze an undefined tensor");
    }
    auto result = source;
    for (std::int64_t dimension = result.dim() - 1; dimension >= 0; --dimension) {
        if (result.size(dimension) == 1) {
            result = squeeze(result, dimension);
        }
    }
    return result;
}

DenseTensor flatten(const DenseTensor& source, std::int64_t start, std::int64_t end) {
    const auto& view = source.view_descriptor();
    if (!source.defined()) {
        throw std::invalid_argument("flatten requires a defined tensor");
    }
    if (!source.is_contiguous()) return flatten(dense_result(ops::contiguous(source)), start, end);
    const auto first = normalize_dimension(start, view.rank);
    const auto last = normalize_dimension(end, view.rank);
    if (first > last) {
        throw std::invalid_argument("flatten start must not exceed end");
    }
    std::vector<std::int64_t> shape;
    for (std::size_t index = 0; index < first; ++index) {
        shape.push_back(view.sizes[index]);
    }
    std::int64_t collapsed = 1;
    for (std::size_t index = first; index <= last; ++index) {
        collapsed *= view.sizes[index];
    }
    shape.push_back(collapsed);
    for (std::size_t index = last + 1; index < view.rank; ++index) {
        shape.push_back(view.sizes[index]);
    }
    return reshape(source, shape);
}

DenseTensor narrow(const DenseTensor& source,
    std::int64_t dimension,
    std::int64_t start,
    std::int64_t length) {
    const auto& view = source.view_descriptor();
    if (!source.defined()) {
        throw std::invalid_argument("cannot narrow an undefined tensor");
    }
    const auto selected = normalize_dimension(dimension, view.rank);
    const auto extent = view.sizes[selected];
    auto normalized_start = start < 0 ? start + extent : start;
    if (normalized_start < 0 || length < 0 || normalized_start + length > extent) {
        throw std::out_of_range("narrow range is out of bounds");
    }
    auto result = view;
    result.sizes[selected] = length;
    const auto offset_elements = normalized_start * view.strides[selected];
    const auto offset_bytes = static_cast<std::size_t>(offset_elements) * source.element_size();
    result.data = static_cast<std::byte*>(view.data) + offset_bytes;
    return source.alias(result, source.byte_offset() + offset_bytes);
}

DenseTensor select(const DenseTensor& source, std::int64_t dimension, std::int64_t index) {
    const auto& view = source.view_descriptor();
    const auto selected = normalize_dimension(dimension, view.rank);
    const auto narrowed = narrow(source, dimension, index, 1);
    TensorView result = narrowed.view_descriptor();
    for (std::size_t position = selected; position + 1 < result.rank; ++position) {
        result.sizes[position] = result.sizes[position + 1];
        result.strides[position] = result.strides[position + 1];
    }
    --result.rank;
    return narrowed.alias(result, narrowed.byte_offset());
}

DenseTensor slice(const DenseTensor& source,
    std::int64_t dimension,
    std::int64_t start,
    std::int64_t end,
    std::int64_t step) {
    const auto& view = source.view_descriptor();
    if (step <= 0) {
        throw std::invalid_argument("native slice requires a positive step");
    }
    const auto selected = normalize_dimension(dimension, view.rank);
    const auto extent = view.sizes[selected];
    auto first = start < 0 ? start + extent : start;
    auto last = end < 0 ? end + extent : end;
    first = std::clamp<std::int64_t>(first, 0, extent);
    last = std::clamp<std::int64_t>(last, 0, extent);
    const auto length = last <= first ? 0 : (last - first + step - 1) / step;
    auto narrowed = narrow(source, dimension, first, std::max<std::int64_t>(0, last - first));
    auto result = narrowed.view_descriptor();
    result.sizes[selected] = length;
    result.strides[selected] *= step;
    return narrowed.alias(result, narrowed.byte_offset());
}

DenseTensor index(const DenseTensor& source, std::initializer_list<TensorIndex> indices) {
    source.require_dense();
    for (const auto& index : indices)
        if (index.kind() == TensorIndex::Kind::tensor) index.tensor().require_dense();
    DenseTensor result = source;
    std::int64_t dimension = 0;
    for (const auto& index : indices) {
        if (index.kind() == TensorIndex::Kind::integer) {
            result = select(result, dimension, index.integer());
            continue;
        }
        if (index.kind() == TensorIndex::Kind::tensor) {
            result = dense_result(ops::index_select(result, dimension, index.tensor()));
            ++dimension;
            continue;
        }
        const auto& selected = index.slice();
        const auto extent = result.size(dimension);
        const auto start = selected.has_start() ? selected.start() : 0;
        const auto end = selected.has_end() ? selected.end() : extent;
        result = slice(result, dimension, start, end, selected.step());
        ++dimension;
    }
    return result;
}

DenseTensor expand(const DenseTensor& source, std::span<const std::int64_t> shape) {
    const auto& view = source.view_descriptor();
    if (shape.size() < view.rank || shape.size() > kMaximumTensorRank) {
        throw std::invalid_argument("expand rank is incompatible with tensor rank");
    }
    TensorView result = view;
    std::array<std::int64_t, kMaximumTensorRank> sizes{};
    std::array<std::int64_t, kMaximumTensorRank> strides{};
    const auto offset = shape.size() - view.rank;
    for (std::size_t output = 0; output < shape.size(); ++output) {
        const auto requested = shape[output];
        if (output < offset) {
            if (requested < 0) throw std::invalid_argument("new expand dimensions must be explicit");
            sizes[output] = requested;
            strides[output] = 0;
            continue;
        }
        const auto input = output - offset;
        const auto extent = view.sizes[input];
        const auto resolved = requested == -1 ? extent : requested;
        if (resolved != extent && extent != 1) {
            throw std::invalid_argument("cannot expand a non-singleton dimension");
        }
        sizes[output] = resolved;
        strides[output] = resolved == extent ? view.strides[input] : 0;
    }
    result.rank = static_cast<std::uint8_t>(shape.size());
    result.sizes = sizes;
    result.strides = strides;
    return source.alias(result, source.byte_offset());
}

DenseTensor unfold(const DenseTensor& source,
    std::int64_t dimension,
    std::int64_t window,
    std::int64_t step) {
    const auto& view = source.view_descriptor();
    if (window <= 0 || step <= 0 || view.rank == kMaximumTensorRank) {
        throw std::invalid_argument("invalid unfold geometry");
    }
    const auto selected = normalize_dimension(dimension, view.rank);
    const auto extent = view.sizes[selected];
    if (window > extent) throw std::invalid_argument("unfold window exceeds dimension");
    TensorView result = view;
    result.sizes[selected] = (extent - window) / step + 1;
    result.strides[selected] *= step;
    result.sizes[result.rank] = window;
    result.strides[result.rank] = view.strides[selected];
    ++result.rank;
    return source.alias(result, source.byte_offset());
}

std::vector<DenseTensor> split_with_sizes(const DenseTensor& source,
    std::span<const std::int64_t> sections,
    std::int64_t dimension) {
    source.require_dense();
    std::vector<DenseTensor> result;
    result.reserve(sections.size());
    std::int64_t offset = 0;
    for (const auto length : sections) {
        result.push_back(narrow(source, dimension, offset, length));
        offset += length;
    }
    if (offset != source.size(dimension)) {
        throw std::invalid_argument("split sizes do not cover the selected dimension");
    }
    return result;
}

std::vector<DenseTensor> chunk(const DenseTensor& source, std::int64_t chunks, std::int64_t dimension) {
    source.require_dense();
    if (chunks <= 0) throw std::invalid_argument("chunk count must be positive");
    const auto extent = source.size(dimension);
    const auto width = (extent + chunks - 1) / chunks;
    std::vector<DenseTensor> result;
    for (std::int64_t start = 0; start < extent; start += width) {
        result.push_back(narrow(source, dimension, start, std::min(width, extent - start)));
    }
    return result;
}

DenseTensor DenseTensor::reshape(std::span<const std::int64_t> shape) const { return tyloi::reshape(*this, shape); }

DenseTensor DenseTensor::transpose(std::int64_t first, std::int64_t second) const { return tyloi::transpose(*this, first, second); }

DenseTensor DenseTensor::permute(std::span<const std::int64_t> order) const { return tyloi::permute(*this, order); }

DenseTensor DenseTensor::unsqueeze(std::int64_t dimension) const { return tyloi::unsqueeze(*this, dimension); }

DenseTensor DenseTensor::squeeze(std::int64_t dimension) const { return tyloi::squeeze(*this, dimension); }

DenseTensor DenseTensor::squeeze() const { return tyloi::squeeze(*this); }

DenseTensor DenseTensor::flatten(std::int64_t start, std::int64_t end) const { return tyloi::flatten(*this, start, end); }

DenseTensor DenseTensor::narrow(
    std::int64_t dimension,
    std::int64_t start,
    std::int64_t length) const { return tyloi::narrow(*this, dimension, start, length); }

DenseTensor DenseTensor::select(std::int64_t dimension, std::int64_t index) const { return tyloi::select(*this, dimension, index); }

DenseTensor DenseTensor::slice(
    std::int64_t dimension,
    std::int64_t start,
    std::int64_t end,
    std::int64_t step) const { return tyloi::slice(*this, dimension, start, end, step); }

DenseTensor DenseTensor::index(std::initializer_list<TensorIndex> indices) const { return tyloi::index(*this, indices); }

DenseTensor DenseTensor::index_select(std::int64_t dimension, const DenseTensor& indices) const {
    return dense_result(ops::index_select(*this, dimension, indices));
}

DenseTensor DenseTensor::gather(std::int64_t dimension, const DenseTensor& indices) const {
    return dense_result(ops::gather(*this, dimension, indices));
}

DenseTensor DenseTensor::expand(std::span<const std::int64_t> shape) const { return tyloi::expand(*this, shape); }

DenseTensor DenseTensor::repeat(std::span<const std::int64_t> repeats) const {
    return dense_result(ops::repeat(*this, repeats));
}

DenseTensor DenseTensor::repeat_interleave(std::int64_t repeats, std::int64_t dimension) const {
    return dense_result(ops::repeat_interleave(*this, repeats, dimension));
}

DenseTensor DenseTensor::unfold(
    std::int64_t dimension,
    std::int64_t window,
    std::int64_t step) const { return tyloi::unfold(*this, dimension, window, step); }

std::vector<DenseTensor> DenseTensor::split_with_sizes(
    std::span<const std::int64_t> sections,
    std::int64_t dimension) const { return tyloi::split_with_sizes(*this, sections, dimension); }

std::vector<DenseTensor> DenseTensor::chunk(std::int64_t chunks, std::int64_t dimension) const { return tyloi::chunk(*this, chunks, dimension); }

DenseTensor DenseTensor::mean(std::int64_t dimension, bool keep_dimension) const {
    return dense_result(ops::mean(*this, dimension, keep_dimension));
}

DenseTensor DenseTensor::mean() const { return reshape({-1}).mean(0, false); }
DenseTensor DenseTensor::sum(std::int64_t dimension, bool keep_dimension) const {
    return dense_result(ops::sum(*this, dimension, keep_dimension));
}
DenseTensor DenseTensor::sum() const { return reshape({-1}).sum(0, false); }
DenseTensor DenseTensor::max() const { return reshape({-1}).amax(0, false); }
DenseTensor DenseTensor::amax(std::int64_t dimension, bool keep_dimension) const {
    return dense_result(ops::amax(*this, dimension, keep_dimension));
}
DenseTensor DenseTensor::norm() const { return square().sum().sqrt(); }
DenseTensor DenseTensor::abs() const { return dense_result(ops::abs(*this)); }
DenseTensor DenseTensor::square() const { return dense_result(ops::square(*this)); }
DenseTensor DenseTensor::exp() const { return dense_result(ops::exp(*this)); }
DenseTensor DenseTensor::sqrt() const { return dense_result(ops::sqrt(*this)); }
DenseTensor DenseTensor::clamp(double minimum, double maximum) const {
    return dense_result(ops::clamp(*this, minimum, maximum));
}
DenseTensor DenseTensor::clamp_min(double minimum) const {
    return dense_result(ops::clamp_min(*this, minimum));
}
DenseTensor DenseTensor::clamp_max(double maximum) const {
    return dense_result(ops::clamp_max(*this, maximum));
}
DenseTensor DenseTensor::remainder(double divisor) const {
    return dense_result(ops::remainder(*this, divisor));
}
DenseTensor DenseTensor::argmax(std::int64_t dimension, bool keep_dimension) const {
    return dense_result(ops::argmax(*this, dimension, keep_dimension));
}
DenseTensor DenseTensor::eq(const DenseTensor& other) const { return dense_result(ops::eq(*this, other)); }
DenseTensor DenseTensor::eq(double other) const { return dense_result(ops::eq(*this, other)); }
DenseTensor DenseTensor::ne(const DenseTensor& other) const { return dense_result(ops::ne(*this, other)); }
DenseTensor DenseTensor::ne(double other) const { return dense_result(ops::ne(*this, other)); }
DenseTensor DenseTensor::all() const { return reshape({-1}).all(0, false); }
DenseTensor DenseTensor::all(std::int64_t dimension, bool keep_dimension) const {
    return dense_result(ops::all(*this, dimension, keep_dimension));
}
DenseTensor DenseTensor::any(std::int64_t dimension, bool keep_dimension) const {
    return dense_result(ops::any(*this, dimension, keep_dimension));
}
DenseTensor DenseTensor::logical_not() const { return dense_result(ops::logical_not(*this)); }
bool DenseTensor::equal(const DenseTensor& other) const { return ops::equal(*this, other); }
DenseTensor DenseTensor::masked_select(const DenseTensor& mask) const { return dense_result(ops::masked_select(*this, mask)); }
DenseTensor DenseTensor::tril(std::int64_t diagonal) const { return dense_result(ops::tril(*this, diagonal)); }
DenseTensor DenseTensor::masked_fill(const DenseTensor& mask, double value) const {
    return dense_result(ops::masked_fill(*this, mask, value));
}
DenseTensor& DenseTensor::masked_fill_(const DenseTensor& mask, double value) {
    Tensor self{*this};
    ops::masked_fill_inplace(self, mask, value);
    return *this;
}

DenseTensor DenseTensor::contiguous() const {
    return dense_result(ops::contiguous(*this));
}

DenseTensor DenseTensor::clone() const {
    return dense_result(ops::clone(*this));
}

DenseTensor DenseTensor::to(ScalarType type) const {
    return dense_result(ops::cast(*this, type));
}

DenseTensor DenseTensor::to(Device target_device) const {
    return dense_result(ops::copy(*this, target_device));
}

DenseTensor& DenseTensor::copy_(const DenseTensor& source, bool non_blocking) {
    Tensor self{*this};
    ops::copy_into(self, source, non_blocking);
    return *this;
}

DenseTensor& DenseTensor::zero_() {
    return fill_(0.0);
}

DenseTensor& DenseTensor::fill_(double value) {
    Tensor self{*this};
    ops::fill(self, value);
    return *this;
}

DenseTensor& DenseTensor::add_(const DenseTensor& value) {
    auto result = ops::add(*this, value);
    Tensor self{*this};
    ops::copy_into(self, result);
    return *this;
}

DenseTensor& DenseTensor::scatter_add_(
    std::int64_t dimension, const DenseTensor& index, const DenseTensor& source) {
    Tensor self{*this};
    ops::scatter_add(self, dimension, index, source);
    return *this;
}

DenseTensor& DenseTensor::scatter_(
    std::int64_t dimension, const DenseTensor& index, const DenseTensor& source) {
    Tensor self{*this};
    ops::scatter(self, dimension, index, source);
    return *this;
}

DenseTensor& DenseTensor::index_copy_(
    std::int64_t dimension, const DenseTensor& index, const DenseTensor& source) {
    Tensor self{*this};
    ops::index_copy(self, dimension, index, source);
    return *this;
}

DenseTensor& DenseTensor::index_fill_(
    std::int64_t dimension, const DenseTensor& index, double value) {
    Tensor self{*this};
    ops::index_fill(self, dimension, index, value);
    return *this;
}

DenseTensor& DenseTensor::index_put_(
    std::initializer_list<TensorIndex> indices, const DenseTensor& value) {
    value.require_dense();
    Tensor view{index(indices)};
    ops::copy_into(view, value);
    return *this;
}

DenseTensor& DenseTensor::index_put_(
    std::initializer_list<TensorIndex> indices, double value) {
    index(indices).fill_(value);
    return *this;
}

}  // namespace tyloi
