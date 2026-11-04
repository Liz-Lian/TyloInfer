#include "env.h"
#include "tensor.h"

#include <cstdlib>
#include <cstring>
#include <bit>
#include <cmath>
#include <limits>
#include <mutex>
#include <new>
#include <cstdio>
#include <string_view>
#include <unordered_set>

namespace tyloi::cuda {
namespace {

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

struct HostStoragePool {
    static constexpr std::size_t minimum=256u<<10,maximum=8u<<20,budget=64u<<20;
    struct Block {void* data=nullptr;std::size_t bytes=0;std::uint64_t order=0;};
    std::mutex mutex;
    std::array<Block,16> blocks{};
    std::size_t idle=0;
    std::uint64_t order=0;
    void erase(Block& block) noexcept {
        if(!block.data)return;
        ::operator delete(block.data,std::align_val_t{64});
        tensor_host_bytes.fetch_sub(block.bytes);
        idle-=block.bytes;block={};
    }
    ~HostStoragePool(){for(auto& block:blocks)erase(block);}
    Block acquire(std::size_t bytes) {
        Block result;
        const auto capacity=(bytes+4095)/4096*4096;
        {
            std::lock_guard lock(mutex);
            auto best=blocks.end();
            for(auto it=blocks.begin();it!=blocks.end();++it)
                if(it->data && it->bytes==capacity){best=it;break;}
            if(best!=blocks.end()){result=*best;*best={};idle-=result.bytes;}
        }
        if(result.data) {
            static std::atomic<bool> reported{false};
            if(!reported.load(std::memory_order_relaxed) && !reported.exchange(true,std::memory_order_relaxed))
                std::fprintf(stderr,"tensor_host_storage_pool scope=first_reuse enabled=1 max_idle_bytes=%zu bytes=%zu\n",budget,result.bytes);
            return result;
        }
        result={::operator new(capacity,std::align_val_t{64}),capacity};
        charge_tensor_host_bytes(capacity);return result;
    }
    void release(void* data,std::size_t bytes) noexcept {
        std::lock_guard lock(mutex);
        for(;;) {
            auto free=blocks.end(),oldest=blocks.end();
            for(auto it=blocks.begin();it!=blocks.end();++it) {
                if(!it->data)free=it;
                else if(oldest==blocks.end() || it->order<oldest->order)oldest=it;
            }
            if(free!=blocks.end() && bytes<=budget-idle) {
                *free={data,bytes,++order};idle+=bytes;return;
            }
            erase(*oldest);
        }
    }
};

std::shared_ptr<TensorStorage> allocate_cpu_storage(std::size_t bytes, bool pinned) {
    if (pinned) {
        throw std::invalid_argument(
            "pinned host allocation is provided by the native CUDA translation unit");
    }
    const auto allocation_bytes = std::max<std::size_t>(bytes, 1);
    void* pointer=nullptr;
    std::shared_ptr<void> owner;
    if(allocation_bytes>=HostStoragePool::minimum && allocation_bytes<=HostStoragePool::maximum) {
        // Keep only a bounded idle working set. SSD expert churn can reuse
        // these pages instead of repeatedly freeing and faulting megabytes.
        // Live views retain the allocation; pinned storage uses its own path.
        static auto pool=std::make_shared<HostStoragePool>();
        const auto block=pool->acquire(allocation_bytes);pointer=block.data;
        owner=std::shared_ptr<void>(pointer,[held=pool,capacity=block.bytes](void* value){held->release(value,capacity);});
    } else {
        pointer=::operator new(allocation_bytes,std::align_val_t{64});
        charge_tensor_host_bytes(allocation_bytes);
        owner=std::shared_ptr<void>(pointer,[allocation_bytes](void* value) {
            ::operator delete(value,std::align_val_t{64});tensor_host_bytes.fetch_sub(allocation_bytes);
        });
    }
    auto storage = std::make_shared<TensorStorage>();
    storage->owner = std::move(owner);
    storage->base = pointer;
    storage->bytes = bytes;
    storage->device = Device{DeviceType::cpu, 0};
    storage->pinned = false;
    return storage;
}

Tensor empty_cpu(std::span<const std::int64_t> shape, const TensorOptions& options) {
    auto descriptor = make_contiguous_view(
        reinterpret_cast<void*>(1), shape, options.scalar_type(), options.target_device());
    auto storage = allocate_cpu_storage(descriptor.nbytes(), options.pinned());
    descriptor.data = storage->base;
    return Tensor(std::move(storage), descriptor);
}

float decode_float16(std::uint16_t bits) {
    const auto sign = static_cast<std::uint32_t>(bits & 0x8000u) << 16;
    auto exponent = static_cast<std::uint32_t>((bits >> 10) & 0x1fu);
    auto mantissa = static_cast<std::uint32_t>(bits & 0x03ffu);
    std::uint32_t value = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            value = sign;
        } else {
            exponent = 127 - 15 + 1;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            value = sign | (exponent << 23) | ((mantissa & 0x03ffu) << 13);
        }
    } else if (exponent == 0x1fu) {
        value = sign | 0x7f800000u | (mantissa << 13);
    } else {
        value = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    return std::bit_cast<float>(value);
}

std::uint16_t encode_float16(float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    const auto sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000u);
    const auto exponent = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
    auto mantissa = bits & 0x007fffffu;
    if (((bits >> 23) & 0xffu) == 0xffu) {
        if (mantissa == 0) return static_cast<std::uint16_t>(sign | 0x7c00u);
        return static_cast<std::uint16_t>(sign | 0x7e00u | (mantissa >> 13));
    }
    if (exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
    if (exponent <= 0) {
        if (exponent < -10) return sign;
        mantissa |= 0x00800000u;
        const auto shift = static_cast<unsigned>(14 - exponent);
        auto rounded = mantissa >> shift;
        const auto remainder = mantissa & ((std::uint32_t{1} << shift) - 1);
        const auto halfway = std::uint32_t{1} << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (rounded & 1u))) ++rounded;
        return static_cast<std::uint16_t>(sign | rounded);
    }
    auto rounded = mantissa >> 13;
    const auto remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (rounded & 1u))) {
        ++rounded;
        if (rounded == 0x0400u) {
            rounded = 0;
            if (exponent + 1 >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
            return static_cast<std::uint16_t>(sign | ((exponent + 1) << 10));
        }
    }
    return static_cast<std::uint16_t>(sign | (exponent << 10) | rounded);
}

float decode_bfloat16(std::uint16_t bits) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
}

std::uint16_t encode_bfloat16(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    const auto lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return static_cast<std::uint16_t>(bits >> 16);
}

double read_cpu_number(const void* pointer, ScalarType type) {
    switch (type) {
        case kBool: return *static_cast<const bool*>(pointer) ? 1.0 : 0.0;
        case kUInt8: return *static_cast<const std::uint8_t*>(pointer);
        case kInt8: return *static_cast<const std::int8_t*>(pointer);
        case kInt16: return *static_cast<const std::int16_t*>(pointer);
        case kInt32: return *static_cast<const std::int32_t*>(pointer);
        case kInt64: return static_cast<double>(*static_cast<const std::int64_t*>(pointer));
        case kFloat16: return decode_float16(*static_cast<const std::uint16_t*>(pointer));
        case kBFloat16: return decode_bfloat16(*static_cast<const std::uint16_t*>(pointer));
        case kFloat32: return *static_cast<const float*>(pointer);
        case kFloat64: return *static_cast<const double*>(pointer);
        case kFloat8E4M3FN:
            throw std::invalid_argument("CPU FP8 conversion is format-specific");
    }
    throw std::invalid_argument("unknown CPU tensor dtype");
}

void write_cpu_number(void* pointer, ScalarType type, double value) {
    switch (type) {
        case kBool: *static_cast<bool*>(pointer) = value != 0.0; return;
        case kUInt8: *static_cast<std::uint8_t*>(pointer) = static_cast<std::uint8_t>(value); return;
        case kInt8: *static_cast<std::int8_t*>(pointer) = static_cast<std::int8_t>(value); return;
        case kInt16: *static_cast<std::int16_t*>(pointer) = static_cast<std::int16_t>(value); return;
        case kInt32: *static_cast<std::int32_t*>(pointer) = static_cast<std::int32_t>(value); return;
        case kInt64: *static_cast<std::int64_t*>(pointer) = static_cast<std::int64_t>(value); return;
        case kFloat16: *static_cast<std::uint16_t*>(pointer) = encode_float16(static_cast<float>(value)); return;
        case kBFloat16: *static_cast<std::uint16_t*>(pointer) = encode_bfloat16(static_cast<float>(value)); return;
        case kFloat32: *static_cast<float*>(pointer) = static_cast<float>(value); return;
        case kFloat64: *static_cast<double*>(pointer) = value; return;
        case kFloat8E4M3FN:
            throw std::invalid_argument("CPU FP8 conversion is format-specific");
    }
    throw std::invalid_argument("unknown CPU tensor dtype");
}

std::int64_t cpu_tensor_offset(const TensorView& view, std::int64_t linear) {
    std::int64_t result = 0;
    for (std::size_t reverse = view.rank; reverse > 0; --reverse) {
        const auto dimension = reverse - 1;
        const auto coordinate = linear % view.sizes[dimension];
        linear /= view.sizes[dimension];
        result += coordinate * view.strides[dimension];
    }
    return result;
}

void copy_contiguous_cpu(Tensor& destination, const Tensor& source) {
    if (!destination.is_cpu() || !source.is_cpu() ||
        destination.numel() != source.numel()) {
        throw std::invalid_argument("CPU copy shape mismatch");
    }
    if (source.numel() == 0) return;
    if (destination.scalar_type() == source.scalar_type() &&
        destination.is_contiguous() && source.is_contiguous()) {
        std::memcpy(destination.data_ptr(), source.data_ptr(), destination.nbytes());
        return;
    }
    const auto source_width = scalar_size(source.scalar_type());
    const auto destination_width = scalar_size(destination.scalar_type());
    for (std::int64_t linear = 0; linear < source.numel(); ++linear) {
        const auto source_offset = cpu_tensor_offset(source.view_descriptor(), linear);
        const auto destination_offset = cpu_tensor_offset(destination.view_descriptor(), linear);
        const auto* input = static_cast<const std::byte*>(source.data_ptr()) +
            source_offset * source_width;
        auto* output = static_cast<std::byte*>(destination.data_ptr()) +
            destination_offset * destination_width;
        if (destination.scalar_type() == source.scalar_type()) {
            std::memcpy(output, input, source_width);
        } else {
            write_cpu_number(output, destination.scalar_type(),
                             read_cpu_number(input, source.scalar_type()));
        }
    }
}

}  // namespace

// Implemented in tensor.cu for CUDA allocations and copies.
Tensor empty_cuda(std::span<const std::int64_t> shape, const TensorOptions& options);
Tensor empty_pinned(std::span<const std::int64_t> shape, const TensorOptions& options);
Tensor copy_or_convert_cuda(const Tensor& source, const TensorOptions& options);
void fill_cuda(Tensor& destination, double value);
void copy_cuda(Tensor& destination, const Tensor& source);
Tensor make_contiguous_cuda(const Tensor& source);
Tensor mean_cuda(const Tensor& source, std::int64_t dimension, bool keep_dimension);
Tensor reduce_cuda(
    const Tensor& source,
    std::int64_t dimension,
    bool keep_dimension,
    int operation);
Tensor unary_cuda(const Tensor& source, int operation, double first = 0.0, double second = 0.0);

Tensor reshape(const Tensor& source, std::span<const std::int64_t> shape) {
    const auto& view = source.view_descriptor();
    if (!source.defined()) {
        throw std::invalid_argument("cannot reshape an undefined tensor");
    }
    if (!source.is_contiguous()) {
        const auto* option=tyloi::cuda::env_lookup("TYLOINFER_TENSOR_VIEW_RESHAPE");
        if(option && option[0]=='0')return reshape(contiguous(source), shape);
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
        return reshape(contiguous(source), shape);
    }
    return source.alias(reshape_view(view, shape), source.byte_offset());
}

Tensor transpose(const Tensor& source, std::int64_t first, std::int64_t second) {
    const auto& view = source.view_descriptor();
    if (!source.defined()) {
        throw std::invalid_argument("cannot transpose an undefined tensor");
    }
    return source.alias(transpose_view(view, first, second), source.byte_offset());
}

Tensor permute(const Tensor& source, std::span<const std::int64_t> order) {
    const auto& view = source.view_descriptor();
    if (!source.defined() || order.size() != view.rank) {
        throw std::invalid_argument("permute order must contain every tensor dimension");
    }
    TensorView result = view;
    std::unordered_set<std::size_t> seen;
    for (std::size_t output = 0; output < order.size(); ++output) {
        const auto input = cuda::normalize_dimension(order[output], view.rank);
        if (!seen.insert(input).second) {
            throw std::invalid_argument("permute order contains a duplicate dimension");
        }
        result.sizes[output] = view.sizes[input];
        result.strides[output] = view.strides[input];
    }
    return source.alias(result, source.byte_offset());
}

Tensor unsqueeze(const Tensor& source, std::int64_t dimension) {
    const auto& view = source.view_descriptor();
    if (!source.defined() || view.rank == kMaximumTensorRank) {
        throw std::invalid_argument("cannot unsqueeze this tensor");
    }
    const auto inserted = cuda::normalize_dimension(dimension, view.rank, true);
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

Tensor squeeze(const Tensor& source, std::int64_t dimension) {
    const auto& view = source.view_descriptor();
    if (!source.defined()) {
        throw std::invalid_argument("cannot squeeze an undefined tensor");
    }
    const auto removed = cuda::normalize_dimension(dimension, view.rank);
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

Tensor squeeze(const Tensor& source) {
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

Tensor flatten(const Tensor& source, std::int64_t start, std::int64_t end) {
    const auto& view = source.view_descriptor();
    if (!source.defined()) {
        throw std::invalid_argument("flatten requires a defined tensor");
    }
    if (!source.is_contiguous()) return flatten(contiguous(source), start, end);
    const auto first = cuda::normalize_dimension(start, view.rank);
    const auto last = cuda::normalize_dimension(end, view.rank);
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

Tensor narrow(const Tensor& source,
    std::int64_t dimension,
    std::int64_t start,
    std::int64_t length) {
    const auto& view = source.view_descriptor();
    if (!source.defined()) {
        throw std::invalid_argument("cannot narrow an undefined tensor");
    }
    const auto selected = cuda::normalize_dimension(dimension, view.rank);
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

Tensor select(const Tensor& source, std::int64_t dimension, std::int64_t index) {
    const auto& view = source.view_descriptor();
    const auto selected = cuda::normalize_dimension(dimension, view.rank);
    const auto narrowed = narrow(source, dimension, index, 1);
    TensorView result = narrowed.view_descriptor();
    for (std::size_t position = selected; position + 1 < result.rank; ++position) {
        result.sizes[position] = result.sizes[position + 1];
        result.strides[position] = result.strides[position + 1];
    }
    --result.rank;
    return narrowed.alias(result, narrowed.byte_offset());
}

Tensor slice(const Tensor& source,
    std::int64_t dimension,
    std::int64_t start,
    std::int64_t end,
    std::int64_t step) {
    const auto& view = source.view_descriptor();
    if (step <= 0) {
        throw std::invalid_argument("native slice requires a positive step");
    }
    const auto selected = cuda::normalize_dimension(dimension, view.rank);
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

Tensor index(const Tensor& source, std::initializer_list<TensorIndex> indices) {
    source.require_dense();
    for (const auto& index : indices)
        if (index.kind() == TensorIndex::Kind::tensor) index.tensor().require_dense();
    Tensor result = source;
    std::int64_t dimension = 0;
    for (const auto& index : indices) {
        if (index.kind() == TensorIndex::Kind::integer) {
            result = select(result, dimension, index.integer());
            continue;
        }
        if (index.kind() == TensorIndex::Kind::tensor) {
            result = index_select_cuda(result, dimension, index.tensor());
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

Tensor expand(const Tensor& source, std::span<const std::int64_t> shape) {
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

Tensor unfold(const Tensor& source,
    std::int64_t dimension,
    std::int64_t window,
    std::int64_t step) {
    const auto& view = source.view_descriptor();
    if (window <= 0 || step <= 0 || view.rank == kMaximumTensorRank) {
        throw std::invalid_argument("invalid unfold geometry");
    }
    const auto selected = cuda::normalize_dimension(dimension, view.rank);
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

std::vector<Tensor> split_with_sizes(const Tensor& source,
    std::span<const std::int64_t> sections,
    std::int64_t dimension) {
    source.require_dense();
    std::vector<Tensor> result;
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

std::vector<Tensor> chunk(const Tensor& source, std::int64_t chunks, std::int64_t dimension) {
    source.require_dense();
    if (chunks <= 0) throw std::invalid_argument("chunk count must be positive");
    const auto extent = source.size(dimension);
    const auto width = (extent + chunks - 1) / chunks;
    std::vector<Tensor> result;
    for (std::int64_t start = 0; start < extent; start += width) {
        result.push_back(narrow(source, dimension, start, std::min(width, extent - start)));
    }
    return result;
}

}  // namespace tyloi::cuda

namespace tyloi {

Tensor Tensor::reshape(std::span<const std::int64_t> shape) const { return cuda::reshape(*this, shape); }

Tensor Tensor::transpose(std::int64_t first, std::int64_t second) const { return cuda::transpose(*this, first, second); }

Tensor Tensor::permute(std::span<const std::int64_t> order) const { return cuda::permute(*this, order); }

Tensor Tensor::unsqueeze(std::int64_t dimension) const { return cuda::unsqueeze(*this, dimension); }

Tensor Tensor::squeeze(std::int64_t dimension) const { return cuda::squeeze(*this, dimension); }

Tensor Tensor::squeeze() const { return cuda::squeeze(*this); }

Tensor Tensor::flatten(std::int64_t start, std::int64_t end) const { return cuda::flatten(*this, start, end); }

Tensor Tensor::narrow(
    std::int64_t dimension,
    std::int64_t start,
    std::int64_t length) const { return cuda::narrow(*this, dimension, start, length); }

Tensor Tensor::select(std::int64_t dimension, std::int64_t index) const { return cuda::select(*this, dimension, index); }

Tensor Tensor::slice(
    std::int64_t dimension,
    std::int64_t start,
    std::int64_t end,
    std::int64_t step) const { return cuda::slice(*this, dimension, start, end, step); }

Tensor Tensor::index(std::initializer_list<TensorIndex> indices) const { return cuda::index(*this, indices); }

Tensor Tensor::index_select(std::int64_t dimension, const Tensor& indices) const {
    return cuda::index_select_cuda(*this, dimension, indices);
}

Tensor Tensor::gather(std::int64_t dimension, const Tensor& indices) const {
    return cuda::gather_cuda(*this, dimension, indices);
}

Tensor Tensor::expand(std::span<const std::int64_t> shape) const { return cuda::expand(*this, shape); }

Tensor Tensor::repeat(std::span<const std::int64_t> repeats) const {
    return cuda::repeat_cuda(*this, repeats);
}

Tensor Tensor::repeat_interleave(std::int64_t repeats, std::int64_t dimension) const {
    return cuda::repeat_interleave_cuda(*this, repeats, dimension);
}

Tensor Tensor::unfold(
    std::int64_t dimension,
    std::int64_t window,
    std::int64_t step) const { return cuda::unfold(*this, dimension, window, step); }

std::vector<Tensor> Tensor::split_with_sizes(
    std::span<const std::int64_t> sections,
    std::int64_t dimension) const { return cuda::split_with_sizes(*this, sections, dimension); }

std::vector<Tensor> Tensor::chunk(std::int64_t chunks, std::int64_t dimension) const { return cuda::chunk(*this, chunks, dimension); }

Tensor Tensor::mean(std::int64_t dimension, bool keep_dimension) const {
    return cuda::mean(*this, dimension, keep_dimension);
}

Tensor Tensor::mean() const { return reshape({-1}).mean(0, false); }
Tensor Tensor::sum(std::int64_t dimension, bool keep_dimension) const {
    return cuda::sum(*this, dimension, keep_dimension);
}
Tensor Tensor::sum() const { return reshape({-1}).sum(0, false); }
Tensor Tensor::max() const { return reshape({-1}).amax(0, false); }
Tensor Tensor::amax(std::int64_t dimension, bool keep_dimension) const {
    return cuda::reduce_cuda(*this, dimension, keep_dimension, 2);
}
Tensor Tensor::norm() const { return square().sum().sqrt(); }
Tensor Tensor::abs() const { return cuda::unary_cuda(*this, 0); }
Tensor Tensor::square() const { return cuda::unary_cuda(*this, 1); }
Tensor Tensor::exp() const { return cuda::exp(*this); }
Tensor Tensor::sqrt() const { return cuda::sqrt(*this); }
Tensor Tensor::clamp(double minimum, double maximum) const {
    return cuda::clamp(*this, minimum, maximum);
}
Tensor Tensor::clamp_min(double minimum) const {
    return cuda::clamp_min(*this, minimum);
}
Tensor Tensor::clamp_max(double maximum) const {
    return cuda::clamp_max(*this, maximum);
}
Tensor Tensor::remainder(double divisor) const {
    return cuda::remainder(*this, divisor);
}
Tensor Tensor::argmax(std::int64_t dimension, bool keep_dimension) const {
    return cuda::argmax(*this, dimension, keep_dimension);
}
Tensor Tensor::eq(const Tensor& other) const { return cuda::binary_cuda(*this, other, 10); }
Tensor Tensor::eq(double other) const { return cuda::scalar_binary_cuda(*this, other, 10); }
Tensor Tensor::ne(const Tensor& other) const { return cuda::binary_cuda(*this, other, 11); }
Tensor Tensor::ne(double other) const { return cuda::scalar_binary_cuda(*this, other, 11); }
Tensor Tensor::all() const { return reshape({-1}).all(0, false); }
Tensor Tensor::all(std::int64_t dimension, bool keep_dimension) const {
    return cuda::reduce_cuda(*this, dimension, keep_dimension, 4);
}
Tensor Tensor::any(std::int64_t dimension, bool keep_dimension) const {
    return cuda::reduce_cuda(*this, dimension, keep_dimension, 5);
}
Tensor Tensor::logical_not() const { return cuda::unary_cuda(*this, 8); }
bool Tensor::equal(const Tensor& other) const { return tyloi::cuda::equal(*this, other); }
Tensor Tensor::masked_select(const Tensor& mask) const { return cuda::masked_select_cuda(*this, mask); }
Tensor Tensor::tril(std::int64_t diagonal) const {
    require_dense();
    if (dim() < 2) throw std::invalid_argument("tril requires at least two dimensions");
    auto rows = cuda::arange(size(-2), TensorOptions{}.dtype(kInt64).device(device())).unsqueeze(-1);
    auto columns = cuda::arange(size(-1), TensorOptions{}.dtype(kInt64).device(device())).unsqueeze(0);
    return cuda::where(columns <= rows + diagonal, *this, 0.0);
}
Tensor Tensor::masked_fill(const Tensor& mask, double value) const {
    mask.require_dense();
    auto result = clone();
    result.masked_fill_(mask, value);
    return result;
}
Tensor& Tensor::masked_fill_(const Tensor& mask, double value) {
    cuda::masked_fill_cuda(*this, mask, value);
    return *this;
}

Tensor Tensor::contiguous() const {
    return cuda::contiguous(*this);
}

Tensor Tensor::clone() const {
    return cuda::clone(*this);
}

Tensor Tensor::to(ScalarType type) const {
    return cuda::cast(*this, type);
}

Tensor Tensor::to(Device target_device) const {
    return cuda::copy(*this, target_device);
}

Tensor& Tensor::copy_(const Tensor& source, bool) {
    cuda::copy_into(*this, source);
    return *this;
}

Tensor& Tensor::zero_() {
    return fill_(0.0);
}

Tensor& Tensor::fill_(double value) {
    require_dense();
    if (!defined()) {
        throw std::invalid_argument("fill_ requires a defined tensor");
    }
    if (is_cuda()) {
        cuda::fill_cuda(*this, value);
        return *this;
    }
    if (!is_contiguous()) {
        throw std::invalid_argument("CPU fill requires a contiguous tensor");
    }
    if (value == 0.0) {
        std::memset(data_ptr(), 0, nbytes());
        return *this;
    }
    switch (scalar_type()) {
        case ScalarType::boolean:
            std::fill_n(data_ptr<bool>(), numel(), value != 0.0);
            break;
        case ScalarType::uint8:
            std::fill_n(data_ptr<std::uint8_t>(), numel(), static_cast<std::uint8_t>(value));
            break;
        case ScalarType::int8:
            std::fill_n(data_ptr<std::int8_t>(), numel(), static_cast<std::int8_t>(value));
            break;
        case ScalarType::int16:
            std::fill_n(data_ptr<std::int16_t>(), numel(), static_cast<std::int16_t>(value));
            break;
        case ScalarType::int32:
            std::fill_n(data_ptr<std::int32_t>(), numel(), static_cast<std::int32_t>(value));
            break;
        case ScalarType::int64:
            std::fill_n(data_ptr<std::int64_t>(), numel(), static_cast<std::int64_t>(value));
            break;
        case ScalarType::float16:
            std::fill_n(data_ptr<std::uint16_t>(), numel(), cuda::encode_float16(static_cast<float>(value)));
            break;
        case ScalarType::bfloat16:
            std::fill_n(data_ptr<std::uint16_t>(), numel(), cuda::encode_bfloat16(static_cast<float>(value)));
            break;
        case ScalarType::float32:
            std::fill_n(data_ptr<float>(), numel(), static_cast<float>(value));
            break;
        case ScalarType::float64:
            std::fill_n(data_ptr<double>(), numel(), value);
            break;
        default:
            throw std::invalid_argument("CPU fill for this scalar type is not implemented yet");
    }
    return *this;
}

Tensor& Tensor::add_(const Tensor& value) {
    auto result = *this + value;
    return copy_(result);
}

Tensor& Tensor::scatter_add_(
    std::int64_t dimension, const Tensor& index, const Tensor& source) {
    cuda::scatter_cuda(*this, dimension, index, source, true);
    return *this;
}

Tensor& Tensor::scatter_(
    std::int64_t dimension, const Tensor& index, const Tensor& source) {
    cuda::scatter_cuda(*this, dimension, index, source, false);
    return *this;
}

Tensor& Tensor::index_copy_(
    std::int64_t dimension, const Tensor& index, const Tensor& source) {
    cuda::index_copy_cuda(*this, dimension, index, source);
    return *this;
}

Tensor& Tensor::index_fill_(
    std::int64_t dimension, const Tensor& index, double value) {
    cuda::index_fill_cuda(*this, dimension, index, value);
    return *this;
}

Tensor& Tensor::index_put_(
    std::initializer_list<TensorIndex> indices, const Tensor& value) {
    value.require_dense();
    {
        auto destination = index(indices);
        cuda::copy_into(destination, value);
    }
    return *this;
}

Tensor& Tensor::index_put_(
    std::initializer_list<TensorIndex> indices, double value) {
    index(indices).fill_(value);
    return *this;
}

}  // namespace tyloi

namespace tyloi::cuda {

Tensor contiguous(const Tensor& input) {
    input.require_dense();
    if (!input.defined()) return {};
    if (input.is_contiguous()) return input;
    if (input.is_cuda()) return make_contiguous_cuda(input);
    auto result = empty(input.sizes(), input.options());
    copy_contiguous_cpu(result, input);
    return result;
}

Tensor clone(const Tensor& input) {
    input.require_dense();
    if (!input.defined()) return {};
    auto result = empty(input.sizes(), input.options());
    copy_into(result, input);
    return result;
}

Tensor cast(const Tensor& input, ScalarType type) {
    auto target = input.options().dtype(type);
    if (type == input.scalar_type()) return input;
    input.require_dense();
    if (input.is_cpu()) {
        auto result = empty(input.sizes(), target);
        copy_contiguous_cpu(result, input);
        return result;
    }
    return copy_or_convert_cuda(input, target);
}

Tensor cast(const Tensor& input, ScalarType type, Device target) {
    return copy(cast(input, type), target);
}

Tensor copy(const Tensor& input, Device target_device) {
    auto target = input.options().device(target_device);
    if (target_device == input.device()) return input;
    input.require_dense();
    return copy_or_convert_cuda(input, target);
}

void copy_into(Tensor& destination, const Tensor& source) {
    destination.require_dense(); source.require_dense();
    if (!destination.defined() || !source.defined()) {
        throw std::invalid_argument("copy_ requires defined tensors");
    }
    if (destination.is_cpu() && source.is_cpu()) {
        copy_contiguous_cpu(destination, source);
    } else {
        copy_cuda(destination, source);
    }
}

Tensor empty(std::span<const std::int64_t> shape, const TensorOptions& options) {
    if (options.target_device().is_cuda()) {
        return empty_cuda(shape, options);
    }
    return options.pinned() ? empty_pinned(shape, options) : empty_cpu(shape, options);
}

Tensor empty(std::initializer_list<std::int64_t> shape, const TensorOptions& options) {
    return empty(std::span<const std::int64_t>(shape.begin(), shape.size()), options);
}

Tensor empty_like(const Tensor& source, const TensorOptions* options) {
    source.require_dense();
    if (options == nullptr) return empty(source.sizes(), source.options());
    auto merged = source.options();
    if (options->has_dtype()) merged.dtype(options->scalar_type());
    if (options->has_device()) merged.device(options->target_device());
    if (options->has_pinned_memory()) merged.pinned_memory(options->pinned());
    return empty(source.sizes(), merged);
}

Tensor zeros(std::span<const std::int64_t> shape, const TensorOptions& options) {
    auto result = empty(shape, options);
    result.zero_();
    return result;
}

Tensor zeros(std::initializer_list<std::int64_t> shape, const TensorOptions& options) {
    return zeros(std::span<const std::int64_t>(shape.begin(), shape.size()), options);
}

Tensor zeros_like(const Tensor& source) {
    source.require_dense();
    return zeros(source.sizes(), source.options());
}

Tensor ones(std::span<const std::int64_t> shape, const TensorOptions& options) {
    return full(shape, 1.0, options);
}

Tensor ones(std::initializer_list<std::int64_t> shape, const TensorOptions& options) {
    return ones(std::span<const std::int64_t>(shape.begin(), shape.size()), options);
}

Tensor full(std::span<const std::int64_t> shape, double value, const TensorOptions& options) {
    auto result = empty(shape, options);
    result.fill_(value);
    return result;
}

Tensor full(
    std::initializer_list<std::int64_t> shape,
    double value,
    const TensorOptions& options) {
    return full(std::span<const std::int64_t>(shape.begin(), shape.size()), value, options);
}

Tensor full_like(const Tensor& source, double value) {
    source.require_dense();
    return full(source.sizes(), value, source.options());
}

Tensor ones_like(const Tensor& source) {
    source.require_dense();
    return ones(source.sizes(), source.options());
}

Tensor from_blob(
    void* data,
    std::span<const std::int64_t> shape,
    const TensorOptions& options) {
    auto descriptor = make_contiguous_view(
        data, shape, options.scalar_type(), options.target_device());
    if (data == nullptr && descriptor.nbytes() != 0) {
        throw std::invalid_argument("from_blob requires non-null storage");
    }
    auto storage = std::make_shared<TensorStorage>();
    storage->base = data;
    storage->bytes = descriptor.nbytes();
    storage->device = options.target_device();
    storage->pinned = options.pinned();
    return Tensor(std::move(storage), descriptor);
}

Tensor from_blob(
    void* data,
    std::initializer_list<std::int64_t> shape,
    const TensorOptions& options) {
    return from_blob(
        data,
        std::span<const std::int64_t>(shape.begin(), shape.size()),
        options);
}

std::vector<char> pickle_save(const Tensor& value) {
    if (!value.defined()) throw std::invalid_argument("cannot serialize an undefined tensor");
    const auto host = contiguous(copy(value, Device{DeviceType::cpu, 0}));
    constexpr std::string_view magic{"MFQTNSR1", 8};
    const auto header_bytes = std::size_t{16} +
        static_cast<std::size_t>(host.dim()) * sizeof(std::int64_t);
    std::vector<char> result(header_bytes + host.nbytes());
    std::memcpy(result.data(), magic.data(), magic.size());
    result[8] = static_cast<char>(host.scalar_type());
    result[9] = static_cast<char>(host.dim());
    std::fill(result.begin() + 10, result.begin() + 16, 0);
    for (std::int64_t axis = 0; axis < host.dim(); ++axis) {
        const auto extent = host.size(axis);
        std::memcpy(
            result.data() + 16 + static_cast<std::size_t>(axis) * sizeof(extent),
            &extent,
            sizeof(extent));
    }
    std::memcpy(result.data() + header_bytes, host.data_ptr(), host.nbytes());
    return result;
}

NativeIValue pickle_load(const std::vector<char>& bytes) {
    constexpr std::string_view magic{"MFQTNSR1", 8};
    if (bytes.size() < 16 ||
        std::string_view(bytes.data(), magic.size()) != magic) {
        throw std::invalid_argument(
            "native tensor file is not MFQTNSR1; legacy Torch pickle files require "
            "the optional mfq-runtime-torch reference runtime");
    }
    const auto raw_type = static_cast<std::uint8_t>(bytes[8]);
    const auto rank = static_cast<std::uint8_t>(bytes[9]);
    if (raw_type > static_cast<std::uint8_t>(ScalarType::float8_e4m3fn) ||
        rank > kMaximumTensorRank) {
        throw std::invalid_argument("native tensor header is invalid");
    }
    const auto header_bytes = std::size_t{16} +
        static_cast<std::size_t>(rank) * sizeof(std::int64_t);
    if (bytes.size() < header_bytes) {
        throw std::invalid_argument("native tensor header is truncated");
    }
    std::vector<std::int64_t> shape(rank);
    for (std::size_t axis = 0; axis < rank; ++axis) {
        std::memcpy(
            &shape[axis],
            bytes.data() + 16 + axis * sizeof(std::int64_t),
            sizeof(std::int64_t));
        if (shape[axis] < 0) throw std::invalid_argument("native tensor shape is invalid");
    }
    auto output = empty(
        shape,
        TensorOptions{}.dtype(static_cast<ScalarType>(raw_type)).device(kCPU));
    if (bytes.size() != header_bytes + output.nbytes()) {
        throw std::invalid_argument("native tensor payload size does not match its shape");
    }
    std::memcpy(output.data_ptr(), bytes.data() + header_bytes, output.nbytes());
    return NativeIValue{std::move(output)};
}

}  // namespace tyloi::cuda
