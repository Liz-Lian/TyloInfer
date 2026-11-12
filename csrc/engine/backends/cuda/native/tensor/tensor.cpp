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

DenseTensor empty_cpu(std::span<const std::int64_t> shape, const TensorOptions& options) {
    auto descriptor = make_contiguous_view(
        reinterpret_cast<void*>(1), shape, options.scalar_type(), options.target_device());
    auto storage = allocate_cpu_storage(descriptor.nbytes(), options.pinned());
    descriptor.data = storage->base;
    return DenseTensor(std::move(storage), descriptor);
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

void copy_contiguous_cpu(DenseTensor& destination, const DenseTensor& source) {
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
DenseTensor empty_cuda(std::span<const std::int64_t> shape, const TensorOptions& options);
DenseTensor empty_pinned(std::span<const std::int64_t> shape, const TensorOptions& options);
DenseTensor copy_or_convert_cuda(const DenseTensor& source, const TensorOptions& options);
DenseTensor make_contiguous_cuda(const DenseTensor& source);

DenseTensor contiguous(const DenseTensor& input) {
    input.require_dense();
    if (!input.defined()) return {};
    if (input.is_contiguous()) return input;
    if (input.is_cuda()) return make_contiguous_cuda(input);
    auto result = empty(input.sizes(), input.options());
    copy_contiguous_cpu(result, input);
    return result;
}

DenseTensor clone(const DenseTensor& input) {
    input.require_dense();
    if (!input.defined()) return {};
    auto result = empty(input.sizes(), input.options());
    copy_into(result, input);
    return result;
}

DenseTensor cast(const DenseTensor& input, ScalarType type) {
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

DenseTensor cast(const DenseTensor& input, ScalarType type, Device target) {
    return copy(cast(input, type), target);
}

DenseTensor copy(const DenseTensor& input, Device target_device) {
    auto target = input.options().device(target_device);
    if (target_device == input.device()) return input;
    input.require_dense();
    return copy_or_convert_cuda(input, target);
}

void copy_into(DenseTensor& destination, const DenseTensor& source, bool non_blocking) {
    destination.require_dense(); source.require_dense();
    if (!destination.defined() || !source.defined()) {
        throw std::invalid_argument("copy_ requires defined tensors");
    }
    if (non_blocking && destination.is_cuda() != source.is_cuda()) {
        const auto& host = destination.is_cpu() ? destination : source;
        if (!host.options().pinned() || !destination.is_contiguous() || !source.is_contiguous() ||
            destination.scalar_type() != source.scalar_type() || destination.numel() != source.numel())
            throw std::invalid_argument("non-blocking host/device copy requires pinned host storage and contiguous tensors with equal dtype and size");
    }
    if (destination.is_cpu() && source.is_cpu()) {
        copy_contiguous_cpu(destination, source);
    } else {
        copy_cuda(destination, source, non_blocking);
    }
}

DenseTensor empty(std::span<const std::int64_t> shape, const TensorOptions& options) {
    if (options.target_device().is_cuda()) {
        return empty_cuda(shape, options);
    }
    return options.pinned() ? empty_pinned(shape, options) : empty_cpu(shape, options);
}

DenseTensor empty(std::initializer_list<std::int64_t> shape, const TensorOptions& options) {
    return empty(std::span<const std::int64_t>(shape.begin(), shape.size()), options);
}

DenseTensor empty_like(const DenseTensor& source, const TensorOptions* options) {
    source.require_dense();
    if (options == nullptr) return empty(source.sizes(), source.options());
    auto merged = source.options();
    if (options->has_dtype()) merged.dtype(options->scalar_type());
    if (options->has_device()) merged.device(options->target_device());
    if (options->has_pinned_memory()) merged.pinned_memory(options->pinned());
    return empty(source.sizes(), merged);
}

DenseTensor zeros(std::span<const std::int64_t> shape, const TensorOptions& options) {
    auto result = empty(shape, options);
    result.zero_();
    return result;
}

DenseTensor zeros(std::initializer_list<std::int64_t> shape, const TensorOptions& options) {
    return zeros(std::span<const std::int64_t>(shape.begin(), shape.size()), options);
}

DenseTensor zeros_like(const DenseTensor& source) {
    source.require_dense();
    return zeros(source.sizes(), source.options());
}

DenseTensor ones(std::span<const std::int64_t> shape, const TensorOptions& options) {
    return full(shape, 1.0, options);
}

DenseTensor ones(std::initializer_list<std::int64_t> shape, const TensorOptions& options) {
    return ones(std::span<const std::int64_t>(shape.begin(), shape.size()), options);
}

DenseTensor full(std::span<const std::int64_t> shape, double value, const TensorOptions& options) {
    auto result = empty(shape, options);
    result.fill_(value);
    return result;
}

DenseTensor full(
    std::initializer_list<std::int64_t> shape,
    double value,
    const TensorOptions& options) {
    return full(std::span<const std::int64_t>(shape.begin(), shape.size()), value, options);
}

DenseTensor full_like(const DenseTensor& source, double value) {
    source.require_dense();
    return full(source.sizes(), value, source.options());
}

DenseTensor ones_like(const DenseTensor& source) {
    source.require_dense();
    return ones(source.sizes(), source.options());
}

DenseTensor from_blob(
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
    return DenseTensor(std::move(storage), descriptor);
}

DenseTensor from_blob(
    void* data,
    std::initializer_list<std::int64_t> shape,
    const TensorOptions& options) {
    return from_blob(
        data,
        std::span<const std::int64_t>(shape.begin(), shape.size()),
        options);
}

std::vector<char> pickle_save(const DenseTensor& value) {
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
