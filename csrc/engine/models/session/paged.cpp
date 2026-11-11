#include "engine/models/session/paged.h"
#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/ops.h"
#include <span>
#else
#include "backend.h"
#endif

#include <cstring>
#include <limits>

void CudaPagedWriter::raw(const void *data, size_t size) {
    const auto *begin = static_cast<const uint8_t *>(data);
    bytes_.insert(bytes_.end(), begin, begin + size);
}

void CudaPagedWriter::tensor(const ::tyloi::Tensor &tensor) {
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& source = tensor;
#else
    const auto* dense = tensor.representation<::tyloi::DenseTensor>();
    if (!dense) throw std::runtime_error("invalid CUDA paged cache tensor");
    const auto& source = *dense;
#endif
    if (!source.defined() || !source.is_cuda() || source.dim() <= 0 || source.dim() > 8) {
        throw std::runtime_error("invalid CUDA paged cache tensor");
    }
    const auto dtype = source.scalar_type();
    if (dtype != ::tyloi::kFloat16 && dtype != ::tyloi::kBFloat16 &&
        dtype != ::tyloi::kFloat32 && dtype != ::tyloi::kInt32 && dtype != ::tyloi::kInt64) {
        throw std::runtime_error("unsupported CUDA paged cache tensor dtype");
    }
#ifdef TYLOINFER_NATIVE_CUDA
    const auto data = ::tyloi::ops::bytes(source);
    scalar<int32_t>(static_cast<int32_t>(dtype));
    scalar<int32_t>(source.device().index);
    scalar<uint32_t>(static_cast<uint32_t>(source.dim()));
    scalar<uint32_t>(0);
    for (int64_t axis = 0; axis < source.dim(); ++axis)
        scalar<int64_t>(source.size(axis));
    scalar<uint64_t>(static_cast<uint64_t>(data.size()));
    raw(data.data(), data.size());
#else
    const auto cpu = source.contiguous().to(::tyloi::kCPU);
    scalar<int32_t>(static_cast<int32_t>(dtype));
    scalar<int32_t>(source.get_device());
    scalar<uint32_t>(static_cast<uint32_t>(cpu.dim()));
    scalar<uint32_t>(0);
    for (const auto dimension : cpu.sizes())
        scalar<int64_t>(dimension);
    const auto bytes = static_cast<uint64_t>(cpu.numel() * cpu.element_size());
    scalar<uint64_t>(bytes);
    raw(cpu.data_ptr(), static_cast<size_t>(bytes));
#endif
}

CudaPagedPayload CudaPagedWriter::finish() && {
    return std::make_shared<const std::vector<uint8_t>>(std::move(bytes_));
}

void CudaPagedReader::raw(void *destination, size_t size, const char *name) {
    if (remaining() < size) {
        throw CudaSessionStateError(std::string("truncated CUDA paged cache ") + name);
    }
    std::memcpy(destination, cursor_, size);
    cursor_ += size;
}

std::pair<::tyloi::Tensor, int> CudaPagedReader::tensor() {
    const auto dtype_value = scalar<int32_t>("dtype");
    const auto device = scalar<int32_t>("device");
    const auto rank = scalar<uint32_t>("rank");
    (void)scalar<uint32_t>("tensor flags");
    if (rank == 0 || rank > 8 || device < 0) {
        throw CudaSessionStateError("invalid CUDA paged cache tensor header");
    }
    const auto dtype = static_cast<::tyloi::ScalarType>(dtype_value);
    if (dtype != ::tyloi::kFloat16 && dtype != ::tyloi::kBFloat16 &&
        dtype != ::tyloi::kFloat32 && dtype != ::tyloi::kInt32 && dtype != ::tyloi::kInt64) {
        throw CudaSessionStateError("invalid CUDA paged cache tensor dtype");
    }
    std::vector<int64_t> shape;
    shape.reserve(rank);
    uint64_t elements = 1;
    for (uint32_t index = 0; index < rank; ++index) {
        const auto dimension = scalar<int64_t>("shape");
        if (dimension <= 0 ||
            elements > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(dimension)) {
            throw CudaSessionStateError("invalid CUDA paged cache tensor shape");
        }
        shape.push_back(dimension);
        elements *= static_cast<uint64_t>(dimension);
    }
    const auto bytes = scalar<uint64_t>("tensor size");
    const uint64_t element_size = dtype == ::tyloi::kInt64 ? 8 :
        (dtype == ::tyloi::kFloat32 || dtype == ::tyloi::kInt32) ? 4 : 2;
    if (elements > std::numeric_limits<uint64_t>::max() / element_size ||
        bytes != elements * element_size || bytes > remaining()) {
        throw CudaSessionStateError("invalid CUDA paged cache tensor size");
    }
#ifdef TYLOINFER_NATIVE_CUDA
    const auto data = std::span<const std::byte>(reinterpret_cast<const std::byte*>(cursor_),
                                                  static_cast<size_t>(bytes));
    auto cpu = ::tyloi::ops::tensor(data, shape,
        ::tyloi::TensorOptions().device(::tyloi::kCPU).dtype(dtype));
    cursor_ += static_cast<size_t>(bytes);
    return {std::move(cpu), device};
#else
    auto cpu = ::tyloi::cuda::empty(
        shape, ::tyloi::TensorOptions().device(::tyloi::kCPU).dtype(dtype));
    raw(cpu.data_ptr(), static_cast<size_t>(bytes), "tensor");
    return {std::make_shared<::tyloi::DenseTensor>(std::move(cpu)), device};
#endif
}

void CudaPagedReader::expect_end() const {
    if (cursor_ != end_) {
        throw CudaSessionStateError("trailing CUDA paged cache payload bytes");
    }
}

size_t CudaPagedReader::remaining() const { return static_cast<size_t>(end_ - cursor_); }
