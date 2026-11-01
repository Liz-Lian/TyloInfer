#include "engine/models/common/session/paged.h"

#include <cstring>
#include <limits>

void CudaPagedWriter::raw(const void *data, size_t size) {
    const auto *begin = static_cast<const uint8_t *>(data);
    bytes_.insert(bytes_.end(), begin, begin + size);
}

void CudaPagedWriter::tensor(const mfq_tensor_backend::Tensor &source) {
    if (!source.defined() || !source.is_cuda() || source.dim() <= 0 || source.dim() > 8) {
        throw std::runtime_error("invalid CUDA paged cache tensor");
    }
    const auto dtype = source.scalar_type();
    if (dtype != mfq_tensor_backend::kFloat16 && dtype != mfq_tensor_backend::kBFloat16 &&
        dtype != mfq_tensor_backend::kFloat32 && dtype != mfq_tensor_backend::kInt32 && dtype != mfq_tensor_backend::kInt64) {
        throw std::runtime_error("unsupported CUDA paged cache tensor dtype");
    }
    const auto cpu = source.contiguous().to(mfq_tensor_backend::kCPU);
    scalar<int32_t>(static_cast<int32_t>(dtype));
    scalar<int32_t>(source.get_device());
    scalar<uint32_t>(static_cast<uint32_t>(cpu.dim()));
    scalar<uint32_t>(0);
    for (const auto dimension : cpu.sizes())
        scalar<int64_t>(dimension);
    const auto bytes = static_cast<uint64_t>(cpu.numel() * cpu.element_size());
    scalar<uint64_t>(bytes);
    raw(cpu.data_ptr(), static_cast<size_t>(bytes));
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

std::pair<mfq_tensor_backend::Tensor, int> CudaPagedReader::tensor() {
    const auto dtype_value = scalar<int32_t>("dtype");
    const auto device = scalar<int32_t>("device");
    const auto rank = scalar<uint32_t>("rank");
    (void)scalar<uint32_t>("tensor flags");
    if (rank == 0 || rank > 8 || device < 0) {
        throw CudaSessionStateError("invalid CUDA paged cache tensor header");
    }
    const auto dtype = static_cast<mfq_tensor_backend::ScalarType>(dtype_value);
    if (dtype != mfq_tensor_backend::kFloat16 && dtype != mfq_tensor_backend::kBFloat16 &&
        dtype != mfq_tensor_backend::kFloat32 && dtype != mfq_tensor_backend::kInt32 && dtype != mfq_tensor_backend::kInt64) {
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
    const uint64_t element_size = dtype == mfq_tensor_backend::kInt64 ? 8 :
        (dtype == mfq_tensor_backend::kFloat32 || dtype == mfq_tensor_backend::kInt32) ? 4 : 2;
    if (elements > std::numeric_limits<uint64_t>::max() / element_size ||
        bytes != elements * element_size || bytes > remaining()) {
        throw CudaSessionStateError("invalid CUDA paged cache tensor size");
    }
    auto cpu = mfq_tensor_backend::empty(
        shape, mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU).dtype(dtype));
    raw(cpu.data_ptr(), static_cast<size_t>(bytes), "tensor");
    return {std::move(cpu), device};
}

void CudaPagedReader::expect_end() const {
    if (cursor_ != end_) {
        throw CudaSessionStateError("trailing CUDA paged cache payload bytes");
    }
}

size_t CudaPagedReader::remaining() const { return static_cast<size_t>(end_ - cursor_); }
