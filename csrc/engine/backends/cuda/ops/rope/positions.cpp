#include "engine/ops/positions.h"

#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/native/resident_allocations.h"
#include "engine/backends/cuda/native/tensor/backend.h"

#include <limits>
#include <optional>
#include <stdexcept>

namespace tyloi::ops {
namespace {
const DenseTensor& dense(const Tensor& value) {
    const auto* result = value.representation<DenseTensor>();
    if (!result || !result->defined() || result->is_encoded())
        throw std::invalid_argument("position input must be dense");
    return *result;
}
void span(std::int64_t begin, std::int64_t count) {
    if (begin < 0 || count <= 0 || begin > std::numeric_limits<std::int64_t>::max() - count)
        throw std::invalid_argument("invalid position span");
}
void device(const TensorOptions& options, std::optional<MfqCudaGuard>& guard) {
    const auto target = options.target_device();
    if (target.is_cuda()) {
        (void)cuda::Executor::current();
        guard.emplace(target.index);
    } else if (!target.is_cpu()) {
        throw std::invalid_argument("unsupported position device");
    }
}
} // namespace

std::array<Tensor, 2> position_span(std::int64_t begin, std::int64_t count, std::int64_t offset,
        const Tensor& history, const TensorOptions& options) {
    span(begin, count);
    std::array<Tensor, 2> result{Tensor{DenseTensor{}}, Tensor{DenseTensor{}}};
    std::optional<MfqCudaGuard> guard;
    device(options, guard);
    cuda::ResidentAllocationScope resident;
    result[0].dense() = cuda::arange(begin, begin + count, options);
    if (offset) result[0].dense() = result[0].dense() + offset;
    result[1].dense() = history.defined()
        ? cuda::cat({dense(history), result[0].dense()}, -1) : result[0].dense();
    return result;
}

std::array<Tensor, 2> position_axes(std::int64_t begin, std::int64_t count, std::int64_t batch,
        std::int64_t axes, const Tensor& supplied, const Tensor& history, const TensorOptions& options) {
    span(begin, count);
    if (batch <= 0 || axes <= 0) throw std::invalid_argument("invalid position axes");
    std::array<Tensor, 2> result{Tensor{DenseTensor{}}, Tensor{DenseTensor{}}};
    std::optional<MfqCudaGuard> guard;
    device(options, guard);
    cuda::ResidentAllocationScope resident;
    result[0].dense() = supplied.defined() ? dense(supplied) : cuda::arange(begin, begin + count, options);
    auto& current = result[0].dense();
    if (current.dim() == 1)
        current = current.reshape({1, 1, count}).expand({axes, 1, count}).contiguous();
    else if (current.dim() == 2)
        current = current.unsqueeze(1);
    if (current.dim() == 3 && current.size(0) == axes + 1)
        current = current.narrow(0, 1, axes);
    if (current.dim() != 3 || current.size(0) != axes || current.size(-1) != count ||
        (current.size(1) != 1 && current.size(1) != batch))
        throw std::invalid_argument("positions require token, axis/token or axis/batch/token rows");
    current = current.to(options.scalar_type()).expand({axes, batch, count}).contiguous();
    result[1].dense() = history.defined() ? cuda::cat({dense(history), current}, -1) : current;
    return result;
}
} // namespace tyloi::ops
