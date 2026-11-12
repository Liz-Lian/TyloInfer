#include "engine/backends/cpu/native/host_parallel.h"
#include "host.h"

#include <thread>

namespace tyloi::cuda::host {
namespace {

template <typename Value>
void multiply(const DenseTensor &left, const DenseTensor &right,
              DenseTensor &output, const std::vector<std::int64_t> &batch,
              std::int64_t rows, std::int64_t columns,
              std::int64_t contraction) {
  const auto a = left.view_descriptor();
  const auto b = right.view_descriptor();
  const auto dst = output.view_descriptor();
  const auto threads =
      static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  ::tyloi::host_parallel_for(
      0, output.numel(), 64, threads, [&](auto begin, auto end) {
        for (auto index = begin; index < end; ++index) {
          const auto column = index % columns;
          const auto row = (index / columns) % rows;
          auto rest = index / (columns * rows);
          std::int64_t ao = row * a.strides[a.rank - 2];
          std::int64_t bo = column * b.strides[b.rank - 1];
          for (std::size_t rev = batch.size(); rev > 0; --rev) {
            const auto axis = rev - 1;
            const auto coordinate = rest % batch[axis];
            rest /= batch[axis];
            const auto ai = static_cast<std::int64_t>(axis) -
                            static_cast<std::int64_t>(batch.size()) + a.rank -
                            2;
            const auto bi = static_cast<std::int64_t>(axis) -
                            static_cast<std::int64_t>(batch.size()) + b.rank -
                            2;
            if (ai >= 0 && a.sizes[ai] != 1)
              ao += coordinate * a.strides[ai];
            if (bi >= 0 && b.sizes[bi] != 1)
              bo += coordinate * b.strides[bi];
          }
          Value sum = 0;
          for (std::int64_t k = 0; k < contraction; ++k) {
            sum += load<Value>(pointer(a, ao + k * a.strides[a.rank - 1]),
                               a.scalar_type) *
                   load<Value>(pointer(b, bo + k * b.strides[b.rank - 2]),
                               b.scalar_type);
          }
          store(pointer(dst, index), dst.scalar_type, sum);
        }
      });
}
} // namespace

DenseTensor matmul(const DenseTensor &left_source,
                   const DenseTensor &right_source) {
  left_source.require_dense();
  right_source.require_dense();
  if (!left_source.is_cpu() || !right_source.is_cpu() ||
      left_source.device() != right_source.device()) {
    throw std::invalid_argument(
        "host matmul requires tensors on the same CPU device");
  }
  if (left_source.dim() < 1 || right_source.dim() < 1 ||
      left_source.scalar_type() != right_source.scalar_type() ||
      !native_ops_detail::floating(left_source.scalar_type())) {
    throw std::invalid_argument(
        "host matmul requires ranked tensors with matching floating dtypes");
  }
  const bool lv = left_source.dim() == 1;
  const bool rv = right_source.dim() == 1;
  const auto left = lv ? left_source.unsqueeze(0) : left_source;
  const auto right = rv ? right_source.unsqueeze(-1) : right_source;
  if (left.size(-1) != right.size(-2)) {
    throw std::invalid_argument("matmul contraction dimensions do not match");
  }
  const auto rank =
      static_cast<std::size_t>(std::max(left.dim(), right.dim()) - 2);
  std::vector<std::int64_t> batch(rank, 1);
  for (std::size_t rev = 0; rev < rank; ++rev) {
    const auto ai = left.dim() - 3 - static_cast<std::int64_t>(rev);
    const auto bi = right.dim() - 3 - static_cast<std::int64_t>(rev);
    const auto a = ai >= 0 ? left.size(ai) : 1;
    const auto b = bi >= 0 ? right.size(bi) : 1;
    if (a != b && a != 1 && b != 1)
      throw std::invalid_argument(
          "matmul batch dimensions cannot be broadcast");
    batch[rank - rev - 1] = a == 1 ? b : a;
  }
  const auto rows = left.size(-2), columns = right.size(-1),
             contraction = left.size(-1);
  auto shape = batch;
  shape.push_back(rows);
  shape.push_back(columns);
  auto output = empty(shape, left.options());
  if (output.numel() != 0) {
    if (left.scalar_type() == kFloat64)
      multiply<double>(left, right, output, batch, rows, columns, contraction);
    else
      multiply<float>(left, right, output, batch, rows, columns, contraction);
  }
  if (lv)
    output = output.squeeze(-2);
  if (rv)
    output = output.squeeze(-1);
  return output;
}
} // namespace tyloi::cuda::host
