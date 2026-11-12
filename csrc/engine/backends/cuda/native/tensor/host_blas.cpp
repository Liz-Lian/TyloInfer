#include "engine/backends/cpu/native/host_parallel.h"
#include "host.h"

#include <thread>

namespace tyloi::cuda::host {
namespace {

/** Reuse host loads across disjoint output tiles while retaining each dot
 * product's K order. */
template <ScalarType Type, typename Value>
void multiply(const DenseTensor &left, const DenseTensor &right,
              DenseTensor &output, const std::vector<std::int64_t> &batch,
              std::int64_t rows, std::int64_t columns,
              std::int64_t contraction) {
  const auto a = left.view_descriptor(), b = right.view_descriptor(),
             dst = output.view_descriptor();
  constexpr std::int64_t tile_rows = 4, tile_columns = 8;
  constexpr auto element_size = static_cast<std::int64_t>(scalar_size(Type));
  const auto row_tiles = 1 + (rows - 1) / tile_rows;
  const auto column_tiles = 1 + (columns - 1) / tile_columns;
  const auto batches = output.numel() / rows / columns;
  const auto tiles = batches * row_tiles * column_tiles;
  const auto *left_data = static_cast<const std::byte *>(a.data);
  const auto *right_data = static_cast<const std::byte *>(b.data);
  const auto threads =
      static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  ::tyloi::host_parallel_for(0, tiles, 1, threads, [&](auto begin, auto end) {
    for (auto tile = begin; tile < end; ++tile) {
      const auto column = (tile % column_tiles) * tile_columns;
      const auto row = ((tile / column_tiles) % row_tiles) * tile_rows;
      auto rest = tile / column_tiles / row_tiles;
      const auto batch_index = rest;
      std::int64_t ao = 0, bo = 0;
      for (std::size_t rev = batch.size(); rev > 0; --rev) {
        const auto axis = rev - 1;
        const auto coordinate = rest % batch[axis];
        rest /= batch[axis];
        const auto ai = static_cast<std::int64_t>(axis) -
                        static_cast<std::int64_t>(batch.size()) + a.rank - 2;
        const auto bi = static_cast<std::int64_t>(axis) -
                        static_cast<std::int64_t>(batch.size()) + b.rank - 2;
        if (ai >= 0 && a.sizes[ai] != 1)
          ao += coordinate * a.strides[ai];
        if (bi >= 0 && b.sizes[bi] != 1)
          bo += coordinate * b.strides[bi];
      }
      const auto active_rows = std::min(tile_rows, rows - row);
      const auto active_columns = std::min(tile_columns, columns - column);
      Value sums[tile_rows][tile_columns]{};
      for (std::int64_t k = 0; k < contraction; ++k) {
        Value x[tile_rows], y[tile_columns];
        for (std::int64_t r = 0; r < active_rows; ++r) {
          const auto index = ao + (row + r) * a.strides[a.rank - 2] +
                             k * a.strides[a.rank - 1];
          x[r] = load<Value>(left_data + index * element_size, Type);
        }
        for (std::int64_t c = 0; c < active_columns; ++c) {
          const auto index = bo + k * b.strides[b.rank - 2] +
                             (column + c) * b.strides[b.rank - 1];
          y[c] = load<Value>(right_data + index * element_size, Type);
        }
        for (std::int64_t r = 0; r < active_rows; ++r)
          for (std::int64_t c = 0; c < active_columns; ++c)
            sums[r][c] += x[r] * y[c];
      }
      for (std::int64_t r = 0; r < active_rows; ++r)
        for (std::int64_t c = 0; c < active_columns; ++c)
          store(pointer(dst,
                        (batch_index * rows + row + r) * columns + column + c),
                Type, sums[r][c]);
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
    switch (left.scalar_type()) {
    case kFloat16:
      multiply<kFloat16, float>(left, right, output, batch, rows, columns,
                                contraction);
      break;
    case kBFloat16:
      multiply<kBFloat16, float>(left, right, output, batch, rows, columns,
                                 contraction);
      break;
    case kFloat32:
      multiply<kFloat32, float>(left, right, output, batch, rows, columns,
                                contraction);
      break;
    case kFloat64:
      multiply<kFloat64, double>(left, right, output, batch, rows, columns,
                                 contraction);
      break;
    default:
      throw std::invalid_argument("unsupported host matmul dtype");
    }
  }
  if (lv)
    output = output.squeeze(-2);
  if (rv)
    output = output.squeeze(-1);
  return output;
}
} // namespace tyloi::cuda::host
