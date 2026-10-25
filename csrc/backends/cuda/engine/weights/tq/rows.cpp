#include "backends/cuda/engine/weights/tq/rows.h"

#include "env.h"
#include "tq.h"
#include "backends/cuda/engine/weights/tq/image.h"
#include "weight_memory.h"

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <unordered_map>

namespace tb = mfq_tensor_backend;

namespace {

void status(int code) {
    if (code < 0) throw std::invalid_argument("invalid TQ CUDA projection arguments");
    if (code >= 1000) {
        throw std::runtime_error("TQ cuBLAS status " + std::to_string(code - 1000));
    }
    TYLOINFER_CUDA_CHECK(static_cast<cudaError_t>(code));
}

std::size_t segment_for(const mfq::TqTable& table, std::uint64_t row) {
    std::size_t segment = 0;
    while (segment + 1 < table.segment_count() &&
           table.segment_end(segment) <= row) {
        ++segment;
    }
    return segment;
}

tb::Tensor lookup_tq_rows(const mfq::TqTable& table,
                          const std::int64_t* ids, std::size_t count,
                          const tb::Device& device) {
    if (!device.is_cuda() || !count || !ids) {
        throw std::invalid_argument("TQ table lookup requires CUDA and row IDs");
    }
    std::vector<std::int64_t> selected(ids, ids + count);
    std::sort(selected.begin(), selected.end());
    selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
    if (selected.front() < 0 ||
        std::uint64_t(selected.back()) >= table.rows()) {
        throw std::out_of_range("TQ table lookup row is out of bounds");
    }
    // Read each compressed row once, coalescing adjacent rows before restoring
    // the caller's order on GPU.
    const MfqCudaGuard guard(device);
    const auto gather = [&](std::size_t segment, std::size_t begin,
                            std::size_t end) {
        const auto start = segment ? table.segment_end(segment - 1) : 0;
        std::vector<std::vector<std::uint8_t>> pieces;
        for (auto run = begin; run < end;) {
            auto next = run + 1;
            while (next < end && selected[next] == selected[next - 1] + 1) ++next;
            pieces.push_back(table.segment(segment).slice_rows(
                selected[run] - start, selected[next - 1] + 1 - start));
            run = next;
        }
        return to_device_tq(mfq::join_rows(pieces), table.width(), device.index);
    };
    const auto first = segment_for(table, selected.front());
    const auto last = segment_for(table, selected.back());
    if (first == last) {
        const auto weight = gather(first, 0, selected.size());
        if (selected.size() != count || !std::equal(selected.begin(), selected.end(), ids)) {
            std::vector<std::int64_t> inverse;
            inverse.reserve(count);
            for (std::size_t i = 0; i < count; ++i) {
                inverse.push_back(std::lower_bound(
                    selected.begin(), selected.end(), ids[i]) - selected.begin());
            }
            const auto index = tb::from_blob(
                inverse.data(), {static_cast<std::int64_t>(count)},
                tb::TensorOptions().dtype(tb::kInt64)).clone().to(device, false, false);
            return weight.embedding(index);
        }
        if (count > 65535) {
            return weight.embedding(tb::arange(
                count, weight.image.options().dtype(tb::kInt64)));
        }
        auto result = tb::empty(
            {static_cast<std::int64_t>(count), table.width()},
            weight.image.options().dtype(tb::kFloat16));
        status(mfq_tq_embedding(weight.view, nullptr, static_cast<int>(count),
                                 result.data_ptr(), 0, mfq_current_cuda_stream()));
        return result;
    }
    auto result = tb::empty(
        {static_cast<std::int64_t>(selected.size()), table.width()},
        tb::TensorOptions().device(device).dtype(tb::kFloat16));
    std::size_t begin = 0;
    for (std::size_t segment = 0;
         segment < table.segment_count() && begin < selected.size(); ++segment) {
        const auto end = std::lower_bound(
            selected.begin() + begin, selected.end(), table.segment_end(segment)) -
            selected.begin();
        if (std::size_t(end) > begin) {
            const auto weight = gather(segment, begin, end);
            const auto rows = end - begin;
            if (rows > 65535) {
                const auto index = tb::arange(
                    rows, weight.image.options().dtype(tb::kInt64));
                result.slice(0, begin, end).copy_(weight.embedding(index));
            } else {
                status(mfq_tq_embedding(
                    weight.view, nullptr, static_cast<int>(rows),
                    static_cast<std::uint8_t*>(result.data_ptr()) +
                        begin * table.width() * sizeof(std::uint16_t),
                    0, mfq_current_cuda_stream()));
            }
        }
        begin = end;
    }
    if (selected.size() != count || !std::equal(selected.begin(), selected.end(), ids)) {
        std::vector<std::int64_t> inverse;
        inverse.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            inverse.push_back(std::lower_bound(
                selected.begin(), selected.end(), ids[i]) - selected.begin());
        }
        const auto index = tb::from_blob(
            inverse.data(), {static_cast<std::int64_t>(count)},
            tb::TensorOptions().dtype(tb::kInt64)).clone().to(device, false, false);
        return result.index_select(0, index);
    }
    return result;
}

} // namespace

tb::Tensor lookup_tq_table(const mfq::TqTable& table,
                           const std::int64_t* ids, std::size_t count,
                           const tb::Device& device) {
    return lookup_tq_rows(table, ids, count, device);
}

tb::Tensor lookup_tq_table(const mfq::TqTable& table, tb::Tensor ids) {
    auto host = ids.contiguous().cpu().to(tb::kInt64);
    auto shape = ids.sizes().vec();
    shape.push_back(table.width());
    if (!ids.numel()) return tb::empty(shape, ids.options().dtype(tb::kFloat16));
    return lookup_tq_rows(
        table, host.data_ptr<std::int64_t>(), host.numel(), ids.device()).reshape(shape);
}

tb::Tensor lookup_tq_shards(
        const std::vector<std::shared_ptr<mfq::TqTable>>& tables,
        std::int64_t rows_per_shard, const std::vector<std::int64_t>& ids,
        const tb::Device& device) {
    if (tables.empty() || !tables.front() || rows_per_shard < 1 || !device.is_cuda()) {
        throw std::invalid_argument("invalid TQ table shard geometry");
    }
    const auto columns = tables.front()->width();
    for (const auto& table : tables) {
        if (!table || table->width() != columns) {
            throw std::invalid_argument("TQ table shard widths differ");
        }
    }
    static std::atomic<bool> reported{false};
    if (mfq::cuda::env_lookup("TYLOINFER_TQ_PROFILE_CACHE") && !reported.exchange(true)) {
        std::fprintf(stderr,
            "tq_table_shards_dispatch enabled=1 shards=%zu lookups=%zu width=%lld CPU_ids=1\n",
            tables.size(), ids.size(), static_cast<long long>(columns));
    }
    const MfqCudaGuard guard(device);
    auto result = tb::empty(
        {static_cast<std::int64_t>(ids.size()), columns},
        tb::TensorOptions().device(device).dtype(tb::kFloat16));
    std::vector<std::vector<std::int64_t>> local(tables.size()), positions(tables.size());
    for (std::size_t index = 0; index < ids.size(); ++index) {
        if (ids[index] < 0 ||
            std::uint64_t(ids[index] / rows_per_shard) >= tables.size()) {
            throw std::out_of_range("TQ embedding shard ID is out of bounds");
        }
        const auto shard = ids[index] / rows_per_shard;
        local[shard].push_back(ids[index] % rows_per_shard);
        positions[shard].push_back(static_cast<std::int64_t>(index));
    }
    for (std::size_t shard = 0; shard < tables.size(); ++shard) {
        if (local[shard].empty()) continue;
        const auto values = lookup_tq_rows(
            *tables[shard], local[shard].data(), local[shard].size(), device);
        const auto index = tb::from_blob(
            positions[shard].data(),
            {static_cast<std::int64_t>(positions[shard].size())},
            tb::TensorOptions().dtype(tb::kInt64)).clone().to(device, false, false);
        result.index_copy_(0, index, values);
    }
    return result;
}

namespace {

class TqRowStage final : public RowStage {
    std::vector<std::shared_ptr<mfq::TqTable>> tables_;
    std::vector<std::int64_t> shape_;
    std::int64_t rows_per_shard_, count_ = 1, columns_ = 0;
    std::size_t image_stride_ = 0, images_offset_ = 0, bytes_ = 0;
    tb::Device device_;
    std::unique_ptr<mfq::cuda::HostBuffer> host_;
    tb::Tensor packet_, output_;
    std::uint8_t* packet_address_ = nullptr;
    bool mapped_ = false, charged_ = false;
    std::vector<MfqCudaStream> streams_;

    void retain_stream() {
        const auto stream = mfq_get_current_cuda_stream(device_.index);
        if (std::none_of(streams_.begin(), streams_.end(),
                         [&](const auto& candidate) {
                             return candidate.stream() == stream.stream();
                         })) {
            streams_.push_back(stream);
        }
    }
    TrellisProjectionView pack_row(
            const std::vector<std::uint8_t>& wire, std::size_t slot) {
        const mfq::TqLayout layout(wire.data(), wire.size());
        const auto image = tq_image(wire, layout);
        if (layout.rows != 1 || columns_ > layout.width || image.size() > image_stride_) {
            throw std::invalid_argument("TQ stage row exceeds its prepared layout");
        }
        const auto offset = images_offset_ + slot * image_stride_;
        std::memcpy(static_cast<std::uint8_t*>(host_->data()) + offset,
                    image.data(), image.size());
        auto* address = packet_address_ + offset;
        const auto words = (layout.packed + 3) & ~std::size_t(3);
        int gslog = 0;
        for (auto gs = layout.gs; gs > 1; gs >>= 1) ++gslog;
        return {reinterpret_cast<const std::uint32_t*>(address + words),
                reinterpret_cast<const std::uint64_t*>(address + 64),
                address + layout.scales, address + words + layout.packed_bytes,
                nullptr, 1, int(layout.width), int(columns_), gslog,
                int(layout.columns)};
    }

public:
    TqRowStage(std::vector<std::shared_ptr<mfq::TqTable>> tables,
               std::int64_t rows_per_shard, std::vector<std::int64_t> shape,
               const tb::Device& device)
        : tables_(std::move(tables)), shape_(std::move(shape)),
          rows_per_shard_(rows_per_shard), device_(device) {
        if (tables_.empty() || !tables_[0] || rows_per_shard_ < 1 || !device.is_cuda()) {
            throw std::invalid_argument("invalid TQ row stage geometry");
        }
        for (const auto size : shape_) {
            if (size < 1 || count_ > INT_MAX / size) {
                throw std::overflow_error("TQ row stage shape exceeds bounds");
            }
            count_ *= size;
        }
        columns_ = tables_[0]->width();
        for (const auto& table : tables_) {
            if (!table || table->width() != columns_) {
                throw std::invalid_argument("TQ row stage table widths differ");
            }
            image_stride_ = std::max(image_stride_, table->row_image_bytes_bound());
        }
        if (columns_ > INT_MAX ||
            std::uint64_t(count_) > (SIZE_MAX - 7) / sizeof(TrellisProjectionView)) {
            throw std::overflow_error("TQ row stage metadata exceeds bounds");
        }
        images_offset_ = (std::size_t(count_) * sizeof(TrellisProjectionView) + 7) &
                         ~std::size_t(7);
        if (std::uint64_t(count_) > (SIZE_MAX - images_offset_) / image_stride_) {
            throw std::overflow_error("TQ row stage packet exceeds bounds");
        }
        bytes_ = images_offset_ + std::size_t(count_) * image_stride_;
        if (bytes_ > INT64_MAX) {
            throw std::overflow_error("TQ row stage packet exceeds tensor limits");
        }
        const MfqCudaGuard guard(device);
        retain_stream();
        int can_map = 0;
        TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(
            &can_map, cudaDevAttrCanMapHostMemory, device.index));
        const auto* setting = mfq::cuda::env_lookup("TYLOINFER_PLE_MAPPED_ROWS");
        mapped_ = can_map && (setting ? setting[0] == '1' : bytes_ <= 16u * 1024);
        host_ = std::make_unique<mfq::cuda::HostBuffer>(bytes_, mapped_);
        if (mapped_) {
            void* address = nullptr;
            const auto error = cudaHostGetDevicePointer(&address, host_->data(), 0);
            if (error == cudaErrorNotSupported || error == cudaErrorInvalidValue) {
                (void)cudaGetLastError();
                mapped_ = false;
            } else {
                TYLOINFER_CUDA_CHECK(error);
                packet_address_ = static_cast<std::uint8_t*>(address);
            }
        }
        const auto options = tb::TensorOptions().device(device).dtype(tb::kFloat16);
        if (!mapped_) {
            packet_ = tb::empty(
                {static_cast<std::int64_t>(bytes_)}, options.dtype(tb::kUInt8));
            packet_address_ = packet_.data_ptr<std::uint8_t>();
        }
        auto output_shape = shape_;
        output_shape.push_back(columns_);
        output_ = tb::zeros(output_shape, options);
        std::memset(host_->data(), 0, bytes_);
        const auto zero_width = (std::uint64_t(columns_) + 255) & ~std::uint64_t(255);
        if (zero_width > INT_MAX) {
            throw std::overflow_error("TQ row stage padding exceeds CUDA limits");
        }
        std::vector<std::uint8_t> zero(72 + zero_width / 32 + zero_width / 8);
        std::memcpy(zero.data(), "RTRELL05", 8);
        mfq::TqLayout::integer(zero.data() + 8, 5, 4);
        mfq::TqLayout::integer(zero.data() + 12, 1, 4);
        mfq::TqLayout::integer(zero.data() + 16, zero_width, 4);
        mfq::TqLayout::integer(zero.data() + 24, 16, 4);
        mfq::TqLayout::integer(zero.data() + 28, zero_width / 8, 4);
        const auto view = pack_row(zero, 0);
        auto* views = static_cast<TrellisProjectionView*>(host_->data());
        std::fill_n(views, count_, view);
        if (!mapped_) {
            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(
                packet_address_, host_->data(), bytes_, cudaMemcpyHostToDevice,
                mfq_current_cuda_stream()));
        }
        TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
        mfq::cuda::charge_tensor_host_bytes(bytes_);
        charged_ = true;
    }

    ~TqRowStage() override {
        int previous = 0;
        const bool restore = cudaGetDevice(&previous) == cudaSuccess;
        (void)cudaSetDevice(device_.index);
        for (const auto& stream : streams_) {
            (void)cudaStreamSynchronize(stream.stream());
        }
        output_ = {};
        packet_ = {};
        if (charged_) mfq::cuda::tensor_host_bytes.fetch_sub(bytes_);
        if (restore) (void)cudaSetDevice(previous);
    }
    tb::Tensor output() const override { return output_; }
    bool mapped_rows() const override { return mapped_; }
    void upload(const std::vector<std::int64_t>& ids,
                const std::vector<std::int64_t>& shape) override {
        if (shape != shape_ || ids.size() != std::size_t(count_)) {
            throw std::invalid_argument("TQ stage upload shape changed");
        }
        const MfqCudaGuard guard(device_);
        retain_stream();
        std::unordered_map<std::int64_t, TrellisProjectionView> unique;
        std::vector<TrellisProjectionView> views;
        views.reserve(ids.size());
        for (const auto id : ids) {
            if (id < 0 || std::uint64_t(id / rows_per_shard_) >= tables_.size()) {
                throw std::out_of_range("TQ stage shard ID is out of bounds");
            }
            const auto [position, inserted] = unique.try_emplace(id);
            if (inserted) {
                const auto shard = id / rows_per_shard_, row = id % rows_per_shard_;
                position->second = pack_row(
                    tables_[shard]->slice_rows(row, row + 1), unique.size() - 1);
            }
            views.push_back(position->second);
        }
        std::memcpy(host_->data(), views.data(), views.size() * sizeof(views[0]));
        if (!mapped_) {
            mfq_cuda_record_stream(packet_, mfq_get_current_cuda_stream(device_.index));
            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(
                packet_address_, host_->data(), bytes_, cudaMemcpyHostToDevice,
                mfq_current_cuda_stream()));
        }
    }
    void decode() override {
        const MfqCudaGuard guard(device_);
        retain_stream();
        mfq_cuda_record_stream(output_, mfq_get_current_cuda_stream(device_.index));
        if (packet_.defined()) {
            mfq_cuda_record_stream(packet_, mfq_get_current_cuda_stream(device_.index));
        }
        status(mfq_tq_embedding_rows(
            reinterpret_cast<const TrellisProjectionView*>(packet_address_),
            int(count_), int(columns_), output_.data_ptr(), 0,
            mfq_current_cuda_stream()));
    }
};

} // namespace

std::shared_ptr<RowStage> make_tq_row_stage(
        const std::vector<std::shared_ptr<mfq::TqTable>>& tables,
        std::int64_t rows_per_shard, const std::vector<std::int64_t>& shape,
        const tb::Device& device) {
    return std::make_shared<TqRowStage>(tables, rows_per_shard, shape, device);
}
