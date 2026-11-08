#include "engram.h"
#include "engine/backends/cuda/native/memory.h"
#include "engine/backends/cuda/native/stream.h"
#include "engine/backends/cpu/native/host_parallel.h"
#include "engine/weights/source/mapped.h"
#include "engine/weights/encoding/quant/nint_rows.h"
#include "engine/weights/nint_row_cache.h"
#include "engine/weights/nint_row_pipeline.h"
#include "engine/weights/tq/source.h"
#include "engine/weights/tq/rows.h"

#include <condition_variable>
#include <future>
#include <map>
#include <thread>

namespace tyloi::cuda::deepseek_v41_runtime {
namespace {
struct RowPacket {
    std::vector<std::uint8_t> packed;
    std::vector<std::int64_t> inverse;
    std::size_t unique = 0;
};

// The pinned packet and completion event follow the PLE row stage lifetime:
// CPU publication must wait for the preceding DMA to finish reading it.
struct RowStage {
    tyloi::cuda::HostBuffer host;
    ::tyloi::DenseTensor packet;
    tyloi::cuda::Event uploaded;
    bool pending = false;
    RowStage(std::size_t bytes, const ::tyloi::Device& device)
        : host(bytes), packet(::tyloi::cuda::empty({static_cast<int64_t>(bytes)},
              ::tyloi::TensorOptions().device(device).dtype(::tyloi::kUInt8))) {
        tyloi::cuda::charge_tensor_host_bytes(bytes);
    }
    ~RowStage() {
        if (pending) {
            try { uploaded.synchronize(); } catch (...) {}
        }
        tyloi::cuda::tensor_host_bytes.fetch_sub(host.size());
    }
};
}

struct EngramTable::Shared {
    const tyloi::source::Source& source;
    std::string name;
    std::uint64_t rows = 0;
    std::uint64_t value_bytes = 0;
    int width = 0;
    std::size_t capacity = 0;
    std::size_t row_bytes = 0;
    std::shared_ptr<const tyloi::TqTable> tq_table;
    std::shared_ptr<const NintRowSource> nint;

    Shared(const tyloi::source::Source& data, std::string tensor, std::uint64_t count,
           int columns, std::uint64_t values, std::size_t cache_rows)
        : source(data), name(std::move(tensor)), rows(count), value_bytes(values), width(columns),
          capacity(cache_rows), row_bytes(columns + columns / 32) {}
};

struct EngramTable::Impl {
    std::shared_ptr<const Shared> shared;
    tyloi::source::Source::TensorBatchReader read_batch;
    struct CacheEntry {
        std::vector<std::uint8_t> packed;
        std::list<std::int64_t>::iterator recency;
    };
    std::list<std::int64_t> recency;
    std::unordered_map<std::int64_t, CacheEntry> cache;
    tyloi::HostParallelPool read_pool{{}, false};
    std::vector<std::int64_t> current_ids;
    std::future<RowPacket> pending;
    std::optional<RowPacket> ready;
    std::mutex mutex;
    std::condition_variable available;
    std::packaged_task<RowPacket()> work;
    std::thread worker;
    bool stopping = false;
    std::unique_ptr<RowStage> stage;
    std::shared_ptr<NintRowPipeline> nint_pipeline;
    std::shared_ptr<NintRowStage> nint_stage;
    std::unique_ptr<tyloi::cuda::Event> nint_done;
    bool nint_pending=false;
    std::size_t nint_stage_rows=0;
    explicit Impl(std::shared_ptr<const Shared> resources) : shared(std::move(resources)) {
        if (shared->tq_table) return;
#if TYLOINFER_ENABLE_MFQ_COMPAT
        if(shared->nint) {
            nint_pipeline=std::make_shared<NintRowPipeline>(shared->nint,
                std::make_shared<tyloi::NintRowCache>(shared->capacity));
            return;
        }
#endif

        read_batch = shared->source.mapped_tensor_batch_reader(shared->name);
        if (!read_batch) read_batch = shared->source.tensor_batch_reader(shared->name);
    }
    ~Impl() {
        if(nint_pending)try {nint_done->synchronize();}catch(...) {}
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        available.notify_one();
        if (worker.joinable()) worker.join();
    }

    RowPacket gather_rows(const std::vector<std::int64_t>& ids) {
        RowPacket out;
        std::map<std::int64_t, std::size_t> locations;
        for (auto id : ids) locations.emplace(id, 0);
        out.unique = locations.size();
        if (out.unique > std::numeric_limits<std::size_t>::max() / shared->row_bytes)
            throw std::overflow_error("DeepSeek-V4.1 Engram packet is too large");
        out.packed.resize(out.unique * shared->row_bytes);
        std::vector<std::pair<std::int64_t, std::size_t>> misses;
        std::size_t index = 0;
        for (auto& [id, location] : locations) {
            location = index++;
            const auto found = cache.find(id);
            if (found == cache.end()) misses.emplace_back(id, location);
            else {
                std::memcpy(out.packed.data() + location * shared->row_bytes,
                            found->second.packed.data(), shared->row_bytes);
                recency.splice(recency.begin(), recency, found->second.recency);
            }
        }
        // Values and scales occupy separate table streams. Submit all selected
        // pages together, exactly as the PLE batch reader does for NINT streams.
        std::vector<tyloi::ReadSpan> spans;
        spans.reserve(misses.size() * 2);
        for (int scale = 0; scale < 2; ++scale) for (auto [id, location] : misses) {
            const auto count = scale ? shared->width / 32 : shared->width;
            const auto offset = engram_detail::kMxHeaderBytes +
                (scale ? shared->value_bytes : 0) + static_cast<std::uint64_t>(id) * count;
            auto* destination = out.packed.data() + location * shared->row_bytes +
                                (scale ? shared->width : 0);
            spans.push_back({offset, reinterpret_cast<std::byte*>(destination), std::size_t(count)});
        }
        if (!spans.empty()) {
            if (read_batch) {
                tyloi::MappedPrefetchScope prefetch(misses.size() > 128);
                read_batch(spans);
            }
            else read_pool.run(0, spans.size(), 32,
                shared->source.supports_parallel_tensor_reads() ? 8 : 1,
                [&](int64_t first, int64_t last) {
                    for (auto i = first; i < last; ++i) {
                        const auto& span = spans[i];
                        shared->source.read_range_into(shared->name, span.offset, span.destination,
                                                       span.size);
                    }
                });
        }
        // Validate once on insertion, retaining the original malformed-row
        // errors without inspecting or expanding cached values on every token.
        for (auto [id, location] : misses) {
            const auto* packed = out.packed.data() + location * shared->row_bytes;
            for (int col = 0; col < shared->width; ++col) if ((packed[col] & 127) == 127)
                throw std::runtime_error("DeepSeek-V4.1 Engram row contains an E4M3 NaN");
            for (int col = 0; col < shared->width / 32; ++col)
                if (packed[shared->width + col] == 255)
                throw std::runtime_error("DeepSeek-V4.1 Engram row contains an invalid E8M0 scale");
            if (!shared->capacity) continue;
            if (cache.size() >= shared->capacity) {
                cache.erase(recency.back());
                recency.pop_back();
            }
            recency.push_front(id);
            cache.emplace(id, CacheEntry{std::vector<std::uint8_t>(
                                         packed, packed + shared->row_bytes), recency.begin()});
        }
        out.inverse.reserve(ids.size());
        for (auto id : ids) out.inverse.push_back(locations.at(id));
        return out;
    }

    void issue(const std::int64_t* ids, std::size_t count) {
        if (!ids || !count) throw std::invalid_argument("DeepSeek-V4.1 Engram lookup is empty");
        for (std::size_t i = 0; i < count; ++i)
            if (ids[i] < 0 || static_cast<std::uint64_t>(ids[i]) >= shared->rows)
                throw std::out_of_range("DeepSeek-V4.1 Engram lookup row is outside the table");
        if (current_ids.size() == count && std::equal(current_ids.begin(), current_ids.end(), ids) &&
            (pending.valid() || ready)) return;
        if (pending.valid()) pending.get();
        ready.reset();
        current_ids.assign(ids, ids + count);
        std::packaged_task<RowPacket()> next([this, selected = current_ids] { return gather_rows(selected); });
        pending = next.get_future();
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (work.valid()) throw std::logic_error("Engram row worker has an unconsumed request");
            if (!worker.joinable()) worker = std::thread([this] {
                for (;;) {
                    std::packaged_task<RowPacket()> next;
                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        available.wait(lock, [this] { return stopping || work.valid(); });
                        if (!work.valid()) return;
                        next = std::move(work);
                    }
                    next();
                }
            });
            work = std::move(next);
        }
        available.notify_one();
    }
};

EngramTable::State::State(std::unique_ptr<EngramTable::Impl> impl) : impl_(std::move(impl)) {}
EngramTable::State::~State() = default;
EngramTable::State::State(EngramTable::State&&) noexcept = default;
EngramTable::State& EngramTable::State::operator=(EngramTable::State&&) noexcept = default;

std::vector<::tyloi::Tensor> EngramTable::State::storage() const {
    std::vector<::tyloi::Tensor> result;
    if (!impl_)
        return result;
    if (impl_->stage && impl_->stage->packet.defined())
        result.emplace_back(impl_->stage->packet);
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (impl_->nint_stage && impl_->nint_stage->output().defined())
        result.emplace_back(impl_->nint_stage->output());
#endif
    return result;
}

EngramTable::EngramTable(std::shared_ptr<const Shared> shared) : shared_(std::move(shared)) {}
EngramTable::~EngramTable() = default;

std::unique_ptr<EngramTable::State> EngramTable::make_state() const {
    return std::unique_ptr<State>(new State(std::make_unique<Impl>(shared_)));
}

void EngramTable::prefetch(State& state, const std::int64_t* rows, std::size_t count) const {
    if (!state.impl_ || state.impl_->shared.get() != shared_.get())
        throw std::invalid_argument("DeepSeek-V4.1 Engram state belongs to another table");
    auto& s = *state.impl_;
    if(s.shared->tq_table)return;
#if TYLOINFER_ENABLE_MFQ_COMPAT
        if(s.nint_pipeline) {
        if(!rows || !count)throw std::invalid_argument("DeepSeek-V4.1 Engram lookup is empty");
        for(std::size_t i=0;i<count;++i)if(rows[i]<0 || static_cast<std::uint64_t>(rows[i])>=s.shared->rows)
            throw std::out_of_range("DeepSeek-V4.1 Engram lookup row is outside the table");
        s.nint_pipeline->issue(std::vector<int64_t>(rows,rows+count),{static_cast<int64_t>(count)});
        return;
    }
#endif

    s.issue(rows, count);
}

EngramTensor EngramTable::gather(State& state, const std::int64_t* rows, int batch, int tokens,
                                int columns,
                                const ::tyloi::Device& device) const {
    if (batch <= 0 || tokens <= 0 || columns <= 0 || !device.is_cuda())
        throw std::invalid_argument("DeepSeek-V4.1 Engram lookup geometry disagrees");
    if (!state.impl_ || state.impl_->shared.get() != shared_.get())
        throw std::invalid_argument("DeepSeek-V4.1 Engram state belongs to another table");
    auto& s = *state.impl_;
    const auto count = std::size_t(batch) * tokens * columns;
    if(s.shared->tq_table) {
        auto result=lookup_tq_table(*s.shared->tq_table,rows,count,device);
        return result.dense()
            .reshape({batch,tokens,static_cast<int64_t>(columns)*s.shared->width});
    }
#if TYLOINFER_ENABLE_MFQ_COMPAT
        if(s.nint_pipeline) {
        if(!rows)throw std::invalid_argument("DeepSeek-V4.1 Engram row IDs are missing");
        const MfqCudaGuard guard(device);
        if(s.nint_pending) {s.nint_done->synchronize();s.nint_pending=false;}
        const std::vector<int64_t> shape{static_cast<int64_t>(count)};
        if(!s.nint_stage || s.nint_stage_rows!=count || s.nint_stage->output().device()!=device) {
            s.nint_stage=s.nint_pipeline->make_stage(shape,device);s.nint_stage_rows=count;
        }
        if(!s.nint_done)s.nint_done=std::make_unique<tyloi::cuda::Event>(cudaEventDisableTiming);
        s.nint_pipeline->upload(*s.nint_stage,std::vector<int64_t>(rows,rows+count),shape);
        s.nint_stage->decode();
        s.nint_done->record(mfq_current_cuda_stream());s.nint_pending=true;
        return s.nint_stage->output().dense()
            .reshape({batch,tokens,static_cast<int64_t>(columns)*s.shared->width});
    }
#endif

    s.issue(rows, count);
    if (s.pending.valid()) s.ready.emplace(s.pending.get());
    const auto& selected = *s.ready;
    const auto inverse_offset = (selected.packed.size() + 7) & ~std::size_t(7);
    if (count > (std::numeric_limits<std::size_t>::max() - inverse_offset) / sizeof(int64_t))
        throw std::overflow_error("DeepSeek-V4.1 Engram inverse packet is too large");
    const auto bytes = inverse_offset + count * sizeof(int64_t);
    const MfqCudaGuard guard(device);
    if (s.stage && s.stage->pending) { s.stage->uploaded.synchronize(); s.stage->pending = false; }
    if (!s.stage || s.stage->host.size() < bytes || s.stage->packet.device() != device)
        s.stage = std::make_unique<RowStage>(bytes, device);
    auto& stage = *s.stage;
    auto* host = static_cast<uint8_t*>(stage.host.data());
    std::memcpy(host, selected.packed.data(), selected.packed.size());
    std::memcpy(host + inverse_offset, selected.inverse.data(), count * sizeof(int64_t));
    TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(stage.packet.data_ptr(), host, bytes, cudaMemcpyHostToDevice,
                                   mfq_current_cuda_stream()));
    auto options = stage.packet.options();
    auto* data = stage.packet.data_ptr<std::uint8_t>();
    auto packed = ::tyloi::cuda::from_blob(data, {static_cast<int64_t>(selected.unique),
                                      static_cast<int64_t>(s.shared->row_bytes)}, options);
    auto inverse = ::tyloi::cuda::from_blob(data + inverse_offset, {static_cast<int64_t>(count)},
                                 options.dtype(::tyloi::kInt64));
    auto output = deepseek_v41_engram_rows_cuda(packed, inverse, s.shared->width)
        .reshape({batch, tokens, static_cast<int64_t>(columns) * s.shared->width});
    stage.uploaded.record(mfq_current_cuda_stream());
    stage.pending = true;
    return output;
}

std::shared_ptr<EngramTable> EngramTable::load(const tyloi::source::Source &model, const std::string &name,
                                             std::int64_t expected_rows, int expected_width,
                                             std::size_t cache_capacity_rows) {
        if(tyloi::is_tq_table(model,name)) {
            auto shared = std::make_shared<Shared>(model, name, expected_rows, expected_width, 0,
                                                   cache_capacity_rows);
            shared->tq_table = tyloi::tq_table(model, name);
            if (shared->tq_table->rows() != shared->rows || shared->tq_table->width() != shared->width)
                throw std::invalid_argument("TQ Engram table shape differs");
            return std::shared_ptr<EngramTable>(new EngramTable(std::move(shared)));
        }
        const auto record = require_tensor(model, name);
#if TYLOINFER_ENABLE_MFQ_COMPAT
        if(record.dtype=="NINT") {
            auto table=load_nint_row_table(model,name,true);
            if(table->rows()!=expected_rows || table->width()!=expected_width)
                throw std::runtime_error("DeepSeek-V4.1 Engram NINT table shape disagrees: "+name);
            std::fprintf(stderr,"deepseek_v41_engram_dispatch format=NINT row_pipeline=ple rows=%lld width=%d\n",
                static_cast<long long>(expected_rows),expected_width);
            auto shared = std::make_shared<Shared>(model, name, expected_rows, expected_width, 0,
                                                   cache_capacity_rows);
            shared->nint = std::make_shared<const NintRowSource>(
                std::vector<std::shared_ptr<tyloi::NintRows>>{std::move(table)});
            return std::shared_ptr<EngramTable>(new EngramTable(std::move(shared)));
        }
#endif

        if (record.dtype != "MXFP8") {
            throw std::runtime_error("DeepSeek-V4.1 Engram table must use native MXFP8: " + name);
        }
        auto raw_header = model.read(name, 0, engram_detail::kMxHeaderBytes);
        std::vector<std::uint8_t> header(raw_header.size());
        std::memcpy(header.data(), raw_header.data(), header.size());
        if (header.size() < engram_detail::kMxHeaderBytes ||
            std::memcmp(header.data(), "MXT1", 4) != 0 || header[4] != 1 || header[5] != 8 ||
            header[6] != 0 || header[7] != 0) {
            throw std::runtime_error("invalid DeepSeek-V4.1 Engram MXFP8 header: " + name);
        }
        const auto rows = engram_detail::read_u64_at(header, 8);
        const auto columns = engram_detail::read_u64_at(header, 16);
        const auto storage_rows = engram_detail::read_u64_at(header, 24);
        const auto storage_columns = engram_detail::read_u64_at(header, 32);
        const auto scale_rows = engram_detail::read_u64_at(header, 40);
        const auto scale_columns = engram_detail::read_u64_at(header, 48);
        if (rows != static_cast<std::uint64_t>(expected_rows) ||
            columns != static_cast<std::uint64_t>(expected_width) || storage_rows != rows ||
            storage_columns != columns || scale_rows != rows || scale_columns != columns / 32 ||
            columns == 0 || columns % 32) {
            throw std::runtime_error("DeepSeek-V4.1 Engram table is not row-wise MXFP8: " + name);
        }
        const auto value_bytes = engram_detail::checked_product(rows, columns, "value");
        const auto scale_bytes = engram_detail::checked_product(scale_rows, scale_columns, "scale");
        if (value_bytes >
                std::numeric_limits<std::uint64_t>::max() - engram_detail::kMxHeaderBytes ||
            scale_bytes > std::numeric_limits<std::uint64_t>::max() -
                              engram_detail::kMxHeaderBytes - value_bytes ||
            engram_detail::kMxHeaderBytes + value_bytes + scale_bytes != record.nbytes) {
            throw std::runtime_error("DeepSeek-V4.1 Engram MXFP8 payload size disagrees: " + name);
        }
        auto shared = std::make_shared<Shared>(model, name, static_cast<std::int64_t>(rows),
                                               static_cast<int>(columns), value_bytes,
                                               cache_capacity_rows);
        return std::shared_ptr<EngramTable>(new EngramTable(std::move(shared)));
    }

} // namespace tyloi::cuda::deepseek_v41_runtime
