#include "gguf_weight.h"
#include "gguf_rotation.h"
#include "../../kernels/gguf.h"
#include "mfq_weight_memory.h"
#include <algorithm>
#include <climits>
#include <cstring>
#include <numeric>
#include <list>
#include <mutex>
#include <unordered_map>

namespace tb = mfq_tensor_backend;

struct GgufRowTable::RowCache {
    struct Entry {std::vector<std::byte> bytes;std::list<std::uint64_t>::iterator recent;};
    std::mutex mutex;
    std::unordered_map<std::uint64_t,Entry> rows;
    std::list<std::uint64_t> recent;
    std::size_t capacity=0,bytes=0;
    ~RowCache(){mfq::cuda::tensor_host_bytes.fetch_sub(bytes);}
};

GgufRowTable::GgufRowTable(const mfq::ModelSource& source, const std::string& name) {
    const auto* record = source.find_tensor(name);
    if (!record || !record->shape || record->shape->size() != 2)
        throw std::invalid_argument("GGUF row table requires a matrix: " + name);
    type_ = mfq::gguf::quant_type(record->dtype);
    rows_ = record->shape->at(0); columns_ = record->shape->at(1);
    if(load_gguf_input_rotation(source,name,columns_,0,columns_,0,false))
        throw std::invalid_argument("GGUF streaming row tables do not yet support an inverse Hadamard transform");
    row_bytes_ = mfq::gguf::row_bytes(type_, columns_);
    if (rows_ <= 0 || std::uint64_t(rows_) > UINT64_MAX / row_bytes_ ||
            record->nbytes != std::uint64_t(rows_) * row_bytes_)
        throw std::invalid_argument("invalid GGUF row table payload: " + name);
    read_ = std::make_shared<mfq::ModelSource::TensorReader>(source.tensor_reader(name));
    batch_ = source.tensor_batch_reader(name);
    cache_ = std::make_unique<RowCache>();
    // Include per-entry metadata in this small optional cache's bound.
    cache_->capacity=row_bytes_>SIZE_MAX-128?0:(8ULL<<20)/(row_bytes_+128);
    mfq::cuda::observe_streamed_ple(read_, record->nbytes);
}
GgufRowTable::~GgufRowTable()=default;

void GgufRowTable::read_row(std::int64_t id, void* destination) const {
    if (id < 0 || id >= rows_) throw std::out_of_range("GGUF table row id");
    read_rows({{std::uint64_t(id)*row_bytes_,static_cast<std::byte*>(destination),std::size_t(row_bytes_)}});
}
void GgufRowTable::read_rows(const std::vector<mfq::ReadSpan>& requests) const {
    std::lock_guard lock(cache_->mutex);
    std::vector<mfq::ReadSpan> missing;
    std::unordered_map<std::uint64_t,std::byte*> unique;
    std::vector<std::pair<std::byte*,const std::byte*>> repeated;
    for(const auto& row:requests) {
        if(row.size!=row_bytes_ || row.offset%row_bytes_ || row.offset/row_bytes_>=std::uint64_t(rows_) || !row.destination)
            throw std::out_of_range("invalid GGUF table row range");
        if(auto found=cache_->rows.find(row.offset);found!=cache_->rows.end()) {
            std::memcpy(row.destination,found->second.bytes.data(),row_bytes_);
            cache_->recent.splice(cache_->recent.begin(),cache_->recent,found->second.recent);
        }else if(auto [first,inserted]=unique.emplace(row.offset,row.destination);inserted) missing.push_back(row);
        else repeated.emplace_back(row.destination,first->second);
    }
    std::sort(missing.begin(),missing.end(),[](const auto& a,const auto& b){return a.offset<b.offset;});
    const auto count=std::max<std::size_t>(1,std::min<std::size_t>(256,(8ULL<<20)/std::max<std::uint64_t>(4096,row_bytes_)));
    for(std::size_t begin=0;begin<missing.size();begin+=count) {
        const auto end=std::min(missing.size(),begin+count);
        if(batch_)batch_({missing.begin()+begin,missing.begin()+end});
        else for(auto i=begin;i<end;++i)(*read_)(missing[i].offset,missing[i].destination,missing[i].size);
    }
    for(const auto& [destination,source]:repeated)std::memcpy(destination,source,row_bytes_);
    if(!cache_->capacity)return;
    // Physical reads are sorted, but retention follows the caller's token
    // order so a long prompt leaves its most recent rows hot for decode.
    for(const auto& row:requests) {
        if(auto found=cache_->rows.find(row.offset);found!=cache_->rows.end()) {
            cache_->recent.splice(cache_->recent.begin(),cache_->recent,found->second.recent);continue;
        }
        if(cache_->rows.size()>=cache_->capacity) {
            const auto old=cache_->recent.back();cache_->recent.pop_back();
            cache_->rows.erase(old);cache_->bytes-=row_bytes_;mfq::cuda::tensor_host_bytes.fetch_sub(row_bytes_);
        }
        std::vector<std::byte> value(row.destination,row.destination+row_bytes_);
        cache_->recent.push_front(row.offset);
        try {cache_->rows.emplace(row.offset,RowCache::Entry{std::move(value),cache_->recent.begin()});}
        catch(...) {cache_->recent.pop_front();throw;}
        cache_->bytes+=row_bytes_;mfq::cuda::charge_tensor_host_bytes(row_bytes_);
    }
}

namespace {
class GgufRowStage final : public RowStage {
    std::vector<std::shared_ptr<GgufRowTable>> tables_;
    std::vector<std::int64_t> shape_;
    std::int64_t rows_per_shard_, count_ = 1, width_;
    std::size_t stride_ = 0, offset_, bytes_;
    tb::Device device_;
    std::unique_ptr<mfq::cuda::HostBuffer> host_;
    tb::Tensor packet_, output_;
    void* packet_address_ = nullptr;
    bool mapped_ = false;
    std::vector<MfqCudaStream> streams_;
    std::unique_ptr<mfq::cuda::Event> done_;
    cudaStream_t last_stream_ = nullptr;
    bool pending_ = false;
    void retain_stream() {
        const auto stream = mfq_get_current_cuda_stream(device_.index);
        if (std::none_of(streams_.begin(), streams_.end(), [&](const auto& prior) { return prior.stream() == stream.stream(); }))
            streams_.push_back(stream);
    }
public:
    GgufRowStage(std::vector<std::shared_ptr<GgufRowTable>> tables, std::int64_t rows,
        std::vector<std::int64_t> shape, tb::Device device)
        : tables_(std::move(tables)), shape_(std::move(shape)), rows_per_shard_(rows), device_(device) {
        if (tables_.empty() || !tables_[0] || rows <= 0 || !device.is_cuda())
            throw std::invalid_argument("invalid GGUF row stage");
        width_ = tables_[0]->width();
        for (const auto size : shape_) {
            if (size <= 0 || count_ > INT64_MAX / size) throw std::overflow_error("GGUF row stage shape");
            count_ *= size;
        }
        for (const auto& table : tables_) {
            if (!table || table->width() != width_) throw std::invalid_argument("GGUF table widths differ");
            stride_ = std::max(stride_, static_cast<std::size_t>(table->row_bytes()));
        }
        stride_ = (stride_ + 7) & ~std::size_t(7);
        if (std::uint64_t(count_) > std::uint64_t(INT64_MAX) /
                (sizeof(mfq::cuda::gguf::StagedRow) + stride_))
            throw std::overflow_error("GGUF row stage packet size");
        offset_ = std::size_t(count_) * sizeof(mfq::cuda::gguf::StagedRow);
        bytes_ = offset_ + std::size_t(count_) * stride_;
        MfqCudaGuard guard(device_);
        retain_stream();
        done_ = std::make_unique<mfq::cuda::Event>();
        int can_map = 0;
        TYLOINFER_NATIVE_CUDA_CHECK(cudaDeviceGetAttribute(&can_map, cudaDevAttrCanMapHostMemory, device_.index));
        // As in the TQ row stage, a small packet is cheaper to read directly
        // than to submit a separate host-to-device transfer for every layer.
        mapped_ = can_map && bytes_ <= 16 * 1024;
        host_ = std::make_unique<mfq::cuda::HostBuffer>(bytes_, mapped_);
        if (mapped_) TYLOINFER_NATIVE_CUDA_CHECK(cudaHostGetDevicePointer(&packet_address_, host_->data(), 0));
        else {
            packet_ = tb::empty({static_cast<std::int64_t>(bytes_)},
                tb::TensorOptions().device(device_).dtype(tb::kUInt8));
            packet_address_ = packet_.data_ptr();
        }
        auto output_shape = shape_; output_shape.push_back(width_);
        output_ = tb::zeros(output_shape, tb::TensorOptions().device(device_).dtype(tb::kFloat16));
        std::memset(host_->data(), 0, bytes_);
        auto* descriptors = static_cast<mfq::cuda::gguf::StagedRow*>(host_->data());
        for (std::int64_t i = 0; i < count_; ++i)
            descriptors[i] = {offset_ + i * stride_, std::int32_t(tables_[0]->type()), 0};
        if (!mapped_) TYLOINFER_NATIVE_CUDA_CHECK(cudaMemcpyAsync(packet_address_, host_->data(), bytes_,
            cudaMemcpyHostToDevice, mfq_current_cuda_stream()));
        TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
        mfq::cuda::charge_tensor_host_bytes(bytes_);
    }
    ~GgufRowStage() override {
        try {
            MfqCudaGuard guard(device_);
            for (const auto& stream : streams_) (void)cudaStreamSynchronize(stream.stream());
        } catch (...) {}
        mfq::cuda::tensor_host_bytes.fetch_sub(bytes_);
    }
    tb::Tensor output() const override { return output_; }
    bool mapped_rows() const override { return mapped_; }
    void upload(const std::vector<std::int64_t>& ids, const std::vector<std::int64_t>& shape) override {
        if (shape != shape_ || ids.size() != static_cast<std::size_t>(count_))
            throw std::invalid_argument("GGUF row stage upload shape differs");
        MfqCudaGuard guard(device_);
        retain_stream();
        if (pending_) done_->synchronize();
        auto* descriptors = static_cast<mfq::cuda::gguf::StagedRow*>(host_->data());
        auto* bytes = static_cast<std::uint8_t*>(host_->data());
        std::vector<std::vector<mfq::ReadSpan>> requests(tables_.size());
        for (std::int64_t i = 0; i < count_; ++i) {
            const auto id = ids[i];
            if (id < 0 || std::uint64_t(id / rows_per_shard_) >= tables_.size())
                throw std::out_of_range("GGUF row stage shard id");
            const auto& table = tables_[id / rows_per_shard_];
            descriptors[i].type = table->type();
            requests[id/rows_per_shard_].push_back({std::uint64_t(id%rows_per_shard_)*table->row_bytes(),
                reinterpret_cast<std::byte*>(bytes+descriptors[i].offset),std::size_t(table->row_bytes())});
        }
        for(std::size_t shard=0;shard<tables_.size();++shard)tables_[shard]->read_rows(requests[shard]);
        if (!mapped_) TYLOINFER_NATIVE_CUDA_CHECK(cudaMemcpyAsync(packet_address_, host_->data(), bytes_,
            cudaMemcpyHostToDevice, mfq_current_cuda_stream()));
        TYLOINFER_NATIVE_CUDA_CHECK(cudaEventRecord(done_->get(), mfq_current_cuda_stream()));
        last_stream_ = mfq_current_cuda_stream(); pending_ = true;
        if (packet_.defined()) packet_.record_stream(reinterpret_cast<std::uintptr_t>(last_stream_));
    }
    void decode() override {
        MfqCudaGuard guard(device_);
        retain_stream();
        const auto stream = mfq_current_cuda_stream();
        cudaStreamCaptureStatus capture;
        TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
        if (!mapped_ && pending_ && stream != last_stream_)
            TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamWaitEvent(stream, done_->get(),
                capture == cudaStreamCaptureStatusNone ? 0 : cudaEventWaitExternal));
        if (packet_.defined()) packet_.record_stream(reinterpret_cast<std::uintptr_t>(stream));
        output_.record_stream(reinterpret_cast<std::uintptr_t>(stream));
        mfq::cuda::gguf::staged_rows(packet_address_, count_, width_, output_.data_ptr(), true,
            mfq_current_cuda_stream());
        // A captured internal event cannot be synchronized by the next host
        // upload. Graph owners fence replay; keep this event for eager/DMA use.
        if (capture == cudaStreamCaptureStatusNone) {
            TYLOINFER_NATIVE_CUDA_CHECK(cudaEventRecord(done_->get(), stream));
            last_stream_ = stream; pending_ = true;
        }
    }
};
}

std::shared_ptr<RowStage> make_gguf_row_stage(const std::vector<std::shared_ptr<GgufRowTable>>& tables,
    std::int64_t rows, const std::vector<std::int64_t>& shape, const tb::Device& device) {
    return std::make_shared<GgufRowStage>(tables, rows, shape, device);
}

tb::Tensor lookup_gguf_shards(const std::vector<std::shared_ptr<GgufRowTable>>& tables,
    std::int64_t rows, const std::vector<std::int64_t>& ids, const tb::Device& device) {
    if (tables.empty() || rows <= 0) throw std::invalid_argument("invalid GGUF table shards");
    if (ids.empty()) return tb::empty({0, tables[0]->width()},
        tb::TensorOptions().device(device).dtype(device.is_cuda() ? tb::kFloat16 : tb::kFloat32));
    if (!device.is_cuda()) {
        if (tables.empty()) throw std::invalid_argument("empty GGUF table shards");
        const auto width = tables[0]->width();
        auto result = tb::empty({static_cast<std::int64_t>(ids.size()), width},
            tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
        std::vector<std::uint8_t> row;
        for (std::size_t i = 0; i < ids.size(); ++i) {
            if (ids[i] < 0 || std::uint64_t(ids[i] / rows) >= tables.size()) throw std::out_of_range("GGUF table id");
            const auto& table = tables[ids[i] / rows];
            row.resize(table->row_bytes()); table->read_row(ids[i] % rows, row.data());
            decode_gguf_cpu(table->type(), row.data(), result.data_ptr<float>() + i * width, width);
        }
        return result;
    }
    auto stage = make_gguf_row_stage(tables, rows, {static_cast<std::int64_t>(ids.size())}, device);
    stage->upload(ids, {static_cast<std::int64_t>(ids.size())}); stage->decode();
    return stage->output();
}

tb::Tensor GgufRowTable::lookup(tb::Tensor ids) const {
    const auto device = ids.device();
    auto shape = ids.sizes().vec(); shape.push_back(columns_);
    auto host = ids.to(tb::kInt64).cpu().contiguous().reshape({-1});
    auto table = std::const_pointer_cast<GgufRowTable>(shared_from_this());
    std::vector<std::int64_t> values(host.data_ptr<std::int64_t>(), host.data_ptr<std::int64_t>() + host.numel());
    return lookup_gguf_shards({table}, rows_, values, device).reshape(shape);
}
