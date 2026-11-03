#include "tyloinfer/row_residual.h"
#include "tyloinfer/model_source.h"
#include "tyloinfer/gguf_quant.h"
#include "tyloinfer/gguf_model_source.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <limits>
#include <stdexcept>
#include <cstring>

namespace mfq {
namespace {
std::uint64_t integer(const std::uint8_t* p, unsigned bytes) {
    std::uint64_t value=0;
    for(unsigned i=0;i<bytes;++i)value|=std::uint64_t(p[i])<<(8*i);
    return value;
}
}
RowResidualFile::RowResidualFile(const std::filesystem::path& path) {
    const auto size=std::filesystem::file_size(path);
    reader_=std::make_shared<FileRangeReader>(path,FileReadMode::Buffered);
    std::uint8_t h[48];reader_->read(0,reinterpret_cast<std::byte*>(h),sizeof(h));
    const auto count=integer(h+16,8), padded=integer(h+32,8);
    columns_=integer(h+24,8);payload_bytes_=integer(h+40,8);
    if(std::memcmp(h,"TQRES001",8) || integer(h+8,4)!=1 || integer(h+12,4) ||
        !count || !columns_ || columns_>std::uint64_t(INT64_MAX)-127 ||
        padded!=(columns_+127)/128*128 || size<48 || count>(size-48)/16)
        throw std::invalid_argument("invalid row residual header");
    payload_offset_=48+count*16;
    if(payload_bytes_!=size-payload_offset_ || count>std::uint64_t(SIZE_MAX)/sizeof(ResidualRow))
        throw std::invalid_argument("row residual extent differs");
    rows_.reserve(std::size_t(count));
    std::uint64_t cursor=0;
    for(std::uint64_t i=0;i<count;++i) {
        std::uint8_t d[16];reader_->read(48+i*16,reinterpret_cast<std::byte*>(d),sizeof(d));
        ResidualRow row{integer(d,8),std::uint32_t(integer(d+8,4)),std::uint32_t(integer(d+12,4))};
        const bool valid=row.type>=144 && row.type<=151;
        const auto expected=row.type?valid?gguf::row_bytes(ggml_type(row.type),std::int64_t(padded)):UINT64_MAX:0;
        if(row.offset!=cursor || row.bytes!=expected || row.bytes>payload_bytes_-cursor)
            throw std::invalid_argument("invalid row residual descriptor");
        cursor+=row.bytes;rows_.push_back(row);
    }
    if(cursor!=payload_bytes_)throw std::invalid_argument("unreferenced residual data");
}
void RowResidualFile::read_payload(std::uint64_t offset,void* output,std::size_t bytes) const {
    if(offset>payload_bytes_ || bytes>payload_bytes_-offset || (bytes && !output))
        throw std::out_of_range("residual payload read is out of bounds");
    reader_->read(payload_offset_+offset,reinterpret_cast<std::byte*>(output),bytes);
}
std::vector<std::uint8_t> RowResidualFile::read_row(std::uint64_t index) const {
    const auto& row=rows_.at(std::size_t(index));
    std::vector<std::uint8_t> data(row.bytes);read_payload(row.offset,data.data(),data.size());return data;
}

namespace {
class ResidualModelSource final:public ModelSource {
    std::shared_ptr<const ModelSource> base_;
    std::unordered_map<std::string,std::shared_ptr<const RowResidualFile>> residuals_;
    std::vector<std::filesystem::path> paths_;
public:
    explicit ResidualModelSource(const std::filesystem::path& path) {
        std::ifstream stream(path);
        const auto document=nlohmann::json::parse(stream);
        if(document.at("format")!="tyloinfer-row-residual-v1")
            throw std::invalid_argument("unsupported residual model manifest");
        auto base=path.parent_path()/std::filesystem::u8path(document.at("base").get<std::string>());
        if(base.extension()==".json")throw std::invalid_argument("residual manifests cannot reference another manifest");
        base_=open_model_source(base);paths_=base_->source_paths();paths_.push_back(path);
        const auto& tensors=document.at("tensors");
        if(!tensors.is_object())throw std::invalid_argument("residual tensors must be an object");
        for(auto it=tensors.begin();it!=tensors.end();++it) {
            auto name=it.key();
            if(!base_->find_tensor(name)) {
                if(auto gguf=std::dynamic_pointer_cast<const GgufModelSource>(base_))name=gguf->canonical_tensor_name(name);
            }
            const auto* tensor=base_->find_tensor(name);
            if(!tensor || !tensor->shape || tensor->shape->size()<2 || !gguf::is_quantized(tensor->dtype))
                throw std::invalid_argument("residual target must be an existing GGUF weight tensor: "+it.key());
            const auto file=path.parent_path()/std::filesystem::u8path(it.value().get<std::string>());
            auto residual=std::make_shared<RowResidualFile>(file);
            std::uint64_t rows=1;
            for(std::size_t i=0;i+1<tensor->shape->size();++i) {
                const auto n=tensor->shape->at(i);
                if(n<=0 || std::uint64_t(n)>UINT64_MAX/rows)throw std::invalid_argument("invalid residual target shape");
                rows*=std::uint64_t(n);
            }
            if(rows!=residual->row_count() || tensor->shape->back()!=std::int64_t(residual->columns()))
                throw std::invalid_argument("residual geometry differs from base tensor: "+it.key());
            if(!residuals_.emplace(name,std::move(residual)).second)
                throw std::invalid_argument("duplicate residual target after name normalization");
            paths_.push_back(file);
        }
    }
    const std::vector<std::filesystem::path>& source_paths() const noexcept override { return paths_; }
    std::string_view architecture() const noexcept override { return base_->architecture(); }
    const std::unordered_map<std::string,std::string>& metadata() const noexcept override { return base_->metadata(); }
    const HadamardTransforms* hadamard_transforms() const noexcept override { return base_->hadamard_transforms(); }
    const std::vector<TensorMetadata>& tensors() const noexcept override { return base_->tensors(); }
    const TensorMetadata* find_tensor(std::string_view name) const noexcept override { return base_->find_tensor(name); }
    const MfqLegacyTensorAliases& legacy_tensor_compatibility() const noexcept override { return base_->legacy_tensor_compatibility(); }
    void read_range_into(std::string_view n,std::uint64_t o,std::byte* p,std::size_t s) const override { base_->read_range_into(n,o,p,s); }
    void drop_file_cache() const noexcept override { base_->drop_file_cache(); }
    FileReadStats file_read_stats() const noexcept override { return base_->file_read_stats(); }
    bool supports_parallel_tensor_reads() const noexcept override { return base_->supports_parallel_tensor_reads(); }
    std::optional<TensorFileRange> tensor_file_range(std::string_view n) const override { return base_->tensor_file_range(n); }
    std::optional<TensorFileRange> tensor_file_range_at(std::string_view n,std::uint64_t o,std::size_t s) const override { return base_->tensor_file_range_at(n,o,s); }
    TensorReader tensor_reader(std::string_view n) const override { return base_->tensor_reader(n); }
    TensorBatchReader tensor_batch_reader(std::string_view n) const override { return base_->tensor_batch_reader(n); }
    TensorReader mapped_tensor_reader(std::string_view n) const override { return base_->mapped_tensor_reader(n); }
    TensorBatchReader mapped_tensor_batch_reader(std::string_view n) const override { return base_->mapped_tensor_batch_reader(n); }
    const std::vector<std::string>& assets() const noexcept override { return base_->assets(); }
    bool has_asset(std::string_view n) const noexcept override { return base_->has_asset(n); }
    std::vector<std::byte> read_asset(std::string_view n) const override { return base_->read_asset(n); }
    std::optional<ModelGraph> model_graph() const override { return base_->model_graph(); }
    std::shared_ptr<const RowResidualFile> row_residual(std::string_view n) const override {
        const auto found=residuals_.find(std::string(n));
        return found==residuals_.end()?nullptr:found->second;
    }
};
}
std::shared_ptr<const ModelSource> open_residual_model_source(const std::filesystem::path& path) {
    return std::make_shared<ResidualModelSource>(path);
}
}
