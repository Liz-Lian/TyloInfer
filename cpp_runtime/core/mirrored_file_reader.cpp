#include "tyloinfer/mirrored_file_reader.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace mfq {
std::vector<std::filesystem::path> model_read_replica_paths(const std::filesystem::path& primary) {
    std::vector<std::filesystem::path> result{primary};
#if defined(_WIN32)
    const auto* value=_wgetenv(L"TYLOINFER_MODEL_MIRROR_DIRS");
    if(!value || !*value)return result;
    const std::wstring settings(value);
#else
    const auto* value=std::getenv("TYLOINFER_MODEL_MIRROR_DIRS");
    if(!value || !*value)return result;
    const std::string settings(value);
#endif
    for(std::size_t begin=0;begin<settings.size();) {
        const auto end=settings.find(';',begin);
        const auto segment=settings.substr(begin,end==settings.npos?end:end-begin);
#if defined(_WIN32)
        const auto directory=std::filesystem::path(segment);
#else
        const auto directory=std::filesystem::u8path(segment);
#endif
        if(directory.empty() || !std::filesystem::is_directory(directory))
            throw std::invalid_argument("TYLOINFER_MODEL_MIRROR_DIRS requires existing directories");
        const auto candidate=directory/primary.filename();
        if(std::filesystem::is_regular_file(candidate) && std::none_of(result.begin(),result.end(),[&](const auto& path){
            return std::filesystem::equivalent(path,candidate);}))result.push_back(candidate);
        if(end==std::string::npos)break;
        begin=end+1;
    }
    return result;
}
MirroredFileReader::MirroredFileReader(std::vector<std::filesystem::path> paths,FileReadMode mode,
    std::uint64_t header_bytes) {
    if(paths.empty())throw std::invalid_argument("model reader requires a primary file");
    size_=std::filesystem::file_size(paths.front());
    if(header_bytes>size_)throw std::invalid_argument("model header exceeds its file size");
    for(std::size_t i=0;i<paths.size();++i) {
        if(std::filesystem::file_size(paths[i])!=size_)
            throw std::invalid_argument("model mirror file size differs from primary: "+paths[i].filename().string());
        for(std::size_t j=0;j<i;++j)if(std::filesystem::equivalent(paths[i],paths[j]))
            throw std::invalid_argument("model mirror duplicates an existing file");
        readers_.push_back(std::make_unique<FileRangeReader>(paths[i],mode));
    }
    if(readers_.size()>1) {
        std::array<std::byte,65536> expected{},actual{};
        for(std::uint64_t offset=0;offset<header_bytes;) {
            const auto count=std::size_t(std::min<std::uint64_t>(expected.size(),header_bytes-offset));
            readers_[0]->read(offset,expected.data(),count);
            for(std::size_t i=1;i<readers_.size();++i) {
                readers_[i]->read(offset,actual.data(),count);
                if(std::memcmp(expected.data(),actual.data(),count))
                    throw std::invalid_argument("model mirror header differs from primary: "+paths[i].filename().string());
            }
            offset+=count;
        }
    }
}
void MirroredFileReader::validate(const ReadSpan& span) const {
    if(span.offset>size_ || span.size>size_-span.offset)
        throw std::out_of_range("model mirror read exceeds file size");
    if(span.size && !span.destination)throw std::invalid_argument("model mirror destination is null");
}
std::size_t MirroredFileReader::replica(std::uint64_t offset) const noexcept {
    // Stable byte-range routing, independent of quantization and tensor shape.
    // Colibri also assigns expert images to replicas by a stable hash. Mix all
    // offset bits here so aligned GGUF byte ranges distribute across replicas.
    offset^=offset>>33;offset*=0xff51afd7ed558ccdULL;
    offset^=offset>>33;offset*=0xc4ceb9fe1a85ec53ULL;offset^=offset>>33;
    return std::size_t(offset%readers_.size());
}
void MirroredFileReader::read(std::uint64_t offset,std::byte* destination,std::size_t size) const {
    validate({offset,destination,size});
    if(size)readers_[readers_.size()==1?0:replica(offset)]->read(offset,destination,size);
}
void MirroredFileReader::read_coalesced(const std::vector<ReadSpan>& spans) const {
    for(const auto& span:spans)validate(span); // Validate the complete request before any writes.
    if(readers_.size()==1){readers_[0]->read_coalesced(spans);return;}
    std::vector<std::vector<ReadSpan>> groups(readers_.size());
    for(const auto& span:spans)if(span.size)groups[replica(span.offset)].push_back(span);
    std::vector<std::size_t> active;
    for(std::size_t i=0;i<groups.size();++i)if(!groups[i].empty())active.push_back(i);
    if(active.empty())return;
    if(active.size()>std::size_t(std::numeric_limits<int>::max()))throw std::length_error("too many model mirrors");
    // Persistent I/O workers submit each SSD's batch concurrently. Each local
    // reader retains its Windows OVERLAPPED or Linux AIO queue and alignment.
    // A failed callback joins the other workers before propagating the error.
    pool_.run(0,std::int64_t(active.size()),1,int(active.size()),[&](auto begin,auto end) {
        for(auto at=begin;at<end;++at) {
            const auto index=active[std::size_t(at)];readers_[index]->read_coalesced(groups[index]);
        }
    });
}
std::vector<FileReadStats> MirroredFileReader::replica_stats() const {
    std::vector<FileReadStats> result;result.reserve(readers_.size());
    for(const auto& reader:readers_)result.push_back(reader->stats());
    return result;
}
}
