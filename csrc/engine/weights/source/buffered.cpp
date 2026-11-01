#include "engine/weights/source/buffered.h"

#include <cstdio>
#include <ios>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

#ifndef _WIN32
#include <sys/types.h>
#endif

namespace mfq {
namespace {

// MSVC's filebuf can split a large xsgetn into thousands of small reads.
// fread sends large contiguous records directly to the buffered file handle.
class BufferedRecordInput {
    std::FILE* file_=nullptr;
public:
    explicit BufferedRecordInput(const std::filesystem::path& path) {
#ifdef _WIN32
        file_=_wfopen(path.c_str(),L"rb");
#else
        file_=std::fopen(path.c_str(),"rb");
#endif
        if(!file_)throw std::runtime_error("cannot open MFQ row source: "+path.string());
    }
    ~BufferedRecordInput(){if(file_)std::fclose(file_);}
    BufferedRecordInput(const BufferedRecordInput&)=delete;
    BufferedRecordInput& operator=(const BufferedRecordInput&)=delete;
    void read(std::uint64_t offset,std::byte* destination,std::size_t size) {
        if(offset>std::uint64_t(std::numeric_limits<std::int64_t>::max()) ||
                size>std::uint64_t(std::numeric_limits<std::int64_t>::max()))
            throw std::overflow_error("model tensor row range overflow");
        if(!size)return;
        std::clearerr(file_);
#ifdef _WIN32
        const auto seek=_fseeki64(file_,static_cast<std::int64_t>(offset),SEEK_SET);
#else
        const auto seek=fseeko(file_,static_cast<off_t>(offset),SEEK_SET);
#endif
        if(seek || std::fread(destination,1,size,file_)!=size)
            throw std::runtime_error("MFQ row source was truncated");
    }
};

} // namespace

void read_buffered(
    const std::filesystem::path& path,
    std::uint64_t offset,
    std::byte* destination,
    std::size_t size) {
    if (size == 0) return;
    if (offset > static_cast<std::uint64_t>(
                     std::numeric_limits<std::streamoff>::max()) ||
        size > static_cast<std::size_t>(
                   std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("model source byte range is too large");
    }
    BufferedRecordInput(path).read(offset,destination,size);
}

std::function<void(std::uint64_t, std::byte*, std::size_t)> buffered_reader(
    const std::filesystem::path& path,
    std::uint64_t offset,
    std::uint64_t bytes) {
    struct Range {
        std::uint64_t offset;
        std::uint64_t bytes;
        BufferedRecordInput input;
        std::mutex mutex;
        Range(const std::filesystem::path& path, std::uint64_t offset, std::uint64_t bytes)
            : offset(offset), bytes(bytes), input(path) {}
    };
    auto range = std::make_shared<Range>(path, offset, bytes);
    return [range](std::uint64_t offset, std::byte* destination, std::size_t size) {
        if (offset > range->bytes || size > range->bytes - offset)
            throw std::out_of_range("model tensor row range is out of bounds");
        if (offset > std::numeric_limits<std::uint64_t>::max() - range->offset ||
            range->offset + offset > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()) ||
            size > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()))
            throw std::overflow_error("model tensor row range overflow");
        if (!size) return;
        std::lock_guard<std::mutex> lock(range->mutex);
        range->input.read(range->offset+offset,destination,size);
    };
}

} // namespace mfq
