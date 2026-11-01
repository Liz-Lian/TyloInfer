#include "engine/weights/source/mapped.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <utility>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cerrno>
#endif

namespace mfq {
namespace {

std::array<std::atomic<std::uint64_t>,5> mapped_read_totals{};
using MappedClock=std::chrono::steady_clock;
thread_local bool mapped_prefetch_context=false;
thread_local bool mapped_release_context=false;

class MappedTensorRange {
    void* view_=nullptr;
    std::size_t mapped_bytes_=0,prefix_=0,page_bytes_=0;
    std::uint64_t bytes_=0;
    bool profile_=false;
    int prefetch_override_=-1;
#ifdef _WIN32
    HANDLE file_=INVALID_HANDLE_VALUE,mapping_=nullptr;
#else
    int file_=-1;
#endif
    void close() noexcept {
#ifdef _WIN32
        if(view_)UnmapViewOfFile(view_);
        if(mapping_)CloseHandle(mapping_);
        if(file_!=INVALID_HANDLE_VALUE)CloseHandle(file_);
#else
        if(view_)::munmap(view_,mapped_bytes_);
        if(file_>=0)::close(file_);
#endif
    }
public:
    explicit MappedTensorRange(const std::filesystem::path& path, std::uint64_t offset, std::uint64_t bytes):bytes_(bytes) {
        const auto* profile=std::getenv("TYLOINFER_PROFILE_SSD_EXPERTS");
        profile_=profile && profile[0]=='1';
        const auto* prefetch=std::getenv("TYLOINFER_MAPPED_BATCH_PREFETCH");
        if(prefetch)prefetch_override_=prefetch[0]!='0';
        if(!bytes_)return;
        try {
            std::uint64_t granularity=0,file_bytes=0;
#ifdef _WIN32
            file_=CreateFileW(path.c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_FLAG_RANDOM_ACCESS,nullptr);
            if(file_==INVALID_HANDLE_VALUE)throw std::system_error(GetLastError(),std::system_category(),"open mapped tensor");
            LARGE_INTEGER size{};
            if(!GetFileSizeEx(file_,&size) || size.QuadPart<0)throw std::runtime_error("mapped tensor file size unavailable");
            file_bytes=static_cast<std::uint64_t>(size.QuadPart);
            SYSTEM_INFO info{};GetSystemInfo(&info);granularity=info.dwAllocationGranularity;
            page_bytes_=info.dwPageSize;
#else
            file_=::open(path.c_str(),O_RDONLY);
            if(file_<0)throw std::system_error(errno,std::generic_category(),"open mapped tensor");
            struct stat size{};
            if(::fstat(file_,&size) || size.st_size<0)throw std::runtime_error("mapped tensor file size unavailable");
            file_bytes=static_cast<std::uint64_t>(size.st_size);
            const auto page=::sysconf(_SC_PAGESIZE);
            if(page<=0)throw std::runtime_error("mapped tensor page size unavailable");
            page_bytes_=static_cast<std::size_t>(page);
            granularity=static_cast<std::uint64_t>(page);
#endif
            if(offset>file_bytes || bytes_>file_bytes-offset)throw std::out_of_range("mapped tensor exceeds file");
            const auto base=offset/granularity*granularity;
            prefix_=static_cast<std::size_t>(offset-base);
            if(bytes_>std::numeric_limits<std::size_t>::max()-prefix_)throw std::overflow_error("mapped tensor exceeds address space");
            mapped_bytes_=prefix_+static_cast<std::size_t>(bytes_);
#ifdef _WIN32
            mapping_=CreateFileMappingW(file_,nullptr,PAGE_READONLY,0,0,nullptr);
            if(!mapping_)throw std::system_error(GetLastError(),std::system_category(),"create tensor mapping");
            view_=MapViewOfFile(mapping_,FILE_MAP_READ,static_cast<DWORD>(base>>32),static_cast<DWORD>(base),mapped_bytes_);
            if(!view_)throw std::system_error(GetLastError(),std::system_category(),"map tensor view");
#else
            if(base>static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()))throw std::overflow_error("mapped tensor offset exceeds platform range");
            view_=::mmap(nullptr,mapped_bytes_,PROT_READ,MAP_SHARED,file_,static_cast<off_t>(base));
            if(view_==MAP_FAILED){view_=nullptr;throw std::system_error(errno,std::generic_category(),"map tensor view");}
#endif
        }catch(...){close();throw;}
    }
    ~MappedTensorRange(){close();}
    MappedTensorRange(const MappedTensorRange&)=delete;
    MappedTensorRange& operator=(const MappedTensorRange&)=delete;
    bool release_pages() const noexcept {
        if(!view_ || !mapped_bytes_)return false;
#ifdef _WIN32
        // Unlocked file pages become reclaimable even when the API reports
        // ERROR_NOT_LOCKED. The read-only view and its file stay open.
        (void)VirtualUnlock(view_,mapped_bytes_);
#else
        (void)::madvise(view_,mapped_bytes_,MADV_DONTNEED);
#endif
        return true;
    }
    void release_read_pages(const std::vector<ReadSpan>& spans) const {
        if(spans.empty())return;
        // Large prefill batches benefit from one view-wide trim. For sparse
        // decode rows, walking a multi-GiB view costs more than the reads:
        // release just the touched pages, merging duplicate/adjacent ranges.
        if(spans.size()>1024 || !page_bytes_) {release_pages();return;}
        std::vector<std::pair<std::uintptr_t,std::uintptr_t>> pages;
        pages.reserve(spans.size());
        for(const auto& span:spans)if(span.size) {
            const auto address=reinterpret_cast<std::uintptr_t>(view_)+prefix_+std::size_t(span.offset);
            const auto begin=address/page_bytes_*page_bytes_;
            const auto end=(address+span.size+page_bytes_-1)/page_bytes_*page_bytes_;
            pages.emplace_back(begin,end);
        }
        if(pages.empty())return;
        std::sort(pages.begin(),pages.end());
        const auto release=[](std::uintptr_t begin,std::uintptr_t end) {
#ifdef _WIN32
            (void)VirtualUnlock(reinterpret_cast<void*>(begin),end-begin);
#else
            (void)::madvise(reinterpret_cast<void*>(begin),end-begin,MADV_DONTNEED);
#endif
        };
        auto begin=pages.front().first,end=pages.front().second;
        for(std::size_t i=1;i<pages.size();++i) {
            if(pages[i].first<=end)end=std::max(end,pages[i].second);
            else {release(begin,end);begin=pages[i].first;end=pages[i].second;}
        }
        release(begin,end);
    }
    void read(std::uint64_t offset,std::byte* destination,std::size_t size)const {
        if(offset>bytes_ || size>bytes_-offset)throw std::out_of_range("mapped tensor row range is out of bounds");
        if(!size)return;
        if(!destination)throw std::invalid_argument("mapped tensor destination is null");
        const auto started=profile_?MappedClock::now():MappedClock::time_point{};
        std::memcpy(destination,static_cast<const std::byte*>(view_)+prefix_+static_cast<std::size_t>(offset),size);
        if(profile_) {
            mapped_read_totals[1].fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(MappedClock::now()-started).count(),std::memory_order_relaxed);
            mapped_read_totals[2].fetch_add(size,std::memory_order_relaxed);
            mapped_read_totals[3].fetch_add(1,std::memory_order_relaxed);
        }
    }
    void read_batch(const std::vector<ReadSpan>& spans)const {
        // Check the complete batch before touching any destination. A failed
        // hint is harmless: the following bounded read faults in that page.
        for(const auto& span:spans) {
            if(span.offset>bytes_ || span.size>bytes_-span.offset)
                throw std::out_of_range("mapped tensor batch is out of bounds");
            if(span.size && !span.destination)
                throw std::invalid_argument("mapped tensor batch destination is null");
        }
        // Like Strata's PLE prefetch, submit the independent pages together.
        // NINT stores q, sub-scale/min and neuron anchors in separate streams;
        // sequential mapped reads otherwise fault those five streams in turn.
        const auto started=profile_?MappedClock::now():MappedClock::time_point{};
        if(prefetch_override_<0?mapped_prefetch_context:prefetch_override_!=0) {
#ifdef _WIN32
        std::vector<WIN32_MEMORY_RANGE_ENTRY> ranges;
        ranges.reserve(std::min<std::size_t>(spans.size(),4096));
        const auto prefetch=[&] {
            if(!ranges.empty())(void)PrefetchVirtualMemory(GetCurrentProcess(),ranges.size(),ranges.data(),0);
            ranges.clear();
        };
        for(const auto& span:spans)if(span.size) {
            ranges.push_back({static_cast<std::byte*>(view_)+prefix_+static_cast<std::size_t>(span.offset),span.size});
            if(ranges.size()==4096)prefetch();
        }
        prefetch();
#else
        const auto page=std::size_t(::sysconf(_SC_PAGESIZE));
        if(page && page!=std::size_t(-1))for(const auto& span:spans)if(span.size) {
            const auto address=reinterpret_cast<std::uintptr_t>(view_)+prefix_+std::size_t(span.offset);
            const auto begin=address/page*page,end=(address+span.size+page-1)/page*page;
            (void)::madvise(reinterpret_cast<void*>(begin),end-begin,MADV_WILLNEED);
        }
#endif
        }
        if(profile_) {
            mapped_read_totals[0].fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(MappedClock::now()-started).count(),std::memory_order_relaxed);
            mapped_read_totals[4].fetch_add(1,std::memory_order_relaxed);
        }
        for(const auto& span:spans)if(span.size)read(span.offset,span.destination,span.size);
        // Selected rows now belong to the bounded row cache/packet. Keeping
        // a second working-set copy of sparse pages across all Engram tables
        // can exhaust RAM despite a bounded expert cache.
        if(mapped_release_context)release_read_pages(spans);
    }
};

std::mutex mapped_ranges_mutex;
std::vector<std::weak_ptr<MappedTensorRange>> mapped_ranges;
std::shared_ptr<MappedTensorRange> map_tensor_range(const std::filesystem::path& path, std::uint64_t offset, std::uint64_t bytes) {
    auto range=std::make_shared<MappedTensorRange>(path,offset,bytes);
    std::lock_guard lock(mapped_ranges_mutex);mapped_ranges.push_back(range);
    return range;
}

} // namespace

std::array<std::uint64_t,5> mapped_tensor_read_profile() {
    std::array<std::uint64_t,5> result{};
    for(std::size_t i=0;i<result.size();++i)result[i]=mapped_read_totals[i].load(std::memory_order_relaxed);
    return result;
}

std::size_t release_mapped_tensor_pages() {
    std::lock_guard lock(mapped_ranges_mutex);
    std::size_t views=0;
    for(auto item=mapped_ranges.begin();item!=mapped_ranges.end();) {
        if(const auto range=item->lock()) {views+=range->release_pages();++item;}
        else item=mapped_ranges.erase(item);
    }
    return views;
}

MappedPrefetchScope::MappedPrefetchScope(bool enabled,bool release_after_read)
        :previous_(mapped_prefetch_context),previous_release_(mapped_release_context) {
    mapped_prefetch_context=enabled;
    mapped_release_context=release_after_read;
}
MappedPrefetchScope::~MappedPrefetchScope(){
    mapped_prefetch_context=previous_;mapped_release_context=previous_release_;
}

std::function<void(std::uint64_t, std::byte*, std::size_t)> mapped_reader(const std::filesystem::path& path, std::uint64_t offset, std::uint64_t bytes) {
    auto range=map_tensor_range(path,offset,bytes);
    return [range](std::uint64_t offset,std::byte* destination,std::size_t size){range->read(offset,destination,size);};
}

std::function<void(const std::vector<ReadSpan>&)> mapped_batch_reader(const std::filesystem::path& path, std::uint64_t offset, std::uint64_t bytes) {
    auto range=map_tensor_range(path,offset,bytes);
    return [range](const std::vector<ReadSpan>& spans){range->read_batch(spans);};
}

} // namespace mfq
