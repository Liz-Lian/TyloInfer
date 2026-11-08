#include "mfq_paged_prefix_cache.h"
#include "paged_session_bindings.h"
#include "tyloinfer/file_read_window.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

std::filesystem::path temporary_directory() {
    const auto parent=std::filesystem::temp_directory_path();
    // Two SHA-256 path components already consume most of MAX_PATH. Use
    // short atomic fixture names under the configured build/test temp root.
    for(std::size_t index=0;;++index) {
        auto path=parent/("p"+std::to_string(index));
        if(std::filesystem::create_directory(path))return path;
    }
}

std::shared_ptr<const std::vector<std::uint8_t>> payload(
    std::initializer_list<std::uint8_t> bytes) {
    return std::make_shared<const std::vector<std::uint8_t>>(bytes);
}

std::size_t benchmark_size(const char* name, std::size_t fallback) {
    const auto* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return fallback;
    char* end = nullptr;
    const auto parsed = std::strtoull(value, &end, 10);
    require(end != value && *end == '\0' && parsed > 0,
            "invalid prefix cache benchmark size");
    return static_cast<std::size_t>(parsed);
}

void check_read_window() {
    using Window=tyloinfer::ScopedFileReadWindow;
    std::vector<std::uint8_t> file(3*Window::capacity+17);
    for(std::size_t i=0;i<file.size();++i)file[i]=std::uint8_t(i*37+i/251);
    std::size_t calls=0,total=0;bool fail=false;
    const auto reader=[&](std::uint64_t at,void* output,std::size_t count) {
        ++calls;total+=count;
        require(at<=file.size() && count<=file.size()-at,"Read window crossed EOF");
        std::memcpy(output,file.data()+at,count);
        if(fail)throw std::runtime_error("Injected window read failure");
    };
    std::array<std::uint8_t,80> bytes{};
    const auto read=[&](std::uint64_t source,std::size_t at) {
        const auto hit=Window::read(source,file.size(),at,bytes.data(),bytes.size(),reader);
        if(hit)require(std::equal(bytes.begin(),bytes.end(),file.begin()+at),"Read window changed bytes");
        return hit;
    };
    require(!read(1,0) && calls==0,"Read-ahead leaked outside a recovery scope");
    {
        Window scope;
        require(read(1,13) && read(1,97) && calls==1 && total==Window::capacity,
            "Sequential ranges did not share one bounded read");
        {
            Window index_only(false);
            require(read(1,200) && !read(1,Window::capacity+13) && calls==1,
                "SSD-only companion read caused unwanted KV prefetch");
        }
        require(read(1,Window::capacity+13) && calls==2,"Nested scope lost its parent's read-ahead mode");
        require(read(2,Window::capacity+13) && calls==3,"File version change reused stale bytes");
        fail=true;bool rejected=false;
        try{read(3,0);}catch(const std::runtime_error&){rejected=true;}
        require(rejected,"Failed window fill did not throw");fail=false;
        require(read(2,Window::capacity+13) && calls==5,"Failed fill left a valid corrupted window");
        require(read(1,file.size()-bytes.size()) && total==5*Window::capacity+bytes.size(),
            "Read window did not clamp its final fill at EOF");
        std::vector<std::uint8_t> bulk(Window::capacity);
        require(!Window::read(1,file.size(),0,bulk.data(),bulk.size(),reader),
            "Bulk reads unexpectedly used read-ahead");
        try{Window::read(1,file.size(),file.size()-1,bytes.data(),2,reader);
            require(false,"Invalid window range was accepted");}catch(const std::out_of_range&){}
        auto isolated=std::async(std::launch::async,[&]{return read(1,0);});
        require(!isolated.get(),"Read-ahead state leaked across threads");
    }
    require(!read(1,0),"Recovery window survived its owning scope");
}

void check_ancestry_eviction() {
    using namespace mfq::cache;
    const auto root = temporary_directory();
    constexpr std::uint64_t file_bytes = 156 + 4;
    PagedPrefixCacheConfig config{root, "durable-prefix-ancestry", 4, 4096, 0, 8};
    config.min_disk_free_bytes = 0;
    config.max_pending_bytes = 0;
    {
        PagedPrefixCache cache(config);
        const std::vector<std::int64_t> tokens{1,2,3,4,5,6,7,8,9,10,11,12};
        const auto first = cache.store({}, tokens.data(), 4, payload({1,2,3,4}));
        const auto second = cache.store(first, tokens.data()+4, 4, payload({5,6,7,8}));
        const auto third = cache.store(second, tokens.data()+8, 4, payload({9,10,11,12}));
        cache.flush();
        require(cache.load(first).has_value(), "Ancestry fixture lost its first page");
        cache.set_disk_limit(2*file_bytes);
        cache.flush();
        require(cache.match(tokens).matched_tokens == 8 && cache.load(second) && !cache.load(third),
            "Disk LRU removed an interior prefix page instead of the branch leaf");
        require(cache.metrics().disk_bytes == 2*file_bytes && cache.metrics().evictions == 1,
            "Ancestry-aware eviction exceeded its byte budget");
        cache.set_disk_limit(file_bytes);
        cache.flush();
        require(cache.match(tokens).matched_tokens == 4 && !cache.load(second),
            "Repeated disk shrink did not retain the usable prefix root");
        cache.set_disk_limit(0);
        cache.flush();
        require(cache.metrics().disk_blocks == 0, "Zero SSD budget left prefix ancestors behind");
    }
    config.compatibility_key = "branched-prefix-ancestry";
    {
        PagedPrefixCache cache(config);
        const std::vector<std::int64_t> trunk{1,2,3,4}, a{5,6,7,8}, b{9,10,11,12};
        const auto first = cache.store({}, trunk.data(), 4, payload({1,2,3,4}));
        const auto old_leaf = cache.store(first, a.data(), 4, payload({5,6,7,8}));
        const auto recent_leaf = cache.store(first, b.data(), 4, payload({9,10,11,12}));
        cache.flush();
        cache.set_disk_limit(2*file_bytes);
        cache.flush();
        require(cache.load(first) && !cache.load(old_leaf) && cache.load(recent_leaf),
            "Branch leaf selection stopped respecting LRU");
    }
    config.compatibility_key = "pinned-prefix-leaf";
    {
        PagedPrefixCache cache(config);
        const std::vector<std::int64_t> tokens{1,2,3,4,5,6,7,8};
        const auto first = cache.store({}, tokens.data(), 4, payload({1,2,3,4}));
        const auto second = cache.store(first, tokens.data()+4, 4, payload({5,6,7,8}));
        cache.flush();
        cache.pin({second});
        cache.set_disk_limit(file_bytes);
        cache.flush();
        require(cache.match(tokens).matched_tokens == 8,
            "Pinned child retained its payload but lost its prefix ancestor");
        cache.unpin({second});
        cache.flush();
        require(cache.match(tokens).matched_tokens == 4,
            "Unpinned leaf prevented deferred ancestor-safe eviction");
    }
    std::filesystem::remove_all(root);
}

void check_deferred_replacement() {
    using namespace mfq::cache;
    const auto root = temporary_directory();
    PagedPrefixCacheConfig config{root, "deferred-current-version", 4, 4096, 16, 8};
    config.spill_hot_on_eviction = true;
    config.max_pending_bytes = 0;
    config.min_disk_free_bytes = 0;
    {
        PagedPrefixCache cache(config);
        const std::vector<std::int64_t> tokens{1,2,3,4};
        const auto hash = cache.store({},tokens.data(),4,payload({1,2,3,4}));
        cache.trim_hot();cache.flush();
        const auto old = cache.borrow(hash);
        require(old && old->payload_bytes==4,"Deferred replacement lost initial SSD version");
        const auto replacement = cache.replace({},tokens.data(),4,payload({9,8,7,6,5,4}));
        cache.flush();
        require(replacement==hash && cache.metrics().writes==1 && cache.metrics().hot_bytes==6,
            "RAM replacement unexpectedly wrote through to SSD");
        require(!cache.borrow(hash) && !cache.borrow_prefix({hash}).front(),
            "Dirty RAM replacement exposed its stale SSD file lease");
        const auto loaded = cache.load(hash);
        require(loaded && *loaded==std::vector<std::uint8_t>({9,8,7,6,5,4}),
            "Current RAM replacement was not readable");
        require(old->read_range(0,4)==std::vector<std::uint8_t>({1,2,3,4}),
            "RAM replacement invalidated an earlier immutable SSD lease");
        cache.trim_hot();cache.flush();
        const auto current = cache.borrow(hash);
        require(current && current!=old && current->payload_bytes==6 &&
                current->read_range(0,6)==*loaded && cache.metrics().writes==2,
            "Evicted RAM replacement did not publish its current SSD version");
        require(old->read_range(0,4)==std::vector<std::uint8_t>({1,2,3,4}),
            "New SSD publication mixed existing file-reader versions");
    }
    std::filesystem::remove_all(root);
}

void run_benchmark() {
    using namespace mfq::cache;
    const auto block_count = benchmark_size(
        "TYLOINFER_PREFIX_CACHE_BENCHMARK_BLOCKS", 16);
    constexpr std::size_t block_tokens = 256;
    const auto payload_bytes = benchmark_size(
        "TYLOINFER_PREFIX_CACHE_BENCHMARK_PAYLOAD_BYTES",
        8ULL * 1024ULL * 1024ULL);
    const auto root = temporary_directory();
    std::vector<std::int64_t> tokens(block_count * block_tokens);
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        tokens[index] = static_cast<std::int64_t>(index * 17 + 3);
    }
    auto bytes = std::make_shared<std::vector<std::uint8_t>>(payload_bytes);
    for (std::size_t index = 0; index < bytes->size(); ++index) {
        (*bytes)[index] = static_cast<std::uint8_t>(index * 29 + 11);
    }
    const std::shared_ptr<const std::vector<std::uint8_t>> shared = bytes;
    require(payload_bytes <= std::numeric_limits<std::size_t>::max() / block_count,
            "prefix cache benchmark size overflow");
    const auto total_bytes = block_count * payload_bytes;
    {
        PagedPrefixCache cache(PagedPrefixCacheConfig{
            root,
            "benchmark-v1",
            block_tokens,
            total_bytes * 2,
            total_bytes * 2,
            block_count,
            total_bytes * 2,
        });
        BlockHash parent{};
        for (std::size_t block = 0; block < block_count; ++block) {
            parent = cache.store(
                parent,
                tokens.data() + block * block_tokens,
                block_tokens,
                shared);
        }
        cache.flush();
        const auto match = cache.match(tokens, {}, false);
        volatile std::uint64_t checksum = 0;
        const auto started = std::chrono::steady_clock::now();
        for (const auto& hash : match.blocks) {
            const auto loaded = cache.load(hash);
            require(loaded.has_value(), "benchmark hot load failed");
            checksum += loaded->front();
            checksum += loaded->back();
        }
        const auto elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        std::cout << "prefix_cache_benchmark phase=legacy_hot blocks="
                  << block_count << " bytes=" << total_bytes
                  << " elapsed_ms=" << elapsed
                  << " checksum=" << checksum << '\n';
        checksum = 0;
        const auto shared_started = std::chrono::steady_clock::now();
        const auto loaded = cache.load_prefix(match.blocks);
        require(loaded.size() == block_count, "benchmark shared hot load failed");
        for (const auto& block : loaded) {
            checksum += block->front();
            checksum += block->back();
        }
        const auto shared_elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - shared_started).count();
        std::cout << "prefix_cache_benchmark phase=shared_hot blocks="
                  << block_count << " bytes=" << total_bytes
                  << " elapsed_ms=" << shared_elapsed
                  << " checksum=" << checksum << '\n';
    }
    const auto benchmark_cold = [&](const char* phase, std::size_t readers) {
        PagedPrefixCache cache(PagedPrefixCacheConfig{
            root,
            "benchmark-v1",
            block_tokens,
            total_bytes * 2,
            total_bytes * 2,
            block_count,
            total_bytes * 2,
            readers,
        });
        const auto match = cache.match(tokens, {}, false);
        volatile std::uint64_t checksum = 0;
        const auto started = std::chrono::steady_clock::now();
        const auto loaded = cache.load_prefix(match.blocks);
        require(loaded.size() == block_count, "benchmark cold load failed");
        for (const auto& block : loaded) {
            checksum += block->front();
            checksum += block->back();
        }
        const auto elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        std::cout << "prefix_cache_benchmark phase=" << phase << " blocks="
                  << block_count << " bytes=" << total_bytes
                  << " elapsed_ms=" << elapsed
                  << " checksum=" << checksum << '\n';
    };
    benchmark_cold("shared_cold_serial", 1);
    benchmark_cold("shared_cold_2", 2);
    benchmark_cold("shared_cold", 4);
    benchmark_cold("shared_cold_8", 8);
    std::filesystem::remove_all(root);
}

} // namespace

int main() try {
    using namespace mfq::cache;
    check_read_window();
    check_ancestry_eviction();
    check_deferred_replacement();
    const auto inspect_root = temporary_directory();
    require(block_hash_hex(sha256(""))=="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "SHA-256 empty known-answer mismatch");
    require(block_hash_hex(sha256("abc"))=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "SHA-256 abc known-answer mismatch");
    {
        auto image=std::make_shared<std::vector<std::uint8_t>>(1000000,'a');
        require(block_hash_hex(sha256(image->data(),image->size()))==
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
            "SHA-256 million-byte known-answer mismatch");
        PagedPrefixCacheConfig config{inspect_root/"streamed-sha","streamed-sha",4,8ULL<<20,0,8};
        config.max_pending_bytes=0;PagedPrefixCache cache(config);
        const std::vector<std::int64_t> tokens{1,2,3,4};
        const auto hash=cache.store({},tokens.data(),tokens.size(),image);cache.flush();
        auto reader=cache.borrow(hash);
        require(reader && reader->read_range(65530,80)==std::vector<std::uint8_t>(80,'a'),
            "Streaming SHA-256 validation rejected a known payload across read boundaries");
        tyloinfer::ScopedFileReadWindow window;
        for(std::size_t at:{std::size_t(0),std::size_t(65530),std::size_t(65580),image->size()-80})
            require(reader->read_range(at,80)==std::vector<std::uint8_t>(80,'a'),
                "Recovery read-ahead changed the borrowed payload");
    }
    {
        PagedPrefixCacheConfig config{inspect_root/"verified-leases","verified-file-lease",4,4096,0,8};
        config.max_pending_bytes=0;
        PagedPrefixCache cache(config);
        const std::vector<std::int64_t> tokens{1,2,3,4};
        const auto hash=cache.store({},tokens.data(),tokens.size(),payload({1,2,3,4}));cache.flush();
        auto incoming=cache.borrow_prefix({hash,hash,{}});
        require(incoming.size()==3 && !incoming.back(),"Batch prefix borrow lost its missing-entry boundary");
        auto first=incoming.front(),second=incoming[1];incoming.clear();
        std::array<uint8_t,6> direct{};direct.fill(0x6a);
        require(first && first->read_range_into,"Verified file lease omitted direct destination reads");
        first->read_range_into(1,direct.data()+1,2);
        require(direct==std::array<uint8_t,6>{0x6a,2,3,0x6a,0x6a,0x6a},"Direct prefix read changed its range or surrounding bytes");
        try {first->read_range_into(3,direct.data(),2);require(false,"Out-of-range direct prefix read accepted");}catch(const std::out_of_range&){}
        require(direct[0]==0x6a,"Rejected direct prefix read changed its destination");
        try {first->read_range_into(0,nullptr,1);require(false,"Null direct prefix destination accepted");}catch(const std::invalid_argument&){}
        tyloinfer::ScopedFileReadWindow window;
        require(first && first==second,"Repeated borrow did not reuse its verified immutable file lease");
        const std::weak_ptr<const PagedPrefixReadLease> weak=first;
        cache.replace({},tokens.data(),tokens.size(),payload({5,6,7,8}));cache.flush();
        auto replaced=cache.borrow(hash);
        require(replaced && replaced!=first && replaced->read_range(0,4)==std::vector<uint8_t>({5,6,7,8}) &&
            first->read_range(0,4)==std::vector<uint8_t>({1,2,3,4}),"Replacement mixed old and new prefix file versions");
        first.reset();second.reset();require(weak.expired(),"Verified-lease memo retained an unused file handle");
        std::atomic<unsigned> failures{0};std::vector<std::thread> readers;
        for(int i=0;i<6;++i)readers.emplace_back([&]{if(cache.borrow(hash)!=replaced)++failures;});
        for(auto& thread:readers)thread.join();require(!failures,"Concurrent readers lost the shared file version");
        cache.clear();require(!cache.borrow(hash),"Clear retained a borrowable cache entry");
        require(replaced->read_range(0,4)==std::vector<uint8_t>({5,6,7,8}),"Clear invalidated a live file lease");
        replaced->read_range_into(0,direct.data(),4);
        require(std::equal(direct.begin(),direct.begin()+4,std::array<uint8_t,4>{5,6,7,8}.begin()),"Clear invalidated a direct prefix reader");
        cache.store({},tokens.data(),tokens.size(),payload({9,10,11,12}));cache.flush();
        auto before_corruption=cache.borrow(hash);require(bool(before_corruption),"Fresh cache lease missing");
        const auto hex=block_hash_hex(hash);
        const auto file=config.cache_dir/block_hash_hex(sha256(config.compatibility_key))/hex.substr(0,2)/(hex+".mfqkv");
        const auto stamp=std::filesystem::last_write_time(file);
        {std::fstream output(file,std::ios::binary|std::ios::in|std::ios::out);output.seekp(-1,std::ios::end);output.put(0);}
        std::filesystem::last_write_time(file,stamp+std::chrono::seconds(2));
        require(!cache.borrow(hash),"Changed prefix file bypassed checksum validation through the lease memo");
    }
    for (const bool disk : {false, true}) {
        PagedPrefixCacheConfig config{inspect_root, disk ? "disposable-ram-ssd" : "disposable-ram", 4,
            disk ? 4096ULL : 0ULL, 8, 8};
        config.spill_hot_on_eviction = true;
        PagedPrefixCache cache(config);
        const std::vector<std::int64_t> a{1,2,3,4}, b{5,6,7,8}, c{9,10,11,12};
        const auto first = cache.store({}, a.data(), a.size(), payload({1,2,3,4}));
        const auto second = cache.store({}, b.data(), b.size(), payload({5,6,7,8}));
        cache.pin({first, second});
        cache.flush();
        require(cache.metrics().disk_bytes == 0, "RAM hits must not write through to SSD");
        require(cache.load(first).has_value(), "first RAM hit missing");
        cache.store({}, c.data(), c.size(), payload({9,10,11,12}));
        cache.flush();
        require(cache.metrics().hot_bytes == 8, "pinned prefixes exceeded disposable RAM budget");
        require(cache.metrics().disk_blocks == (disk ? 1 : 0), "LRU eviction did not spill to enabled SSD");
        require(cache.match(a).matched_tokens == 4, "most recent RAM prefix evicted");
        require(cache.match(b).matched_tokens == (disk ? 4 : 0), "LRU victim retention incorrect");
        require(cache.set_hot_limit(4) == 4, "partial RAM reclaim failed");
        require(cache.set_hot_limit(0) == 4, "complete RAM reclaim failed");
        require(cache.metrics().hot_bytes == 0 && cache.metrics().pending_bytes == 0,
            "reclaimed prefixes still hold RAM");
        require(cache.load(second).has_value() == disk, "SSD restore/discard behavior incorrect");
        cache.set_hot_limit(8);
        cache.set_disk_limit(0);
        cache.flush();
        cache.store({}, a.data(), a.size(), payload({1,2,3,4}));
        cache.set_hot_limit(0);
        require(cache.metrics().disk_bytes == 0 && !cache.load(first), "disabling SSD still retained an evicted prefix");
    }
    {
        PagedPrefixCacheConfig config{inspect_root, "ram-pressure", 4, 0, 8, 8};
        config.spill_hot_on_eviction = true;
        std::atomic<std::uint64_t> available{100};
        config.available_ram_bytes = [&] { return available.load(); };
        config.min_free_ram_bytes = 8;
        PagedPrefixCache cache(config);
        const std::vector<std::int64_t> tokens{1,2,3,4};
        auto hash = cache.store({}, tokens.data(), tokens.size(), payload({1,2,3,4}));
        cache.pin({hash});
        require(cache.reserve_ram(96) == 4 && cache.metrics().hot_bytes == 0,
            "higher priority RAM allocation did not reclaim prefixes first");
        hash = cache.store({}, tokens.data(), tokens.size(), payload({1,2,3,4}));
        available = 0;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (cache.metrics().hot_bytes && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        require(cache.metrics().hot_bytes == 0 && !cache.load(hash), "host pressure must reclaim pinned RAM without a request");
    }
    {
        PagedPrefixCacheConfig config{inspect_root, "text-record-test", 4, 4096, 4096, 8};
        config.model_path = inspect_root / "sample.mfq";
        PagedPrefixCache cache(config);
        const std::vector<std::int64_t> sequence{1, 2, 3, 4};
        const auto hash = cache.store({}, sequence.data(), 4, payload({1, 2}));
        cache.flush();
        const auto name = block_hash_hex(hash);
        const auto directory = inspect_root / block_hash_hex(cache.compatibility_hash());
        const auto block = directory / name.substr(0, 2) / (name + ".mfqkv");
        const auto text = std::filesystem::path(block.string() + ".tokens");
        require(std::filesystem::file_size(text) == 108 + 4 * 8, "token record has incorrect size");
#ifndef _WIN32
        require((std::filesystem::status(text).permissions() & std::filesystem::perms::others_read)
            == std::filesystem::perms::none, "token record is publicly readable");
#endif
        std::ifstream record(text, std::ios::binary);
        std::array<char, 8> magic{};
        record.read(magic.data(), magic.size());
        require(std::string(magic.data(), 7) == "MFQTXT1", "token record has incorrect header");
        record.seekg(108);
        for (const auto token : sequence) {
            std::array<unsigned char, 8> bytes{};
            record.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
            require(record.good() && bytes[0] == token && std::all_of(bytes.begin() + 1, bytes.end(),
                [](unsigned char value) { return value == 0; }), "token record did not retain original IDs");
        }
        require(std::filesystem::exists(directory / "identity.txt"), "model cache identity is missing");
        record.close();
        std::filesystem::remove(block);
        std::filesystem::remove(text);
        require(cache.refresh_disk_index() == 1, "external deletion did not refresh native cache index");
        require(cache.metrics().hot_blocks == 0 && cache.metrics().disk_blocks == 0,
            "deleted cache retained stale hot payloads");
        require(cache.match(sequence).matched_tokens == 0, "deleted cache still matched");
        cache.store({}, sequence.data(), 4, payload({1, 2}));
        cache.flush();
        cache.clear();
        require(!std::filesystem::exists(text), "clearing left a token sidecar behind");
    }
    {
        PagedPrefixCacheConfig config{inspect_root, "disk-reserve-test", 4, 4096, 4096, 8};
        config.min_disk_free_bytes = std::numeric_limits<std::uint64_t>::max();
        PagedPrefixCache cache(config);
        const std::vector<std::int64_t> sequence{1, 2, 3, 4};
        cache.store({}, sequence.data(), 4, payload({1, 2}));
        cache.flush();
        require(cache.metrics().low_disk_space_skips == 1 && cache.metrics().failed_writes == 1,
            "disk reserve did not reject the write");
        require(cache.metrics().disk_blocks == 0 && cache.match(sequence).matched_tokens == 4,
            "disk-full guard broke the RAM cache");
    }
    std::filesystem::remove_all(inspect_root);
    const auto tail_root = temporary_directory();
    for (const bool disk : {false, true}) {
        const PagedPrefixCacheConfig config{tail_root, disk ? "tail-disk" : "tail-ram", 4,
            disk ? 4096U : 0U, disk ? 0U : 4096U, 8};
        const std::vector<std::int64_t> sequence{1, 2, 3, 4, 5, 6, 7};
        {
            PagedPrefixCache cache(config);
            const auto parent = cache.store({}, sequence.data(), 4, payload({1}));
            cache.store(parent, sequence.data() + 4, 2, payload({2}));
            cache.store(parent, sequence.data() + 4, 3, payload({3}));
            require(cache.match(sequence).matched_tokens == 7, "longest tail did not match");
            require(cache.match({1, 2, 3, 4, 5, 6, 8}).matched_tokens == 6,
                "divergent tail did not fall back");
            require(cache.match({1, 2, 3, 4, 5}).matched_tokens == 4,
                "tail matched an incomplete token sequence");
            cache.flush();
        }
        if (disk) {
            PagedPrefixCache cache(config);
            const auto match = cache.match(sequence);
            require(match.matched_tokens == 7 && match.blocks.size() == 2,
                "tail index did not survive restart");
            require(cache.load_prefix(match.blocks).size() == 2, "tail payload did not reload");
            cache.clear();
            require(cache.match(sequence).matched_tokens == 0, "cleared tail still matched");
        }
    }
    std::filesystem::remove_all(tail_root);
    const auto growing_root = temporary_directory();
    {
        PagedPrefixCache cache({growing_root, "growing-tails", 8, 4096, 64, 8});
        const std::vector<std::int64_t> sequence{1, 2, 3, 4, 5, 6};
        const std::vector<std::int64_t> branch{1, 2, 99};
        cache.store({}, branch.data(), branch.size(), payload({9}));
        for (std::size_t count = 2; count <= sequence.size(); ++count) {
            cache.store({}, sequence.data(), count, payload({static_cast<std::uint8_t>(count)}));
            cache.flush();
        }
        require(cache.metrics().disk_blocks == 3, "growing tails retained obsolete snapshots");
        require(cache.match(sequence).matched_tokens == 6, "newest tail was discarded");
        require(cache.match({1, 2, 3, 4, 5, 99}).matched_tokens == 5, "previous tail fallback was discarded");
        require(cache.match(branch).matched_tokens == 3, "unrelated branch tail was discarded");
        require(cache.set_hot_limit(0) > 0 && cache.metrics().hot_bytes == 0, "online hot budget did not trim RAM");
        require(cache.load_prefix(cache.match(sequence).blocks).size() == 1, "hot budget change deleted durable cache");
        require(cache.metrics().hot_bytes == 0, "disabled hot tier retained restored payloads");
        cache.set_disk_limit(0);
        cache.flush();
        require(cache.metrics().disk_bytes == 0,
            "online SSD disable retained durable cache");
        cache.set_disk_limit(4096);
        cache.store({}, sequence.data(), sequence.size(), payload({6}));
        cache.store({}, sequence.data(), sequence.size() - 1, payload({5}));
        cache.store({}, branch.data(), branch.size(), payload({9}));
        cache.flush();
        require(cache.metrics().disk_blocks == 3, "online SSD enable did not resume writes");
    }
    {
        PagedPrefixCache cache({growing_root, "growing-tails", 8, 4096, 0, 8});
        require(cache.match({1, 2, 3, 4, 5, 6}).matched_tokens == 6, "tail cleanup broke restart reuse");
        require(cache.metrics().disk_blocks == 3, "restart restored discarded tail versions");
        const std::vector<std::int64_t> full{1, 2, 3, 4, 5, 6, 7, 8};
        cache.store({}, full.data(), full.size(), payload({8}));
        cache.flush();
        require(cache.metrics().disk_blocks == 3, "full block left superseded tail versions");
        require(cache.match(full).matched_tokens == 8, "full block was not reusable");
    }
    std::filesystem::remove_all(growing_root);
    const auto budgets_root = temporary_directory();
    {
        PagedPrefixCache cache({budgets_root, "concurrent-budgets", 4, 0, 4096, 64});
        std::thread budgets([&] {
            for (std::size_t index = 0; index < 64; ++index) {
                cache.set_disk_limit(index % 2 ? 4096 : 16384);
                cache.set_hot_limit(index % 2 ? 0 : 4096);
                cache.trim_hot(0);
            }
        });
        for (std::int64_t index = 0; index < 64; ++index) {
            const std::vector<std::int64_t> sequence{index, 1, 2, 3};
            cache.store({}, sequence.data(), sequence.size(), payload({1, 2, 3}));
            (void)cache.metrics();
        }
        budgets.join();
        cache.set_disk_limit(0);
        cache.set_hot_limit(0);
        cache.flush();
        const auto stats = cache.metrics();
        require(stats.disk_bytes == 0 && stats.hot_bytes == 0 && stats.pending_bytes == 0,
            "concurrent budget updates left pending or over-budget cache data");
        const auto identity = budgets_root /
            block_hash_hex(sha256("concurrent-budgets")) / "identity.txt";
        require(std::filesystem::exists(identity), "enabling SSD cache did not persist its identity");
    }
    std::filesystem::remove_all(budgets_root);
    const auto binding_root = temporary_directory();
    {
        auto cache = std::make_shared<PagedPrefixCache>(PagedPrefixCacheConfig{binding_root, "bound-disk", 4, 200, 0, 8});
        mfq::engine::PagedSessionBindings bindings(cache, 16);
        const std::vector<std::int64_t> a{1, 2, 3, 4}, b{5, 6, 7, 8};
        const auto first_block = cache->store({}, a.data(), 4, payload({1}));
        cache->flush();
        bindings.bind("idle-a", {first_block}, 4);
        const auto second_block = cache->store({}, b.data(), 4, payload({2}));
        cache->flush();
        bindings.bind("idle-b", {second_block}, 4);
        require(cache->metrics().disk_bytes <= 200, "idle session pins bypassed the SSD budget");
        require(cache->match(a).matched_tokens == 0 && cache->match(b).matched_tokens == 4, "SSD LRU did not evict the older idle session");
        require(bindings.clear() == 2, "stale bindings could not be released safely");
    }
    {
        auto cache = std::make_shared<PagedPrefixCache>(PagedPrefixCacheConfig{binding_root, "bound-ram", 4, 0, 4, 8});
        mfq::engine::PagedSessionBindings bindings(cache, 16);
        const std::vector<std::int64_t> sequence{1, 2, 3, 4};
        const auto block = cache->store({}, sequence.data(), 4, payload({1, 2, 3, 4}));
        bindings.bind("idle-ram", {block}, 4);
        require(cache->set_hot_limit(0) == 4, "idle session bypassed the RAM budget");
        require(cache->match(sequence).matched_tokens == 0, "evicted RAM-only prefix still matched");
        require(bindings.close("idle-ram") == 1, "evicted RAM-only binding could not be closed");
    }
    std::filesystem::remove_all(binding_root);
    require(
        block_hash_hex(sha256("abc")) ==
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "SHA-256 implementation mismatch");

    const auto root = temporary_directory();
    const std::vector<std::int64_t> tokens{1, 2, 3, 4, 5, 6, 7, 8, 9};
    BlockHash first{};
    BlockHash second{};
    {
        PagedPrefixCache cache(PagedPrefixCacheConfig{
            root,
            "model-sha|layout-v1|fp16",
            4,
            4096,
            64,
            2,
        });
        BlockHash parent{};
        first = cache.store(parent, tokens.data(), 4, payload({10, 11, 12}));
        second = cache.store(
            first, tokens.data() + 4, 4, payload({20, 21, 22, 23}));
        cache.flush();

        const auto match = cache.match(tokens);
        require(match.matched_tokens == 8, "longest prefix match failed");
        require(match.blocks.size() == 2, "prefix block chain length mismatch");
        require(match.blocks[0] == first && match.blocks[1] == second,
                "prefix block chain mismatch");
        const auto shared_chain = cache.load_prefix(match.blocks);
        require(shared_chain.size() == 2,
                "shared prefix chain load length mismatch");
        require(*shared_chain[0] == std::vector<std::uint8_t>({10, 11, 12}) &&
                    *shared_chain[1] ==
                        std::vector<std::uint8_t>({20, 21, 22, 23}),
                "shared prefix chain payload mismatch");
        const auto loaded = cache.load(second);
        require(loaded && *loaded == std::vector<std::uint8_t>({20, 21, 22, 23}),
                "hot block load mismatch");

        const auto duplicate = cache.store(
            first, tokens.data() + 4, 4, payload({99}));
        require(duplicate == second, "content address changed on duplicate store");
        const auto original = cache.load(second);
        require(original && *original == std::vector<std::uint8_t>({20, 21, 22, 23}),
                "duplicate store replaced deterministic block content");
        const auto replaced = cache.replace(
            first,
            tokens.data() + 4,
            4,
            payload({30, 31, 32, 33, 34}));
        require(replaced == second,
                "payload refresh changed the content-addressed block hash");
        const auto refreshed = cache.load(second);
        require(refreshed &&
                    *refreshed ==
                        std::vector<std::uint8_t>({30, 31, 32, 33, 34}),
                "pending payload refresh was not immediately visible");
        cache.flush();
        const auto stats = cache.metrics();
        require(stats.writes == 3, "unexpected physical write count");
        require(stats.deduplicated_writes == 1, "deduplication was not counted");
        require(stats.disk_blocks == 2, "disk block count mismatch");
    }

    {
        auto cache = std::make_shared<PagedPrefixCache>(
            PagedPrefixCacheConfig{
                root,
                "model-sha|layout-v1|fp16",
                4,
                4096,
                64,
                2,
            });
        mfq::engine::PagedSessionBindings bindings(cache, 2);
        bindings.bind("first", {first}, 4);
        require(bindings.fork("first", "fork") == 1,
                "paged session fork failed");
        require(bindings.sessions() == 2 && bindings.tokens() == 8,
                "paged session telemetry mismatch");
        bindings.bind("new", {first, second}, 8);
        require(bindings.close("first") == 0,
                "paged session LRU eviction failed");
        require(bindings.sessions() == 2 && bindings.tokens() == 12,
                "paged session replacement telemetry mismatch");
        require(bindings.clear() == 2 && bindings.sessions() == 0 &&
                    bindings.tokens() == 0,
                "paged session clear failed");
    }

    const auto namespace_dir =
        root / block_hash_hex(sha256("model-sha|layout-v1|fp16"));
    const auto stale_temporary = namespace_dir / "stale.mfqkv.tmp.1";
    {
        std::ofstream output(stale_temporary, std::ios::binary);
        output << "incomplete";
    }
    {
        PagedPrefixCache cache(PagedPrefixCacheConfig{
            root,
            "model-sha|layout-v1|fp16",
            4,
            4096,
            0,
            2,
        });
        const auto match = cache.match(tokens);
        require(match.matched_tokens == 8, "restart prefix recovery failed");
        const auto loaded = cache.load(first);
        require(loaded && *loaded == std::vector<std::uint8_t>({10, 11, 12}),
                "restart disk load mismatch");
        const auto replaced = cache.load(second);
        require(replaced &&
                    *replaced ==
                        std::vector<std::uint8_t>({30, 31, 32, 33, 34}),
                "refreshed payload did not survive restart");
        require(!std::filesystem::exists(stale_temporary),
                "stale temporary write was not removed during recovery");
    }

    {
        PagedPrefixCache incompatible(PagedPrefixCacheConfig{
            root,
            "different-model",
            4,
            4096,
            0,
            2,
        });
        require(
            incompatible.match(tokens).matched_tokens == 0,
            "compatibility namespace isolation failed");
    }

    const auto semantic_root = temporary_directory();
    {
        PagedPrefixCache cache(PagedPrefixCacheConfig{
            semantic_root,
            "semantic-invalidation",
            4,
            0,
            64,
            2,
        });
        const auto block = cache.store(
            BlockHash{}, tokens.data(), 4, payload({1, 3, 5, 7}));
        require(cache.match(tokens).matched_tokens == 4,
                "hot-only block was not indexed");
        require(cache.invalidate(block),
                "semantic invalidation did not remove the block");
        require(cache.match(tokens).matched_tokens == 0,
                "semantically invalid block remained matchable");
        require(!cache.invalidate(block),
                "semantic invalidation reported a missing block");
        require(cache.metrics().corrupt_blocks == 1,
                "semantic invalidation metric was not stable");
    }
    std::filesystem::remove_all(semantic_root);

    const auto effective_metrics_root = temporary_directory();
    {
        PagedPrefixCache cache(PagedPrefixCacheConfig{
            effective_metrics_root,
            "effective-match-metrics",
            4,
            0,
            64,
            2,
        });
        (void)cache.store(
            BlockHash{}, tokens.data(), 4, payload({2, 4, 6, 8}));
        require(
            cache.match(tokens, {}, false).matched_tokens == 4,
            "unrecorded prefix probe failed");
        require(cache.metrics().queries == 0,
                "unrecorded prefix probe changed query metrics");
        cache.record_match(4);
        cache.record_match(0);
        const auto stats = cache.metrics();
        require(stats.queries == 2 && stats.hits == 1 &&
                    stats.hit_tokens == 4,
                "effective prefix metrics were not recorded exactly");
    }
    std::filesystem::remove_all(effective_metrics_root);

    const auto multiprocess_root = temporary_directory();
    {
        const PagedPrefixCacheConfig config{
            multiprocess_root,
            "cross-process-upgrade",
            4,
            4096,
            0,
            2,
        };
        PagedPrefixCache upgrading_writer(config);
        PagedPrefixCache stale_writer(config);
        const auto block = upgrading_writer.replace(
            BlockHash{},
            tokens.data(),
            4,
            payload({21, 22, 23, 24, 25}));
        upgrading_writer.flush();
        require(
            stale_writer.store(
                BlockHash{},
                tokens.data(),
                4,
                payload({1, 2})) == block,
            "stale writer changed the shared block identity");
        stale_writer.flush();
        PagedPrefixCache reader(config);
        const auto loaded = reader.load(block);
        require(
            loaded &&
                *loaded == std::vector<std::uint8_t>({21, 22, 23, 24, 25}),
            "stale writer downgraded a cross-process checkpoint payload");
    }
    std::filesystem::remove_all(multiprocess_root);

    {
        const auto header_root = root / "header-chain-validation";
        const std::vector<std::int64_t> header_tokens{61, 62, 63, 64};
        BlockHash block{};
        {
            PagedPrefixCache cache(PagedPrefixCacheConfig{
                header_root,
                "header-chain-validation",
                4,
                4096,
                0,
                2,
            });
            BlockHash parent{};
            block = cache.store(
                parent,
                header_tokens.data(),
                4,
                payload({31, 32, 33}));
            cache.flush();
        }
        const auto block_text = block_hash_hex(block);
        const auto block_path =
            header_root /
            block_hash_hex(sha256("header-chain-validation")) /
            block_text.substr(0, 2) /
            (block_text + ".mfqkv");
        {
            std::fstream file(
                block_path,
                std::ios::binary | std::ios::in | std::ios::out);
            require(
                static_cast<bool>(file),
                "cannot open block header for corruption test");
            constexpr std::streamoff token_count_offset =
                8 + 4 + 32 + 32 + 32 + 4;
            const std::array<char, 4> partial_count{1, 0, 0, 0};
            file.seekp(token_count_offset);
            file.write(partial_count.data(), partial_count.size());
        }
        PagedPrefixCache cache(PagedPrefixCacheConfig{
            header_root,
            "header-chain-validation",
            4,
            4096,
            0,
            2,
        });
        require(cache.match(header_tokens).matched_tokens == 0,
                "a partial disk block was accepted as a full prefix block");
        require(cache.metrics().corrupt_blocks == 1,
                "invalid disk block metadata was not rejected on match");
        require(!std::filesystem::exists(block_path),
                "invalid disk block metadata was not removed");
    }

    {
        const auto lru_root = root / "restart-lru-test";
        const std::vector<std::int64_t> old_tokens{31, 32, 33, 34};
        const std::vector<std::int64_t> new_tokens{41, 42, 43, 44};
        std::uint64_t one_file_bytes = 0;
        {
            PagedPrefixCache cache(PagedPrefixCacheConfig{
                lru_root,
                "restart-lru",
                4,
                4096,
                0,
                2,
            });
            BlockHash parent{};
            (void)cache.store(
                parent, old_tokens.data(), old_tokens.size(), payload({1, 2, 3}));
            cache.flush();
            one_file_bytes = cache.metrics().disk_bytes;
            require(one_file_bytes > 0, "restart LRU fixture was not persisted");
        }
        {
            PagedPrefixCache cache(PagedPrefixCacheConfig{
                lru_root,
                "restart-lru",
                4,
                one_file_bytes,
                0,
                2,
            });
            BlockHash parent{};
            (void)cache.store(
                parent, new_tokens.data(), new_tokens.size(), payload({4, 5, 6}));
            cache.flush();
            require(cache.match(new_tokens).matched_tokens == 4,
                    "new block was treated as oldest after restart");
            require(cache.match(old_tokens).matched_tokens == 0,
                    "restart LRU did not evict the older persisted block");
        }
    }

    {
        const auto recovery_root = root / "scan-recovery-test";
        std::filesystem::path canonical;
        {
            PagedPrefixCache cache(PagedPrefixCacheConfig{
                recovery_root,
                "scan-recovery",
                4,
                4096,
                0,
                2,
            });
            BlockHash parent{};
            const auto hash = cache.store(
                parent, tokens.data(), 4, payload({7, 8, 9}));
            cache.flush();
            const auto text = block_hash_hex(hash);
            canonical = recovery_root /
                block_hash_hex(sha256("scan-recovery")) /
                text.substr(0, 2) /
                (text + ".mfqkv");
        }
        const auto invalid_dir = canonical.parent_path().parent_path() / "ff";
        std::filesystem::create_directories(invalid_dir);
        const auto duplicate = invalid_dir /
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff.mfqkv";
        std::filesystem::copy_file(canonical, duplicate);
        const auto truncated = invalid_dir /
            "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee.mfqkv";
        {
            std::ofstream output(truncated, std::ios::binary);
            output << "incomplete";
        }
        {
            PagedPrefixCache cache(PagedPrefixCacheConfig{
                recovery_root,
                "scan-recovery",
                4,
                4096,
                0,
                2,
            });
            require(!std::filesystem::exists(duplicate),
                    "non-canonical duplicate cache block survived recovery");
            require(!std::filesystem::exists(truncated),
                    "truncated cache block survived recovery");
            require(cache.metrics().corrupt_blocks == 2,
                    "rejected startup cache blocks were not counted");
            require(cache.match(tokens).matched_tokens == 4,
                    "valid block was lost during startup recovery");
        }
    }

    {
        const auto pin_root = root / "pin-test";
        PagedPrefixCache cache(PagedPrefixCacheConfig{
            pin_root,
            "pin-before-write",
            4,
            1,
            0,
            2,
        });
        BlockHash parent{};
        const auto pinned = cache.store(
            parent, tokens.data(), 4, payload({1, 2, 3, 4}));
        cache.pin({pinned});
        cache.flush();
        require(cache.metrics().disk_blocks == 1,
                "pending pinned block was evicted after write");
        cache.unpin({pinned});
        require(cache.metrics().disk_blocks == 0,
                "unpinned over-budget block was not evicted");
    }

    {
        const auto hot_only_root = root / "hot-only-test";
        PagedPrefixCache cache(PagedPrefixCacheConfig{
            hot_only_root,
            "hot-only",
            4,
            0,
            4096,
            2,
        });
        BlockHash parent{};
        const auto hot = cache.store(
            parent, tokens.data(), 4, payload({5, 6, 7, 8}));
        const auto duplicate = cache.store(
            parent, tokens.data(), 4, payload({9, 9, 9, 9}));
        cache.flush();
        const auto stats = cache.metrics();
        require(duplicate == hot, "RAM-only duplicate changed its block hash");
        require(stats.disk_blocks == 0 && stats.hot_blocks == 1,
                "RAM-only prefix block did not survive disk eviction");
        require(stats.writes == 0 && stats.failed_writes == 0,
                "RAM-only prefix cache performed a disk write");
        require(stats.deduplicated_writes == 1,
                "RAM-only duplicate store was not deduplicated");
        require(cache.match(tokens).matched_tokens == 4,
                "RAM-only prefix block was not matched");
        const auto loaded = cache.load(hot);
        require(loaded && *loaded == std::vector<std::uint8_t>({5, 6, 7, 8}),
                "RAM-only prefix block could not be loaded");
        require(
            cache.replace(
                parent,
                tokens.data(),
                4,
                payload({10, 11, 12, 13})) == hot,
            "RAM-only payload refresh changed its block hash");
        const auto refreshed = cache.load(hot);
        require(refreshed &&
                    *refreshed == std::vector<std::uint8_t>({10, 11, 12, 13}),
                "RAM-only payload refresh was not visible");
        cache.pin({hot});
        require(cache.trim_hot() == 0,
                "trim discarded a pinned RAM-only prefix block");
        cache.unpin({hot});
        require(cache.trim_hot() == 4,
                "trim did not reclaim an unpinned RAM-only prefix block");
        require(!cache.load(hot),
                "trimmed RAM-only prefix block remained readable");
        require(std::none_of(
                    std::filesystem::recursive_directory_iterator(hot_only_root),
                    std::filesystem::recursive_directory_iterator(),
                    [](const auto& entry) { return entry.is_regular_file(); }),
                "RAM-only prefix cache created a disk payload");
    }

    {
        const auto trim_root = root / "hot-trim-test";
        PagedPrefixCache cache(PagedPrefixCacheConfig{
            trim_root,
            "hot-trim",
            4,
            4096,
            4096,
            2,
        });
        BlockHash parent{};
        const auto block = cache.store(
            parent, tokens.data(), 4, payload({12, 13, 14, 15}));
        const auto second = cache.store(
            block, tokens.data(), 4, payload({16, 17, 18, 19}));
        cache.flush();
        require(cache.metrics().hot_bytes == 8,
                "durable block was not promoted into the hot tier");
        require(cache.trim_hot(4) == 4,
                "hot trim did not report released durable payload bytes");
        const auto trimmed = cache.metrics();
        require(trimmed.hot_bytes == 4 && trimmed.disk_blocks == 2,
                "hot trim deleted or retained the wrong cache tier");
        const auto restored = cache.load_prefix({block, second});
        require(restored.size() == 2 &&
                    *restored[0] == std::vector<std::uint8_t>({12, 13, 14, 15}) &&
                    *restored[1] == std::vector<std::uint8_t>({16, 17, 18, 19}),
                "trimmed hot chain could not be restored from SSD");
        require(cache.metrics().disk_hits == 1,
                "cold restore after hot trim was not recorded");
    }

    {
        const auto byte_root = root / "pending-byte-test";
        PagedPrefixCache cache(PagedPrefixCacheConfig{
            byte_root,
            "pending-byte-budget",
            4,
            4096,
            64,
            64,
            3,
        });
        BlockHash parent{};
        (void)cache.store(
            parent, tokens.data(), 4, payload({1, 2, 3, 4}));
        const auto stats = cache.metrics();
        require(stats.writes == 1,
                "oversized pending payload did not write inline");
        require(stats.pending_writes == 0 && stats.pending_bytes == 0,
                "inline write retained pending payload memory");
        require(stats.pending_max_bytes == 3,
                "pending payload byte budget was not reported");
    }

    {
        const auto clear_root = root / "concurrent-clear-test";
        constexpr std::size_t payload_size = 16ULL * 1024ULL * 1024ULL;
        PagedPrefixCache cache(PagedPrefixCacheConfig{
            clear_root,
            "concurrent-clear",
            4,
            64ULL * 1024ULL * 1024ULL,
            0,
            1,
            1,
            1,
        });
        const auto mutable_payload =
            std::make_shared<std::vector<std::uint8_t>>(payload_size, 0x5a);
        const std::shared_ptr<const std::vector<std::uint8_t>> large_payload =
            mutable_payload;
        std::atomic<bool> store_finished{false};
        std::thread writer([&] {
            BlockHash parent{};
            (void)cache.store(
                parent, tokens.data(), 4, large_payload);
            store_finished = true;
        });
        const auto deadline = std::chrono::steady_clock::now() +
            std::chrono::seconds(2);
        bool observed_inline_write = false;
        while (!store_finished && std::chrono::steady_clock::now() < deadline) {
            if (cache.metrics().pending_writes == 1) {
                observed_inline_write = true;
                break;
            }
            std::this_thread::yield();
        }
        if (!observed_inline_write) writer.join();
        require(observed_inline_write,
                "concurrent clear fixture did not overlap an inline write");
        (void)cache.clear();
        writer.join();
        const auto stats = cache.metrics();
        require(stats.pending_writes == 0 && stats.disk_blocks == 0 &&
                    stats.hot_blocks == 0,
                "clear returned before an overlapping inline write drained");
        require(cache.match(tokens).matched_tokens == 0,
                "overlapping inline write survived clear");
    }

    {
        const auto clear_root = root / "queued-clear-test";
        PagedPrefixCache cache(PagedPrefixCacheConfig{
            clear_root,
            "queued-clear",
            4,
            8ULL * 1024ULL * 1024ULL,
            0,
            128,
            8ULL * 1024ULL * 1024ULL,
        });
        const auto mutable_payload =
            std::make_shared<std::vector<std::uint8_t>>(
                64ULL * 1024ULL, 0x6b);
        const std::shared_ptr<const std::vector<std::uint8_t>> queued_payload =
            mutable_payload;
        std::array<std::int64_t, 4> queued_tokens{};
        std::size_t submitted = 0;
        for (; submitted < 128; ++submitted) {
            for (std::size_t token = 0; token < queued_tokens.size(); ++token) {
                queued_tokens[token] = static_cast<std::int64_t>(
                    submitted * queued_tokens.size() + token + 1000);
            }
            BlockHash parent{};
            (void)cache.store(
                parent,
                queued_tokens.data(),
                queued_tokens.size(),
                queued_payload);
            if (cache.metrics().pending_writes >= 4) {
                ++submitted;
                break;
            }
        }
        require(cache.metrics().pending_writes >= 4,
                "queued clear fixture did not build a write backlog");
        (void)cache.clear();
        const auto stats = cache.metrics();
        require(stats.pending_writes == 0 && stats.pending_bytes == 0 &&
                    stats.disk_blocks == 0 && stats.hot_blocks == 0,
                "clear retained a queued prefix write");
        require(stats.writes < submitted,
                "clear flushed queued prefix writes before deleting them");
    }

    {
        const auto concurrent_root = root / "concurrent-load-test";
        std::vector<std::int64_t> concurrent_tokens(32);
        for (std::size_t index = 0; index < concurrent_tokens.size(); ++index) {
            concurrent_tokens[index] = static_cast<std::int64_t>(index + 100);
        }
        {
            PagedPrefixCache cache(PagedPrefixCacheConfig{
                concurrent_root,
                "concurrent-load",
                4,
                1024 * 1024,
                0,
                16,
            });
            BlockHash parent{};
            for (std::size_t offset = 0; offset < concurrent_tokens.size();
                 offset += 4) {
                parent = cache.store(
                    parent,
                    concurrent_tokens.data() + offset,
                    4,
                    payload({1, 3, 5, 7, 9}));
            }
            cache.flush();
        }
        PagedPrefixCache cache(PagedPrefixCacheConfig{
            concurrent_root,
            "concurrent-load",
            4,
            1024 * 1024,
            0,
            16,
        });
        std::atomic<int> failures{0};
        std::vector<std::thread> callers;
        for (int caller = 0; caller < 4; ++caller) {
            callers.emplace_back([&] {
                for (int iteration = 0; iteration < 4; ++iteration) {
                    const auto match = cache.match(concurrent_tokens);
                    const auto loaded = cache.load_prefix(match.blocks);
                    if (loaded.size() != 8 || loaded.front()->at(2) != 5) {
                        ++failures;
                    }
                }
            });
        }
        for (auto& caller : callers) caller.join();
        require(failures.load() == 0,
                "concurrent shared prefix loads were inconsistent");
    }

    const auto second_text = block_hash_hex(second);
    const auto block_file =
        root /
        block_hash_hex(sha256("model-sha|layout-v1|fp16")) /
        second_text.substr(0, 2) /
        (second_text + ".mfqkv");
    require(std::filesystem::is_regular_file(block_file),
            "cache block file is missing");
    {
        std::fstream file(block_file, std::ios::binary | std::ios::in | std::ios::out);
        require(static_cast<bool>(file), "cannot open block for corruption test");
        file.seekp(-1, std::ios::end);
        char byte = 0;
        file.write(&byte, 1);
    }
    {
        PagedPrefixCache cache(PagedPrefixCacheConfig{
            root,
            "model-sha|layout-v1|fp16",
            4,
            4096,
            0,
            2,
        });
        const auto match = cache.match(tokens);
        require(match.matched_tokens == 8,
                "corrupt block was not indexed for validation");
        const auto loaded = cache.load_prefix(match.blocks);
        require(loaded.size() == 1 &&
                    *loaded.front() == std::vector<std::uint8_t>({10, 11, 12}),
                "batch load did not stop at the corrupt block");
        require(cache.metrics().corrupt_blocks == 1,
                "corrupt block was not counted");
    }

    std::filesystem::remove_all(root);
    if (const auto* enabled = std::getenv("TYLOINFER_PREFIX_CACHE_BENCHMARK");
        enabled != nullptr && enabled[0] == '1') {
        run_benchmark();
    }
    std::cout << "MFQ paged prefix cache tests passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "mfq-paged-prefix-cache-test: " << error.what() << '\n';
    return 1;
}
