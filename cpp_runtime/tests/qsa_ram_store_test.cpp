#include "tyloinfer/qsa_ram_store.h"
#include "tyloinfer/disk_budget.h"
#if defined(TYLOINFER_TEST_PREFIX_BORROW)
#include "mfq_paged_prefix_cache.h"
#endif
#include <cassert>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>

namespace tq=tyloinfer::qsa;
tq::RamKvImagePtr allocate(const tq::MicroblockImage& input) {
    auto bytes=std::make_shared<std::vector<std::uint8_t>>(input.key);
    bytes->insert(bytes->end(),input.value.begin(),input.value.end());
    auto image=std::make_shared<tq::RamKvImage>();image->owner=bytes;
    image->key=bytes->data();image->value=bytes->data()+input.key.size();
    image->key_bytes=input.key.size();image->value_bytes=input.value.size();
    image->key_bits=input.key_bits;image->value_bits=input.value_bits;
    image->key_state_bits=input.key_state_bits;image->value_state_bits=input.value_state_bits;
    return image;
}
tq::RamKvImagePtr image(unsigned seed,int bytes=1024,unsigned bits=0) {
    tq::MicroblockImage src;src.key.resize(bytes/2);src.value.resize(bytes/2);
    for(int i=0;i<bytes/2;++i){src.key[i]=std::uint8_t(i*29+seed);src.value[i]=std::uint8_t(i*13+seed+7);}
    src.key_bits=src.value_bits=std::uint8_t(bits);return allocate(src);
}
std::vector<tq::RamKvImagePtr> encode(const std::vector<tq::RamKvImagePtr>& input,tq::RamKvTier tier) {
    std::vector<tq::RamKvImagePtr> result;
    for(const auto& old:input) {
        tq::MicroblockImage src;const auto bytes=tier==tq::RamKvTier::Affine8?256:128;
        src.key.assign(old->key,old->key+bytes);src.value.assign(old->value,old->value+bytes);
        src.key_bits=src.value_bits=tier==tq::RamKvTier::Affine8?254:4;
        result.push_back(allocate(src));
    }
    return result;
}
std::shared_ptr<mfq::cache::QsaKvStore> disk(const std::filesystem::path& path) {
    mfq::cache::QsaKvStoreConfig c;c.directory=path.string();c.buffer_bytes=c.budget_bytes=64<<10;
    return std::make_shared<mfq::cache::QsaKvStore>(c);
}
void check_batched_ssd(const std::filesystem::path& root) {
    std::vector<std::future<void>> jobs;
    for(unsigned layer=0;layer<3;++layer)jobs.push_back(std::async(std::launch::async,[&,layer]{
        mfq::cache::QsaKvStoreConfig cfg;cfg.directory=(root/std::to_string(layer)).string();
        cfg.buffer_bytes=cfg.budget_bytes=8192;mfq::cache::QsaKvStore store(cfg);
        using Ref=mfq::cache::QsaKvStore::BlockPtr;
        std::vector<Ref> initial,replacements,retained;
        const auto payload=[&](unsigned id){return std::vector<std::uint8_t>(256+128*(id%3),std::uint8_t(id+layer*19));};
        std::size_t total=0;
        for(unsigned id=0;id<192;++id){auto bytes=payload(id);total+=bytes.size();initial.push_back(store.write(std::move(bytes)));}
        store.flush();
        for(unsigned id=0;id<192;++id)if(id%2)retained.push_back(initial[id]);else initial[id].reset();
        auto reader=std::async(std::launch::async,[&]{
            for(unsigned round=0;round<4;++round)for(unsigned id=0;id<retained.size();++id)
                assert(*store.read(retained[id])==payload(2*id+1));
        });
        // Reuse differently positioned holes while another thread reads the
        // surviving blocks through this same descriptor's seek cursor.
        for(unsigned id=192;id<384;++id){auto bytes=payload(id);total+=bytes.size();replacements.push_back(store.write(std::move(bytes)));}
        store.flush();reader.get();
        for(unsigned id=0;id<replacements.size();++id)assert(*store.read(replacements[id])==payload(192+id));
        assert(store.stats().written_bytes==total && !store.stats().pending_bytes && !store.stats().hot_bytes);
        initial.clear();retained.clear();replacements.clear();
        auto restarted=store.write(payload(997));store.flush();assert(*store.read(restarted)==payload(997));
    }));
    for(auto& job:jobs)job.get();
    // Reserve the entire volume through the real download reservation API.
    // An asynchronous batch must propagate that failure through flush().
    const auto budget=root/"blocked-budget";std::filesystem::create_directories(budget);
    const auto lease_path=budget/"blocked.lease";
    {std::ofstream file(lease_path,std::ios::binary);file<<'#'<<mfq::cache::disk_volume(root)<<' '<<std::numeric_limits<std::uint64_t>::max();}
    mfq::cache::DiskBudgetLock lease(lease_path);assert(lease.locked());
    const auto* previous=std::getenv("TYLOINFER_DISK_BUDGET_DIR");const auto saved=previous?std::string(previous):std::string{};
    const auto set_budget=[](const std::string& value){
#ifdef _WIN32
        _putenv_s("TYLOINFER_DISK_BUDGET_DIR",value.c_str());
#else
        if(value.empty())unsetenv("TYLOINFER_DISK_BUDGET_DIR");else setenv("TYLOINFER_DISK_BUDGET_DIR",value.c_str(),1);
#endif
    };
    set_budget(budget.string());
    try {
        auto blocked=disk(root/"blocked");bool failed=false;
        try{(void)blocked->write(std::vector<std::uint8_t>(256,7));blocked->flush();}catch(const std::runtime_error&){failed=true;}
        assert(failed && !blocked->stats().written_bytes);
    }catch(...){set_budget(saved);throw;}
    set_budget(saved);
    std::cout<<"Batched SSD mixed sizes, reused holes, concurrent stores/readers, restart and disk-budget failure PASS\n";
}
#if defined(TYLOINFER_TEST_PREFIX_BORROW)
void check_prefix_reclamation(const std::filesystem::path& root) {
    assert(!mfq::cache::make_ram_prefix_reclaimer({}));
    for(bool ssd:{false,true}) {
        mfq::cache::PagedPrefixCacheConfig cfg;
        cfg.cache_dir=root/(ssd?"ssd":"ram");cfg.compatibility_key="ram-pressure";
        cfg.block_size_tokens=4;cfg.max_hot_bytes=1024;cfg.max_disk_bytes=ssd?1<<20:0;
        cfg.spill_hot_on_eviction=true;cfg.min_disk_free_bytes=0;
        auto cache=std::make_shared<mfq::cache::PagedPrefixCache>(cfg);
        auto reclaim=mfq::cache::make_ram_prefix_reclaimer(cache);
        std::vector<mfq::cache::BlockHash> hashes;
        for(int i=0;i<4;++i) {
            const std::int64_t tokens[]{i*4,i*4+1,i*4+2,i*4+3};
            hashes.push_back(cache->store({},tokens,4,std::make_shared<std::vector<std::uint8_t>>(256,std::uint8_t(i))));
        }
        cache->pin(hashes);assert(cache->metrics().hot_bytes==1024);
        assert(reclaim(0)==0 && cache->metrics().hot_bytes==1024);
        assert(reclaim(300)==512 && cache->metrics().hot_bytes==512);
        assert(reclaim(4096)==512 && cache->metrics().hot_bytes==0);
        for(int i=0;i<4;++i) {
            const auto bytes=cache->load(hashes[i]);
            assert(bool(bytes)==ssd);
            if(bytes)assert(*bytes==std::vector<std::uint8_t>(256,std::uint8_t(i)));
        }
        std::weak_ptr<mfq::cache::PagedPrefixCache> weak=cache;
        cache.reset();assert(weak.expired() && reclaim(1024)==0);
    }
    std::cout<<"RAM prefix pressure: partial/full LRU eviction, SSD spill, RAM-only discard and weak lifetime PASS\n";
}
#endif
int main(int argc,char** argv)try {
    if(argc!=2)throw std::invalid_argument("Supply the test SSD directory");
    const std::filesystem::path root=argv[1];std::filesystem::create_directories(root);
    check_batched_ssd(root/"batch-io");
#if defined(TYLOINFER_TEST_PREFIX_BORROW)
    check_prefix_reclamation(root/"prefix-pressure");
#endif
    {
        std::promise<void> started,resume;auto started_future=started.get_future();auto resume_future=resume.get_future().share();
        tq::RamKvStoreConfig config;
        tq::RamKvStore live_floor(config,allocate,[&](const auto& input,auto tier){started.set_value();resume_future.wait();return encode(input,tier);},[](const auto&){});
        for(unsigned i=0;i<32;++i)live_floor.append(image(i),{1024,512,256},1);
        for(unsigned i=0;i<608;++i)live_floor.append_at_tier(image(i,512,254),{1024,512,256},tq::RamKvTier::Affine8,1);
        auto work=std::async(std::launch::async,[&]{return live_floor.maintain(32*1024+96*512+512*256);});
        started_future.wait();live_floor.append(image(99),{1024,512,256},2);resume.set_value();
        const auto result=work.get();const auto usage=live_floor.stats();
        assert(usage.tokens[0]==33*4 && usage.tokens[1]==608*4 && usage.tokens[2]==0 && result.needs_readers);
    }
    {
        tq::RamKvSnapshot snapshot;
        snapshot.blocks.push_back({image(13),{},{1024,512,256},0,1,tq::RamKvTier::Ssd});
        tq::RamKvStoreConfig config;config.pressure.canonical_tier=tq::RamKvTier::Tq4;
        tq::RamKvStore imported(config,allocate,encode,[](const auto&){});
        imported.restore(snapshot,256,false);
        assert(imported.resident(0)->key_bits==4 && imported.resident(0)->bytes()==256);
    }
    auto session=disk(root/"session"),prefix=disk(root/"prefix");
    {
        tq::RamKvStoreConfig config;config.pressure.canonical_tier=tq::RamKvTier::Tq4;
        config.pressure.ssd_enabled=true;config.session_ssd=disk(root/"selected-leases");
        tq::RamKvStore selected(config,allocate,encode,[](const auto&){});
        for(unsigned id=0;id<3;++id)selected.append(image(id+7,256,4),{1024,512,256},1);
        auto leases=selected.lease({0,1},10);assert(leases.size()==2 && leases[0].generation==selected.generation(0));
        const auto pressure=selected.maintain(256);
        assert(!pressure.fits() && pressure.needs_readers && selected.resident(0) && selected.resident(1) && !selected.resident(2));
        try{(void)selected.lease({0,3},100);assert(false);}catch(const std::out_of_range&){}
        leases.clear();assert(selected.maintain(256).fits());
        assert(!selected.resident(0) && selected.resident(1));
        leases=selected.lease({0,1,1},20);assert(leases.size()==2 && leases[0].id==1 && leases[1].image==leases[0].image);
        const auto old=leases[0].generation;selected.truncate(0);selected.append(image(19,256,4),{1024,512,256},1);
        assert(leases[0].image->key[0]==8 && selected.resident(0)->key[0]==19 && selected.generation(0)>old);
    }
    {
        tq::RamKvStoreConfig config;config.pressure.canonical_tier=tq::RamKvTier::Tq4;
        config.pressure.ssd_enabled=true;config.session_ssd=disk(root/"index-publication");
        bool reject=false;
        tq::RamKvStore indexed(config,allocate,encode,[&](const auto&){if(reject)throw std::runtime_error("reject index publication");});
        indexed.append(image(7,256,4),{1024,512,256},1);indexed.append(image(8,256,4),{1024,512,256},1);
        assert(indexed.maintain(256).fits());indexed.touch({0},2);
        reject=true;try{(void)indexed.read(0,256);assert(false);}catch(const std::runtime_error&){}
        assert(!indexed.resident(0) && indexed.resident(1));reject=false;
        assert(indexed.read(0,256)->key[0]==7 && !indexed.resident(1));
        indexed.truncate(0);indexed.append(image(9,256,4),{1024,512,256},1);
        indexed.touch({0},3);assert(indexed.maintain(0).fits());
        assert(indexed.read(0,256)->key[0]==9);
    }
    {
        std::promise<void> started,resume;auto ready=started.get_future();auto resumed=resume.get_future().share();
        std::atomic<bool> first{true};auto seed=image(77,256,4);
        auto packet=std::vector<uint8_t>(seed->key,seed->key+seed->key_bytes);packet.insert(packet.end(),seed->value,seed->value+seed->value_bytes);seed.reset();
        auto backing=std::make_shared<tq::RamKvBacking>();backing->key_bytes=backing->value_bytes=128;
        backing->key_bits=backing->value_bits=4;backing->tier=tq::RamKvTier::Tq4;
        backing->read=[&]{if(first.exchange(false)){started.set_value();resumed.wait();}return packet;};
        tq::RamKvStoreConfig config;config.pressure.canonical_tier=tq::RamKvTier::Tq4;
        config.pressure.ssd_enabled=true;config.prefix_enabled=true;config.session_ssd=session;
        tq::RamKvStore indexed(config,allocate,encode,[](const auto&){});
        indexed.append(image(77,256,4),{1024,512,256},1,backing);indexed.append(image(77,256,4),{1024,512,256},1,backing);
        assert(indexed.maintain(256).fits());indexed.touch({0},2);
        auto read=std::async(std::launch::async,[&]{return indexed.read(0,256);});
        ready.wait();indexed.touch({0},50);resume.set_value();auto image=read.get();
        indexed.touch({0},60);indexed.touch({1},40);(void)indexed.read(1,256);
        assert(indexed.resident(0)==image && !indexed.resident(1));
    }
    auto first=image(0);std::vector<std::uint8_t> packed(first->key,first->key+first->key_bytes);
    packed.insert(packed.end(),first->value,first->value+first->value_bytes);
    auto backing=std::make_shared<tq::RamKvBacking>();backing->store=prefix;backing->block=prefix->write(packed);
    backing->key_bytes=backing->value_bytes=512;backing->key_bits=backing->value_bits=0;backing->tier=tq::RamKvTier::Fp16;
    prefix->flush();
    tq::RamKvStoreConfig c;c.pressure.ssd_enabled=true;c.prefix_enabled=true;c.session_ssd=session;
    std::size_t publications=0;bool fail=false;
    tq::RamKvStore store(c,allocate,encode,[&](const auto& changes){if(fail)throw std::runtime_error("publication failure");publications+=changes.size();});
    for(unsigned id=0;id<10240;++id)assert(store.append(image(id),{1024,512,256},1,id==0?backing:nullptr)==id);
    first.reset();
    const auto before=store.stats();assert(before.ram_bytes==10240*1024 && before.tokens[0]==40960);
    const auto result=store.maintain(2<<20);const auto after=store.stats();
    assert(result.fits() && result.ram_bytes==after.ram_bytes && after.ram_bytes<=2<<20);
    assert(after.tokens[0]>=2048 && after.tokens[1]>=6144 && after.tokens[3]>0);
    assert(after.tokens[0]<2048+2048 && after.tokens[1]<6144+2048);
    {
        auto snapshot=store.snapshot(10240);
        auto restore_config=c;restore_config.session_ssd=disk(root/"restore-policy");
        tq::RamKvStore restored(restore_config,allocate,encode,[](const auto&){});
        restored.restore(*snapshot,1536<<10,true);
        const auto stats=restored.stats();
        assert(stats.blocks==10240 && stats.ram_bytes<=1536<<10);
        assert(stats.tokens[0]>=2048 && stats.tokens[1]>=6144 && stats.tokens[3]>0);
        assert(restored.read(0,0,false)->key[0]==0);
        try {restored.restore(*snapshot,64<<10,true);assert(false);}catch(const std::runtime_error&){}
        assert(restored.stats().blocks==10240);
    }
    assert(after.prefix_reused_bytes==256 && store.backing(0)==backing);
    assert(!store.resident(0));
    session->flush();assert(session->stats().written_bytes==after.ssd_written_bytes);
    // Borrowed prefix storage is kept alive without any second SSD write.
    const auto prefix_written=prefix->stats().written_bytes;
    prefix.reset();backing.reset();
    store.touch({0},100);const auto saved=store.stats();
    auto restored=store.read(0,2<<20);assert(restored && restored->key[0]==0 && restored->value[0]==7);
    assert(store.resident(0) && store.stats().ram_bytes<=2<<20 && store.stats().promoted_blocks==1);
    assert(store.stats().spilled_blocks>saved.spilled_blocks);
    assert(store.backing(0)->store->stats().written_bytes==prefix_written);
    restored.reset();
    // A caller-held reader cannot be quantized or selected as an admission victim.
    const auto pinned=store.resident(0);store.touch({0},0);
    store.maintain(1<<20);assert(store.resident(0)==pinned);
    const auto old_generation=store.generation(0);
    const auto offered_prefix=store.backing(0);assert(offered_prefix);
    store.truncate(0);assert(store.stats().ram_bytes==0);
    store.append(image(17),{1024,512,256},1);assert(store.generation(0)>old_generation);
    fail=true;
    try {store.append(image(18),{1024,512,256},1);assert(false);}catch(const std::runtime_error&){}
    assert(store.stats().blocks==1 && store.resident(0)->key[0]==17);fail=false;
    // With prefix disabled, an offered prefix extent is ignored. Spilling
    // and re-reading use only this session's SSD store.
    tq::RamKvStoreConfig fixed;
    fixed.pressure.canonical_tier=tq::RamKvTier::Tq4;fixed.pressure.ssd_enabled=true;fixed.session_ssd=session;
    tq::RamKvStore no_prefix(fixed,allocate,encode,[](const auto&){});
    no_prefix.append(image(9,256,4),{1024,512,256},1,offered_prefix);
    assert(!no_prefix.backing(0));assert(no_prefix.maintain(0).fits());
    assert(no_prefix.backing(0)->store==session);
    assert(no_prefix.read(0,0)->key[0]==9 && !no_prefix.resident(0));
    assert(no_prefix.stats().prefix_reused_bytes==0);
    // New online blocks accumulate in Affine8. A TQ4 target consumes one
    // complete2048-token batch and leaves the partial next batch untouched.
    tq::RamKvStoreConfig batched;batched.pressure.canonical_tier=tq::RamKvTier::Tq4;
    tq::RamKvStore pending(batched,allocate,encode,[](const auto&){});
    for(unsigned i=0;i<513;++i)pending.append_at_tier(image(i,512,254),{1024,512,256},tq::RamKvTier::Affine8,i);
    const auto batch=pending.maintain(512*256+512);
    assert(batch.fits() && pending.stats().tokens[1]==4 && pending.stats().tokens[2]==2048);
    assert((batch.tokens==std::array<std::uint64_t,4>{0,4,2048,0}));
    assert(pending.stats().quantized_blocks==512 && pending.resident(512)->key_bits==254);
    pending.truncate(0);
    const auto empty=pending.maintain(0);
    assert(empty.fits() && empty.ram_bytes==0 && (empty.tokens==std::array<std::uint64_t,4>{}));
    // A touch while the background codec runs invalidates its LRU decision.
    std::mutex mutex;std::condition_variable cv;bool entered=false,resume=false;
    tq::RamKvStoreConfig tiny;tiny.pressure.quantize_tokens=4;
    tq::RamKvStore concurrent(tiny,allocate,[&](const auto& src,auto tier){
        {std::unique_lock<std::mutex> lock(mutex);entered=true;cv.notify_all();cv.wait(lock,[&]{return resume;});}
        return encode(src,tier);
    },[](const auto&){});
    for(unsigned i=0;i<4;++i)concurrent.append(image(i),{1024,512,256},1);
    auto task=std::async(std::launch::async,[&]{return concurrent.maintain(2048);});
    {std::unique_lock<std::mutex> lock(mutex);cv.wait(lock,[&]{return entered;});}
    concurrent.touch({0,1,2,3},200);
    {std::lock_guard<std::mutex> lock(mutex);resume=true;cv.notify_all();}
    const auto interrupted=task.get();assert(!interrupted.fits() && interrupted.ram_bytes==4096 && interrupted.needs_readers);
    assert(concurrent.stats().quantized_blocks==0);
    // Growth during a completed batch requires another plan. It must not be
    // reported as an impossible protected-floor budget while savings remain.
    entered=false;resume=false;
    tq::RamKvStore growing(tiny,allocate,[&](const auto& src,auto tier){
        {std::unique_lock<std::mutex> lock(mutex);entered=true;cv.notify_all();cv.wait(lock,[&]{return resume;});}
        return encode(src,tier);
    },[](const auto&){});
    for(unsigned i=0;i<4;++i)growing.append(image(i),{1024,512,256},1);
    auto growth=std::async(std::launch::async,[&]{return growing.maintain(3072);});
    {std::unique_lock<std::mutex> lock(mutex);cv.wait(lock,[&]{return entered;});}
    growing.append(image(55),{1024,512,256},2);
    {std::lock_guard<std::mutex> lock(mutex);resume=true;cv.notify_all();}
    assert(growth.get().fits() && growing.stats().ram_bytes<=3072 && growing.stats().blocks==5);
    // An optional TQ4 conversion target must never spill SSD data when a
    // concurrent append leaves less than one complete next codec batch.
    entered=false;resume=false;
    auto optional_config=fixed;optional_config.pressure.quantize_tokens=2048;
    tq::RamKvStore optional(optional_config,allocate,[&](const auto& src,auto tier){
        {std::unique_lock<std::mutex> lock(mutex);entered=true;cv.notify_all();cv.wait(lock,[&]{return resume;});}
        return encode(src,tier);
    },[](const auto&){});
    for(unsigned i=0;i<512;++i)optional.append_at_tier(image(i,512,254),{1024,512,256},tq::RamKvTier::Affine8,i);
    auto conversion=std::async(std::launch::async,[&]{return optional.maintain(512*256,false);});
    {std::unique_lock<std::mutex> lock(mutex);cv.wait(lock,[&]{return entered;});}
    optional.append_at_tier(image(99,512,254),{1024,512,256},tq::RamKvTier::Affine8,1024);
    {std::lock_guard<std::mutex> lock(mutex);resume=true;cv.notify_all();}
    const auto partial=conversion.get();
    assert(!partial.fits() && optional.stats().tokens[1]==4 && optional.stats().tokens[2]==2048);
    assert(optional.stats().spilled_blocks==0 && optional.stats().ssd_written_bytes==0);
    assert(optional.maintain(512*256+512).fits());
#if defined(TYLOINFER_TEST_PREFIX_BORROW)
    // Borrow an actual paged prefix's packed KV range. The open file lease
    // survives replacement, cache clear and destruction without a RAM copy
    // of the complete prefix or any duplicate session SSD write.
    mfq::cache::PagedPrefixCacheConfig paged_config;
    paged_config.cache_dir=root/"paged";paged_config.compatibility_key="qsa-ram-range-test";
    paged_config.block_size_tokens=4;paged_config.min_disk_free_bytes=0;
    auto paged=std::make_unique<mfq::cache::PagedPrefixCache>(paged_config);
    auto packed_image=image(37,256,4);
    auto payload=std::make_shared<std::vector<std::uint8_t>>(4096,0x6a);
    std::memcpy(payload->data()+101,packed_image->key,128);std::memcpy(payload->data()+229,packed_image->value,128);
    const std::int64_t tokens[]{11,12,13,14};
    const auto hash=paged->store({},tokens,4,payload);paged->flush();
    auto range=paged->borrow(hash);assert(range && range->payload_bytes==payload->size());
    assert(range->read_range(101,256)==std::vector<std::uint8_t>(payload->begin()+101,payload->begin()+357));
    try {range->read_range(4090,7);assert(false);}catch(const std::out_of_range&){}
    auto borrowed=std::make_shared<tq::RamKvBacking>();borrowed->key_bytes=borrowed->value_bytes=128;
    borrowed->key_bits=borrowed->value_bits=4;borrowed->tier=tq::RamKvTier::Tq4;
    borrowed->read=[range]{return range->read_range(101,256);};
    auto borrowed_config=fixed;borrowed_config.prefix_enabled=true;
    tq::RamKvStore borrowed_store(borrowed_config,allocate,encode,[](const auto&){});
    borrowed_store.append(packed_image,{1024,512,256},1,borrowed);packed_image.reset();
    const auto session_bytes=session->stats().written_bytes;
    assert(borrowed_store.maintain(0).fits() && borrowed_store.stats().prefix_reused_bytes==256);
    assert(borrowed_store.stats().ssd_written_bytes==0 && session->stats().written_bytes==session_bytes);
    auto replacement=std::make_shared<std::vector<std::uint8_t>>(4096,0x55);
    paged->replace({},tokens,4,replacement);paged->flush();
    assert(paged->borrow(hash)->read_range(101,1)[0]==0x55);
    paged->clear();paged.reset();payload.reset();range.reset();borrowed.reset();
    auto borrowed_image=borrowed_store.read(0,0,false);
    assert(borrowed_image->key[0]==37 && borrowed_image->value[0]==44);
    assert(!borrowed_store.resident(0) && borrowed_store.stats().ssd_read_bytes==256);
    auto snapshot=borrowed_store.snapshot(1);const auto captured_generation=borrowed_store.generation(0);
    borrowed_store.truncate(0);borrowed_store.restore(*snapshot,0,true);
    assert(borrowed_store.stats().blocks==1 && borrowed_store.generation(0)>captured_generation);
    assert(borrowed_store.stats().ssd_written_bytes==0 && borrowed_store.read(0,0,false)->key[0]==37);
    tq::RamKvStore session_only(fixed,allocate,encode,[](const auto&){});
    session_only.restore(*snapshot,0,false);
    assert(session_only.backing(0)->store==session && session_only.stats().ssd_written_bytes==256);
    assert(session_only.stats().prefix_reused_bytes==0 && session_only.read(0,0,false)->key[0]==37);
    // Publish a prefix backing to an immutable live version. A later reset
    // and ID reuse must never attach that old backing to the new KV block.
    tq::RamKvStore live(fixed,allocate,encode,[](const auto&){});
    live.append(image(37,256,4),{1024,512,256},1);auto live_snapshot=live.snapshot(1);
    auto durable=snapshot->blocks[0].backing;
    live_snapshot->bind_prefix(0,durable);assert(live.backing(0)==durable && !live_snapshot->blocks[0].image);
    const auto written=session->stats().written_bytes;
    assert(live.maintain(0).fits() && live.stats().ssd_written_bytes==0 && session->stats().written_bytes==written);
    live.truncate(0);live.append(image(9,256,4),{1024,512,256},2);
    live_snapshot->bind_prefix(0,durable);assert(!live.backing(0) && live.resident(0)->key[0]==9);
#endif
    std::cout<<"QSA RAM pressure execution, real SSD/prefix reuse, hot promotion, reader leases, rollback generations and concurrent touch PASS publications="<<publications<<"\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
