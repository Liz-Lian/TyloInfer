#ifdef NDEBUG
#undef NDEBUG
#endif
#include "tyloinfer/qsa_adaptive_cache.h"
#include <cassert>
#include <condition_variable>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>

using namespace tyloinfer::qsa;

static MicroblockImage image(Precision precision) {
    MicroblockImage result;result.key_bits=precision.key_bits;result.value_bits=precision.value_bits;
    for(auto pair:{std::make_pair(&result.key,precision.key_bits),std::make_pair(&result.value,precision.value_bits)}) {
        if(pair.second==8) {
            pair.first->resize(24+8*256);
            for(int row=0;row<8;++row)(*pair.first)[2*row+1]=0x3c;
            continue;
        }
        // A valid uniform image with eight independent closed TQ rings.
        pair.first->resize(20+8*32*pair.second);
        for(int row=0;row<8;++row){(*pair.first)[2*row]=0;(*pair.first)[2*row+1]=0x3c;}
        for(int byte=16;byte<20;++byte)(*pair.first)[byte]=0x77;
    }
    if(precision.key_bits==8)result.key_bits=affine8_tag;
    if(precision.value_bits==8)result.value_bits=affine8_tag;
    return result;
}

static void check_batch_append() {
    AdaptiveCacheConfig config;config.rows_per_kind=8;config.policy={8,128,128};
    const auto canonical=image({8,8});
    const auto encode=[](const MicroblockImage&,Precision p){return image(p);};
    std::map<std::size_t,std::uint8_t> singles,batches;
    std::size_t publications=0;
    AdaptiveCache single(config,encode,[&](const auto& changes){
        for(const auto& c:changes)singles[c.block]=c.tier;
    });
    AdaptiveCache batch(config,encode,[&](const auto& changes){
        ++publications;for(const auto& c:changes)batches[c.block]=c.tier;
    });
    assert(batch.append_batch({},0)==0);
    std::size_t total=0;
    for(int count:{1,3,9,65,129,1024}) {
        for(int i=0;i<count;++i)assert(single.append(canonical,(total+i)*4)==total+i);
        assert(batch.append_batch(std::vector<MicroblockImage>(count,canonical),total*4)==total);
        total+=count;
    }
    assert(publications==6 && singles==batches && batch.stats().blocks==total);
    for(std::size_t i=0;i<total;++i) {
        const auto a=single.reconstruction_image(i),b=batch.reconstruction_image(i);
        assert(a.key==b.key && a.value==b.value);
    }
    auto invalid=std::vector<MicroblockImage>(2,canonical);invalid.back().key_bits=253;
    bool rejected=false;
    try{batch.append_batch(std::move(invalid),total*4);}catch(const std::invalid_argument&){rejected=true;}
    assert(rejected && batch.stats().blocks==total && publications==6);
    for(auto position:{std::uint64_t(1),std::uint64_t(0),std::numeric_limits<std::uint64_t>::max()-3}) {
        rejected=false;
        try{batch.append_batch({canonical},position);}catch(const std::invalid_argument&){rejected=true;}
        assert(rejected && batch.stats().blocks==total);
    }
    const auto costs=batch.tier_bytes();
    const auto budget=(total-2)*costs.front()+2*costs.back();
    auto hits=std::vector<double>(total,1.),mass=std::vector<double>(total,.5);
    single.update(total*4,budget,hits,mass);batch.update(total*4,budget,hits,mass);
    single.wait();batch.wait();
    assert(single.stats().resident_bytes==batch.stats().resident_bytes);
    assert(single.stats().blocks_per_tier==batch.stats().blocks_per_tier && singles==batches);
    // wait() includes destruction of the retired host images.
    std::weak_ptr<const MicroblockImage> old;
    AdaptiveCache retirement(config,encode,[&](const auto& changes){
        for(const auto& c:changes)if(c.block==0)old=c.image;
    });
    retirement.append_batch(std::vector<MicroblockImage>(32,canonical),0);
    assert(!old.expired());retirement.truncate(0);retirement.wait();
    assert(old.expired() && retirement.stats().blocks==0);
    for(int round=0;round<3;++round) {
        retirement.append_batch(std::vector<MicroblockImage>(17,canonical),0);
        retirement.truncate(0);
    }
    retirement.wait();assert(old.expired());
    auto small_window=config;small_window.io_buffer_bytes=4*costs.back();
    AdaptiveCache bounded(small_window,encode,[&](const auto& changes){
        for(const auto& c:changes)if(c.block==0)old=c.image;
    });
    bounded.append_batch(std::vector<MicroblockImage>(32,canonical),0);
    bounded.truncate(0);assert(old.expired());bounded.wait();
    auto oversized=canonical;oversized.key.reserve(3*config.io_buffer_bytes);
    std::vector<MicroblockImage> oversized_batch;oversized_batch.push_back(std::move(oversized));
    retirement.append_batch(std::move(oversized_batch),0);
    retirement.truncate(0);assert(old.expired());retirement.wait();
    bool fail=true;
    AdaptiveCache throwing(config,encode,[&](const auto&){if(fail)throw std::runtime_error("publisher failure");});
    rejected=false;
    try{throwing.append_batch({canonical,canonical},0);}catch(const std::runtime_error&){rejected=true;}
    assert(rejected && throwing.stats().blocks==0);
    fail=false;assert(throwing.append_batch({canonical,canonical},0)==0);
    std::cout<<"QSA batch/single parity, atomic publication, positions and host retirement PASS\n";
}
int main(int argc,char** argv) {
    assert(argc==2);
    check_batch_append();
    AdaptiveCacheConfig config;config.directory=argv[1];config.rows_per_kind=8;config.ssd_cache=true;
    config.policy={8,128,128};
    std::map<std::size_t,std::uint8_t> published;
    std::mutex mutex;std::size_t encodes=0;
    const auto canonical=image({8,8});
    AdaptiveCache cache(config,[&](const MicroblockImage& source,Precision p) {
        // A controlled codec callback verifies that demotion and subsequent
        // promotion always receive the canonical bytes, never the cold image.
        assert(source.key_bits==affine8_tag && source.value_bits==affine8_tag);
        assert(source.key==canonical.key && source.value==canonical.value);
        ++encodes;return image(p);
    },[&](const std::vector<BlockReplacement>& changes) {
        std::lock_guard<std::mutex> lock(mutex);
        for(const auto& change:changes)published[change.block]=change.tier;
    });
    auto cost=cache.tier_bytes();
    for(int first:{0,20,40}) {
        const int count=first==40?24:20;
        assert(cache.append_batch(std::vector<MicroblockImage>(count,canonical),first*4)==std::size_t(first));
    }
    cache.wait();
    auto initial=cache.stats();
    assert(initial.blocks==64 && initial.blocks_per_tier[5]==64);
    assert(initial.canonical.written_bytes==64*cost[5] && initial.canonical.hot_bytes==0);
    std::vector<double> hits(64),mass(64);hits[0]=100;mass[0]=80;
    const auto budget=62*cost[0]+2*cost[5]+cost[4]-cost[0];
    // Observations arriving before the scheduled interval must survive.
    cache.update(64,budget,hits,mass);
    hits.assign(64,0);mass.assign(64,0);
    cache.update(256,budget,hits,mass);cache.wait();
    auto first=cache.stats();
    assert(first.resident_bytes<=budget && first.demotions>0 && first.updates==1);
    assert(published[0]==4 && published[62]==5 && published[63]==5);
    for(int i=64;i<96;++i)cache.append(canonical,i*4);
    hits.assign(96,0);mass.assign(96,0);hits[4]=100;mass[4]=90;
    const auto budget2=94*cost[0]+2*cost[5]+cost[4]-cost[0];
    cache.update(384,budget2,hits,mass);cache.wait();
    auto drift=cache.stats();
    assert(drift.resident_bytes<=budget2 && drift.promotions>0 && drift.updates==2);
    assert(published[4]>0 && published[0]<4 && published[4]>=published[0]);
    assert(published[94]==5 && published[95]==5);
    assert(drift.canonical_read_bytes>0 && drift.canonical.read_bytes>0 && encodes>0);
    const auto original=cache.canonical_image(0);
    assert(original.key==canonical.key && original.value==canonical.value);
    bool outside=false;try{(void)cache.canonical_image(96);}catch(const std::out_of_range&){outside=true;}
    assert(outside);
    cache.truncate(64);
    assert(cache.stats().blocks==64);
    // Rollback must not wait for a background fitter. Kept blocks may still
    // receive an update, while reused tail IDs must reject stale publications.
    std::mutex gate_mutex;std::condition_variable gate;
    bool entered=false,released=false;
    std::map<std::size_t,std::uint8_t> rollback_published;
    AdaptiveCache rollback(config,AdaptiveCache::ReencodeBatch([&](
        const std::vector<MicroblockImage>& source,const std::vector<Precision>& target) {
        {
            std::unique_lock<std::mutex> lock(gate_mutex);entered=true;gate.notify_all();
            gate.wait(lock,[&]{return released;});
        }
        std::vector<MicroblockImage> result;
        for(auto precision:target) {
            assert(precision.key_bits<8 && precision.value_bits<8);
            result.push_back(image(precision));
        }
        return result;
    }),[&](const std::vector<BlockReplacement>& changes) {
        for(const auto& change:changes)rollback_published[change.block]=change.tier;
    });
    for(int i=0;i<64;++i)rollback.append(canonical,i*4);
    hits.assign(64,0);mass.assign(64,0);
    rollback.update(256,budget,hits,mass);
    {std::unique_lock<std::mutex> lock(gate_mutex);gate.wait(lock,[&]{return entered;});}
    rollback.truncate(4);
    for(int i=4;i<64;++i)rollback.append(canonical,i*4);
    {std::lock_guard<std::mutex> lock(gate_mutex);released=true;gate.notify_all();}
    rollback.wait();
    for(int i=4;i<64;++i)assert(rollback_published[i]==5);
    assert(rollback.stats().demotions>0);
    {std::lock_guard<std::mutex> lock(gate_mutex);entered=false;released=false;}
    rollback.update(384,budget,std::vector<double>(64),std::vector<double>(64));
    {std::unique_lock<std::mutex> lock(gate_mutex);gate.wait(lock,[&]{return entered;});}
    rollback.truncate(0);
    assert(rollback.append_batch(std::vector<MicroblockImage>(64,canonical),0)==0);
    {std::lock_guard<std::mutex> lock(gate_mutex);released=true;gate.notify_all();}
    rollback.wait();
    assert(rollback.stats().blocks_per_tier[5]==64);
    for(int i=0;i<64;++i)assert(rollback_published[i]==5);
    auto fast_config=config;fast_config.tq_state_bits=14;
    std::size_t fast_publications=0;
    AdaptiveCache fast(fast_config,[](const MicroblockImage& source,Precision p){
        assert(source.key_state_bits==16 && source.value_state_bits==16);
        auto result=image(p);result.key_state_bits=result.value_state_bits=14;return result;
    },[&](const std::vector<BlockReplacement>& changes){
        for(const auto& change:changes) {
            assert(change.image->key_state_bits==(change.tier==5?16:14));
            assert(change.image->value_state_bits==(change.tier==5?16:14));
            fast_publications+=change.tier<5;
        }
    });
    for(int i=0;i<64;++i)fast.append(canonical,i*4);
    fast.update(256,budget,std::vector<double>(64),std::vector<double>(64));fast.wait();
    assert(fast_publications>0);
    AdaptiveCache invalid_profile(fast_config,[](const MicroblockImage&,Precision p){return image(p);},
        [](const std::vector<BlockReplacement>&){});
    for(int i=0;i<64;++i)invalid_profile.append(canonical,i*4);
    invalid_profile.update(256,budget,std::vector<double>(64),std::vector<double>(64));
    bool rejected=false;try{invalid_profile.wait();}catch(const std::invalid_argument&){rejected=true;}
    assert(rejected);
    AdaptiveCache late(config,[](const MicroblockImage&,Precision p){return image(p);},
        [](const std::vector<BlockReplacement>&){});
    for(int i=0;i<96;++i)late.append(canonical,i*4);
    // A worker handling a queued old observation may see a newer append.
    late.update(256,budget,std::vector<double>(96),std::vector<double>(96));late.wait();
    assert(late.stats().blocks_per_tier.back()>=32 && late.stats().demotions>0);
    auto ram_config=config;ram_config.ssd_cache=false;ram_config.directory=std::string(argv[1])+"-no-ssd";
    std::size_t from_cold=0;
    AdaptiveCache ram(ram_config,[&](const MicroblockImage& source,Precision p) {
        from_cold+=source.key_bits!=affine8_tag;
        return image(p);
    },[](const std::vector<BlockReplacement>&){});
    for(int i=0;i<64;++i)ram.append(canonical,i*4);
    ram.update(256,budget,std::vector<double>(64),std::vector<double>(64));ram.wait();
    const auto low=ram.stats();assert(low.demotions>0 && low.promotions==0);
    std::vector<std::uint8_t> before;
    for(int i=0;i<64;++i)before.push_back(ram.reconstruction_image(i).key_bits);
    // More capacity and a drifted query distribution cannot recreate originals.
    auto drift_hits=std::vector<double>(64,1e6);
    ram.update(384,64*cost.back(),drift_hits,drift_hits);ram.wait();
    for(int i=0;i<64;++i)assert(ram.reconstruction_image(i).key_bits==before[i]);
    assert(ram.stats().promotions==0 && ram.stats().canonical_read_bytes==0);
    assert(ram.stats().canonical.written_bytes==0 && !std::filesystem::exists(ram_config.directory));
    // Further pressure must encode the current compressed image, then release
    // the former image instead of retaining a hidden high-precision replica.
    for(int i=64;i<128;++i)ram.append(canonical,i*4);
    const auto minimum=126*cost.front()+2*cost.back();
    ram.update(512,minimum,std::vector<double>(128,1e6),std::vector<double>(128,1e6));ram.wait();
    assert(from_cold>0 && ram.stats().resident_bytes<=minimum);
    bool no_original=false;try{ram.canonical_image(0);}catch(const std::logic_error&){no_original=true;}
    assert(no_original);
    std::cout<<"QSA SSD disabled: no files, no promotions, current-image demotion PASS\n";
    std::cout<<"QSA canonical SSD, TQ14 profile validation, async requantization and truncation PASS\n";
}
