#include "qsa_reader_fixture.h"
#include "tyloinfer/qsa_ram_store.h"
#include "tyloinfer/disk_budget.h"
#include "tyloinfer/file_read_window.h"
#if defined(TYLOINFER_TEST_PREFIX_BORROW)
#include "mfq_paged_prefix_cache.h"
#endif
#include <algorithm>
#include <cassert>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <fstream>
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
void check_deferred_reads(const std::filesystem::path& root) {
    tq::RamKvStoreConfig cfg;cfg.pressure.canonical_tier=tq::RamKvTier::Tq4;
    cfg.pressure.ssd_enabled=true;cfg.prefix_enabled=true;cfg.session_ssd=disk(root);
    bool device_waiting=false;unsigned allocations=0,encodes=0;
    tq::RamKvStore store(cfg,[&](const auto& source){
        if(device_waiting)throw std::runtime_error("Backend allocation while graph waits for CPU");
        ++allocations;return allocate(source);
    },[&](const auto& source,auto tier){
        if(device_waiting)throw std::runtime_error("Backend encoding while graph waits for CPU");
        ++encodes;return encode(source,tier);
    },[](const auto&){});
    store.append(image(19,256,4),{1024,512,256},1);
    assert(store.stats().resident_target_bytes==256);
    store.truncate(0);assert(store.stats().resident_target_bytes==0);
    store.append(image(19,256,4),{1024,512,256},1);
    assert(store.maintain(0).fits() && store.graph_reads_supported());
    store.touch({0},90);const auto source=store.source(0);
    const auto before_allocations=allocations,before_encodes=encodes;
    device_waiting=true;const auto bytes=store.read_cpu(source);device_waiting=false;
    assert(bytes.key.front()==19 && !store.resident(0));
    assert(allocations==before_allocations && encodes==before_encodes && store.stats().ssd_read_bytes==256);
    const auto reads=cfg.session_ssd->stats().read_bytes;
    store.admit_read(0,source.generation,bytes,0);
    store.admit_read(0,source.generation,bytes,255);
    assert(allocations==before_allocations && !store.resident(0));
    store.append_at_tier(image(23,1024,0),{1024,512,256},tq::RamKvTier::Fp16,1);
    store.admit_read(0,source.generation,bytes,1024);
    assert(allocations==before_allocations && !store.resident(0) && store.stats().ram_bytes==1024);
    store.truncate(1);
    store.append(image(31,256,4),{1024,512,256},1);
    {
        auto selected=store.lease({1},40);
        store.admit_read(0,source.generation,bytes,256);
        assert(allocations==before_allocations && !store.resident(0));
    }
    store.admit_read(0,source.generation,bytes,256);
    assert(store.resident(0) && store.resident(0)->key[0]==19 && store.stats().promoted_blocks==1);
    assert(allocations==before_allocations+1 && !store.resident(1));
    store.truncate(1);
    assert(cfg.session_ssd->stats().read_bytes==reads && store.stats().ssd_read_bytes==256);
    assert(store.maintain(0).fits());const auto old=store.source(0);
    device_waiting=true;const auto stale=store.read_cpu(old);device_waiting=false;
    store.truncate(0);store.append(image(71,256,4),{1024,512,256},99);
    store.admit_read(0,old.generation,stale,256);
    assert(store.resident(0)->key[0]==71);
    store.truncate(0);
    auto fp16=std::make_shared<tq::RamKvBacking>();fp16->key_bytes=fp16->value_bytes=512;
    fp16->tier=tq::RamKvTier::Fp16;
    qsa_test::reader(*fp16).read_range=[](std::size_t,std::size_t){return std::vector<std::uint8_t>(1024,37);};
    store.append(image(37,256,4),{1024,512,256},1,fp16);
    assert(store.maintain(0).fits() && store.graph_reads_supported());
    const auto packed=store.snapshot(1);store.truncate(0);assert(store.graph_reads_supported());
    store.restore(*packed,0,true);assert(store.graph_reads_supported());
    const auto original=store.read(0,256);
    assert(original->key[0]==37 && original->key_bits==0 && original->bytes()==1024);
    assert(!store.resident(0) && store.graph_reads_supported());
    assert(store.maintain(0).fits() && store.graph_reads_supported());
    store.truncate(0);assert(store.graph_reads_supported());
    std::cout<<"Deferred SSD reads: no backend allocation/encoding, single read/admission, stale generation and original-precision prefix eligibility PASS\n";
}
void check_session_read_window(const std::filesystem::path& root) {
    mfq::cache::QsaKvStoreConfig cfg;cfg.directory=(root/"windows").string();
    cfg.buffer_bytes=cfg.budget_bytes=8192;
    mfq::cache::QsaKvStore store(cfg);
    std::vector<mfq::cache::QsaKvStore::BlockPtr> blocks;
    for(unsigned i=0;i<96;++i)blocks.push_back(store.write(std::vector<std::uint8_t>(1024,std::uint8_t(i))));
    store.flush();
    std::vector<std::vector<std::uint8_t>> destinations(49,std::vector<std::uint8_t>(1056,0xc7));
    std::vector<mfq::cache::QsaKvStore::ReadRequest> requests;
    for(unsigned i=0;i<48;++i)requests.push_back({blocks[2*i],destinations[i].data()+16,1024});
    requests.push_back({blocks[0],destinations.back().data()+16,1024});
    std::reverse(requests.begin(),requests.end());
    assert(!tyloinfer::ScopedFileReadWindow::read_ahead_enabled());
    const auto before=store.stats();store.read_into_batch(requests);const auto random=store.stats();
    {
        tyloinfer::ScopedFileReadWindow recovery;
        assert(tyloinfer::ScopedFileReadWindow::read_ahead_enabled());
        {tyloinfer::ScopedFileReadWindow nested(false);assert(!tyloinfer::ScopedFileReadWindow::read_ahead_enabled());}
        assert(tyloinfer::ScopedFileReadWindow::read_ahead_enabled());
        store.read_into_batch(requests);
    }
    const auto sequential=store.stats();
    assert(!tyloinfer::ScopedFileReadWindow::read_ahead_enabled());
    assert(sequential.reads-random.reads<random.reads-before.reads);
    assert(sequential.read_bytes-random.read_bytes<=2*requests.size()*1024);
    for(unsigned i=0;i<destinations.size();++i) {
        const auto expected=i<48?std::uint8_t(2*i):0;
        for(unsigned at=0;at<1056;++at)assert(destinations[i][at]==(at<16 || at>=1040?0xc7:expected));
    }
    auto invalid=requests;invalid.back().bytes++;
    const auto reads=store.stats().reads;
    try{tyloinfer::ScopedFileReadWindow recovery;store.read_into_batch(invalid);assert(false);}catch(const std::invalid_argument&){}
    assert(store.stats().reads==reads);
    std::cout<<"Session recovery: exact gaps/duplicates/reversed destinations/guards, bounded amplification, nested disable and prevalidation PASS\n";
}

void check_shared_read_source(const std::filesystem::path& root) {
    auto data=std::make_shared<std::vector<std::uint8_t>>(541,17);
    std::fill(data->begin()+7,data->begin()+136,31);
    std::fill(data->begin()+136,data->begin()+263,53);
    std::fill(data->begin()+277,data->begin()+406,71);
    std::fill(data->begin()+406,data->begin()+533,89);
    std::size_t vectors=0,direct=0;
    auto source=std::make_shared<tq::RamKvReadSource>();source->bytes=data->size();
    source->read_range=[data,&vectors](std::size_t at,std::size_t n) {
        ++vectors;return std::vector<std::uint8_t>(data->begin()+at,data->begin()+at+n);
    };
    source->read_range_into=[data,&direct](std::size_t at,std::uint8_t* out,std::size_t n) {
        ++direct;std::memcpy(out,data->data()+at,n);
    };
    tq::RamKvSnapshot state;
    for(const auto at:{std::size_t(7),std::size_t(277)}) {
        auto backing=std::make_shared<tq::RamKvBacking>();backing->source=source;backing->offset=at;
        backing->key_bytes=129;backing->value_bytes=127;backing->key_bits=backing->value_bits=4;
        backing->tier=tq::RamKvTier::Tq4;
        state.blocks.push_back({{},backing,{1024,512,256},0,at,tq::RamKvTier::Ssd});
    }
    auto cfg=tq::RamKvStoreConfig{};cfg.pressure.ssd_enabled=cfg.prefix_enabled=true;
    cfg.session_ssd=disk(root);tq::RamKvStore store(cfg,allocate,encode,[](const auto&){});
    store.restore(state,0,true);assert(!vectors && !direct);
    assert(store.backing(0)->source==store.backing(1)->source);
    std::weak_ptr<const tq::RamKvReadSource> retained=source;
    source.reset();data.reset();state.blocks.clear();
    for(std::size_t id=0;id<2;++id) {
        std::vector<std::uint8_t> output(273,0xa5);const auto block=store.source(id);
        store.read_cpu_into(block,output.data(),271);
        assert(direct==id+1 && !vectors && output[0]==(id?71:31) && output[128]==output[0]);
        assert(output[144]==(id?89:53) && output[270]==output[144] && output[271]==0xa5 && output[272]==0xa5);
        try {store.read_cpu_into(block,output.data(),270);assert(false);}catch(const std::out_of_range&){}
    }
    {
        const auto owner=retained.lock();
        const auto before=direct;owner->read_into(owner->bytes,nullptr,0);assert(direct==before);
        assert(owner->read(owner->bytes,0).empty());
        try {owner->read_into(0,nullptr,1);assert(false);}catch(const std::invalid_argument&){}
        try {owner->read(SIZE_MAX,1);assert(false);}catch(const std::out_of_range&){}
        try {owner->read(owner->bytes-1,2);assert(false);}catch(const std::out_of_range&){}
        auto fallback=*owner;fallback.read_range_into={};std::vector<std::uint8_t> output(256);
        fallback.read_into(7,output.data(),output.size());assert(vectors==1 && output[0]==31 && output[255]==53);
        fallback.read_range=[](std::size_t,std::size_t){return std::vector<std::uint8_t>(1);};
        try {fallback.read(7,256);assert(false);}catch(const std::runtime_error&){}
        try {fallback.read_into(7,output.data(),256);assert(false);}catch(const std::runtime_error&){}
        tq::RamKvReadSource only_direct=*owner;only_direct.read_range={};
        assert(only_direct.read(277,256).front()==71);
    }
    store.truncate(0);assert(retained.expired());
    std::cout<<"Shared immutable byte sources, independent slices, retained lifetime, padded direct/vector reads and bounds PASS\n";
}
void check_mixed_precision_recall(const std::filesystem::path& root) {
    for(unsigned tier=0;tier<3;++tier) {
        const auto size=tier==0?1024:tier==1?512:256,bits=tier==0?0:tier==1?254:4;
        auto original=image(43,size,bits);
        auto backing=std::make_shared<tq::RamKvBacking>();
        backing->key_bytes=backing->value_bytes=size/2;backing->key_bits=backing->value_bits=bits;
        backing->tier=tq::RamKvTier(tier);
        const auto data=std::make_shared<std::vector<std::uint8_t>>(original->key,original->key+size/2);
        data->insert(data->end(),original->value,original->value+size/2);
        qsa_test::reader(*backing).read_range=[data](std::size_t,std::size_t){return *data;};
        tq::RamKvStoreConfig cfg;cfg.pressure.canonical_tier=tq::RamKvTier::Tq4;
        cfg.pressure.quantize_tokens=4;cfg.pressure.ssd_enabled=true;cfg.prefix_enabled=true;
        cfg.session_ssd=disk(root/std::to_string(tier));
        unsigned encodes=0,allocations=0;
        tq::RamKvStore store(cfg,[&](const auto& input){++allocations;return allocate(input);},
            [&](const auto& input,auto target){++encodes;return encode(input,target);},[](const auto&){});
        tq::RamKvSnapshot state;state.blocks.push_back({{},backing,{1024,512,256},0,1,tq::RamKvTier::Ssd});
        store.restore(state,0,true);const auto source=store.source(0);
        const auto raw=store.read_cpu(source);const auto before=allocations;
        store.admit_read(0,source.generation,raw,size-1);
        assert(allocations==before && !store.resident(0));
        auto recalled=store.read(0,size-1,false);
        assert(recalled->bytes()==size && recalled->key_bits==bits && !std::memcmp(recalled->key,original->key,size/2));
        assert(!std::memcmp(recalled->value,original->value,size/2) && !encodes && !store.stats().ssd_written_bytes);
        store.admit_read(0,source.generation,raw,size);
        assert(store.resident(0)->bytes()==size && store.stats().tokens[tier]==4 && store.generation(0)==source.generation);
        assert(store.graph_reads_supported() && !encodes);
        if(tier<2) {
            const auto promoted=store.generation(0);
            assert(store.maintain(256,false).fits() && store.stats().tokens[2]==4);
            const auto compact=store.generation(0);assert(compact>promoted);
            assert(store.maintain(0).fits() && store.generation(0)>compact);
            assert(store.backing(0)==backing && !store.stats().ssd_written_bytes);
            store.admit_read(0,compact,raw,size);assert(!store.resident(0));
        }
    }
    std::cout<<"Mixed FP16/Affine8/TQ4 SSD recall: original bytes, no fitting/copy, actual tier budget, promotion and precision-version invalidation PASS\n";
}
void check_sparse_restore_batches(const std::filesystem::path& root) {
    tq::RamKvStoreConfig config;config.pressure.canonical_tier=tq::RamKvTier::Tq4;
    config.pressure.ssd_enabled=true;config.session_ssd=disk(root);
    constexpr std::size_t blocks=4096,batch=512,budget=batch*2048;
    tq::RamKvSnapshot snapshot;
    for(std::size_t id=0;id<blocks;++id) {
        if(id%8==0)snapshot.blocks.push_back({image(unsigned(id),4096,254),{}, {8192,4096,2048},0,id,tq::RamKvTier::Affine8});
        else {
            auto backing=std::make_shared<tq::RamKvBacking>();backing->key_bytes=backing->value_bytes=1024;
            backing->key_bits=backing->value_bits=4;backing->tier=tq::RamKvTier::Tq4;
            qsa_test::reader(*backing).read_range=[](std::size_t,std::size_t)->std::vector<std::uint8_t>{
                throw std::runtime_error("Sparse restore read an SSD-only block");};
            snapshot.blocks.push_back({{},backing,{8192,4096,2048},0,id,tq::RamKvTier::Ssd});
        }
    }
    std::size_t live=0,peak=0,codec_calls=0,largest=0,visited=0,published=0;
    bool recording=false,reject=false,fail=false;
    struct Counted {std::shared_ptr<const void> owner;std::size_t bytes;std::size_t* live;~Counted(){*live-=bytes;}};
    const auto counted=[&](const tq::MicroblockImage& input) {
        auto result=std::make_shared<tq::RamKvImage>(*allocate(input));const auto bytes=result->bytes();
        live+=bytes;peak=std::max(peak,live);
        result->owner=std::shared_ptr<Counted>(new Counted{result->owner,bytes,&live});return result;
    };
    tq::RamKvStore store(config,counted,[&](const auto& input,tq::RamKvTier tier) {
        ++codec_calls;largest=std::max(largest,input.size());
        assert(input.size()<=batch);
        std::vector<tq::RamKvImagePtr> result;
        for(const auto& source:input) {
            const auto value=image(source->key[0],tier==tq::RamKvTier::Affine8?4096:2048,tier==tq::RamKvTier::Affine8?254:4);
            tq::MicroblockImage bytes;bytes.key.assign(value->key,value->key+value->key_bytes);bytes.value.assign(value->value,value->value+value->value_bytes);
            bytes.key_bits=value->key_bits;bytes.value_bits=value->value_bits;result.push_back(counted(bytes));
        }
        return result;
    },[&](const auto& changes){if(recording){if(reject)throw std::runtime_error("Sparse restore publication failure");
        for(const auto& change:changes)assert(change.id==published++);}});
    const auto observer=[&](std::size_t id){assert(id==visited++ && store.stats().blocks<=id);
        if(fail && id==1024)throw std::runtime_error("Sparse restore companion read failure");};
    const auto restore=[&] {
        recording=false;store.truncate(0);visited=published=codec_calls=largest=0;recording=true;
        store.restore(snapshot,budget,true,false,observer);recording=false;
    };
    restore();
    if(codec_calls>2)throw std::runtime_error("Sparse restore submitted too many codec batches: "+std::to_string(codec_calls));
    assert(largest==batch && visited==blocks && published==blocks);
    assert(store.stats().ram_bytes==budget && store.stats().tokens[2]==batch*4 && store.stats().tokens[3]==(blocks-batch)*4);
    assert(store.stats().ssd_written_bytes==0 && peak<=budget+batch*(8192+4096));
    for(std::size_t id=0;id<blocks;++id)if(id%8==0) {
        const auto expected=image(unsigned(id),2048,4),actual=store.resident(id);
        assert(actual && actual->key_bytes==expected->key_bytes && actual->value_bytes==expected->value_bytes);
        assert(!std::memcmp(actual->key,expected->key,actual->key_bytes) && !std::memcmp(actual->value,expected->value,actual->value_bytes));
    }
    fail=true;try{restore();assert(false);}catch(const std::runtime_error&){}
    assert(store.stats().blocks<=visited && live==0);
    fail=false;reject=true;try{restore();assert(false);}catch(const std::runtime_error&){}
    assert(store.stats().blocks==0 && live==0);
    reject=false;restore();assert(store.stats().blocks==blocks);
    recording=false;store.truncate(0);assert(live==0);
    std::cout<<"Sparse restore batches: bounded codec calls/temporary bytes, ordered observers/publication, exact data and failure/retry PASS\n";
}
void check_restore_observer(const std::filesystem::path& root) {
    tq::RamKvStoreConfig config;config.pressure.canonical_tier=tq::RamKvTier::Tq4;
    config.pressure.ssd_enabled=true;config.prefix_enabled=true;config.session_ssd=disk(root);
    unsigned reads=0;bool fail=false;
    tq::RamKvSnapshot snapshot;
    for(unsigned id=0;id<513;++id) {
        auto backing=std::make_shared<tq::RamKvBacking>();
        backing->key_bytes=backing->value_bytes=128;backing->key_bits=backing->value_bits=4;
        backing->tier=tq::RamKvTier::Tq4;
        qsa_test::reader(*backing).read_range=[&,id](std::size_t,std::size_t){++reads;return std::vector<std::uint8_t>(256,std::uint8_t(id));};
        snapshot.blocks.push_back({{},backing,{1024,512,256},0,id,tq::RamKvTier::Tq4});
    }
    bool reject=false;std::size_t publications=0;
    tq::RamKvStore store(config,allocate,encode,[&](const auto& changes) {
        if(changes.empty())return;
        if(reject)throw std::runtime_error("Injected restore publication failure");
        ++publications;
    });
    std::size_t visited=0;
    const auto observer=[&](std::size_t id) {
        assert(id==visited++);
        // Observers run outside the entry mutex, even for SSD-only blocks.
        assert(store.stats().blocks<=id);
        if(fail && id==512)throw std::runtime_error("Injected companion archive failure");
    };
    store.restore(snapshot,0,true,false,observer);
    assert(visited==513 && reads==0 && store.stats().tokens[3]==513*4 && publications==2);
    visited=0;std::size_t allocations=0;
    const auto batched=[&](const auto& bytes,std::size_t remaining) {
        assert(remaining==(allocations<512?512-allocations:1));++allocations;return allocate(bytes);
    };
    store.restore(snapshot,0,true,false,observer,batched);
    assert(allocations==0 && visited==513 && reads==0);
    visited=0;store.restore(snapshot,513*256,true,false,observer,batched);
    assert(allocations==513 && reads==513 && visited==513);
    reads=0;
    visited=0;store.restore(snapshot,513*256,true,false,observer);
    assert(visited==513 && reads==513 && store.resident(512)->key[0]==0);
    fail=true;visited=0;
    try{store.restore(snapshot,513*256,true,false,observer);assert(false);}catch(const std::runtime_error&){}
    assert(store.stats().blocks==512);
    fail=false;visited=0;store.restore(snapshot,513*256,true,false,observer);
    assert(visited==513 && store.stats().blocks==513 && store.resident(17)->key[0]==17);
    store.truncate(0);reject=true;visited=0;
    try{store.restore(snapshot,513*256,true,false,observer);assert(false);}catch(const std::runtime_error&){}
    assert(store.stats().blocks==0 && store.stats().ram_bytes==0);
    reject=false;visited=0;store.restore(snapshot,513*256,true,false,observer);
    assert(visited==513 && store.stats().blocks==513 && store.maintain(0).fits());
    std::cout<<"Interleaved restore: ordered RAM/SSD observers, no needless SSD reads, failure/retry PASS\n";
}
void check_admit_view(const std::filesystem::path& root) {
    for(unsigned tier=0;tier<3;++tier) {
        const std::size_t size=tier==0?1024:tier==1?512:256,kb=size/2-3,vb=size-kb;
        std::vector<std::uint8_t> payload(size+11,31);
        std::fill(payload.begin()+kb+11,payload.end(),73);
        tq::RamKvImage view;view.key=payload.data();view.value=payload.data()+kb+11;
        view.key_bytes=kb;view.value_bytes=vb;
        view.key_bits=view.value_bits=tier==0?0:tier==1?254:4;
        view.key_state_bits=14;view.value_state_bits=16;
        auto backing=std::make_shared<tq::RamKvBacking>();
        backing->tier=tq::RamKvTier(tier);backing->key_bytes=kb;backing->value_bytes=vb;
        backing->key_bits=view.key_bits;backing->value_bits=view.value_bits;
        backing->key_state_bits=14;backing->value_state_bits=16;
        qsa_test::reader(*backing).read_range=[kb,vb](std::size_t,std::size_t){std::vector<std::uint8_t> bytes(kb+vb,31);std::fill(bytes.begin()+kb,bytes.end(),73);return bytes;};
        tq::RamKvStoreConfig config;config.pressure.ssd_enabled=true;
        config.session_ssd=disk(root/std::to_string(tier));
        unsigned allocations=0;bool fail=false,reject=false;
        tq::RamKvStore store(config,[](const auto&)->tq::RamKvImagePtr {
            throw std::runtime_error("Deferred view admission used temporary K/V arrays");
        },encode,[&](const auto&){if(reject)throw std::runtime_error("View publication failure");},
        [&](const tq::RamKvImage& source)->tq::RamKvImagePtr {
            assert(source.key==view.key && source.value==view.value && source.key_state_bits==14);
            ++allocations;if(fail)throw std::runtime_error("View allocator failure");
            auto bytes=std::make_shared<std::vector<std::uint8_t>>(source.bytes());
            std::memcpy(bytes->data(),source.key,kb);std::memcpy(bytes->data()+kb,source.value,vb);
            auto result=std::make_shared<tq::RamKvImage>(source);result->owner=bytes;
            result->key=bytes->data();result->value=bytes->data()+kb;return result;
        });
        tq::RamKvSnapshot snapshot;snapshot.blocks.push_back({{},backing,{1024,512,256},0,10,tq::RamKvTier::Ssd});
        store.restore(snapshot,0,true);const auto generation=store.generation(0);
        store.admit_read(0,generation,view,size-1);
        store.admit_read(0,generation+1,view,size);
        assert(allocations==0 && !store.resident(0));
        fail=true;
        try{store.admit_read(0,generation,view,size);assert(false);}catch(const std::runtime_error&){}
        assert(allocations==1 && !store.resident(0) && store.stats().ram_bytes==0);
        fail=false;reject=true;
        try{store.admit_read(0,generation,view,size);assert(false);}catch(const std::runtime_error&){}
        assert(allocations==2 && !store.resident(0) && store.generation(0)==generation);
        reject=false;store.admit_read(0,generation,view,size);
        assert(allocations==3 && store.stats().tokens[tier]==4 && store.stats().ram_bytes==size);
        const auto resident=store.resident(0);
        assert(resident->key[0]==31 && resident->value[0]==73 && resident->key_state_bits==14 && resident->value_state_bits==16);
    }
    std::cout<<"Direct admission views: original padded spans, all precisions, no temporary arrays, stale/budget/failure publication PASS\n";
}
void check_read_views(const std::filesystem::path& root) {
    for(bool borrowed:{false,true}) {
        tq::RamKvStoreConfig cfg;cfg.pressure.canonical_tier=tq::RamKvTier::Tq4;
        cfg.pressure.ssd_enabled=true;cfg.prefix_enabled=borrowed;
        mfq::cache::QsaKvStoreConfig disk_config;
        disk_config.directory=(root/(borrowed?"prefix":"session")).string();
        disk_config.buffer_bytes=disk_config.budget_bytes=4096;
        cfg.session_ssd=std::make_shared<mfq::cache::QsaKvStore>(disk_config);
        bool waiting=false,fail=false;unsigned allocations=0;std::size_t backing_capacity=0;
        auto backing=std::make_shared<tq::RamKvBacking>();
        backing->key_bytes=backing->value_bytes=128;backing->key_bits=backing->value_bits=4;
        backing->tier=tq::RamKvTier::Tq4;
        qsa_test::reader(*backing).read_range=[&](std::size_t,std::size_t){
            std::vector<std::uint8_t> bytes;bytes.reserve(4096);bytes.resize(256,19);
            std::fill(bytes.begin()+128,bytes.end(),26);backing_capacity=bytes.capacity();return bytes;
        };
        tq::RamKvStore store(cfg,[&](const auto& source) {
            if(waiting)throw std::runtime_error("Backend allocation during retained CPU read");
            ++allocations;return allocate(source);
        },encode,[&](const auto&) {if(fail)throw std::runtime_error("View publication failed");});
        store.append(image(19,256,4),{1024,512,256},1,borrowed?backing:nullptr);
        assert(store.maintain(0).fits());cfg.session_ssd->flush();
        auto source=store.source(0);const auto allocated=allocations;
        std::size_t retained=0;waiting=true;
        auto view=store.read_cpu_view(source,&retained);waiting=false;
        assert(allocations==allocated && view.owner && retained>=view.bytes());
        if(borrowed)assert(retained==backing_capacity && retained>view.bytes());
        assert(view.key_bytes==128 && view.value_bytes==128 && view.value==view.key+128);
        assert(view.key[0]==19 && view.value[0]==26 && view.key_bits==4 && view.value_bits==4);
        assert(store.stats().ssd_read_bytes==256);
        store.admit_read(0,source.generation,view,255);
        assert(allocations==allocated && !store.resident(0));
        fail=true;
        try {store.admit_read(0,source.generation,view,256);assert(false);}catch(const std::runtime_error&){}
        assert(!store.resident(0) && store.stats().ram_bytes==0 && store.stats().promoted_blocks==0);
        fail=false;store.admit_read(0,source.generation,view,256);
        assert(store.resident(0)->key[0]==19 && store.resident(0)->value[0]==26);
        const auto before=store.stats().ssd_read_bytes;auto resident_source=store.source(0);
        std::size_t resident_charge=99;auto resident_view=store.read_cpu_view(resident_source,&resident_charge);
        assert(!resident_charge && resident_view.key==resident_source.image->key && store.stats().ssd_read_bytes==before);
        auto weak=std::weak_ptr<const void>(view.owner);
        store.truncate(0);source={};resident_source={};backing.reset();
        assert(view.key[0]==19 && resident_view.value[0]==26 && !weak.expired());
        view={};if(borrowed)assert(weak.expired());
        tq::RamKvSnapshotBlock invalid;retained=99;
        try {store.read_cpu_view(invalid,&retained);assert(false);}catch(const std::runtime_error&){}
        assert(!retained && store.stats().ssd_read_bytes==before);
    }
    std::cout<<"Retained CPU KV views: session/prefix lifetime, allocation capacity, resident accounting and failed admission PASS\n";
}
void check_read_into(const std::filesystem::path& root) {
    for(bool borrowed:{false,true}) {
        tq::RamKvStoreConfig cfg;cfg.pressure.canonical_tier=tq::RamKvTier::Tq4;
        cfg.pressure.ssd_enabled=true;cfg.prefix_enabled=borrowed;
        mfq::cache::QsaKvStoreConfig file;
        file.directory=(root/(borrowed?"prefix":"session")).string();
        file.buffer_bytes=file.budget_bytes=4096;cfg.session_ssd=std::make_shared<mfq::cache::QsaKvStore>(file);
        tq::MicroblockImage raw;raw.key.assign(129,37);raw.value.assign(127,53);
        raw.key_bits=raw.value_bits=4;raw.key_state_bits=14;raw.value_state_bits=12;
        auto original=allocate(raw);unsigned allocations=0,direct_reads=0,vector_reads=0;bool fail=false;
        auto backing=std::make_shared<tq::RamKvBacking>();
        backing->key_bytes=129;backing->value_bytes=127;backing->key_bits=backing->value_bits=4;
        backing->key_state_bits=14;backing->value_state_bits=12;backing->tier=tq::RamKvTier::Tq4;
        qsa_test::reader(*backing).read_range=[&](std::size_t,std::size_t){++vector_reads;std::vector<std::uint8_t> bytes(256,37);std::fill(bytes.begin()+129,bytes.end(),53);return bytes;};
        qsa_test::reader(*backing).read_range_into=[&](std::size_t,std::uint8_t* output,std::size_t count){
            assert(count==256);++direct_reads;std::fill(output,output+129,37);std::fill(output+129,output+count,53);
        };
        tq::RamKvStore store(cfg,[&](const auto& source){++allocations;return allocate(source);},encode,
            [&](const auto&){if(fail)throw std::runtime_error("Direct read publication failed");});
        store.append(original,{256,256,256},1,borrowed?backing:nullptr);original.reset();
        assert(store.maintain(0).fits());cfg.session_ssd->flush();
        const auto source=store.source(0);auto buffer=std::make_shared<std::vector<std::uint8_t>>(273,0x6a);
        const auto reads=store.stats().ssd_read_bytes;
        try {store.read_cpu_into(source,buffer->data(),270);assert(false);}catch(const std::out_of_range&){}
        assert(store.stats().ssd_read_bytes==reads && buffer->front()==0x6a);
        store.read_cpu_into(source,buffer->data(),buffer->size());
        assert(!vector_reads && direct_reads==unsigned(borrowed));
        assert(buffer->at(128)==37 && buffer->at(144)==53 && buffer->at(270)==53);
        assert(buffer->at(271)==0x6a && buffer->at(272)==0x6a && !allocations);
        assert(store.stats().ssd_read_bytes==256);
        auto image=std::make_shared<tq::RamKvImage>();image->owner=buffer;
        image->key=buffer->data();image->value=buffer->data()+144;image->key_bytes=129;image->value_bytes=127;
        image->key_bits=image->value_bits=4;image->key_state_bits=14;image->value_state_bits=12;
        fail=true;
        try {store.admit_read(0,source.generation,image,256);assert(false);}catch(const std::runtime_error&){}
        assert(!store.resident(0) && store.stats().ram_bytes==0 && !allocations);
        fail=false;store.admit_read(0,source.generation,image,256);
        assert(store.resident(0)==image && !allocations && store.stats().promoted_blocks==1);
        std::vector<std::uint8_t> resident(273,0x6a);
        store.read_cpu_into(store.source(0),resident.data(),resident.size());
        assert(std::equal(resident.begin(),resident.begin()+129,buffer->begin()));
        assert(std::equal(resident.begin()+144,resident.begin()+271,buffer->begin()+144));
        assert(resident[271]==0x6a && resident[272]==0x6a && store.stats().ssd_read_bytes==256);
    }
    std::cout<<"Direct CPU KV reads: session/prefix, padded K/V, bounded writes and allocation-free admission PASS\n";
}
void check_batch_reads(const std::filesystem::path& root) {
    mfq::cache::QsaKvStoreConfig cfg;cfg.directory=root.string();cfg.buffer_bytes=cfg.budget_bytes=4096;
    auto file=std::make_shared<mfq::cache::QsaKvStore>(cfg);
    std::vector<mfq::cache::QsaKvStore::BlockPtr> blocks;
    for(unsigned i=0;i<9;++i)blocks.push_back(file->write(std::vector<std::uint8_t>(129+i,std::uint8_t(31+i))));
    file->flush();const auto before=file->stats();
    std::vector<std::vector<std::uint8_t>> outputs(9,std::vector<std::uint8_t>(160,0xa7));
    std::vector<mfq::cache::QsaKvStore::ReadRequest> reads;
    for(unsigned i=9;i-->0;)reads.push_back({blocks[i],outputs[i].data(),129+i});
    auto invalid=reads;invalid.back().bytes--;
    try{file->read_into_batch(invalid);assert(false);}catch(const std::invalid_argument&){}
    assert(file->stats().reads==before.reads && outputs.back().front()==0xa7);
    file->read_into_batch(reads);
    assert(file->stats().reads==before.reads+1);
    assert(file->stats().read_bytes-before.read_bytes==9*129+36);
    for(unsigned i=0;i<9;++i) {
        assert(std::all_of(outputs[i].begin(),outputs[i].begin()+129+i,[&](auto b){return b==31+i;}));
        assert(outputs[i][129+i]==0xa7);
    }
    // Duplicates and gaps retain their own destinations without reading gaps.
    std::vector<std::uint8_t> duplicate(129),gap(137);
    file->read_into_batch({{blocks[0],duplicate.data(),129},{blocks[8],gap.data(),137},{blocks[0],outputs[0].data(),129}});
    assert(duplicate.front()==31 && gap.back()==39);
    tq::RamKvStoreConfig ram;ram.pressure.canonical_tier=tq::RamKvTier::Tq4;
    tq::RamKvStore store(ram,allocate,encode,[](const auto&){});
    tq::RamKvSnapshotBlock resident;resident.image=image(11,256,4);
    tq::RamKvSnapshotBlock session;auto backing=std::make_shared<tq::RamKvBacking>();
    backing->store=file;backing->block=blocks[1];backing->key_bytes=67;backing->value_bytes=63;
    backing->key_bits=backing->value_bits=4;session.backing=backing;session.tier=tq::RamKvTier::Ssd;
    tq::RamKvSnapshotBlock prefix;auto borrowed=std::make_shared<tq::RamKvBacking>(*backing);
    unsigned prefix_reads=0;auto& reader=qsa_test::reader(*borrowed);
    reader.read_range_into=[&](std::size_t,std::uint8_t* p,std::size_t n){++prefix_reads;std::fill(p,p+n,55);};
    prefix.backing=borrowed;prefix.tier=tq::RamKvTier::Ssd;
    std::vector<std::uint8_t> a(258,0xa7),b(145,0xa7),c(145,0xa7);
    auto requests=std::vector<tq::RamKvReadInto>{{&resident,a.data(),a.size()},{&session,b.data(),b.size()},{&prefix,c.data(),c.size()}};
    auto undersized=requests;undersized.back().capacity=142;
    try{store.read_cpu_into_batch(undersized);assert(false);}catch(const std::out_of_range&){}
    assert(a.front()==0xa7 && !prefix_reads && !store.stats().ssd_read_bytes);
    store.read_cpu_into_batch(requests);
    assert(a[0]==11 && a[128]==18 && a[256]==0xa7);
    assert(b[66]==32 && b[80]==32 && b[142]==32 && b[143]==0xa7);
    assert(c[66]==55 && c[80]==55 && c[142]==55 && c[143]==0xa7);
    assert(prefix_reads==1 && store.stats().ssd_read_bytes==260);
    store.read_cpu_into_batch({});file->read_into_batch({});
    std::cout<<"Batched original KV reads: contiguous coalescing, reversed order, duplicates, mixed RAM/session/prefix and padding PASS\n";
}
void check_bulk_touches(const std::filesystem::path& root) {
    tq::RamKvStoreConfig cfg;cfg.pressure.canonical_tier=tq::RamKvTier::Tq4;
    cfg.pressure.ssd_enabled=true;cfg.session_ssd=disk(root);
    tq::RamKvStore store(cfg,allocate,encode,[](const auto&){});
    constexpr unsigned count=257;
    std::vector<std::uint64_t> accesses(count);
    for(unsigned id=0;id<count;++id) {
        accesses[id]=id%11;
        store.append(image(id,256,4),{1024,512,256},accesses[id]);
    }
    for(unsigned round=0;round<96;++round) {
        std::vector<std::size_t> ids;
        for(unsigned i=0;i<=round%64;++i)ids.push_back((i*193+round*7)%count);
        ids.push_back(ids.front());
        const auto epoch=2+(round*29)%73;
        if(round%2) {
            const auto leases=store.lease(ids,epoch);assert(leases.size()==ids.size());
            for(unsigned i=0;i<ids.size();++i)assert(leases[i].id==ids[i] && leases[i].image==store.resident(ids[i]));
        }else store.touch(ids,epoch);
        for(auto id:ids)accesses[id]=std::max(accesses[id],std::uint64_t(epoch));
        const auto snapshot=store.snapshot(count);
        for(unsigned id=0;id<count;++id)assert(snapshot->blocks[id].access==accesses[id]);
    }
    try{store.touch({0,count},10000);assert(false);}catch(const std::out_of_range&){}
    try{(void)store.lease({1,count},10000);assert(false);}catch(const std::out_of_range&){}
    std::vector<std::size_t> order;
    for(unsigned id=0;id<count;++id)order.push_back(id);
    std::sort(order.begin(),order.end(),[&](auto a,auto b){
        return accesses[a]!=accesses[b]?accesses[a]<accesses[b]:a<b;
    });
    for(unsigned i=0;i<count;++i) {
        assert(store.maintain((count-i-1)*256).fits());
        assert(!store.resident(order[i]));
        if(i+1<count)assert(store.resident(order[i+1]));
    }
    store.truncate(0);
    store.append(image(19,256,4),{1024,512,256},1);
    store.touch({0,0},90);assert(store.maintain(0).fits());
    assert(store.read(0,256)->key[0]==19 && store.resident(0));
    std::cout<<"Bulk RAM leases: duplicate/order preservation, interleaved and old epochs, invalid IDs, exact LRU ties and ID reuse PASS\n";
}
void check_random_prefix_batch() {
    auto source=std::make_shared<tq::RamKvReadSource>();source->bytes=1024;
    std::vector<std::uint8_t> bytes(1024);for(std::size_t i=0;i<bytes.size();++i)bytes[i]=std::uint8_t(i*13+7);
    unsigned single=0,batches=0;std::size_t transferred=0;
    source->read_range_into=[&](std::size_t offset,std::uint8_t* destination,std::size_t count){++single;transferred+=count;std::memcpy(destination,bytes.data()+offset,count);};
    source->read_ranges_into=[&](const std::vector<mfq::ReadSpan>& spans){
        ++batches;for(const auto& span:spans){transferred+=span.size;std::memcpy(span.destination,bytes.data()+span.offset,span.size);}
    };
    tq::RamKvStoreConfig config;tq::RamKvStore store(config,allocate,encode,[](const auto&){});
    std::vector<tq::RamKvSnapshotBlock> blocks(3);std::vector<std::vector<std::uint8_t>> output(3,std::vector<std::uint8_t>(64,0xa5));
    std::vector<tq::RamKvReadInto> reads;
    for(std::size_t at=0;at<blocks.size();++at){
        auto backing=std::make_shared<tq::RamKvBacking>();backing->source=source;backing->offset=at==0?700:at==1?100:700;
        backing->key_bytes=17;backing->value_bytes=19;backing->key_bits=backing->value_bits=4;
        blocks[at].backing=backing;blocks[at].tier=tq::RamKvTier::Ssd;reads.push_back({&blocks[at],output[at].data(),64});
    }
    auto invalid=reads;invalid.back().capacity=50;
    try{store.read_cpu_into_batch(invalid,false);assert(false);}catch(const std::out_of_range&){}
    assert(!single && !batches && !transferred && output[0][0]==0xa5);
    store.read_cpu_into_batch(reads,false);assert(single==0 && batches==1 && transferred==108 && store.stats().ssd_read_bytes==108);
    for(std::size_t at=0;at<reads.size();++at){const auto offset=blocks[at].backing->offset;
        assert(std::equal(output[at].begin(),output[at].begin()+17,bytes.begin()+offset));
        assert(std::equal(output[at].begin()+32,output[at].begin()+51,bytes.begin()+offset+17));
        assert(output[at][51]==0xa5 && output[at].back()==0xa5);
    }
    source->read_ranges_into={};store.read_cpu_into_batch(reads,false);assert(single==3 && batches==1 && transferred==216);
    const auto before=single;std::vector<mfq::ReadSpan> bad{{0,reinterpret_cast<std::byte*>(output[0].data()),1},{0,nullptr,1}};
    try{source->read_into_batch(bad);assert(false);}catch(const std::invalid_argument&){}
    assert(single==before);
    source->read_ranges_into=[](const auto&){throw std::runtime_error("expected batch source error");};
    try{store.read_cpu_into_batch(reads,false);assert(false);}catch(const std::runtime_error& e){assert(std::string(e.what())=="expected batch source error");}
    assert(store.stats().ssd_read_bytes==216);
    std::cout<<"Random prefix batch exact ranges/duplicates/padding, prevalidation, legacy fallback and failure propagation PASS\n";
}
void check_relocation() {
    bool fail=false;std::vector<tq::RamKvChange> published;
    tq::RamKvStore store({},allocate,encode,[&](const auto& changes){
        if(fail)throw std::runtime_error("relocation publication failure");published=changes;
    });
    for(unsigned i=0;i<4;++i)store.append(image(31+i,256,4),{1024,512,256},7+i);
    published.clear();const auto before=store.snapshot(4);const auto stats=store.stats();
    const auto relocate=[](const auto& images){return std::vector<tq::RamKvRelocation>{{1,allocate(tq::MicroblockImage{
        std::vector<std::uint8_t>(images[1]->key,images[1]->key+images[1]->key_bytes),
        std::vector<std::uint8_t>(images[1]->value,images[1]->value+images[1]->value_bytes),
        images[1]->key_bits,images[1]->value_bits,images[1]->key_state_bits,images[1]->value_state_bits})}};};
    fail=true;
    try{store.relocate(relocate);assert(false);}catch(const std::runtime_error&){}
    assert(store.resident(1)==before->blocks[1].image && store.stats().ram_bytes==stats.ram_bytes);
    fail=false;assert(store.relocate(relocate)==1);
    const auto after=store.snapshot(4);
    for(unsigned i=0;i<4;++i) {
        assert(after->blocks[i].generation==before->blocks[i].generation && after->blocks[i].access==before->blocks[i].access);
        assert(after->blocks[i].tier==before->blocks[i].tier && after->blocks[i].read().key==before->blocks[i].read().key);
    }
    assert(after->blocks[1].image!=before->blocks[1].image && published.size()==1 &&
        published[0].previous==before->blocks[1].image && published[0].image==after->blocks[1].image);
    assert(store.stats().tokens==stats.tokens && store.stats().ram_bytes==stats.ram_bytes);
    try{store.relocate([](const auto& images){return std::vector<tq::RamKvRelocation>{{0,images[0]},{0,images[0]}};});assert(false);}
    catch(const std::invalid_argument&){}
    try{store.relocate([](const auto&){return std::vector<tq::RamKvRelocation>{{0,image(1,512,4)}};});assert(false);}
    catch(const std::invalid_argument&){}
    assert(store.resident(1)==after->blocks[1].image && store.stats().ram_bytes==stats.ram_bytes);
    std::cout<<"Immutable RAM relocation: exact encoding, generations, LRU, old reader lifetime and publication failure/retry PASS\n";
}
void check_native_file_batch(const std::filesystem::path& root,bool packed=false) {
    std::filesystem::create_directories(root);
    std::array<std::vector<std::uint8_t>,3> data;
    std::array<std::shared_ptr<tq::RamKvReadSource>,3> sources;
    for(unsigned i=0;i<data.size();++i) {
        data[i].resize(4096);for(unsigned j=0;j<data[i].size();++j)data[i][j]=std::uint8_t(i*59+j*7);
        const auto path=root/std::to_string(i);
        {std::ofstream file(path,std::ios::binary);file.write(reinterpret_cast<const char*>(data[i].data()),data[i].size());}
        sources[i]=std::make_shared<tq::RamKvReadSource>();sources[i]->bytes=data[i].size()-31;
        sources[i]->file_reader=std::make_shared<mfq::FileRangeReader>(path,mfq::FileReadMode::Buffered);
        sources[i]->file_offset=31;
    }
    tq::RamKvStore store({},allocate,encode,[](const auto&){});
    const auto read_batch=[&](const std::vector<tq::RamKvReadInto>& requests,bool sequential) {
        if(packed)store.read_packed_cpu_into_batch(requests,sequential);
        else store.read_cpu_into_batch(requests,sequential);
    };
    std::vector<tq::RamKvSnapshotBlock> blocks(137);
    std::vector<std::array<std::uint8_t,66>> output(blocks.size());std::vector<tq::RamKvReadInto> reads;
    for(std::size_t i=0;i<blocks.size();++i) {
        output[i].fill(0xa5);auto backing=std::make_shared<tq::RamKvBacking>();
        backing->source=sources[i%3];backing->offset=(i*53)%3000;backing->key_bytes=17;backing->value_bytes=19;
        backing->key_bits=backing->value_bits=4;blocks[i].backing=backing;blocks[i].tier=tq::RamKvTier::Ssd;
        reads.push_back({&blocks[i],output[i].data()+1,64});
    }
    for(bool sequential:{false,true}) {
        for(auto& bytes:output)bytes.fill(0xa5);
        const auto before=store.stats().ssd_read_bytes;
        read_batch(reads,sequential);
        assert(store.stats().ssd_read_bytes-before==blocks.size()*36);
        for(std::size_t i=0;i<blocks.size();++i) {
            const auto offset=31+blocks[i].backing->offset;const auto& input=data[i%3];
            assert(output[i].front()==0xa5 && output[i].back()==0xa5);
            assert(std::equal(output[i].begin()+1,output[i].begin()+18,input.begin()+offset));
            assert(std::equal(output[i].begin()+(packed?18:33),output[i].begin()+(packed?37:52),input.begin()+offset+17));
        }
    }
    const auto scalar=sources[0]->read(3,37);
    assert(std::equal(scalar.begin(),scalar.end(),data[0].begin()+34));
    sources[0]->file_offset=UINT64_MAX;
    try{read_batch(reads,false);assert(false);}catch(const std::out_of_range&){}
    assert(store.stats().ssd_read_bytes==blocks.size()*72);
    std::cout<<"Retained multi-file QSA batch, unaligned K/V, exact accounting and offset overflow PASS\n";
}
void check_sequential_file_reads(const std::filesystem::path& root,bool packed=false) {
    std::filesystem::create_directories(root);
    std::array<std::vector<std::uint8_t>,3> data;
    std::array<std::shared_ptr<tq::RamKvReadSource>,3> sources;
    for(std::size_t file=0;file<data.size();++file) {
        data[file].resize(4096);
        for(std::size_t i=0;i<data[file].size();++i)data[file][i]=std::uint8_t(file*59+i*7);
        const auto path=root/std::to_string(file);
        {std::ofstream stream(path,std::ios::binary);
            stream.write(reinterpret_cast<const char*>(data[file].data()),data[file].size());}
        sources[file]=std::make_shared<tq::RamKvReadSource>();
        sources[file]->bytes=data[file].size()-31;
        sources[file]->file_reader=std::make_shared<mfq::FileRangeReader>(path,mfq::FileReadMode::Buffered);
        sources[file]->file_offset=31;
    }
    tq::RamKvStore store({},allocate,encode,[](const auto&){});
    const auto read_batch=[&](const std::vector<tq::RamKvReadInto>& requests,bool sequential) {
        if(packed)store.read_packed_cpu_into_batch(requests,sequential);
        else store.read_cpu_into_batch(requests,sequential);
    };
    std::vector<tq::RamKvSnapshotBlock> blocks(3*65);
    std::vector<std::array<std::uint8_t,66>> output(blocks.size());
    std::vector<tq::RamKvReadInto> reads;
    for(std::size_t i=0;i<blocks.size();++i) {
        if(i%65==64) {
            tq::MicroblockImage src;
            src.key.assign(17,std::uint8_t(i));src.value.assign(19,std::uint8_t(i+7));
            blocks[i].image=allocate(src);
        }else {
            auto backing=std::make_shared<tq::RamKvBacking>();
            backing->source=sources[i/65];backing->offset=(i%65)*44;
            backing->key_bytes=17;backing->value_bytes=19;
            backing->key_bits=backing->value_bits=4;
            blocks[i].backing=backing;blocks[i].tier=tq::RamKvTier::Ssd;
        }
        reads.push_back({&blocks[i],output[i].data()+1,64});
    }
    const auto check=[&] {
        for(std::size_t i=0;i<blocks.size();++i) {
            const auto* key=blocks[i].image?blocks[i].image->key:
                data[i/65].data()+31+blocks[i].backing->offset;
            const auto* value=blocks[i].image?blocks[i].image->value:key+17;
            assert(output[i].front()==0xa5 && output[i].back()==0xa5);
            assert(std::equal(output[i].begin()+1,output[i].begin()+18,key));
            assert(std::equal(output[i].begin()+(packed?18:33),output[i].begin()+(packed?37:52),value));
        }
    };
    for(auto& bytes:output)bytes.fill(0xa5);
    read_batch(reads,true);check();
    assert(store.stats().ssd_read_bytes==3*64*36);
    mfq::FileReadStats file_stats;
    for(const auto& source:sources)mfq::add_file_read_stats(file_stats,source->file_reader->stats());
    assert(file_stats.calls==3 && file_stats.logical_bytes>=3*64*36 && file_stats.logical_bytes<=2*3*64*36);
    assert(tq::RamKvStore::sequential_read_workspace(reads.size())>=reads.size()*sizeof(mfq::ReadSpan));
    try{(void)tq::RamKvStore::sequential_read_workspace(SIZE_MAX);assert(false);}
    catch(const std::overflow_error&){}
    // Unordered destinations use the original serial scatter path.
    std::reverse(reads.begin(),reads.end());
    for(auto& bytes:output)bytes.fill(0xa5);
    read_batch(reads,true);check();
    std::reverse(reads.begin(),reads.end());
    // Overlapping destinations preserve the last writer in caller order.
    auto first=reads[0],last=reads[65];last.destination=first.destination;
    read_batch({first,last},true);
    assert(std::equal(output[0].begin()+1,output[0].begin()+18,data[1].begin()+31));
    assert(std::equal(output[0].begin()+(packed?18:33),output[0].begin()+(packed?37:52),data[1].begin()+48));
    // A failed native read joins every worker before destinations are reused.
    std::filesystem::resize_file(root/"1",32);
    const auto before_failure=store.stats().ssd_read_bytes;
    try{read_batch(reads,true);assert(false);}catch(const std::runtime_error&){}
    const auto after_failure=store.stats().ssd_read_bytes;
    assert(after_failure>=before_failure && after_failure-before_failure<=2*64*36);
    {std::ofstream stream(root/"1",std::ios::binary|std::ios::trunc);
        stream.write(reinterpret_cast<const char*>(data[1].data()),data[1].size());}
    for(auto& bytes:output)bytes.fill(0xa5);
    read_batch(reads,true);check();
    assert(store.stats().ssd_read_bytes-after_failure==3*64*36);
    std::cout<<"Sequential file KV: bounded gapped reads, mixed RAM, scatter order and failure/retry PASS\n";
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
void check_indexed_precision_planning() {
    unsigned cases=0;
    for(const auto canonical:{tq::RamKvTier::Fp16,tq::RamKvTier::Affine8,tq::RamKvTier::Tq4})
    for(int pins=0;pins<3;++pins)
    for(const auto budget:{std::size_t(0),std::size_t(10000),std::size_t(35000),std::size_t(70000)}) {
        tq::RamKvStoreConfig cfg;cfg.pressure.canonical_tier=canonical;cfg.pressure.quantize_tokens=16;
        bool recording=false;
        std::vector<std::pair<std::size_t,tq::RamKvTier>> changes,expected_changes;
        tq::RamKvStore store(cfg,allocate,encode,[&](const auto& rows){
            if(recording)for(const auto& row:rows)changes.emplace_back(row.id,row.tier);
        });
        std::vector<tq::RamPressureBlock> blocks;
        std::vector<tq::RamKvImagePtr> readers;
        for(std::size_t id=0;id<96;++id) {
            const auto tier=tq::RamKvTier(id%3);const auto access=id*17%101;
            auto value=image(unsigned(id),1024>>unsigned(tier),tier==tq::RamKvTier::Fp16?0:tier==tq::RamKvTier::Affine8?254:4);
            store.append_at_tier(value,{1024,512,256},tier,access);
            const bool pinned=pins==2 || (pins==1 && id%7!=0);
            if(pinned)readers.push_back(value);
            blocks.push_back({id,access,4,tier,{1024,512,256},pinned});
        }
        const std::vector<std::size_t> touched{2,7,7,14,21,25,42,61,80};
        store.touch(touched,300);for(const auto id:touched)blocks[id].last_access=300;
        const auto expected=tq::plan_ram_pressure(blocks,budget,cfg.pressure);
        for(const auto& action:expected.actions)for(const auto id:action.blocks)
            expected_changes.emplace_back(id,action.to);
        recording=true;const auto actual=store.maintain(budget,false);
        assert(changes==expected_changes);
        assert(actual.ram_bytes==expected.ram_bytes && actual.tokens==expected.tokens);
        assert(actual.minimum_fp16==expected.minimum_fp16 && actual.minimum_affine8==expected.minimum_affine8);
        assert(actual.needs_ssd==expected.needs_ssd && actual.needs_readers==expected.needs_readers);
        ++cases;
    }
    std::cout<<"Indexed mixed precision planning matches full reference: multi-stage batches/touch order/readers/floors "<<cases<<" cases PASS\n";
}
void check_protected_floor_planning() {
    for(const auto canonical:{tq::RamKvTier::Fp16,tq::RamKvTier::Affine8,tq::RamKvTier::Tq4})
    for(int pin_mode=0;pin_mode<3;++pin_mode)
    for(const auto budget:{std::size_t(0),std::size_t(1024),std::size_t(16384)}) {
        tq::RamKvStoreConfig cfg;cfg.pressure.canonical_tier=canonical;cfg.pressure.quantize_tokens=16;
        tq::RamKvStore store(cfg,allocate,encode,[](const auto&){});
        std::vector<tq::RamPressureBlock> blocks;
        std::vector<tq::RamKvImagePtr> readers;
        for(std::size_t id=0;id<22;++id) {
            // Both higher tiers contain less than a complete conversion
            // batch. The cold TQ4 population includes optional readers.
            const auto tier=id<3?tq::RamKvTier::Fp16:id<6?tq::RamKvTier::Affine8:tq::RamKvTier::Tq4;
            const auto bytes=tier==tq::RamKvTier::Fp16?1024:tier==tq::RamKvTier::Affine8?512:256;
            auto value=image(unsigned(id),bytes,tier==tq::RamKvTier::Fp16?0:tier==tq::RamKvTier::Affine8?254:4);
            store.append_at_tier(value,{1024,512,256},tier,id);
            const bool pinned=pin_mode==2 || (pin_mode==1 && id%2==0);
            if(pinned)readers.push_back(value);
            blocks.push_back({id,id,4,tier,{1024,512,256},pinned});
        }
        const auto expected=tq::plan_ram_pressure(blocks,budget,cfg.pressure);
        const auto actual=store.maintain(budget,false);
        assert(expected.actions.empty() && actual.actions.empty());
        assert(actual.ram_bytes==expected.ram_bytes && actual.tokens==expected.tokens);
        assert(actual.minimum_fp16==expected.minimum_fp16 && actual.minimum_affine8==expected.minimum_affine8);
        assert(actual.needs_ssd==expected.needs_ssd && actual.needs_readers==expected.needs_readers);
        assert(store.stats().quantized_blocks==0 && store.stats().spilled_blocks==0);
    }
    std::cout<<"Protected precision floors: RAM-only planning, partial batches and reader eligibility PASS\n";
}
int main(int argc,char** argv)try {
    check_indexed_precision_planning();
    check_protected_floor_planning();
    {
        tq::RamKvStoreConfig config;config.pressure.canonical_tier=tq::RamKvTier::Tq4;
        tq::RamKvStore store(config,allocate,encode,[](const auto&){});
        for(unsigned id=0;id<257;++id)store.append(image(id,256,4),{1024,512,256},id);
        const auto snapshot=store.snapshot(257);const auto generation=store.generation(0);
        const auto resident=store.resident(17);const auto before_reserve=store.stats();
        store.reserve_metadata(4096);store.reserve_metadata(128);
        assert(store.stats().blocks==before_reserve.blocks && store.stats().ram_bytes==before_reserve.ram_bytes);
        assert(store.generation(0)==generation && store.resident(17)==resident);
        store.touch({17},5000);assert(store.lease({17},5001).front().image==resident);
        assert(store.release_idle_metadata()==0 && store.stats().blocks==257);
        store.truncate(0);assert(store.release_idle_metadata()>0 && !store.release_idle_metadata());
        store.reserve_metadata(4096);
        assert(store.stats().blocks==0 && store.stats().ram_bytes==0 && store.release_idle_metadata()>0);
        assert(snapshot->blocks[17].read().key.front()==17);
        // An append preference must not reject a checkpoint which fits the
        // real RAM cap when its current format has no lower tier available.
        store.restore(*snapshot,257*256,false,false,{},{},256);
        assert(store.stats().blocks==257 && store.generation(0)>generation && store.resident(17)->key[0]==17);
        store.truncate(0);assert(store.release_idle_metadata()>0);
        std::cout<<"Idle metadata returns capacity without changing active data, snapshot readers or rollback generations PASS\n";
    }
    if(argc!=2)throw std::invalid_argument("Supply the test SSD directory");
    // CTest reuses its output directory. Old durable prefix files must not
    // turn a fresh hot-cache fixture into a cold, already persisted cache.
    const auto parent=std::filesystem::path(argv[1]);std::filesystem::create_directories(parent);
    std::filesystem::path root;
    for(std::size_t id=0;;++id) {
        // Short names also keep the two SHA-256 path components below the
        // Windows path limit in a nested CMake build directory.
        root=parent/("r"+std::to_string(id));
        if(std::filesystem::create_directory(root))break;
    }
    check_deferred_reads(root/"deferred");
    check_session_read_window(root/"session-window");
    check_native_file_batch(root/"packed-native-files",true);
    check_sequential_file_reads(root/"packed-sequential-files",true);
    check_shared_read_source(root/"shared-reader");
    check_mixed_precision_recall(root/"mixed-recall");
    check_read_views(root/"read-views");
    check_restore_observer(root/"restore-observer");
    check_sparse_restore_batches(root/"sparse-restore");
    check_read_into(root/"read-into");
    check_batch_reads(root/"batch-reads");
    check_random_prefix_batch();
    check_relocation();
    check_native_file_batch(root/"native-files");
    check_sequential_file_reads(root/"sequential-files");
    check_admit_view(root/"admit-view");
    check_bulk_touches(root/"bulk-touches");
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
        // An append cannot raise a budget-capped floor by increasing SSD
        // history. Publish the affordable batch while retaining the new row.
        assert(usage.tokens[0]==33*4 && usage.tokens[1]==96*4 && usage.tokens[2]==512*4);
        assert(!result.fits() && !result.needs_readers &&
            usage.ram_bytes==32*1024+96*512+512*256+1024 && live_floor.resident(640)->key[0]==99);
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
        qsa_test::reader(*backing).read_range=[&](std::size_t,std::size_t){if(first.exchange(false)){started.set_value();resumed.wait();}return packet;};
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
    assert(after.tokens[0]>=result.minimum_fp16 && after.tokens[1]>=result.minimum_affine8 && after.tokens[3]>0);
    assert(after.tokens[0]<result.minimum_fp16+2048 && after.tokens[1]<result.minimum_affine8+2048);
    {
        auto snapshot=store.snapshot(10240);
        auto restore_config=c;restore_config.session_ssd=disk(root/"restore-policy");
        tq::RamKvStore restored(restore_config,allocate,encode,[](const auto&){});
        restored.restore(*snapshot,1536<<10,true);
        const auto stats=restored.stats();
        assert(stats.blocks==10240 && stats.ram_bytes<=1536<<10);
        const auto restored_policy=restored.maintain(1536<<10);
        assert(restored_policy.fits() && stats.tokens[0]>=restored_policy.minimum_fp16 &&
            stats.tokens[1]>=restored_policy.minimum_affine8 && stats.tokens[3]>0);
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
    qsa_test::reader(*borrowed).read_range=[range](std::size_t,std::size_t){return range->read_range(101,256);};
    auto borrowed_config=fixed;borrowed_config.prefix_enabled=true;
    tq::RamKvStore borrowed_store(borrowed_config,allocate,encode,[](const auto&){});
    borrowed_store.append(packed_image,{1024,512,256},1,borrowed);packed_image.reset();
    const auto session_bytes=session->stats().written_bytes;
    assert(borrowed_store.maintain(0).fits() && borrowed_store.stats().prefix_reused_bytes==256);
    assert(borrowed_store.stats().ssd_written_bytes==0 && session->stats().written_bytes==session_bytes);
    auto replacement=std::make_shared<std::vector<std::uint8_t>>(4096,0x55);
    paged->replace({},tokens,4,replacement);paged->flush();
    assert(paged->borrow(hash)->read_range(101,1)[0]==0x55);
    std::vector<std::uint8_t> original_key(128),original_value(128);
    assert(range->read_ranges_into);
    range->read_ranges_into({{229,reinterpret_cast<std::byte*>(original_value.data()),128},
        {101,reinterpret_cast<std::byte*>(original_key.data()),128}});
    assert(original_key[0]==37 && original_value[0]==44);
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
