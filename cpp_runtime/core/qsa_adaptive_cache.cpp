#include "tyloinfer/qsa_adaptive_cache.h"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <thread>

namespace tyloinfer::qsa {
namespace {
std::size_t metadata(std::size_t rows,bool mixed=false) {
    const auto bytes=2*rows+(rows+1)/2+(mixed?(3*rows+7)/8:0);
    return (bytes+3)&~std::size_t(3);
}
}
struct AdaptiveCache::Impl {
    struct Block { BlockPrecisionHistory history;mfq::cache::QsaKvStore::BlockPtr canonical;std::uint64_t serial=0;
        std::shared_ptr<const MicroblockImage> resident; };
    explicit Impl(AdaptiveCacheConfig value,ReencodeBatch encoder,Publish publisher)
        :config(std::move(value)),encode(std::move(encoder)),publish(std::move(publisher)) {
        config.policy.validate();
        if(!config.rows_per_kind || config.rows_per_kind%4 ||
           config.rows_per_kind>std::numeric_limits<std::size_t>::max()/520 ||
           (config.canonical_tag!=affine8_tag && config.canonical_tag!=8) ||
           config.tq_state_bits<8 || config.tq_state_bits>16 ||
           config.io_buffer_bytes<4096 || (config.ssd_cache && config.directory.empty()) || !encode || !publish)
            throw std::invalid_argument("invalid adaptive QSA cache configuration");
        for(std::size_t tier=0;tier<precisions.size();++tier) {
            const auto p=precisions[tier];
            costs[tier]=32*config.rows_per_kind*(p.key_bits+p.value_bits)+
                2*metadata(config.rows_per_kind,tier<5 && config.row_allocation==RowPrecisionAllocation::Energy);
            distortion[tier]=tier==5?0:.5*(std::exp2(-2.*p.key_bits)+std::exp2(-2.*p.value_bits));
        }
        if(config.canonical_tag==affine8_tag)
            costs.back()=2*((3*config.rows_per_kind+3)&~std::size_t(3))+512*config.rows_per_kind;
        if(costs.back()>config.io_buffer_bytes/4)
            throw std::invalid_argument("adaptive QSA I/O buffer cannot hold a canonical microblock");
        mfq::cache::QsaKvStoreConfig disk;
        // Zero hot limit ensures every canonical image reaches SSD, even
        // while an independent resident image remains in GPU memory.
        disk.budget_bytes=config.io_buffer_bytes;disk.buffer_bytes=config.io_buffer_bytes;disk.directory=config.directory;
        if(config.ssd_cache)store=std::make_shared<mfq::cache::QsaKvStore>(std::move(disk));
        worker=std::thread([this]{run();});
    }
    ~Impl() {
        {std::lock_guard<std::mutex> lock(mutex);stopping=true;changed.notify_all();}
        if(worker.joinable())worker.join();
    }
    void check() const {if(error)std::rethrow_exception(error);}
    void validate(const MicroblockImage& image,std::uint8_t tier,bool canonical=false) const {
        if(tier>=precisions.size() || image.key.empty() || image.value.empty() ||
           image.key.size()+image.value.size()>costs[tier])
            throw std::invalid_argument("invalid adaptive QSA encoded microblock size");
        const auto p=precisions[tier];
        const auto expected_state_bits=tier==5?16:config.tq_state_bits;
        if(image.key_state_bits!=expected_state_bits || image.value_state_bits!=expected_state_bits)
            throw std::invalid_argument("adaptive QSA trellis profile mismatch");
        const auto check_kind=[&](const std::vector<std::uint8_t>& bytes,std::uint8_t bits,int expected) {
            if(expected==8 && bits==config.canonical_tag && bits==affine8_tag) {
                if(bytes.size()!=((3*config.rows_per_kind+3)&~std::size_t(3))+256*config.rows_per_kind)
                    throw std::invalid_argument("invalid canonical QSA Affine8 image size");
                return;
            }
            const bool mixed=bits==255 && expected<8 && config.row_allocation==RowPrecisionAllocation::Energy;
            if(!mixed && bits!=expected)throw std::invalid_argument("adaptive QSA precision tag mismatch");
            const auto rows=config.rows_per_kind;
            const auto meta=metadata(rows,mixed);
            if(bytes.size()<meta)throw std::invalid_argument("truncated adaptive QSA metadata");
            std::size_t payload=32*rows*bits;
            if(mixed) {
                payload=0;const auto map=2*rows+(rows+1)/2;
                for(std::size_t row=0;row<rows;++row) {
                    const auto at=3*row,shift=at&7;
                    unsigned code=bytes[map+at/8]>>shift;
                    if(shift>5)code|=unsigned(bytes[map+at/8+1])<<(8-shift);
                    payload+=32*(1+(code&7));
                }
            }
            if(bytes.size()!=meta+payload || payload>32*rows*expected)
                throw std::invalid_argument("adaptive QSA packed row budget mismatch");
        };
        check_kind(image.key,image.key_bits,p.key_bits);check_kind(image.value,image.value_bits,p.value_bits);
        if(canonical && (image.key_bits!=config.canonical_tag || image.value_bits!=config.canonical_tag ||
                         image.key.size()+image.value.size()!=costs.back()))
            throw std::invalid_argument("adaptive QSA canonical image encoding mismatch");
    }
    MicroblockImage read(const mfq::cache::QsaKvStore::BlockPtr& block) {
        const auto bytes=store->read(block);
        if(bytes->size()!=costs.back())throw std::runtime_error("invalid canonical QSA image");
        const auto split=bytes->size()/2;
        MicroblockImage result;
        result.key_bits=result.value_bits=config.canonical_tag;
        result.key.assign(bytes->begin(),bytes->begin()+split);result.value.assign(bytes->begin()+split,bytes->end());
        return result;
    }
    void run() {
        PrecisionPolicy policy(config.policy);std::uint64_t policy_epoch=0;
        for(;;) {
            std::vector<double> hits,mass;std::vector<Block> snapshot;
            std::uint64_t tokens=0,epoch=0;std::size_t budget=0;
            {
                std::unique_lock<std::mutex> lock(mutex);
                changed.wait(lock,[&]{return stopping || pending || !retired_blocks.empty();});
                if(stopping)return;
                if(!pending) {
                    std::vector<Block> retired;retired.swap(retired_blocks);
                    working=true;lock.unlock();
                    std::vector<Block>().swap(retired);
                    lock.lock();working=false;changed.notify_all();
                    continue;
                }
                working=true;pending=false;tokens=pending_tokens;budget=pending_budget;epoch=mutation_epoch;
                hits.swap(pending_hits);mass.swap(pending_mass);
                // Appends can outrun a coalesced observation while the worker
                // is fitting an earlier plan. Future blocks do not belong to
                // this observation's timeline and must remain untouched.
                const auto end=std::lower_bound(blocks.begin(),blocks.end(),tokens/4*4,
                    [](const Block& b,std::uint64_t time){return b.history.token_begin<time;});
                snapshot.assign(blocks.begin(),end);
                hits.resize(snapshot.size());mass.resize(snapshot.size());
            }
            const auto started=std::chrono::steady_clock::now();
            try {
                if(epoch!=policy_epoch){policy=PrecisionPolicy(config.policy);policy_epoch=epoch;}
                std::vector<BlockPrecisionHistory> history;history.reserve(snapshot.size());
                for(const auto& block:snapshot)history.push_back(block.history);
                policy.observe(history,hits,mass,tokens);
                const auto plan=policy.plan(history,tokens,budget,config.ssd_cache);
                // Release storage before allocating promotions. Publish bounded
                // batches so a large plan cannot retain a full RAM KV replica.
                for(int direction=0;direction<2;++direction) {
                    std::vector<std::size_t> ids;
                    std::vector<MicroblockImage> source;std::vector<Precision> targets;
                    std::vector<std::uint8_t> tiers;
                    std::size_t batch_bytes=0;
                    const auto commit=[&] {
                        if(ids.empty())return;
                        std::vector<BlockReplacement> batch;
                        std::vector<MicroblockImage> images;
                        // Canonical promotions never pass through a fitter,
                        // including batches that also contain lower tiers.
                        images.resize(ids.size());
                        std::vector<MicroblockImage> fit_source;
                        std::vector<Precision> fit_target;
                        std::vector<std::size_t> fit_ids;
                        for(std::size_t i=0;i<ids.size();++i) {
                            if(tiers[i]==5)images[i]=std::move(source[i]);
                            else {fit_ids.push_back(i);fit_source.push_back(std::move(source[i]));fit_target.push_back(targets[i]);}
                        }
                        if(!fit_ids.empty()) {
                            auto fitted=encode(fit_source,fit_target);
                            if(fitted.size()!=fit_ids.size())throw std::runtime_error("adaptive QSA reencoder returned the wrong batch length");
                            for(std::size_t i=0;i<fit_ids.size();++i)images[fit_ids[i]]=std::move(fitted[i]);
                        }
                        for(std::size_t i=0;i<ids.size();++i) {
                            validate(images[i],tiers[i]);
                            batch.push_back({ids[i],tiers[i],std::make_shared<MicroblockImage>(std::move(images[i]))});
                        }
                        std::lock_guard<std::mutex> lock(mutex);
                        std::vector<BlockReplacement> live;
                        for(const auto& replacement:batch) {
                            const auto id=replacement.block;
                            if(id>=blocks.size() || blocks[id].serial!=snapshot[id].serial)continue;
                            const bool protected_now=live_tokens<=config.policy.recent_tokens ||
                                blocks[id].history.token_begin+4>live_tokens-config.policy.recent_tokens;
                            if(replacement.tier<5 && protected_now)continue;
                            live.push_back(replacement);
                        }
                        if(!live.empty()) {
                            publish(live);
                            for(const auto& replacement:live) {
                                auto& block=blocks[replacement.block];
                                metrics.promotions+=replacement.tier>block.history.current;
                                metrics.demotions+=replacement.tier<block.history.current;
                                block.history.current=replacement.tier;
                                if(!config.ssd_cache)block.resident=replacement.image;
                            }
                        }
                        ids.clear();source.clear();targets.clear();tiers.clear();batch_bytes=0;
                    };
                    for(std::size_t i=0;i<snapshot.size();++i) {
                        const auto to=plan.tiers[i],from=snapshot[i].history.current;
                        if(to==from || (direction==0)!=(to<from))continue;
                        if(batch_bytes+costs.back()>config.io_buffer_bytes/2)commit();
                        ids.push_back(i);source.push_back(config.ssd_cache?read(snapshot[i].canonical):*snapshot[i].resident);
                        targets.push_back(precisions[to]);tiers.push_back(to);batch_bytes+=costs.back();
                        if(config.ssd_cache){std::lock_guard<std::mutex> lock(mutex);metrics.canonical_read_bytes+=costs.back();}
                    }
                    commit();
                }
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    bool applied=false;
                    for(std::size_t i=0;i<history.size() && i<blocks.size();++i) {
                        if(blocks[i].serial==snapshot[i].serial) {
                            blocks[i].history.hits=history[i].hits;
                            blocks[i].history.attention_mass=history[i].attention_mass;
                            applied=true;
                        }
                    }
                    if(applied) {
                        ++metrics.updates;
                        metrics.last_update_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
                    }
                }
            } catch(...) {
                std::lock_guard<std::mutex> lock(mutex);
                if(epoch==mutation_epoch){error=std::current_exception();pending=false;}
            }
            {std::lock_guard<std::mutex> lock(mutex);working=false;changed.notify_all();}
        }
    }
    AdaptiveCacheConfig config;ReencodeBatch encode;Publish publish;
    std::array<std::size_t,6> costs{};
    std::array<double,6> distortion{};
    std::shared_ptr<mfq::cache::QsaKvStore> store;
    std::vector<Block> blocks,retired_blocks;
    mutable std::mutex mutex;std::condition_variable changed;std::thread worker;std::exception_ptr error;
    bool pending=false,working=false,stopping=false;
    std::uint64_t mutation_epoch=0,pending_tokens=0,last_requested_tokens=0,next_serial=0,live_tokens=0;
    std::size_t pending_budget=0;
    std::vector<double> pending_hits,pending_mass;
    AdaptiveCacheStats metrics;
};

AdaptiveCache::AdaptiveCache(AdaptiveCacheConfig config,Reencode encode,Publish publish)
    :AdaptiveCache(std::move(config),ReencodeBatch([encode=std::move(encode)](
        const std::vector<MicroblockImage>& source,const std::vector<Precision>& targets) {
        std::vector<MicroblockImage> result;result.reserve(source.size());
        for(std::size_t i=0;i<source.size();++i)
            result.push_back(targets[i].key_bits==8 && targets[i].value_bits==8?source[i]:encode(source[i],targets[i]));
        return result;
    }),std::move(publish)) {}
AdaptiveCache::AdaptiveCache(AdaptiveCacheConfig config,ReencodeBatch encode,Publish publish)
    :impl_(std::make_unique<Impl>(std::move(config),std::move(encode),std::move(publish))) {}
AdaptiveCache::~AdaptiveCache()=default;
std::array<std::size_t,6> AdaptiveCache::tier_bytes() const {return impl_->costs;}
std::size_t AdaptiveCache::append(MicroblockImage image,std::uint64_t token_begin) {
    impl_->validate(image,5,true);
    mfq::cache::QsaKvStore::BlockPtr canonical;
    if(impl_->store) {
        std::vector<std::uint8_t> payload;payload.reserve(impl_->costs.back());
        payload.insert(payload.end(),image.key.begin(),image.key.end());payload.insert(payload.end(),image.value.begin(),image.value.end());
        canonical=impl_->store->write(std::move(payload));
    }
    auto shared=std::make_shared<MicroblockImage>(std::move(image));
    std::lock_guard<std::mutex> lock(impl_->mutex);impl_->check();
    if(token_begin%4 || (!impl_->blocks.empty() && token_begin<=impl_->blocks.back().history.token_begin))
        throw std::invalid_argument("adaptive QSA microblocks must be appended in position order");
    const auto id=impl_->blocks.size();
    BlockPrecisionHistory history;history.token_begin=token_begin;history.bytes=impl_->costs;
    history.distortion=impl_->distortion;
    impl_->publish({BlockReplacement{id,5,shared}});
    impl_->blocks.push_back({history,std::move(canonical),++impl_->next_serial,impl_->config.ssd_cache?nullptr:std::move(shared)});
    impl_->live_tokens=token_begin+4;
    return id;
}
std::size_t AdaptiveCache::append_batch(std::vector<MicroblockImage> images,std::uint64_t token_begin) {
    if(!images.empty() && (token_begin%4 ||
       images.size()>(std::numeric_limits<std::uint64_t>::max()-token_begin)/4))
        throw std::invalid_argument("invalid adaptive QSA microblock batch positions");
    for(const auto& image:images)impl_->validate(image,5,true);
    std::vector<mfq::cache::QsaKvStore::BlockPtr> canonical(images.size());
    std::vector<std::shared_ptr<const MicroblockImage>> shared;
    shared.reserve(images.size());
    for(std::size_t i=0;i<images.size();++i) {
        auto& image=images[i];
        if(impl_->store) {
            std::vector<std::uint8_t> payload;payload.reserve(impl_->costs.back());
            payload.insert(payload.end(),image.key.begin(),image.key.end());
            payload.insert(payload.end(),image.value.begin(),image.value.end());
            canonical[i]=impl_->store->write(std::move(payload));
        }
        shared.push_back(std::make_shared<MicroblockImage>(std::move(image)));
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);impl_->check();
    const auto first=impl_->blocks.size();
    if(shared.empty())return first;
    if(!impl_->blocks.empty() && token_begin<=impl_->blocks.back().history.token_begin)
        throw std::invalid_argument("adaptive QSA microblocks must be appended in position order");
    // Finish allocations before publication so a failed publisher cannot
    // leave a partial batch visible in the logical cache.
    if(shared.size()>impl_->blocks.max_size()-first)
        throw std::length_error("adaptive QSA microblock batch is too large");
    const auto required=first+shared.size(),capacity=impl_->blocks.capacity();
    if(required>capacity) {
        const auto growth=std::min(capacity/2,impl_->blocks.max_size()-capacity);
        impl_->blocks.reserve(std::max(required,capacity+growth));
    }
    std::vector<BlockReplacement> changes;changes.reserve(shared.size());
    for(std::size_t i=0;i<shared.size();++i)changes.push_back({first+i,5,shared[i]});
    impl_->publish(changes);
    for(std::size_t i=0;i<shared.size();++i) {
        BlockPrecisionHistory history;history.token_begin=token_begin+4*i;
        history.bytes=impl_->costs;history.distortion=impl_->distortion;
        impl_->blocks.push_back({history,std::move(canonical[i]),++impl_->next_serial,
            impl_->config.ssd_cache?nullptr:std::move(shared[i])});
    }
    impl_->live_tokens=token_begin+4*shared.size();
    return first;
}
void AdaptiveCache::update(std::uint64_t tokens,std::size_t budget,std::vector<double> hits,std::vector<double> mass) {
    std::lock_guard<std::mutex> lock(impl_->mutex);impl_->check();
    if(hits.size()!=impl_->blocks.size() || mass.size()!=hits.size())
        throw std::invalid_argument("adaptive QSA observation length mismatch");
    if(tokens<impl_->last_requested_tokens)throw std::invalid_argument("adaptive QSA update time moved backwards");
    if(impl_->pending && tokens<impl_->pending_tokens)throw std::invalid_argument("adaptive QSA update time moved backwards");
    for(std::size_t i=0;i<hits.size();++i)
        if(!std::isfinite(hits[i]) || !std::isfinite(mass[i]) || hits[i]<0 || mass[i]<0)
            throw std::invalid_argument("invalid adaptive QSA observation");
    impl_->pending_hits.resize(hits.size());impl_->pending_mass.resize(mass.size());
    for(std::size_t i=0;i<hits.size();++i) {impl_->pending_hits[i]+=hits[i];impl_->pending_mass[i]+=mass[i];}
    if(tokens-impl_->last_requested_tokens<impl_->config.policy.update_tokens)return;
    impl_->last_requested_tokens=tokens;impl_->pending_tokens=tokens;impl_->pending_budget=budget;impl_->pending=true;impl_->changed.notify_all();
}
void AdaptiveCache::wait() {
    std::unique_lock<std::mutex> lock(impl_->mutex);
    impl_->changed.wait(lock,[&]{return !impl_->working && !impl_->pending && impl_->retired_blocks.empty();});impl_->check();
    lock.unlock();if(impl_->store)impl_->store->flush();
}
void AdaptiveCache::truncate(std::size_t count) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if(count>impl_->blocks.size())throw std::out_of_range("adaptive QSA truncation beyond prefix");
    const auto window=impl_->config.io_buffer_bytes;
    const auto retirement_budget=window<=std::numeric_limits<std::size_t>::max()/2
        ?2*window:std::numeric_limits<std::size_t>::max();
    bool defer=!count && !impl_->blocks.empty() &&
        impl_->blocks.capacity()<=retirement_budget/sizeof(Impl::Block);
    if(defer) {
        auto bytes=impl_->blocks.capacity()*sizeof(Impl::Block);
        const auto account=[&](std::size_t n) {
            if(n>retirement_budget-bytes)defer=false;else bytes+=n;
        };
        for(const auto& block:impl_->blocks) {
            if(!defer)break;
            if(block.resident) {
                account(block.resident->key.capacity());
                if(defer)account(block.resident->value.capacity());
            }
        }
    }
    if(defer) {
        // Bound deferred metadata and encoded payloads to two I/O windows.
        // Larger prefixes release their host memory before reset returns.
        impl_->retired_blocks.swap(impl_->blocks);
        impl_->blocks.clear();
    } else impl_->blocks.resize(count);
    ++impl_->mutation_epoch;impl_->pending=false;
    impl_->live_tokens=count?impl_->blocks.back().history.token_begin+4:0;
    // Position rollback starts a new observation timeline.
    impl_->last_requested_tokens=0;impl_->pending_hits.clear();impl_->pending_mass.clear();
    impl_->changed.notify_all();
}
AdaptiveCacheStats AdaptiveCache::stats() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);impl_->check();
    auto stats=impl_->metrics;stats.blocks=impl_->blocks.size();
    for(const auto& block:impl_->blocks) {++stats.blocks_per_tier[block.history.current];stats.resident_bytes+=block.history.bytes[block.history.current];}
    if(impl_->store)stats.canonical=impl_->store->stats();return stats;
}
MicroblockImage AdaptiveCache::canonical_image(std::size_t id) const {
    if(!impl_->config.ssd_cache)throw std::logic_error("QSA original precision is unavailable with SSD cache disabled");
    mfq::cache::QsaKvStore::BlockPtr source;
    {std::lock_guard<std::mutex> lock(impl_->mutex);impl_->check();
     if(id>=impl_->blocks.size())throw std::out_of_range("QSA canonical block outside prefix");
     source=impl_->blocks[id].canonical;}
    return impl_->read(source);
}
MicroblockImage AdaptiveCache::reconstruction_image(std::size_t id) const {
    if(impl_->config.ssd_cache)return canonical_image(id);
    std::lock_guard<std::mutex> lock(impl_->mutex);impl_->check();
    if(id>=impl_->blocks.size())throw std::out_of_range("QSA resident block outside prefix");
    return *impl_->blocks[id].resident;
}
} // namespace tyloinfer::qsa
