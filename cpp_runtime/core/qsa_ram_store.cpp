#include "tyloinfer/qsa_ram_store.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <mutex>
#include <numeric>
#include <set>
#include <stdexcept>

namespace tyloinfer::qsa {
namespace {
void validate(const RamKvImagePtr& image) {
    if(!image || !image->owner || !image->key || !image->value || !image->key_bytes || !image->value_bytes ||
        image->key_bytes>std::numeric_limits<std::size_t>::max()-image->value_bytes)
        throw std::invalid_argument("Invalid QSA RAM image");
}
MicroblockImage read_backing(const RamKvBacking& backing) {
    if((!backing.read && (!backing.store || !backing.block)) || !backing.key_bytes || !backing.value_bytes ||
            backing.key_bytes>std::numeric_limits<std::size_t>::max()-backing.value_bytes)
        throw std::runtime_error("Invalid QSA SSD backing");
    const auto bytes=backing.read?std::make_shared<const std::vector<std::uint8_t>>(backing.read()):backing.store->read(backing.block);
    if(bytes->size()!=backing.key_bytes+backing.value_bytes)throw std::runtime_error("QSA SSD image length changed");
    MicroblockImage image;
    image.key.assign(bytes->begin(),bytes->begin()+backing.key_bytes);
    image.value.assign(bytes->begin()+backing.key_bytes,bytes->end());
    image.key_bits=backing.key_bits;image.value_bits=backing.value_bits;
    image.key_state_bits=backing.key_state_bits;image.value_state_bits=backing.value_state_bits;
    return image;
}
}
struct RamKvStore::Impl {
    struct Entry {
        RamKvImagePtr image;
        std::shared_ptr<const RamKvBacking> backing;
        std::array<std::size_t,3> bytes{};
        std::uint64_t generation=0,access=0;
        RamKvTier tier=RamKvTier::Fp16;
    };
    RamKvStoreConfig config;
    Allocate allocate;Encode encode;Publish publish;
    mutable std::mutex mutex;
    // Serialize maintenance and admissions; append/touch can proceed while
    // an encoder or SSD read is running. Truncation invalidates its versions.
    std::mutex maintenance;
    std::vector<Entry> entries;
    using Lru=std::set<std::pair<std::uint64_t,std::size_t>>;
    Lru tq4_lru;
    struct IndexInsert {
        Impl& store;std::vector<Lru::iterator> inserted;bool committed=false;
        IndexInsert(Impl& s,std::size_t count):store(s){inserted.reserve(count);}
        void add(std::uint64_t access,std::size_t id) {
            const auto [it,ok]=store.tq4_lru.emplace(access,id);
            if(!ok)throw std::logic_error("Duplicate QSA RAM LRU entry");
            inserted.push_back(it);
        }
        void commit(){committed=true;}
        ~IndexInsert(){if(!committed)for(auto it:inserted)store.tq4_lru.erase(it);}
    };
    std::uint64_t serial=0;
    RamKvStoreStats counters;
    Impl(RamKvStoreConfig c,Allocate a,Encode e,Publish p)
        :config(std::move(c)),allocate(std::move(a)),encode(std::move(e)),publish(std::move(p)) {
        if(!allocate || !encode || !publish || (config.pressure.ssd_enabled && !config.session_ssd))
            throw std::invalid_argument("QSA RAM store requires an allocator, codec, publisher and enabled SSD store");
    }
    std::uint64_t next() {
        if(serial==std::numeric_limits<std::uint64_t>::max()-1)
            throw std::overflow_error("QSA RAM image generation exhausted");
        return ++serial;
    }
    void touch(std::size_t id,std::uint64_t epoch) {
        auto& entry=entries[id];if(epoch<=entry.access)return;
        if(entry.tier==RamKvTier::Tq4) {
            auto node=tq4_lru.extract({entry.access,id});
            if(node.empty())throw std::logic_error("Missing QSA RAM LRU entry");
            node.value().first=epoch;tq4_lru.insert(std::move(node));
        }
        entry.access=epoch;
    }
    std::vector<RamPressureBlock> pressure()const {
        std::vector<RamPressureBlock> result;result.reserve(entries.size());
        for(std::size_t i=0;i<entries.size();++i) {
            const auto& e=entries[i];
            auto bytes=e.bytes;if(e.image)bytes[unsigned(e.tier)]=e.image->bytes();
            result.push_back({i,e.access,4,e.tier,bytes,e.image && e.image.use_count()>1});
        }
        return result;
    }
    std::size_t ram_bytes()const {
        return counters.ram_bytes;
    }
    RamPressurePlan plan(std::size_t budget,const RamPressureConfig& policy)const {
        if(!policy.quantize_tokens || policy.quantize_tokens%4 || unsigned(policy.canonical_tier)>2)
            throw std::invalid_argument("QSA RAM quantization batch must contain complete four-token blocks");
        RamPressurePlan result;result.budget_bytes=budget;result.ram_bytes=counters.ram_bytes;result.tokens=counters.tokens;
        const auto blocks=std::uint64_t(entries.size());
        const auto floor=[&](unsigned n){return 4*(blocks/20*n+(blocks%20*n+19)/20);};
        result.minimum_fp16=policy.canonical_tier==RamKvTier::Fp16?floor(1):0;
        result.minimum_affine8=policy.canonical_tier!=RamKvTier::Tq4?std::min(4*blocks-result.minimum_fp16,floor(3)):0;
        if(result.fits())return result;
        const auto spare16=result.tokens[0]>result.minimum_fp16?result.tokens[0]-result.minimum_fp16:0;
        const auto spare8=result.tokens[1]>result.minimum_affine8?result.tokens[1]-result.minimum_affine8:0;
        if(policy.ssd_enabled && spare16<policy.quantize_tokens && spare8<policy.quantize_tokens) {
            // Between full quantization batches, growth usually needs just a
            // few cold4-bit evictions. Avoid rebuilding all context heaps.
            RamPressureAction action;action.from=RamKvTier::Tq4;action.to=RamKvTier::Ssd;
            for(const auto& [access,id]:tq4_lru) {
                if(result.fits())break;
                const auto& e=entries[id];
                if(e.image.use_count()!=1){result.needs_readers=true;continue;}
                action.blocks.push_back(id);action.tokens+=4;action.saved_bytes+=e.image->bytes();
                result.ram_bytes-=e.image->bytes();result.tokens[2]-=4;result.tokens[3]+=4;
            }
            if(!action.blocks.empty())result.actions.push_back(std::move(action));
            return result;
        }
        return plan_ram_pressure(pressure(),budget,policy);
    }
    RamPressurePlan observed(std::size_t budget,bool changed=false,bool allow_ssd=true)const {
        auto policy=config.pressure;policy.ssd_enabled=policy.ssd_enabled && allow_ssd;
        auto result=plan(budget,policy);
        // The planner describes possible savings, not completed writes or
        // quantization. Report the bytes currently owned by this store.
        result.actions.clear();result.ram_bytes=ram_bytes();result.tokens=counters.tokens;
        result.needs_readers|=changed;return result;
    }
    std::shared_ptr<const RamKvBacking> spill(const Entry& e) {
        if(e.backing)return e.backing;
        if(!config.pressure.ssd_enabled || !config.session_ssd)
            throw std::runtime_error("QSA RAM exhausted with session SSD KV disabled");
        validate(e.image);
        std::vector<std::uint8_t> bytes(e.image->bytes());
        std::memcpy(bytes.data(),e.image->key,e.image->key_bytes);
        std::memcpy(bytes.data()+e.image->key_bytes,e.image->value,e.image->value_bytes);
        auto result=std::make_shared<RamKvBacking>();
        result->store=config.session_ssd;result->block=result->store->write(std::move(bytes));
        result->key_bytes=e.image->key_bytes;result->value_bytes=e.image->value_bytes;
        result->key_bits=e.image->key_bits;result->value_bits=e.image->value_bits;
        result->key_state_bits=e.image->key_state_bits;result->value_state_bits=e.image->value_state_bits;
        result->tier=e.tier;
        return result;
    }
    RamKvImagePtr load(const RamKvBacking& backing) {
        auto result=allocate(read_backing(backing));validate(result);return result;
    }
};
RamKvStore::RamKvStore(RamKvStoreConfig c,Allocate a,Encode e,Publish p)
    :impl_(std::make_shared<Impl>(std::move(c),std::move(a),std::move(e),std::move(p))){}
RamKvStore::~RamKvStore()=default;

std::size_t RamKvStore::append(RamKvImagePtr image,std::array<std::size_t,3> sizes,
        std::uint64_t epoch,std::shared_ptr<const RamKvBacking> prefix) {
    return append_at_tier(std::move(image),sizes,impl_->config.pressure.canonical_tier,epoch,std::move(prefix));
}
std::size_t RamKvStore::append_at_tier(RamKvImagePtr image,std::array<std::size_t,3> sizes,
        RamKvTier initial,std::uint64_t epoch,std::shared_ptr<const RamKvBacking> prefix) {
    validate(image);auto& s=*impl_;const auto tier=unsigned(initial);
    if(tier>2 || !sizes[2] || sizes[0]<sizes[1] || sizes[1]<sizes[2] || image->bytes()>sizes[tier])
        throw std::invalid_argument("QSA RAM tier size does not match its canonical image");
    if(prefix && ((!prefix->read && (!prefix->store || !prefix->block)) || !prefix->key_bytes || !prefix->value_bytes ||
                  unsigned(prefix->tier)>2))throw std::invalid_argument("Invalid QSA prefix backing");
    std::lock_guard<std::mutex> lock(s.mutex);const auto id=s.entries.size(),generation=s.next();
    if(image->bytes()>std::numeric_limits<std::size_t>::max()-s.counters.ram_bytes)
        throw std::overflow_error("QSA RAM byte count overflow");
    if(id==s.entries.capacity())s.entries.reserve(id?std::max(id+1,id+id/2):1);
    Impl::IndexInsert index(s,tier==2?1:0);if(tier==2)index.add(epoch,id);
    s.publish({{id,generation,image,{},initial}});
    s.counters.ram_bytes+=image->bytes();s.counters.tokens[tier]+=4;
    s.entries.push_back({std::move(image),s.config.prefix_enabled?std::move(prefix):nullptr,sizes,generation,epoch,initial});
    index.commit();
    return id;
}
void RamKvStore::touch(const std::vector<std::size_t>& ids,std::uint64_t epoch) {
    auto& s=*impl_;std::lock_guard<std::mutex> lock(s.mutex);
    for(auto id:ids)if(id>=s.entries.size())throw std::out_of_range("QSA RAM touch block");
    for(auto id:ids)s.touch(id,epoch);
}
std::vector<RamKvChange> RamKvStore::lease(const std::vector<std::size_t>& ids,std::uint64_t epoch) {
    auto& s=*impl_;std::vector<RamKvChange> result;result.reserve(ids.size());
    std::lock_guard<std::mutex> lock(s.mutex);
    for(auto id:ids)if(id>=s.entries.size())throw std::out_of_range("QSA RAM lease block");
    for(auto id:ids) {
        s.touch(id,epoch);const auto& entry=s.entries[id];
        if(entry.image)result.push_back({id,entry.generation,entry.image,{},entry.tier});
    }
    return result;
}
RamPressurePlan RamKvStore::maintain(std::size_t budget,bool allow_ssd) {
    auto& s=*impl_;std::lock_guard<std::mutex> worker(s.maintenance);
    auto pressure=s.config.pressure;
    pressure.ssd_enabled=pressure.ssd_enabled && allow_ssd;
    for(;;) {
    RamPressurePlan plan;
    {std::lock_guard<std::mutex> lock(s.mutex);plan=s.plan(budget,pressure);}
    for(const auto& action:plan.actions) {
        std::vector<Impl::Entry> old;old.reserve(action.blocks.size());
        std::vector<long> references;references.reserve(action.blocks.size());
        {std::lock_guard<std::mutex> lock(s.mutex);
            for(auto id:action.blocks) {
                if(id>=s.entries.size() || s.entries[id].tier!=action.from)return s.observed(budget,true,allow_ssd);
                references.push_back(s.entries[id].image.use_count());
                old.push_back(s.entries[id]);
            }
        }
        std::vector<RamKvImagePtr> encoded;
        std::vector<std::shared_ptr<const RamKvBacking>> backings;
        if(action.to==RamKvTier::Ssd) {
            backings.reserve(old.size());for(const auto& e:old)backings.push_back(s.spill(e));
        }else {
            std::vector<RamKvImagePtr> input;input.reserve(old.size());for(const auto& e:old)input.push_back(e.image);
            encoded=s.encode(input,action.to);
            if(encoded.size()!=old.size())throw std::runtime_error("QSA RAM encoder changed batch size");
            for(std::size_t i=0;i<encoded.size();++i) {
                validate(encoded[i]);
                if(encoded[i]->bytes()>old[i].bytes[unsigned(action.to)])
                    throw std::runtime_error("QSA RAM encoded image exceeded its planned precision budget");
            }
        }
        std::lock_guard<std::mutex> lock(s.mutex);
        for(std::size_t i=0;i<old.size();++i) {
            const auto id=action.blocks[i];
            if(id>=s.entries.size() || s.entries[id].generation!=old[i].generation || s.entries[id].access!=old[i].access ||
               s.entries[id].image.use_count()>references[i]+1)
                return s.observed(budget,true,allow_ssd);
        }
        if(action.from==RamKvTier::Fp16 || action.from==RamKvTier::Affine8) {
            // Appends can raise a protected floor while the codec runs.
            // Check the live population before publishing that old plan.
            const auto live=s.plan(s.ram_bytes(),pressure);
            const auto tier=unsigned(action.from);
            const auto minimum=tier==0?live.minimum_fp16:live.minimum_affine8;
            if(live.tokens[tier]<minimum || action.tokens>live.tokens[tier]-minimum)
                return s.observed(budget,true,allow_ssd);
        }
        Impl::IndexInsert index(s,action.to==RamKvTier::Tq4?old.size():0);
        if(action.to==RamKvTier::Tq4)for(std::size_t i=0;i<old.size();++i)index.add(old[i].access,action.blocks[i]);
        std::vector<RamKvChange> changes;changes.reserve(old.size());
        for(std::size_t i=0;i<old.size();++i)changes.push_back({action.blocks[i],
            action.to==RamKvTier::Ssd?old[i].generation:s.next(),
            action.to==RamKvTier::Ssd?RamKvImagePtr{}:encoded[i],old[i].image,action.to});
        s.publish(changes);
        index.commit();
        for(std::size_t i=0;i<old.size();++i) {
            auto& e=s.entries[action.blocks[i]];
            if(e.tier==RamKvTier::Tq4)s.tq4_lru.erase({e.access,action.blocks[i]});
            s.counters.ram_bytes-=e.image->bytes();
            s.counters.tokens[unsigned(e.tier)]-=4;s.counters.tokens[unsigned(action.to)]+=4;
            e.image=changes[i].image;e.generation=changes[i].generation;e.tier=action.to;
            if(e.image)s.counters.ram_bytes+=e.image->bytes();
            if(action.to==RamKvTier::Ssd) {
                e.backing=backings[i];++s.counters.spilled_blocks;
                if(!old[i].backing)s.counters.ssd_written_bytes+=old[i].image->bytes();
                else if(old[i].backing->store!=s.config.session_ssd)s.counters.prefix_reused_bytes+=old[i].image->bytes();
            }else ++s.counters.quantized_blocks;
        }
    }
    std::lock_guard<std::mutex> lock(s.mutex);
    // Decoding may append new microblocks while a codec batch runs. Apply
    // any remaining feasible actions before declaring protected floors to
    // be the limiting factor. A reader-blocked plan returns for a later retry.
    if(s.plan(budget,pressure).actions.empty())return s.observed(budget,false,allow_ssd);
    }
}
RamKvImagePtr RamKvStore::read(std::size_t id,std::size_t budget,bool admit) {
    auto& s=*impl_;std::lock_guard<std::mutex> worker(s.maintenance);
    Impl::Entry original;
    {std::lock_guard<std::mutex> lock(s.mutex);original=s.entries.at(id);if(original.image)return original.image;}
    if(!original.backing)throw std::runtime_error("QSA SSD block has no retained backing");
    auto image=s.load(*original.backing);
    if(original.backing->tier!=RamKvTier::Tq4) {
        auto converted=s.encode({image},RamKvTier::Tq4);
        if(converted.size()!=1)throw std::runtime_error("QSA SSD promotion changed batch size");
        image=std::move(converted[0]);validate(image);
    }
    if(image->bytes()>original.bytes[2])throw std::runtime_error("QSA SSD promotion exceeded its RAM image size");
    {std::lock_guard<std::mutex> lock(s.mutex);s.counters.ssd_read_bytes+=original.backing->key_bytes+original.backing->value_bytes;}
    if(!admit)return image;
    std::vector<std::pair<std::size_t,Impl::Entry>> victims;
    {std::lock_guard<std::mutex> lock(s.mutex);
        if(id>=s.entries.size() || s.entries[id].generation!=original.generation || s.entries[id].tier!=RamKvTier::Ssd)return image;
        auto used=s.ram_bytes();
        if(image->bytes()>budget)return image;
        for(const auto& [access,victim]:s.tq4_lru) {
            if(used<=budget-image->bytes() || access>=original.access)break;
            const auto& e=s.entries[victim];if(e.image.use_count()!=1)continue;
            victims.emplace_back(victim,e);used-=e.image->bytes();
        }
        if(used>budget-image->bytes())return image;
    }
    std::vector<std::shared_ptr<const RamKvBacking>> backing;backing.reserve(victims.size());
    for(const auto& victim:victims)backing.push_back(s.spill(victim.second));
    std::lock_guard<std::mutex> lock(s.mutex);
    if(id>=s.entries.size() || s.entries[id].generation!=original.generation || s.entries[id].tier!=RamKvTier::Ssd)return image;
    for(const auto& victim:victims)if(victim.first>=s.entries.size() ||
        s.entries[victim.first].generation!=victim.second.generation || s.entries[victim.first].access!=victim.second.access ||
        s.entries[victim.first].image.use_count()>2)return image;
    auto remaining=s.ram_bytes();for(const auto& victim:victims)remaining-=victim.second.image->bytes();
    if(remaining>budget-image->bytes())return image;
    const auto generation=original.backing->tier==RamKvTier::Tq4?original.generation:s.next();
    Impl::IndexInsert index(s,1);index.add(s.entries[id].access,id);
    std::vector<RamKvChange> changes;
    for(const auto& victim:victims)changes.push_back({victim.first,victim.second.generation,{},victim.second.image,RamKvTier::Ssd});
    changes.push_back({id,generation,image,{},RamKvTier::Tq4});s.publish(changes);
    index.commit();
    for(std::size_t i=0;i<victims.size();++i) {
        auto& e=s.entries[victims[i].first];s.tq4_lru.erase({e.access,victims[i].first});
        e.image.reset();e.tier=RamKvTier::Ssd;e.backing=backing[i];
        s.counters.ram_bytes-=victims[i].second.image->bytes();s.counters.tokens[2]-=4;s.counters.tokens[3]+=4;
        ++s.counters.spilled_blocks;
        if(!victims[i].second.backing)s.counters.ssd_written_bytes+=victims[i].second.image->bytes();
        else if(victims[i].second.backing->store!=s.config.session_ssd)s.counters.prefix_reused_bytes+=victims[i].second.image->bytes();
    }
    auto& e=s.entries[id];e.image=image;e.tier=RamKvTier::Tq4;e.generation=generation;++s.counters.promoted_blocks;
    s.counters.ram_bytes+=image->bytes();s.counters.tokens[3]-=4;s.counters.tokens[2]+=4;
    return image;
}
RamKvImagePtr RamKvStore::resident(std::size_t id)const {
    auto& s=*impl_;std::lock_guard<std::mutex> lock(s.mutex);return s.entries.at(id).image;
}
std::uint64_t RamKvStore::generation(std::size_t id)const {
    auto& s=*impl_;std::lock_guard<std::mutex> lock(s.mutex);return s.entries.at(id).generation;
}
std::shared_ptr<const RamKvBacking> RamKvStore::backing(std::size_t id)const {
    auto& s=*impl_;std::lock_guard<std::mutex> lock(s.mutex);return s.entries.at(id).backing;
}
void RamKvStore::truncate(std::size_t blocks) {
    auto& s=*impl_;std::lock_guard<std::mutex> lock(s.mutex);
    if(blocks>s.entries.size())throw std::out_of_range("QSA RAM truncation grows context");
    std::vector<RamKvChange> changes;
    for(auto i=blocks;i<s.entries.size();++i)changes.push_back({i,s.next(),{},s.entries[i].image,RamKvTier::Ssd});
    s.publish(changes);
    for(auto i=blocks;i<s.entries.size();++i) {
        const auto& e=s.entries[i];if(e.image)s.counters.ram_bytes-=e.image->bytes();s.counters.tokens[unsigned(e.tier)]-=4;
        if(e.tier==RamKvTier::Tq4)s.tq4_lru.erase({e.access,i});
    }
    s.entries.resize(blocks);
}
RamKvStoreStats RamKvStore::stats()const {
    auto& s=*impl_;std::lock_guard<std::mutex> lock(s.mutex);auto result=s.counters;
    result.blocks=s.entries.size();
    return result;
}
MicroblockImage RamKvSnapshotBlock::read()const {
    if(!image) {
        if(!backing)throw std::runtime_error("QSA snapshot has no retained image");
        return read_backing(*backing);
    }
    validate(image);MicroblockImage result;
    result.key.assign(image->key,image->key+image->key_bytes);result.value.assign(image->value,image->value+image->value_bytes);
    result.key_bits=image->key_bits;result.value_bits=image->value_bits;
    result.key_state_bits=image->key_state_bits;result.value_state_bits=image->value_state_bits;return result;
}
void RamKvSnapshot::bind_prefix(std::size_t id,std::shared_ptr<const RamKvBacking> backing) {
    auto& block=blocks.at(id);
    if(!backing || (!backing->read && (!backing->store || !backing->block)) ||
            !backing->key_bytes || !backing->value_bytes || unsigned(backing->tier)>2)
        throw std::invalid_argument("Invalid QSA snapshot prefix backing");
    const auto* previous=block.image?nullptr:block.backing.get();
    const auto kb=block.image?block.image->key_bytes:previous?previous->key_bytes:0;
    const auto vb=block.image?block.image->value_bytes:previous?previous->value_bytes:0;
    const auto kbits=block.image?block.image->key_bits:previous?previous->key_bits:0;
    const auto vbits=block.image?block.image->value_bits:previous?previous->value_bits:0;
    const auto ks=block.image?block.image->key_state_bits:previous?previous->key_state_bits:0;
    const auto vs=block.image?block.image->value_state_bits:previous?previous->value_state_bits:0;
    if(backing->key_bytes!=kb || backing->value_bytes!=vb || backing->key_bits!=kbits ||
            backing->value_bits!=vbits || backing->key_state_bits!=ks || backing->value_state_bits!=vs)
        throw std::invalid_argument("QSA prefix backing changed its snapshot precision");
    if(bind_live)bind_live(id,block.generation,backing);
    block.backing=std::move(backing);block.image.reset();
}
std::shared_ptr<RamKvSnapshot> RamKvStore::snapshot(std::size_t count)const {
    auto& s=*impl_;std::lock_guard<std::mutex> lock(s.mutex);
    if(count>s.entries.size())throw std::out_of_range("QSA snapshot exceeds context");
    auto result=std::make_shared<RamKvSnapshot>();result->blocks.reserve(count);
    for(std::size_t i=0;i<count;++i) {
        const auto& e=s.entries[i];result->blocks.push_back({e.image,e.backing,e.bytes,e.generation,e.access,e.tier});
    }
    result->bind_live=[weak=std::weak_ptr<Impl>(impl_)](auto id,auto generation,auto backing) {
        if(auto store=weak.lock()) {
            std::lock_guard<std::mutex> lock(store->mutex);
            if(id<store->entries.size() && store->entries[id].generation==generation) {
                store->config.prefix_enabled=true;store->entries[id].backing=std::move(backing);
            }
        }
    };
    return result;
}
void RamKvStore::restore(const RamKvSnapshot& state,std::size_t budget,bool borrow_prefix,bool reuse_resident) {
    auto& s=*impl_;std::lock_guard<std::mutex> worker(s.maintenance);
    std::vector<RamPressureBlock> metadata;metadata.reserve(state.blocks.size());
    for(std::size_t i=0;i<state.blocks.size();++i) {
        const auto& b=state.blocks[i];auto tier=b.tier;
        if(tier==RamKvTier::Ssd && !s.config.pressure.ssd_enabled)tier=RamKvTier::Tq4;
        auto bytes=b.bytes;if(b.image && unsigned(b.tier)<3)bytes[unsigned(b.tier)]=b.image->bytes();
        metadata.push_back({i,b.access,4,tier,bytes,false});
    }
    const auto plan=plan_ram_pressure(metadata,budget,s.config.pressure);
    if(!plan.fits())throw std::runtime_error("QSA prefix cannot fit the RAM budget and protected precision floors");
    for(const auto& action:plan.actions)for(auto id:action.blocks)metadata[id].tier=action.to;
    truncate(0);
    {std::lock_guard<std::mutex> lock(s.mutex);s.config.prefix_enabled=borrow_prefix;}
    // Stage at most one policy batch. File-only cold blocks retain their
    // backing directly, while admitted blocks use the destination allocator.
    const std::size_t batch=s.config.pressure.quantize_tokens/4;
    for(std::size_t begin=0;begin<metadata.size();begin+=batch) {
        const auto end=std::min(begin+batch,metadata.size()),count=end-begin;
        std::vector<RamKvImagePtr> images(count);
        std::vector<std::shared_ptr<const RamKvBacking>> backing(count);
        for(std::size_t j=0;j<count;++j) {
            const auto& src=state.blocks[begin+j];
            const bool keep=src.backing && (borrow_prefix || src.backing->store==s.config.session_ssd);
            if(keep)backing[j]=src.backing;
            if(metadata[begin+j].tier==RamKvTier::Ssd && backing[j])continue;
            images[j]=reuse_resident && src.image?src.image:s.allocate(src.read());validate(images[j]);
        }
        for(unsigned tier=0;tier<3;++tier) {
            std::vector<RamKvImagePtr> source;std::vector<std::size_t> ids;
            for(std::size_t j=0;j<count;++j)if(images[j]) {
                const auto target=metadata[begin+j].tier==RamKvTier::Ssd?RamKvTier::Tq4:metadata[begin+j].tier;
                const auto& src=state.blocks[begin+j];
                // SSD is a placement, not an encoding. A RAM-only paged
                // decoder may own the retained SSD image in a host buffer.
                const auto actual=src.image?(src.tier!=RamKvTier::Ssd?src.tier:
                    src.image->key_bits==0 && src.image->value_bits==0?RamKvTier::Fp16:
                    src.image->key_bits==affine8_tag && src.image->value_bits==affine8_tag?RamKvTier::Affine8:RamKvTier::Tq4):
                    src.backing->tier;
                if(unsigned(target)>=tier && unsigned(actual)<tier){source.push_back(images[j]);ids.push_back(j);}
            }
            if(source.empty())continue;
            const auto encoded=s.encode(source,RamKvTier(tier));
            if(encoded.size()!=ids.size())throw std::runtime_error("QSA prefix codec changed its block count");
            for(std::size_t j=0;j<ids.size();++j){validate(encoded[j]);images[ids[j]]=encoded[j];}
        }
        for(std::size_t j=0;j<count;++j) {
            const auto id=begin+j;const auto& src=state.blocks[id];const auto target=metadata[id].tier;
            if(target!=RamKvTier::Ssd) {
                append_at_tier(images[j],src.bytes,target,src.access,backing[j]);continue;
            }
            const bool written=!backing[j];
            if(written)backing[j]=s.spill({images[j],{},src.bytes,0,src.access,RamKvTier::Tq4});
            std::lock_guard<std::mutex> lock(s.mutex);const auto generation=s.next();
            if(s.entries.size()==s.entries.capacity())s.entries.reserve(std::max(id+1,id+id/2));
            s.publish({{id,generation,{},{},RamKvTier::Ssd}});
            s.entries.push_back({{},backing[j],src.bytes,generation,src.access,RamKvTier::Ssd});s.counters.tokens[3]+=4;
            if(written)s.counters.ssd_written_bytes+=backing[j]->key_bytes+backing[j]->value_bytes;
        }
    }
}
} // namespace tyloinfer::qsa
