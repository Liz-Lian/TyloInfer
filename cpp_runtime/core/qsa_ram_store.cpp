#include "tyloinfer/qsa_ram_store.h"
#include "tyloinfer/file_read_window.h"
#include "tyloinfer/host_parallel.h"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <queue>
#include <optional>
#include <stdexcept>
#include <thread>

namespace tyloinfer::qsa {
namespace {
struct SequentialFileSpan {
    const mfq::FileRangeReader* reader;
    mfq::ReadSpan span;
};
using SequentialFileGroup=std::pair<std::size_t,std::size_t>;
constexpr std::size_t sequential_read_window=ScopedFileReadWindow::capacity;
constexpr unsigned sequential_read_threads=4;

bool read_sequential_files(const std::vector<RamKvReadInto>& requests,std::uint64_t& completed,bool padded) {
    completed=0;
    bool files=false;
    std::uintptr_t previous=0;
    for(const auto& request:requests) {
        const auto& source=*request.source;
        const auto* image=source.image.get();
        const auto* backing=source.backing.get();
        const auto key_bytes=image?image->key_bytes:backing->key_bytes;
        const auto value_bytes=image?image->value_bytes:backing->value_bytes;
        const auto extent=padded?((key_bytes+15)&~std::size_t(15)):key_bytes;
        const auto begin=reinterpret_cast<std::uintptr_t>(request.destination);
        // Parallel reads require disjoint destinations in caller order.
        if(begin<previous || extent+value_bytes>UINTPTR_MAX-begin)return false;
        previous=begin+extent+value_bytes;
        if(image)continue;
        if(!backing->source || !backing->source->file_reader)return false;
        const auto& input=*backing->source;
        input.validate(backing->offset,key_bytes+value_bytes);
        if(backing->offset>UINT64_MAX-input.file_offset)
            throw std::out_of_range("QSA sequential file offset overflow");
        const auto offset=input.file_offset+backing->offset;
        if(offset>input.file_reader->size() || key_bytes+value_bytes>input.file_reader->size()-offset)
            throw std::out_of_range("QSA sequential file extent changed");
        files=true;
    }
    if(!files)return false;
    std::vector<SequentialFileSpan> spans;
    std::vector<SequentialFileGroup> groups;
    spans.reserve(requests.size());
    groups.reserve(requests.size());
    for(const auto& request:requests) {
        const auto& source=*request.source;
        if(source.image) {
            const auto& image=*source.image;
            const auto extent=padded?((image.key_bytes+15)&~std::size_t(15)):image.key_bytes;
            std::memcpy(request.destination,image.key,image.key_bytes);
            std::memcpy(request.destination+extent,image.value,image.value_bytes);
            continue;
        }
        const auto& backing=*source.backing;
        const auto& input=*backing.source;
        spans.push_back({input.file_reader.get(),{input.file_offset+backing.offset,
            reinterpret_cast<std::byte*>(request.destination),backing.key_bytes+backing.value_bytes}});
        if(groups.empty() || spans[groups.back().first].reader!=input.file_reader.get())
            groups.emplace_back(spans.size()-1,spans.size());
        else groups.back().second=spans.size();
    }
    std::atomic<std::uint64_t> done{0};
    const auto threads=unsigned(std::min<std::size_t>({sequential_read_threads,
        std::max(1u,std::thread::hardware_concurrency()),groups.size()}));
    const auto read_group=[&](std::int64_t first,std::int64_t last) {
        // Release each worker's window before its workspace reservation ends.
        std::unique_ptr<std::uint8_t[]> buffer;
        for(auto group=first;group<last;++group) {
            const auto [begin,end]=groups[std::size_t(group)];
            const auto* reader=spans[begin].reader;
            for(auto i=begin;i<end;) {
                auto after=i+1;
                const auto start=spans[i].span.offset;
                auto finish=start+spans[i].span.size;
                std::size_t useful=spans[i].span.size;
                while(after<end) {
                    const auto& next=spans[after].span;
                    if(next.offset<start || next.size>SIZE_MAX-useful)break;
                    const auto high=std::max(finish,next.offset+next.size);
                    const auto bytes=high-start;
                    if(bytes>sequential_read_window || (bytes+1)/2>useful+next.size)break;
                    useful+=next.size;
                    finish=high;
                    ++after;
                }
                if(after==i+1)reader->read(start,spans[i].span.destination,spans[i].span.size);
                else {
                    if(!buffer)buffer=std::make_unique<std::uint8_t[]>(sequential_read_window);
                    reader->read(start,reinterpret_cast<std::byte*>(buffer.get()),std::size_t(finish-start));
                    for(auto at=i;at<after;++at) {
                        const auto& span=spans[at].span;
                        std::memcpy(span.destination,buffer.get()+std::size_t(span.offset-start),span.size);
                    }
                }
                done.fetch_add(useful,std::memory_order_relaxed);
                i=after;
            }
        }
    };
    try {
        static thread_local mfq::HostParallelPool readers;
        readers.run(0,static_cast<std::int64_t>(groups.size()),1,int(threads),read_group);
    }catch(...) {
        completed=done.load(std::memory_order_relaxed);
        throw;
    }
    completed=done.load(std::memory_order_relaxed);
    for(const auto& request:requests)if(!request.source->image) {
        const auto& backing=*request.source->backing;
        const auto extent=padded?((backing.key_bytes+15)&~std::size_t(15)):backing.key_bytes;
        if(extent!=backing.key_bytes)
            std::memmove(request.destination+extent,request.destination+backing.key_bytes,backing.value_bytes);
    }
    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        static std::atomic<bool> reported{false};
        if(!reported.exchange(true))std::fprintf(stderr,
            "native_qsa_sequential_files actual_match=1 requests=%zu groups=%zu threads=%u window_bytes=%zu max_read_amplification=2\n",
            requests.size(),groups.size(),threads,sequential_read_window);
    }
    return true;
}
} // namespace

std::size_t RamKvStore::sequential_read_workspace(std::size_t requests) {
    constexpr auto per_request=sizeof(SequentialFileSpan)+sizeof(SequentialFileGroup);
    constexpr auto windows=sequential_read_threads*sequential_read_window;
    if(requests>(SIZE_MAX-windows)/per_request)
        throw std::overflow_error("QSA sequential read workspace overflow");
    return requests*per_request+windows;
}

void RamKvReadSource::validate(std::size_t offset,std::size_t count)const {
    if(!valid())throw std::runtime_error("QSA byte source has no reader");
    if(offset>bytes || count>bytes-offset)throw std::out_of_range("QSA byte source extent changed");
}
std::vector<std::uint8_t> RamKvReadSource::read(std::size_t offset,std::size_t count)const {
    validate(offset,count);
    if(!count)return {};
    if(read_range) {
        auto result=read_range(offset,count);
        if(result.size()!=count)throw std::runtime_error("QSA byte source read length changed");
        return result;
    }
    std::vector<std::uint8_t> result(count);
    read_into(offset,result.data(),count);return result;
}
void RamKvReadSource::read_into(std::size_t offset,std::uint8_t* destination,std::size_t count)const {
    validate(offset,count);
    if(count && !destination)throw std::invalid_argument("QSA byte source destination is null");
    if(!count)return;
    if(read_range_into)read_range_into(offset,destination,count);
    else if(file_reader) {
        if(offset>UINT64_MAX-file_offset)throw std::out_of_range("QSA file offset overflow");
        file_reader->read(file_offset+offset,reinterpret_cast<std::byte*>(destination),count);
    }
    else {
        const auto data=read(offset,count);
        if(count)std::memcpy(destination,data.data(),count);
    }
}
void RamKvReadSource::read_into_batch(const std::vector<mfq::ReadSpan>& spans)const {
    for(const auto& span:spans) {
        if(span.offset>bytes)throw std::out_of_range("QSA byte source extent changed");
        validate(span.offset,span.size);
        if(span.size && !span.destination)throw std::invalid_argument("QSA byte source destination is null");
    }
    if(spans.empty())return;
    if(read_ranges_into)read_ranges_into(spans);
    else for(const auto& span:spans)read_into(span.offset,reinterpret_cast<std::uint8_t*>(span.destination),span.size);
}
namespace {
void validate(const RamKvImagePtr& image) {
    if(!image || !image->owner || !image->key || !image->value || !image->key_bytes || !image->value_bytes ||
        image->key_bytes>std::numeric_limits<std::size_t>::max()-image->value_bytes)
        throw std::invalid_argument("Invalid QSA RAM image");
}
bool same_encoding(const RamKvImage& image,const RamKvBacking& backing) {
    return image.key_bytes==backing.key_bytes && image.value_bytes==backing.value_bytes &&
        image.key_bits==backing.key_bits && image.value_bits==backing.value_bits &&
        image.key_state_bits==backing.key_state_bits && image.value_state_bits==backing.value_state_bits;
}
bool valid_backing(const RamKvBacking& backing) {
    if(!backing.key_bytes || !backing.value_bytes ||
            backing.key_bytes>std::numeric_limits<std::size_t>::max()-backing.value_bytes)return false;
    if(!backing.source)return backing.store && backing.block;
    return backing.source->valid() && backing.offset<=backing.source->bytes &&
        backing.key_bytes+backing.value_bytes<=backing.source->bytes-backing.offset;
}
RamKvImage read_backing_view(const RamKvBacking& backing,std::size_t* retained_bytes=nullptr) {
    if(!valid_backing(backing))throw std::runtime_error("Invalid QSA SSD backing");
    std::shared_ptr<const std::vector<std::uint8_t>> bytes;
    if(backing.source)bytes=std::make_shared<const std::vector<std::uint8_t>>(
        backing.source->read(backing.offset,backing.key_bytes+backing.value_bytes));
    else bytes=backing.store->read(backing.block);
    if(bytes->size()!=backing.key_bytes+backing.value_bytes)throw std::runtime_error("QSA SSD image length changed");
    RamKvImage image;
    image.owner=bytes;image.key=bytes->data();image.value=bytes->data()+backing.key_bytes;
    image.key_bytes=backing.key_bytes;image.value_bytes=backing.value_bytes;
    image.key_bits=backing.key_bits;image.value_bits=backing.value_bits;
    image.key_state_bits=backing.key_state_bits;image.value_state_bits=backing.value_state_bits;
    if(retained_bytes)*retained_bytes=bytes->capacity();
    return image;
}
MicroblockImage copy_image(const RamKvImage& view) {
    MicroblockImage image;
    image.key.assign(view.key,view.key+view.key_bytes);
    image.value.assign(view.value,view.value+view.value_bytes);
    image.key_bits=view.key_bits;image.value_bits=view.value_bits;
    image.key_state_bits=view.key_state_bits;image.value_state_bits=view.value_state_bits;
    return image;
}
MicroblockImage read_backing(const RamKvBacking& backing) {
    return copy_image(read_backing_view(backing));
}
}
struct RamKvStore::Impl {
    using Lru=std::set<std::pair<std::uint64_t,std::size_t>>;
    struct Entry {
        RamKvImagePtr image;
        std::shared_ptr<const RamKvBacking> backing;
        std::array<std::size_t,3> bytes{};
        std::uint64_t generation=0,access=0;
        RamKvTier tier=RamKvTier::Fp16;
        // Valid for every resident tier. Node handles keep this stable across
        // entry-vector growth and avoid a tree lookup on touch or eviction.
        Lru::iterator lru{};
    };
    RamKvStoreConfig config;
    Allocate allocate;Encode encode;Publish publish;AllocateView allocate_view;
    mutable std::mutex mutex;
    // Serialize maintenance and admissions; append/touch can proceed while
    // an encoder or SSD read is running. Truncation invalidates its versions.
    std::mutex maintenance;
    std::vector<Entry> entries;
    // Admissions are serialized by maintenance; release image owners after
    // each publication while retaining only the descriptor vector capacity.
    std::vector<RamKvChange> admission_changes;
    Lru tq4_lru;
    std::array<Lru,2> higher_lru;
    Lru& lru_for(unsigned tier){return tier==2?tq4_lru:higher_lru.at(tier);}
    const Lru& lru_for(unsigned tier)const{return tier==2?tq4_lru:higher_lru.at(tier);}
    struct IndexInsert {
        Lru& tree;
        Lru::iterator single;
        std::vector<Lru::iterator> inserted;
        std::size_t capacity=0,used=0;
        bool committed=false;
        Lru::const_iterator hint;
        IndexInsert(Impl& s,std::size_t count,unsigned tier=2)
            :tree(s.lru_for(tier)),capacity(count),hint(tree.end()){
            if(count>1)inserted.reserve(count);
        }
        Lru::iterator at(std::size_t i)const{return capacity==1?single:inserted[i];}
        IndexInsert(const IndexInsert&)=delete;
        IndexInsert& operator=(const IndexInsert&)=delete;
        Lru::iterator add(std::uint64_t access,std::size_t id) {
            if(used==capacity)throw std::logic_error("QSA LRU transaction exceeded its prepared count");
            const auto size=tree.size();
            const auto it=tree.emplace_hint(hint,access,id);
            if(tree.size()==size)throw std::logic_error("Duplicate QSA RAM LRU entry");
            if(capacity==1)single=it;else inserted.push_back(it);
            ++used;
            hint=it;++hint;
            return it;
        }
        void commit(){committed=true;}
        ~IndexInsert(){if(!committed)for(std::size_t i=0;i<used;++i)tree.erase(at(i));}
    };
    std::uint64_t serial=0;
    RamKvStoreStats counters;
    std::array<long double,3> precision_bytes{};
    void account_precision(const Entry& e,int direction) {
        for(unsigned i=0;i<3;++i) {
            const auto bytes=e.image && unsigned(e.tier)==i?e.image->bytes():e.bytes[i];
            precision_bytes[i]+=direction*static_cast<long double>(bytes);
        }
    }
    Impl(RamKvStoreConfig c,Allocate a,Encode e,Publish p,AllocateView v)
        :config(std::move(c)),allocate(std::move(a)),encode(std::move(e)),publish(std::move(p)),allocate_view(std::move(v)) {
        if(!allocate || !encode || !publish || (config.pressure.ssd_enabled && !config.session_ssd))
            throw std::invalid_argument("QSA RAM store requires an allocator, codec, publisher and enabled SSD store");
    }
    std::uint64_t next() {
        if(serial==std::numeric_limits<std::uint64_t>::max()-1)
            throw std::overflow_error("QSA RAM image generation exhausted");
        return ++serial;
    }
    void touch_resident(const std::vector<std::size_t>& ids,std::uint64_t epoch) {
        // Prepare all scratch storage before mutating access metadata. Sorting
        // only changed resident IDs gives consecutive (epoch,id) insertion hints;
        // the caller's lease order and duplicate images remain unchanged.
        std::array<std::vector<std::size_t>,3> changed;
        for(auto id:ids)if(unsigned(entries[id].tier)<3 && epoch>entries[id].access) {
            auto& group=changed[unsigned(entries[id].tier)];
            if(group.empty())group.reserve(ids.size());
            group.push_back(id);
        }
        for(unsigned tier=0;tier<3;++tier)if(!changed[tier].empty()) {
            auto& tree=lru_for(tier);auto& group=changed[tier];
            std::sort(group.begin(),group.end());
            group.erase(std::unique(group.begin(),group.end()),group.end());
            auto hint=tree.lower_bound({epoch,group.front()});
            for(auto id:group) {
                auto& entry=entries[id];auto node=tree.extract(entry.lru);
                node.value().first=epoch;
                entry.lru=tree.insert(hint,std::move(node));
                hint=entry.lru;++hint;
            }
        }
    }
    std::size_t ram_bytes()const {
        return counters.ram_bytes;
    }
    RamPressurePlan plan(std::size_t budget,const RamPressureConfig& policy)const {
        if(!policy.quantize_tokens || policy.quantize_tokens%4 || unsigned(policy.canonical_tier)>2)
            throw std::invalid_argument("QSA RAM quantization batch must contain complete four-token blocks");
        RamPressurePlan result;result.budget_bytes=budget;result.ram_bytes=counters.ram_bytes;result.tokens=counters.tokens;
        const auto floors=ram_precision_floors(4*std::uint64_t(entries.size()),budget,precision_bytes,policy.canonical_tier,policy.quantize_tokens);
        result.minimum_fp16=floors[0];result.minimum_affine8=floors[1];
        if(result.fits())return result;
        const auto spare16=result.tokens[0]>result.minimum_fp16?result.tokens[0]-result.minimum_fp16:0;
        const auto spare8=result.tokens[1]>result.minimum_affine8?result.tokens[1]-result.minimum_affine8:0;
        if(spare16<policy.quantize_tokens && spare8<policy.quantize_tokens) {
            if(!policy.ssd_enabled) {
                // No complete higher-precision batch can be converted. The
                // existing TQ4 index answers whether SSD would be needed;
                // rebuilding every context block and three heaps cannot
                // produce an action when this call forbids SSD eviction.
                result.needs_ssd=std::any_of(tq4_lru.begin(),tq4_lru.end(),[&](const auto& item) {
                    return entries[item.second].image.use_count()==1;
                });
                return result;
            }
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
        // Merge each resident LRU with only the blocks demoted by this plan.
        // A small conversion batch never rebuilds full-context metadata/heaps.
        const auto newer=[&](std::size_t a,std::size_t b){return entries[a].access!=entries[b].access?
            entries[a].access>entries[b].access:a>b;};
        using Heap=std::priority_queue<std::size_t,std::vector<std::size_t>,decltype(newer)>;
        std::array<Heap,3> added{Heap(newer),Heap(newer),Heap(newer)};
        std::array<Lru::const_iterator,3> next{higher_lru[0].begin(),higher_lru[1].begin(),tq4_lru.begin()};
        const auto pop=[&](unsigned tier)->std::optional<std::size_t>{
            const auto end=lru_for(tier).end();auto& it=next[tier];
            while(it!=end && entries[it->second].image.use_count()!=1)++it;
            if(it==end && added[tier].empty())return {};
            if(!added[tier].empty() && (it==end || newer(it->second,added[tier].top()))) {
                const auto id=added[tier].top();added[tier].pop();return id;
            }
            const auto id=it->second;++it;return id;
        };
        const auto size_at=[&](std::size_t id,unsigned tier){const auto& e=entries[id];
            return unsigned(e.tier)==tier && e.image?e.image->bytes():e.bytes[tier];};
        const std::array<std::uint64_t,2> minimum{result.minimum_fp16,result.minimum_affine8};
        const auto batch=policy.quantize_tokens/4;
        for(unsigned tier=0;tier<2 && !result.fits();++tier) {
            while(!result.fits() && result.tokens[tier]>=minimum[tier] && result.tokens[tier]-minimum[tier]>=policy.quantize_tokens) {
                RamPressureAction action{RamKvTier(tier),RamKvTier(tier+1)};action.blocks.reserve(batch);
                while(action.blocks.size()<batch) {
                    const auto id=pop(tier);if(!id)break;
                    action.blocks.push_back(*id);
                }
                if(action.blocks.size()!=batch){result.needs_readers=true;break;}
                for(const auto id:action.blocks) {
                    const auto before=size_at(id,tier),after=size_at(id,tier+1);
                    if(before<after)throw std::invalid_argument("Invalid QSA RAM pressure precision sizes");
                    action.saved_bytes+=before-after;action.tokens+=4;added[tier+1].push(id);
                }
                result.ram_bytes-=action.saved_bytes;
                result.tokens[tier]-=action.tokens;result.tokens[tier+1]+=action.tokens;
                result.actions.push_back(std::move(action));
            }
        }
        if(result.fits())return result;
        if(result.tokens[0]>result.minimum_fp16 && result.tokens[0]-result.minimum_fp16>=policy.quantize_tokens)return result;
        if(result.tokens[1]>result.minimum_affine8 && result.tokens[1]-result.minimum_affine8>=policy.quantize_tokens)return result;
        if(!policy.ssd_enabled){result.needs_ssd=pop(2).has_value();return result;}
        RamPressureAction spill{RamKvTier::Tq4,RamKvTier::Ssd};
        while(!result.fits()) {
            const auto id=pop(2);if(!id)break;
            const auto bytes=size_at(*id,2);spill.blocks.push_back(*id);spill.tokens+=4;spill.saved_bytes+=bytes;
            result.ram_bytes-=bytes;result.tokens[2]-=4;result.tokens[3]+=4;
        }
        if(!spill.blocks.empty())result.actions.push_back(std::move(spill));
        if(!result.fits() && result.tokens[2])result.needs_readers=true;
        return result;
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
    RamKvImagePtr admit_loaded(std::size_t id,const Entry& original,RamKvImagePtr image,std::size_t budget,
        const RamKvImage* pending=nullptr);
    void admit_read(std::size_t id,std::uint64_t generation,const RamKvImage& view,
        RamKvImagePtr image,std::size_t budget);
};
RamKvStore::RamKvStore(RamKvStoreConfig c,Allocate a,Encode e,Publish p)
    :RamKvStore(std::move(c),std::move(a),std::move(e),std::move(p),{}){}
RamKvStore::RamKvStore(RamKvStoreConfig c,Allocate a,Encode e,Publish p,AllocateView v)
    :impl_(std::make_shared<Impl>(std::move(c),std::move(a),std::move(e),std::move(p),std::move(v))){}
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
    if(prefix && (!valid_backing(*prefix) || unsigned(prefix->tier)>2))
        throw std::invalid_argument("Invalid QSA prefix backing");
    std::lock_guard<std::mutex> lock(s.mutex);const auto id=s.entries.size(),generation=s.next();
    if(image->bytes()>std::numeric_limits<std::size_t>::max()-s.counters.ram_bytes)
        throw std::overflow_error("QSA RAM byte count overflow");
    if(id==s.entries.capacity())s.entries.reserve(id?std::max(id+1,id+id/2):1);
    Impl::IndexInsert index(s,1,tier);
    const auto lru=index.add(epoch,id);
    s.publish({{id,generation,image,{},initial}});
    s.counters.ram_bytes+=image->bytes();s.counters.tokens[tier]+=4;
    s.entries.push_back({std::move(image),s.config.prefix_enabled?std::move(prefix):nullptr,sizes,generation,epoch,initial,lru});
    s.account_precision(s.entries.back(),1);
    index.commit();
    return id;
}
void RamKvStore::touch(const std::vector<std::size_t>& ids,std::uint64_t epoch) {
    auto& s=*impl_;std::lock_guard<std::mutex> lock(s.mutex);
    for(auto id:ids)if(id>=s.entries.size())throw std::out_of_range("QSA RAM touch block");
    s.touch_resident(ids,epoch);
    for(auto id:ids)s.entries[id].access=std::max(s.entries[id].access,epoch);
}
std::vector<RamKvChange> RamKvStore::lease(const std::vector<std::size_t>& ids,std::uint64_t epoch) {
    auto& s=*impl_;std::vector<RamKvChange> result;result.reserve(ids.size());
    std::lock_guard<std::mutex> lock(s.mutex);
    for(auto id:ids)if(id>=s.entries.size())throw std::out_of_range("QSA RAM lease block");
    s.touch_resident(ids,epoch);
    for(auto id:ids) {
        auto& entry=s.entries[id];entry.access=std::max(entry.access,epoch);
        if(entry.image)result.push_back({id,entry.generation,entry.image,{},entry.tier});
    }
    return result;
}
std::size_t RamKvStore::relocate(const Relocate& relocate) {
    if(!relocate)return 0;
    auto& s=*impl_;std::lock_guard<std::mutex> maintenance(s.maintenance);
    std::vector<std::size_t> ids;
    std::vector<RamKvImagePtr> images;
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        for(std::size_t id=0;id<s.entries.size();++id)if(s.entries[id].image) {
            ids.push_back(id);images.push_back(s.entries[id].image);
        }
    }
    const auto replacements=relocate(images);
    std::set<std::size_t> changed;
    for(const auto& replacement:replacements) {
        if(replacement.index>=images.size() || !changed.insert(replacement.index).second)
            throw std::invalid_argument("Invalid QSA RAM relocation index");
        validate(replacement.image);
        const auto& old=*images[replacement.index];const auto& next=*replacement.image;
        if(next.key_bytes!=old.key_bytes || next.value_bytes!=old.value_bytes ||
                next.key_bits!=old.key_bits || next.value_bits!=old.value_bits ||
                next.key_state_bits!=old.key_state_bits || next.value_state_bits!=old.value_state_bits)
            throw std::invalid_argument("QSA RAM relocation changed its encoding");
    }
    std::vector<RamKvChange> changes;changes.reserve(replacements.size());
    std::lock_guard<std::mutex> lock(s.mutex);
    for(const auto& replacement:replacements) {
        const auto id=ids[replacement.index];
        // Appends/touches may run while the backend copies bytes. A replaced
        // or truncated source must never overwrite a newer live version.
        if(id>=s.entries.size() || s.entries[id].image!=images[replacement.index])continue;
        const auto& entry=s.entries[id];
        if(replacement.image==entry.image)continue;
        changes.push_back({id,entry.generation,replacement.image,entry.image,entry.tier,entry.backing});
    }
    if(changes.empty())return 0;
    s.publish(changes);
    for(const auto& change:changes)s.entries[change.id].image=change.image;
    return changes.size();
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
            // Check the live population against the same target budget.
            const auto floors=ram_precision_floors(4*std::uint64_t(s.entries.size()),budget,
                s.precision_bytes,pressure.canonical_tier,pressure.quantize_tokens);
            const auto tier=unsigned(action.from);
            const auto minimum=floors[tier];
            if(s.counters.tokens[tier]<minimum || action.tokens>s.counters.tokens[tier]-minimum)
                return s.observed(budget,true,allow_ssd);
        }
        const auto target_tier=unsigned(action.to);
        Impl::IndexInsert index(s,target_tier<3?old.size():0,std::min(target_tier,2u));
        if(target_tier<3)for(std::size_t i=0;i<old.size();++i)index.add(old[i].access,action.blocks[i]);
        std::vector<RamKvChange> changes;changes.reserve(old.size());
        for(std::size_t i=0;i<old.size();++i)changes.push_back({action.blocks[i],
            action.to==RamKvTier::Ssd && same_encoding(*old[i].image,*backings[i])?old[i].generation:s.next(),
            action.to==RamKvTier::Ssd?RamKvImagePtr{}:encoded[i],old[i].image,action.to,
            action.to==RamKvTier::Ssd?backings[i]:nullptr});
        s.publish(changes);
        index.commit();
        for(std::size_t i=0;i<old.size();++i) {
            auto& e=s.entries[action.blocks[i]];
            s.account_precision(e,-1);
            if(unsigned(e.tier)<3)s.lru_for(unsigned(e.tier)).erase(e.lru);
            s.counters.ram_bytes-=e.image->bytes();
            s.counters.tokens[unsigned(e.tier)]-=4;s.counters.tokens[unsigned(action.to)]+=4;
            e.image=changes[i].image;e.generation=changes[i].generation;e.tier=action.to;
            e.lru=target_tier<3?index.at(i):Impl::Lru::iterator{};
            s.account_precision(e,1);
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
    const auto tier=unsigned(original.backing->tier);
    if(tier>2 || image->bytes()>original.bytes[tier])throw std::runtime_error("QSA SSD recall exceeded its encoded image size");
    {std::lock_guard<std::mutex> lock(s.mutex);s.counters.ssd_read_bytes+=original.backing->key_bytes+original.backing->value_bytes;}
    if(!admit)return image;
    return s.admit_loaded(id,original,std::move(image),budget);
}
RamKvImagePtr RamKvStore::Impl::admit_loaded(std::size_t id,const Entry& original,RamKvImagePtr image,std::size_t budget,
        const RamKvImage* pending) {
    auto& s=*this;
    const auto bytes=image?image->bytes():pending->bytes();
    const auto tier=original.backing->tier;
    if(unsigned(tier)>2)throw std::invalid_argument("Invalid QSA SSD precision tier");
    std::vector<std::pair<std::size_t,Impl::Entry>> victims;
    {std::lock_guard<std::mutex> lock(s.mutex);
        if(id>=s.entries.size() || s.entries[id].generation!=original.generation || s.entries[id].tier!=RamKvTier::Ssd)return image;
        auto used=s.ram_bytes();
        if(bytes>budget)return image;
        for(const auto& [access,victim]:s.tq4_lru) {
            if(used<=budget-bytes || access>=original.access)break;
            const auto& e=s.entries[victim];if(e.image.use_count()!=1)continue;
            victims.emplace_back(victim,e);used-=e.image->bytes();
        }
        if(used>budget-bytes)return image;
    }
    // Copy only after admission is feasible. Backends accepting a borrowed
    // image can populate their final storage without temporary K/V vectors.
    if(!image) {
        image=s.allocate_view?s.allocate_view(*pending):s.allocate(copy_image(*pending));validate(image);
        if(!same_encoding(*image,*original.backing))
            throw std::runtime_error("QSA admission allocator changed the retained encoding");
        if(image->bytes()>original.bytes[unsigned(tier)])throw std::runtime_error("QSA deferred admission exceeded its RAM image size");
        if(image->bytes()>budget)return image;
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
    const auto generation=original.generation;
    Impl::IndexInsert index(s,1,unsigned(tier));
    index.add(s.entries[id].access,id);
    auto& changes=s.admission_changes;
    struct ReleaseChanges {
        std::vector<RamKvChange>& changes;
        ~ReleaseChanges(){changes.clear();}
    } release{changes};
    changes.reserve(victims.size()+1);
    for(std::size_t i=0;i<victims.size();++i) {
        const auto& victim=victims[i];
        const auto version=same_encoding(*victim.second.image,*backing[i])?victim.second.generation:s.next();
        changes.push_back({victim.first,version,{},victim.second.image,RamKvTier::Ssd,backing[i]});
    }
    changes.push_back({id,generation,image,{},tier});s.publish(changes);
    index.commit();
    for(std::size_t i=0;i<victims.size();++i) {
        auto& e=s.entries[victims[i].first];s.account_precision(e,-1);
        s.tq4_lru.erase(e.lru);e.lru={};
        e.image.reset();e.tier=RamKvTier::Ssd;e.backing=backing[i];e.generation=changes[i].generation;
        s.account_precision(e,1);
        s.counters.ram_bytes-=victims[i].second.image->bytes();s.counters.tokens[2]-=4;s.counters.tokens[3]+=4;
        ++s.counters.spilled_blocks;
        if(!victims[i].second.backing)s.counters.ssd_written_bytes+=victims[i].second.image->bytes();
        else if(victims[i].second.backing->store!=s.config.session_ssd)s.counters.prefix_reused_bytes+=victims[i].second.image->bytes();
    }
    auto& e=s.entries[id];s.account_precision(e,-1);
    e.image=image;e.tier=tier;e.generation=generation;++s.counters.promoted_blocks;
    s.account_precision(e,1);
    e.lru=index.at(0);
    s.counters.ram_bytes+=image->bytes();s.counters.tokens[3]-=4;s.counters.tokens[unsigned(tier)]+=4;
    return image;
}
RamKvSnapshotBlock RamKvStore::source(std::size_t id)const {
    auto& s=*impl_;std::lock_guard<std::mutex> lock(s.mutex);const auto& e=s.entries.at(id);
    return {e.image,e.backing,e.bytes,e.generation,e.access,e.tier};
}
MicroblockImage RamKvStore::read_cpu(const RamKvSnapshotBlock& source) {
    return copy_image(read_cpu_view(source));
}
RamKvImage RamKvStore::read_cpu_view(const RamKvSnapshotBlock& source,std::size_t* retained_bytes) {
    if(retained_bytes)*retained_bytes=0;
    if(source.image){validate(source.image);return *source.image;}
    if(!source.backing)throw std::runtime_error("QSA snapshot has no retained image");
    auto image=read_backing_view(*source.backing,retained_bytes);
    auto& s=*impl_;std::lock_guard<std::mutex> lock(s.mutex);
    s.counters.ssd_read_bytes+=image.bytes();
    return image;
}
void RamKvStore::read_cpu_into(const RamKvSnapshotBlock& source,std::uint8_t* destination,std::size_t capacity) {
    read_cpu_image(source,destination,capacity,true);
}
void RamKvStore::read_cpu_image(const RamKvSnapshotBlock& source,std::uint8_t* destination,std::size_t capacity,bool padded) {
    if(!destination)throw std::invalid_argument("Missing QSA CPU read destination");
    const auto* resident=source.image.get();const auto* backing=source.backing.get();
    if(resident)validate(source.image);
    else if(!backing || !valid_backing(*backing))
        throw std::runtime_error("QSA snapshot has no retained image");
    const auto kb=resident?resident->key_bytes:backing->key_bytes;
    const auto vb=resident?resident->value_bytes:backing->value_bytes;
    if(!kb || !vb || kb>std::numeric_limits<std::size_t>::max()-15)
        throw std::runtime_error("Invalid QSA CPU read extent");
    const auto extent=padded?((kb+15)&~std::size_t(15)):kb;
    if(extent>capacity || vb>capacity-extent)throw std::out_of_range("QSA CPU read destination is too small");
    if(resident) {
        std::memcpy(destination,resident->key,kb);std::memcpy(destination+extent,resident->value,vb);return;
    }
    if(backing->source) {
        backing->source->read_into(backing->offset,destination,kb+vb);
        if(extent!=kb)std::memmove(destination+extent,destination+kb,vb);
    }else {
        backing->store->read_into(backing->block,destination,kb+vb);
        if(extent!=kb)std::memmove(destination+extent,destination+kb,vb);
    }
    auto& s=*impl_;std::lock_guard<std::mutex> lock(s.mutex);s.counters.ssd_read_bytes+=kb+vb;
}
void RamKvStore::admit_read(std::size_t id,std::uint64_t generation,const MicroblockImage& image,std::size_t budget) {
    RamKvImage view;
    view.key=image.key.data();view.value=image.value.data();
    view.key_bytes=image.key.size();view.value_bytes=image.value.size();
    view.key_bits=image.key_bits;view.value_bits=image.value_bits;
    view.key_state_bits=image.key_state_bits;view.value_state_bits=image.value_state_bits;
    admit_read(id,generation,view,budget);
}

void RamKvStore::read_cpu_into_batch(const std::vector<RamKvReadInto>& requests) {
    read_cpu_into_batch(requests,true);
}
void RamKvStore::read_cpu_into_batch(const std::vector<RamKvReadInto>& requests,bool sequential) {
    read_cpu_batch(requests,sequential,true);
}
void RamKvStore::read_packed_cpu_into_batch(const std::vector<RamKvReadInto>& requests,bool sequential) {
    read_cpu_batch(requests,sequential,false);
}
void RamKvStore::read_cpu_batch(const std::vector<RamKvReadInto>& requests,bool sequential,bool padded) {
    for(const auto& r:requests) {
        if(!r.source || !r.destination)throw std::invalid_argument("Missing QSA batch read destination");
        const auto& s=*r.source;
        if(s.image)validate(s.image);
        else if(!s.backing || !valid_backing(*s.backing))throw std::runtime_error("QSA batch has no retained image");
        const auto kb=s.image?s.image->key_bytes:s.backing->key_bytes;
        const auto vb=s.image?s.image->value_bytes:s.backing->value_bytes;
        if(kb>SIZE_MAX-15)throw std::out_of_range("QSA batch image size overflow");
        const auto extent=padded?((kb+15)&~std::size_t(15)):kb;
        if(extent>r.capacity || vb>r.capacity-extent)throw std::out_of_range("QSA batch read destination is too small");
    }
    if(sequential) {
        std::uint64_t completed=0;
        try {
            if(read_sequential_files(requests,completed,padded)) {
                std::lock_guard<std::mutex> lock(impl_->mutex);
                impl_->counters.ssd_read_bytes+=completed;
                return;
            }
        }catch(...) {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            impl_->counters.ssd_read_bytes+=completed;
            throw;
        }
    }
    std::map<mfq::cache::QsaKvStore*,std::vector<const RamKvReadInto*>> groups;
    std::map<const RamKvReadSource*,std::vector<const RamKvReadInto*>> sources;
    std::vector<mfq::FileReadRange> file_ranges;file_ranges.reserve(requests.size());
    std::uint64_t file_bytes=0;
    std::optional<tyloinfer::ScopedFileReadWindow> read_window;
    if(sequential)read_window.emplace();
    for(const auto& r:requests) {
        const auto& source=*r.source;
        if(source.image)read_cpu_image(source,r.destination,r.capacity,padded);
        else if(!sequential && source.backing->source && source.backing->source->file_reader) {
            const auto& backing=*source.backing;const auto& input=*backing.source;
            const auto bytes=backing.key_bytes+backing.value_bytes;
            input.validate(backing.offset,bytes);
            if(backing.offset>UINT64_MAX-input.file_offset)throw std::out_of_range("QSA file offset overflow");
            file_ranges.push_back({input.file_reader,{input.file_offset+backing.offset,
                reinterpret_cast<std::byte*>(r.destination),bytes}});
            file_bytes+=bytes;
        }
        else if(!sequential && source.backing->source)sources[source.backing->source.get()].push_back(&r);
        else if(source.backing->source)read_cpu_image(source,r.destination,r.capacity,padded);
        else groups[source.backing->store.get()].push_back(&r);
    }

    for(const auto& [source,group]:sources) {
        std::vector<mfq::ReadSpan> spans;spans.reserve(group.size());std::uint64_t bytes=0;
        for(const auto* r:group) {
            const auto& b=*r->source->backing;
            spans.push_back({b.offset,reinterpret_cast<std::byte*>(r->destination),b.key_bytes+b.value_bytes});
            bytes+=b.key_bytes+b.value_bytes;
        }
        if(source->file_reader) {
            for(const auto& span:spans) {
                source->validate(span.offset,span.size);
                if(span.offset>UINT64_MAX-source->file_offset)throw std::out_of_range("QSA file offset overflow");
                file_ranges.push_back({source->file_reader,{source->file_offset+span.offset,span.destination,span.size}});
            }
            file_bytes+=bytes;continue;
        }
        source->read_into_batch(spans);
        for(const auto* r:group) {
            const auto& b=*r->source->backing;const auto extent=padded?((b.key_bytes+15)&~std::size_t(15)):b.key_bytes;
            if(extent!=b.key_bytes)std::memmove(r->destination+extent,r->destination+b.key_bytes,b.value_bytes);
        }
        std::lock_guard<std::mutex> lock(impl_->mutex);impl_->counters.ssd_read_bytes+=bytes;
    }
    if(!file_ranges.empty()) {
        mfq::FileRangeReader::read_many(file_ranges);
        for(const auto& r:requests)if(!r.source->image && r.source->backing->source && r.source->backing->source->file_reader) {
            const auto& b=*r.source->backing;const auto extent=padded?((b.key_bytes+15)&~std::size_t(15)):b.key_bytes;
            if(extent!=b.key_bytes)std::memmove(r.destination+extent,r.destination+b.key_bytes,b.value_bytes);
        }
        std::lock_guard<std::mutex> lock(impl_->mutex);impl_->counters.ssd_read_bytes+=file_bytes;
    }
    for(const auto& [store,group]:groups) {
        std::vector<mfq::cache::QsaKvStore::ReadRequest> reads;reads.reserve(group.size());
        std::uint64_t bytes=0;
        for(const auto* r:group) {
            const auto& b=*r->source->backing;
            reads.push_back({b.block,r->destination,b.key_bytes+b.value_bytes});bytes+=b.key_bytes+b.value_bytes;
        }
        store->read_into_batch(reads);
        for(const auto* r:group) {
            const auto& b=*r->source->backing;const auto extent=padded?((b.key_bytes+15)&~std::size_t(15)):b.key_bytes;
            if(extent!=b.key_bytes)std::memmove(r->destination+extent,r->destination+b.key_bytes,b.value_bytes);
        }
        std::lock_guard<std::mutex> lock(impl_->mutex);impl_->counters.ssd_read_bytes+=bytes;
    }
}
void RamKvStore::admit_read(std::size_t id,std::uint64_t generation,const RamKvImage& image,std::size_t budget) {
    impl_->admit_read(id,generation,image,{},budget);
}
void RamKvStore::admit_read(std::size_t id,std::uint64_t generation,RamKvImagePtr image,std::size_t budget) {
    validate(image);impl_->admit_read(id,generation,*image,image,budget);
}
void RamKvStore::Impl::admit_read(std::size_t id,std::uint64_t generation,const RamKvImage& image,
        RamKvImagePtr resident,std::size_t budget) {
    auto& s=*this;std::lock_guard<std::mutex> worker(s.maintenance);Entry original;
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        if(id>=s.entries.size() || s.entries[id].generation!=generation || s.entries[id].tier!=RamKvTier::Ssd)return;
        original=s.entries[id];
    }
    if(!image.key || !image.value || !original.backing || !same_encoding(image,*original.backing))
        throw std::invalid_argument("QSA deferred admission changed the retained SSD encoding");
    const auto tier=unsigned(original.backing->tier);
    if(tier>2 || image.key_bytes>original.bytes[tier] || image.value_bytes>original.bytes[tier]-image.key_bytes)
        throw std::runtime_error("QSA deferred admission exceeded its RAM image size");
    s.admit_loaded(id,original,std::move(resident),budget,&image);
}
bool RamKvStore::graph_reads_supported()const {
    // All stored encodings have immutable CPU byte readers. Device kernels
    // consume the original per-block precision; no backend codec is needed
    // while a captured graph waits for its CPU I/O task.
    return true;
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
        const auto& e=s.entries[i];s.account_precision(e,-1);
        if(e.image)s.counters.ram_bytes-=e.image->bytes();s.counters.tokens[unsigned(e.tier)]-=4;
        if(unsigned(e.tier)<3)s.lru_for(unsigned(e.tier)).erase(e.lru);
    }
    s.entries.resize(blocks);
}
void RamKvStore::reserve_metadata(std::size_t blocks) {
    auto& s=*impl_;std::lock_guard<std::mutex> maintenance(s.maintenance);
    std::lock_guard<std::mutex> lock(s.mutex);
    if(blocks<=s.entries.capacity())return;
    const auto live=s.entries.size();
    // Value initialization commits the pages before the host memory sample.
    // Merely reserving an uninitialized extent can leave its physical cost
    // invisible until payload restoration is already using that budget.
    s.entries.resize(blocks);s.entries.resize(live);
}
std::size_t RamKvStore::release_idle_metadata() {
    auto& s=*impl_;std::lock_guard<std::mutex> maintenance(s.maintenance);
    std::lock_guard<std::mutex> lock(s.mutex);
    if(!s.entries.empty())return 0;
    const auto bytes=s.entries.capacity()*sizeof(Impl::Entry);
    std::vector<Impl::Entry>().swap(s.entries);return bytes;
}
RamKvStoreStats RamKvStore::stats()const {
    auto& s=*impl_;std::lock_guard<std::mutex> lock(s.mutex);auto result=s.counters;
    result.blocks=s.entries.size();
    if(s.config.session_ssd)result.session_disk_bytes=s.config.session_ssd->stats().disk_bytes;
    result.resident_target_bytes=ram_precision_target(result.blocks,s.precision_bytes,
        s.config.pressure.canonical_tier,s.config.pressure.quantize_tokens);
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
    if(!backing || !valid_backing(*backing) || unsigned(backing->tier)>2)
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
                auto& e=store->entries[id];
                store->config.prefix_enabled=true;e.backing=std::move(backing);
            }
        }
    };
    return result;
}
void RamKvStore::restore(const RamKvSnapshot& state,std::size_t budget,bool borrow_prefix,bool reuse_resident) {
    restore(state,budget,borrow_prefix,reuse_resident,{});
}
void RamKvStore::restore(const RamKvSnapshot& state,std::size_t budget,bool borrow_prefix,bool reuse_resident,
                         const std::function<void(std::size_t)>& block_read,const RestoreAllocate& allocate_batch) {
    restore(state,budget,borrow_prefix,reuse_resident,block_read,allocate_batch,0);
}
void RamKvStore::restore(const RamKvSnapshot& state,std::size_t budget,bool borrow_prefix,bool reuse_resident,
                         const std::function<void(std::size_t)>& block_read,const RestoreAllocate& allocate_batch,
                         std::size_t append_bytes) {
    auto& s=*impl_;std::lock_guard<std::mutex> worker(s.maintenance);
    std::vector<RamPressureBlock> metadata;metadata.reserve(state.blocks.size());
    for(std::size_t i=0;i<state.blocks.size();++i) {
        const auto& b=state.blocks[i];auto tier=b.tier;
        if(tier==RamKvTier::Ssd && !s.config.pressure.ssd_enabled) {
            if(b.backing)tier=b.backing->tier;
            else if(b.image)tier=b.image->key_bits==0 && b.image->value_bits==0?RamKvTier::Fp16:
                b.image->key_bits==affine8_tag && b.image->value_bits==affine8_tag?RamKvTier::Affine8:RamKvTier::Tq4;
            else throw std::invalid_argument("QSA SSD prefix has no encoded source");
            // An explicitly selected lower canonical precision still applies
            // when hydrating a RAM-only cache; recalling SSD itself never fits.
            tier=RamKvTier(std::max(unsigned(tier),unsigned(s.config.pressure.canonical_tier)));
        }
        auto bytes=b.bytes;if(b.image && unsigned(b.tier)<3)bytes[unsigned(b.tier)]=b.image->bytes();
        metadata.push_back({i,b.access,4,tier,bytes,false});
    }
    const auto preferred=append_bytes<budget?budget-append_bytes:budget;
    auto plan=plan_ram_pressure(metadata,preferred,s.config.pressure);
    // Resolve a soft append margin before truncating entries or reading data.
    // A valid checkpoint must not fail solely because of this preference.
    if(plan.fits())budget=preferred;
    else if(preferred!=budget)plan=plan_ram_pressure(metadata,budget,s.config.pressure);
    if(!plan.fits())throw std::runtime_error("QSA prefix cannot fit the RAM budget and protected precision floors");
    for(const auto& action:plan.actions)for(auto id:action.blocks)metadata[id].tier=action.to;
    truncate(0);
    {std::lock_guard<std::mutex> lock(s.mutex);s.config.prefix_enabled=borrow_prefix;}
    // Sparse conversions may span many SSD-only blocks. Bound source bytes
    // and transaction metadata by one full-precision policy batch, rather
    // than submitting a tiny codec job for every fixed token-position window.
    const std::size_t batch=s.config.pressure.quantize_tokens/4;
    const auto source_bytes=[](const RamKvSnapshotBlock& block) {
        const auto key=block.image?block.image->key_bytes:block.backing?block.backing->key_bytes:0;
        const auto value=block.image?block.image->value_bytes:block.backing?block.backing->value_bytes:0;
        if(value>std::numeric_limits<std::size_t>::max()-key)throw std::overflow_error("QSA restore source size overflow");
        return key+value;
    };
    const auto actual_tier=[](const RamKvSnapshotBlock& src) {
        if(!src.image && !src.backing)throw std::runtime_error("QSA prefix has no encoded source");
        return src.image?(src.tier!=RamKvTier::Ssd?src.tier:
            src.image->key_bits==0 && src.image->value_bits==0?RamKvTier::Fp16:
            src.image->key_bits==affine8_tag && src.image->value_bits==affine8_tag?RamKvTier::Affine8:RamKvTier::Tq4):
            src.backing->tier;
    };
    std::size_t maximum_source=0;
    for(std::size_t id=0;id<metadata.size();++id)
        maximum_source=std::max({maximum_source,metadata[id].bytes[0],source_bytes(state.blocks[id])});
    constexpr std::size_t metadata_charge=sizeof(Impl::Entry)+sizeof(RamKvChange)+2*sizeof(RamKvImagePtr)+
        sizeof(Impl::Lru::value_type)+4*sizeof(void*);
    if(maximum_source>std::numeric_limits<std::size_t>::max()-metadata_charge ||
            batch>std::numeric_limits<std::size_t>::max()/(maximum_source+metadata_charge))
        throw std::overflow_error("QSA restore staging size overflow");
    const auto staging_limit=batch*(maximum_source+metadata_charge);
    for(std::size_t begin=0;begin<metadata.size();) {
        std::size_t end=begin,staged=0;std::array<std::size_t,2> conversions{};
        while(end<metadata.size()) {
            const bool full=end-begin>=batch;
            if(full && ((!conversions[0] && !conversions[1]) || conversions[0]>=batch || conversions[1]>=batch))break;
            const auto& src=state.blocks[end];const auto target=metadata[end].tier;
            const bool keep=src.backing && (borrow_prefix || src.backing->store==s.config.session_ssd);
            const bool read=target!=RamKvTier::Ssd || !keep;
            const auto charge=metadata_charge+(read?source_bytes(src):0);
            if(full && charge>staging_limit-staged)break;
            staged+=charge;
            if(read) {
                const auto actual=unsigned(actual_tier(src));const auto desired=std::min(unsigned(target),2u);
                for(unsigned tier=1;tier<=desired;++tier)if(actual<tier)++conversions[tier-1];
            }
            ++end;
        }
        const auto count=end-begin;
        std::vector<RamKvImagePtr> images(count);
        std::vector<std::shared_ptr<const RamKvBacking>> backing(count);
        // Only count images which will actually be allocated in this batch.
        // This lets a pinned allocator reserve a shared slab without reserving
        // the entire prefix, or any blocks which remain SSD-only.
        std::map<std::pair<std::size_t,std::size_t>,std::size_t> pending;
        if(allocate_batch)for(std::size_t j=0;j<count;++j) {
            const auto& src=state.blocks[begin+j];
            const bool keep=src.backing && (borrow_prefix || src.backing->store==s.config.session_ssd);
            if((metadata[begin+j].tier==RamKvTier::Ssd && keep) || (reuse_resident && src.image))continue;
            if(src.image)++pending[{src.image->key_bytes,src.image->value_bytes}];
            else if(src.backing)++pending[{src.backing->key_bytes,src.backing->value_bytes}];
        }
        for(std::size_t j=0;j<count;++j) {
            const auto& src=state.blocks[begin+j];
            const bool keep=src.backing && (borrow_prefix || src.backing->store==s.config.session_ssd);
            if(keep)backing[j]=src.backing;
            if(metadata[begin+j].tier!=RamKvTier::Ssd || !backing[j]) {
                if(reuse_resident && src.image)images[j]=src.image;
                else {
                    const auto image=src.read();
                    images[j]=allocate_batch?allocate_batch(image,pending.at({image.key.size(),image.value.size()})--):s.allocate(image);
                }
                validate(images[j]);
            }
            if(block_read)block_read(begin+j);
        }
        for(unsigned tier=0;tier<3;++tier) {
            std::vector<RamKvImagePtr> source;std::vector<std::size_t> ids;
            for(std::size_t j=0;j<count;++j)if(images[j]) {
                const auto target=metadata[begin+j].tier==RamKvTier::Ssd?RamKvTier::Tq4:metadata[begin+j].tier;
                const auto& src=state.blocks[begin+j];
                // SSD is a placement, not an encoding. A RAM-only paged
                // decoder may own the retained SSD image in a host buffer.
                const auto actual=actual_tier(src);
                if(unsigned(target)>=tier && unsigned(actual)<tier){source.push_back(images[j]);ids.push_back(j);}
            }
            if(source.empty())continue;
            const auto encoded=s.encode(source,RamKvTier(tier));
            if(encoded.size()!=ids.size())throw std::runtime_error("QSA prefix codec changed its block count");
            for(std::size_t j=0;j<ids.size();++j){validate(encoded[j]);images[ids[j]]=encoded[j];}
        }
        std::vector<Impl::Entry> entries;entries.reserve(count);
        std::vector<RamKvChange> changes;changes.reserve(count);
        std::size_t ram_bytes=0,written_bytes=0;
        for(std::size_t j=0;j<count;++j) {
            const auto id=begin+j;const auto& src=state.blocks[id];const auto target=metadata[id].tier;
            if(target==RamKvTier::Ssd) {
                if(!backing[j]) {
                    backing[j]=s.spill({images[j],{},src.bytes,0,src.access,RamKvTier::Tq4});
                    written_bytes+=backing[j]->key_bytes+backing[j]->value_bytes;
                }
                entries.push_back({{},backing[j],src.bytes,0,src.access,target});
            }else {
                validate(images[j]);const auto tier=unsigned(target);
                if(!src.bytes[2] || src.bytes[0]<src.bytes[1] || src.bytes[1]<src.bytes[2] || images[j]->bytes()>src.bytes[tier])
                    throw std::invalid_argument("QSA RAM tier size does not match its canonical image");
                if(backing[j] && (!valid_backing(*backing[j]) || unsigned(backing[j]->tier)>2))
                    throw std::invalid_argument("Invalid QSA prefix backing");
                if(images[j]->bytes()>std::numeric_limits<std::size_t>::max()-ram_bytes)
                    throw std::overflow_error("QSA RAM byte count overflow");
                ram_bytes+=images[j]->bytes();
                entries.push_back({images[j],borrow_prefix?backing[j]:nullptr,src.bytes,0,src.access,target});
            }
        }
        // Publish each complete policy batch once. All storage and LRU nodes
        // are prepared before publication, so rejection leaves no new entries
        // or LRU nodes. The final commit only moves already allocated entries.
        std::lock_guard<std::mutex> lock(s.mutex);
        if(s.entries.size()!=begin)throw std::logic_error("QSA restore changed during publication");
        if(ram_bytes>std::numeric_limits<std::size_t>::max()-s.counters.ram_bytes)
            throw std::overflow_error("QSA RAM byte count overflow");
        if(end>s.entries.capacity())s.entries.reserve(std::max(end,s.entries.capacity()+s.entries.capacity()/2));
        std::array<std::size_t,3> tier_counts{};
        for(const auto& entry:entries)if(unsigned(entry.tier)<3)++tier_counts[unsigned(entry.tier)];
        std::array<Impl::IndexInsert,3> indices{Impl::IndexInsert(s,tier_counts[0],0),
            Impl::IndexInsert(s,tier_counts[1],1),Impl::IndexInsert(s,tier_counts[2],2)};
        for(std::size_t j=0;j<count;++j) {
            auto& entry=entries[j];const auto id=begin+j;entry.generation=s.next();
            if(unsigned(entry.tier)<3)entry.lru=indices[unsigned(entry.tier)].add(entry.access,id);
            changes.push_back({id,entry.generation,entry.image,{},entry.tier,entry.tier==RamKvTier::Ssd?entry.backing:nullptr});
        }
        s.publish(changes);
        s.counters.ram_bytes+=ram_bytes;s.counters.ssd_written_bytes+=written_bytes;
        for(auto& entry:entries) {
            s.counters.tokens[unsigned(entry.tier)]+=4;s.account_precision(entry,1);
            s.entries.push_back(std::move(entry));
        }
        for(auto& index:indices)index.commit();
        begin=end;
    }
}
} // namespace tyloinfer::qsa
