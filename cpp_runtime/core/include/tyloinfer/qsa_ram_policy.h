#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <vector>

namespace tyloinfer::qsa {
enum class RamKvTier : unsigned {Fp16,Affine8,Tq4,Ssd};
struct RamPressureBlock {
    std::uint64_t id=0,last_access=0;
    unsigned tokens=4;
    RamKvTier tier=RamKvTier::Fp16;
    // Actual packed K+V image sizes at each RAM precision.
    std::array<std::size_t,3> bytes{};
    bool pinned=false;
};
struct RamPressureConfig {
    unsigned quantize_tokens=2048;
    bool ssd_enabled=false;
    RamKvTier canonical_tier=RamKvTier::Fp16;
};
struct RamPressureAction {
    RamKvTier from,to;
    std::vector<std::size_t> blocks;
    std::uint64_t tokens=0;
    std::size_t saved_bytes=0;
};
struct RamPressurePlan {
    std::vector<RamPressureAction> actions;
    std::array<std::uint64_t,4> tokens{};
    std::uint64_t minimum_fp16=0,minimum_affine8=0;
    std::size_t ram_bytes=0,budget_bytes=0;
    bool needs_readers=false,needs_ssd=false;
    bool fits()const{return ram_bytes<=budget_bytes;}
};

// Add back this cache's own RAM before reserving 20% of the available pool.
// Using only current free RAM would make the reserve shrink as KV grows.
inline std::size_t ram_kv_budget(std::size_t system_free,std::size_t owned_kv) {
    if(system_free>std::numeric_limits<std::size_t>::max()-owned_kv)
        throw std::overflow_error("QSA RAM budget overflow");
    const auto pool=system_free+owned_kv;
    return pool/5*4+pool%5*4/5;
}

// Minimum additional free RAM for the current images and their allocation
// overhead to fit while leaving 20% of the combined pool available. Return a
// saturated shortage if the required pool cannot be represented.
inline std::size_t ram_kv_reclaim_bytes(std::size_t system_free,
        std::size_t physical_kv,std::size_t logical_kv) {
    const auto target=std::max(physical_kv,logical_kv);
    const auto reserve=target/4+(target%4!=0);
    const auto extra=target-physical_kv;
    const auto maximum=std::numeric_limits<std::size_t>::max();
    const auto needed=extra>maximum-reserve ? maximum : extra+reserve;
    return needed>system_free ? needed-system_free : 0;
}

inline long double ram_batch_tail_bytes(std::size_t blocks,const std::array<long double,3>& b,
        RamKvTier canonical,unsigned quantum) {
    if(!blocks || canonical==RamKvTier::Tq4)return 0;
    const auto excess=canonical==RamKvTier::Fp16?b[0]+b[1]-2.L*b[2]:b[1]-b[2];
    return (quantum/4-1)*std::max(0.L,excess)/blocks;
}

// Nominal compressed capacity, including bounded complete-batch tails. It
// describes the full context even when some blocks currently live on SSD.
inline std::size_t ram_precision_target(std::size_t blocks,const std::array<long double,3>& b,
        RamKvTier canonical,unsigned quantize_tokens=2048) {
    if(unsigned(canonical)>2 || !quantize_tokens || quantize_tokens%4)
        throw std::invalid_argument("Invalid RAM precision target");
    const auto mixed=canonical==RamKvTier::Fp16?(b[0]+3.L*b[1]+16.L*b[2])/20.L:
        canonical==RamKvTier::Affine8?(3.L*b[1]+17.L*b[2])/20.L:b[2];
    const auto slack=ram_batch_tail_bytes(blocks,b,canonical,quantize_tokens);
    const auto target=std::min(b[unsigned(canonical)],mixed+slack);
    const auto maximum=std::numeric_limits<std::size_t>::max();
    return target>=static_cast<long double>(maximum)?maximum:
        std::size_t(target)+(target>static_cast<long double>(std::size_t(target)));
}

inline std::array<std::uint64_t,2> ram_precision_floors(std::uint64_t total,
        std::size_t budget,const std::array<long double,3>& bytes,RamKvTier canonical,unsigned quantum=2048) {
    if(!total || canonical==RamKvTier::Tq4)return {0,0};
    const auto mixed=canonical==RamKvTier::Fp16?(bytes[0]+3.L*bytes[1]+16.L*bytes[2])/20.L:
        (3.L*bytes[1]+17.L*bytes[2])/20.L;
    const auto base=bytes[2];
    // All low-precision blocks still occupy RAM until an actual SSD eviction.
    // Only the remaining capacity can protect higher-precision images. Leave
    // room for full quantization batches before deriving their token floors.
    const auto slack=ram_batch_tail_bytes(std::size_t(total/4+(total%4!=0)),bytes,canonical,quantum);
    const bool constrained=static_cast<long double>(budget)<std::min(bytes[unsigned(canonical)],mixed+slack);
    auto population=total;
    if(constrained && mixed>base) {
        const auto extra=static_cast<long double>(budget)-base-slack;
        // If even the low-bit population cannot fit, SSD is unavoidable:
        // protect the affordable hot RAM population at the nominal fractions.
        // When it can fit, spend only its remaining capacity on higher tiers.
        const auto affordable=extra>=0?static_cast<long double>(total)*extra/(mixed-base):
            static_cast<long double>(total)*budget/mixed;
        if(affordable<static_cast<long double>(total))population=static_cast<std::uint64_t>(affordable);
    }
    const auto floor=[&](unsigned numerator) {
        const auto n=population/20*numerator+population%20*numerator/20;
        if(constrained)return n/4*4;
        const auto rounded=population/20*numerator+(population%20*numerator+19)/20;
        return std::min(total,rounded/4*4+(rounded%4?4u:0u));
    };
    const auto fp16=canonical==RamKvTier::Fp16?floor(1):0;
    return {fp16,std::min(total-fp16,floor(3))};
}

// Call only under pressure; the store maintains ordinary append/read counters.
// Build LRU heaps once in linear time, then pop complete quantization batches.
// No payloads are copied. The publisher applies actions in order and retains
// each old immutable image until its GPU readers and SSD writes have retired.
inline RamPressurePlan plan_ram_pressure(const std::vector<RamPressureBlock>& blocks,
        std::size_t budget,RamPressureConfig config={}) {
    if(!config.quantize_tokens || config.quantize_tokens%4 || unsigned(config.canonical_tier)>2)
        throw std::invalid_argument("QSA RAM quantization batch must contain complete four-token blocks");
    RamPressurePlan result;result.budget_bytes=budget;
    std::uint64_t total=0;
    std::array<long double,3> precision_bytes{};
    const auto add_bytes=[](std::size_t& sum,std::size_t n) {
        if(n>std::numeric_limits<std::size_t>::max()-sum)
            throw std::overflow_error("QSA RAM image size overflow");
        sum+=n;
    };
    for(const auto& block:blocks) {
        const auto tier=unsigned(block.tier);
        if(tier>3 || !block.tokens || block.tokens>4 || !block.bytes[2] ||
            block.bytes[0]<block.bytes[1] || block.bytes[1]<block.bytes[2])
            throw std::invalid_argument("Invalid QSA RAM pressure block");
        if(block.tokens>std::numeric_limits<std::uint64_t>::max()-total)
            throw std::overflow_error("QSA RAM token count overflow");
        total+=block.tokens;result.tokens[tier]+=block.tokens;
        if(tier<3)add_bytes(result.ram_bytes,block.bytes[tier]);
        for(unsigned i=0;i<3;++i)precision_bytes[i]+=block.bytes[i];
    }
    const auto floors=ram_precision_floors(total,budget,precision_bytes,config.canonical_tier,config.quantize_tokens);
    result.minimum_fp16=floors[0];result.minimum_affine8=floors[1];
    if(result.fits())return result;

    const auto newer=[&](std::size_t a,std::size_t b) {
        return blocks[a].last_access!=blocks[b].last_access?
            blocks[a].last_access>blocks[b].last_access:blocks[a].id>blocks[b].id;
    };
    using Heap=std::priority_queue<std::size_t,std::vector<std::size_t>,decltype(newer)>;
    std::array<std::vector<std::size_t>,3> eligible;
    for(std::size_t i=0;i<blocks.size();++i) {
        const auto& b=blocks[i];const auto tier=unsigned(b.tier);
        if(tier<3 && !b.pinned && b.tokens==4)eligible[tier].push_back(i);
    }
    std::array<Heap,3> heaps{Heap(newer,std::move(eligible[0])),
        Heap(newer,std::move(eligible[1])),Heap(newer,std::move(eligible[2]))};
    const auto batch=config.quantize_tokens/4;
    const std::array<std::uint64_t,2> minimum{result.minimum_fp16,result.minimum_affine8};
    for(unsigned tier=0;tier<2 && !result.fits();++tier) {
        while(!result.fits() && result.tokens[tier]>=minimum[tier] &&
              result.tokens[tier]-minimum[tier]>=config.quantize_tokens && heaps[tier].size()>=batch) {
            RamPressureAction action{RamKvTier(tier),RamKvTier(tier+1)};
            action.blocks.reserve(batch);
            for(unsigned j=0;j<batch;++j) {
                const auto i=heaps[tier].top();heaps[tier].pop();heaps[tier+1].push(i);
                action.blocks.push_back(i);action.tokens+=4;
                add_bytes(action.saved_bytes,blocks[i].bytes[tier]-blocks[i].bytes[tier+1]);
            }
            result.ram_bytes-=action.saved_bytes;
            result.tokens[tier]-=action.tokens;result.tokens[tier+1]+=action.tokens;
            result.actions.push_back(std::move(action));
        }
        if(result.tokens[tier]>=minimum[tier] &&
            result.tokens[tier]-minimum[tier]>=config.quantize_tokens && heaps[tier].size()<batch)
            result.needs_readers=true;
    }
    if(result.fits())return result;
    const bool fp16_at_floor=result.tokens[0]<=result.minimum_fp16 ||
        result.tokens[0]-result.minimum_fp16<config.quantize_tokens;
    const bool affine8_at_floor=result.tokens[1]<=result.minimum_affine8 ||
        result.tokens[1]-result.minimum_affine8<config.quantize_tokens;
    // Spill TQ4 only after both higher tiers approach their protected floors.
    if(!fp16_at_floor || !affine8_at_floor)return result;
    if(!config.ssd_enabled){result.needs_ssd=!heaps[2].empty();return result;}
    RamPressureAction spill{RamKvTier::Tq4,RamKvTier::Ssd};
    while(!result.fits() && !heaps[2].empty()) {
        const auto i=heaps[2].top();heaps[2].pop();spill.blocks.push_back(i);
        spill.tokens+=4;add_bytes(spill.saved_bytes,blocks[i].bytes[2]);
        result.ram_bytes-=blocks[i].bytes[2];result.tokens[2]-=4;result.tokens[3]+=4;
    }
    if(!spill.blocks.empty())result.actions.push_back(std::move(spill));
    if(!result.fits() && result.tokens[2])result.needs_readers=true;
    return result;
}
} // namespace tyloinfer::qsa
