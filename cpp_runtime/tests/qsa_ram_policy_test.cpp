#include "tyloinfer/qsa_ram_policy.h"
#include <algorithm>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>
namespace tq=tyloinfer::qsa;
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
std::vector<tq::RamPressureBlock> fixture(std::size_t count=8192) {
    std::vector<tq::RamPressureBlock> b(count);
    for(std::size_t i=0;i<count;++i) {
        b[i].id=i;b[i].last_access=(i*257)%count;
        b[i].bytes={8192,4144,2088};
    }
    return b;
}
void check_plan(const std::vector<tq::RamPressureBlock>& original,const tq::RamPressurePlan& plan,
                tq::RamPressureConfig config) {
    auto current=original;std::array<std::uint64_t,4> tokens{};std::size_t bytes=0;
    for(const auto& b:current){tokens[unsigned(b.tier)]+=b.tokens;if(unsigned(b.tier)<3)bytes+=b.bytes[unsigned(b.tier)];}
    for(const auto& action:plan.actions) {
        const auto from=unsigned(action.from),to=unsigned(action.to);
        require(to==from+1,"A pressure action skipped a precision tier");
        std::vector<std::size_t> eligible;
        for(std::size_t i=0;i<current.size();++i)
            if(current[i].tier==action.from && !current[i].pinned && current[i].tokens==4)eligible.push_back(i);
        std::sort(eligible.begin(),eligible.end(),[&](auto a,auto b) {
            return current[a].last_access!=current[b].last_access?current[a].last_access<current[b].last_access:current[a].id<current[b].id;
        });
        require(action.blocks.size()<=eligible.size() &&
            std::equal(action.blocks.begin(),action.blocks.end(),eligible.begin()),"A pressure action violated LRU order");
        if(to<3)require(action.tokens==config.quantize_tokens,"Quantization did not use the configured complete batch");
        else {
            require(config.ssd_enabled,"Disabled SSD received an eviction");
            require(tokens[0]<=plan.minimum_fp16 || tokens[0]-plan.minimum_fp16<config.quantize_tokens,"FP16 was not near its floor before SSD eviction");
            require(tokens[1]<=plan.minimum_affine8 || tokens[1]-plan.minimum_affine8<config.quantize_tokens,"Affine8 was not near its floor before SSD eviction");
        }
        std::size_t saved=0;std::uint64_t moved=0;
        for(auto i:action.blocks) {
            auto& b=current[i];require(b.tier==action.from && !b.pinned,"An action reused a stale block");
            saved+=b.bytes[from]-(to<3?b.bytes[to]:0);moved+=b.tokens;b.tier=action.to;
        }
        require(saved==action.saved_bytes && moved==action.tokens,"Pressure action accounting differs");
        bytes-=saved;tokens[from]-=moved;tokens[to]+=moved;
        if(from==0)require(tokens[0]>=plan.minimum_fp16,"FP16 retention fell below its budget-adjusted floor");
        if(from==1)require(tokens[1]>=plan.minimum_affine8,"Affine8 retention fell below its budget-adjusted floor");
    }
    require(bytes==plan.ram_bytes && tokens==plan.tokens,"Final pressure accounting differs");
}
int main()try {
    constexpr std::size_t MiB=1ULL<<20;
    require(tq::ram_kv_budget(8*MiB,32*MiB)==32*MiB,"RAM reserve shrank with KV growth");
    require(tq::ram_kv_budget(3*MiB,2*MiB)==4*MiB,"RAM reserve did not track other process usage");
    // Independently test the smallest reclaim against the existing budget,
    // including allocator overhead and logical images awaiting allocation.
    for(std::size_t physical=0;physical<100;++physical)
        for(std::size_t logical=0;logical<100;++logical)
            for(std::size_t free=0;free<100;++free) {
                const auto n=tq::ram_kv_reclaim_bytes(free,physical,logical);
                const auto target=std::max(physical,logical);
                require(tq::ram_kv_budget(free+n,physical)>=target,"Reclaimed RAM still violates the reserve");
                if(n)require(tq::ram_kv_budget(free+n-1,physical)<target,"RAM prefix reclamation exceeded the required shortage");
            }
    require(tq::ram_kv_reclaim_bytes(8*MiB,32*MiB,30*MiB)==0,"A fitting KV pool requested prefix eviction");
    require(tq::ram_kv_reclaim_bytes(7*MiB,32*MiB,30*MiB)==MiB,"Pinned allocation overhead was omitted from the reserve");
    const auto maximum=std::numeric_limits<std::size_t>::max();
    require(tq::ram_kv_reclaim_bytes(0,maximum,maximum)==maximum/4+1,"Large reserve arithmetic overflowed");
    require(tq::ram_kv_reclaim_bytes(0,0,maximum)==maximum,"Unrepresentable reclaim did not saturate");
    auto blocks=fixture();int cases=0;
    const auto unconstrained=tq::plan_ram_pressure(blocks,64*MiB,{2048,true});
    require(unconstrained.minimum_fp16==1640 && unconstrained.minimum_affine8==4916,
        "An affordable cache changed its protected fractions");++cases;
    // Keep only a short live tail in RAM while growing SSD history to 1M.
    // Historical tokens must not create an unbounded high-precision floor.
    std::uint64_t previous_fp16=0,previous_affine8=0;
    for(std::size_t count:{std::size_t(16384),std::size_t(65536),std::size_t(262144)}) {
        auto input=fixture(count);
        for(std::size_t i=0;i<count-4096;++i)input[i].tier=tq::RamKvTier::Ssd;
        const auto bounded=tq::plan_ram_pressure(input,16*MiB,{2048,true});
        require(bounded.fits(),"SSD history made budget-adjusted precision floors unfit");
        require(bounded.tokens[0]>=bounded.minimum_fp16 && bounded.tokens[1]>=bounded.minimum_affine8,
            "Long-context pressure discarded a protected tier");
        if(previous_fp16)require(bounded.minimum_fp16==previous_fp16 &&
            bounded.minimum_affine8==previous_affine8,"Precision floors grew with SSD-only history");
        previous_fp16=bounded.minimum_fp16;previous_affine8=bounded.minimum_affine8;
        const auto smaller=tq::plan_ram_pressure(input,8*MiB,{2048,true});
        require(smaller.minimum_fp16<=bounded.minimum_fp16 &&
            smaller.minimum_affine8<=bounded.minimum_affine8,"Precision floors ignored a smaller RAM budget");
        const auto disabled=tq::plan_ram_pressure(fixture(count),16*MiB,{2048,false});
        require(!disabled.fits() && disabled.needs_ssd,"Disabled SSD accepted excess live RAM");
        ++cases;
    }
    // Full low-bit residency has priority over higher-precision protection.
    // An independently sized budget fits TQ4 plus two complete batch tails.
    {
        const auto input=fixture(32768);
        const auto base=input.size()*input.front().bytes[2];
        const auto tails=2*512*input.front().bytes[0];
        for(bool ssd:{false,true}) {
            const auto plan=tq::plan_ram_pressure(input,base+tails,{2048,ssd});
            check_plan(input,plan,{2048,ssd});
            require(plan.fits() && plan.tokens[3]==0,"A fitting all-TQ4 context was evicted to SSD for precision floors");
        }
    }
    for(bool ssd:{false,true})for(auto canonical:{tq::RamKvTier::Fp16,tq::RamKvTier::Affine8,tq::RamKvTier::Tq4}) {
        auto input=blocks;for(auto& b:input)b.tier=canonical;
        tq::RamPressureConfig config;config.ssd_enabled=ssd;config.canonical_tier=canonical;
        for(std::size_t budget:{64*MiB,48*MiB,30*MiB,16*MiB,8*MiB,std::size_t(0)}) {
            const auto plan=tq::plan_ram_pressure(input,budget,config);check_plan(input,plan,config);
            if(budget==64*MiB)require(plan.fits() && plan.actions.empty(),"Fitting RAM cache was needlessly quantized");
            if(canonical==tq::RamKvTier::Fp16 && budget==16*MiB)
                require(ssd?plan.fits():plan.needs_ssd,"SSD overflow decision differs");
            if(canonical==tq::RamKvTier::Tq4 && ssd)require(plan.fits(),"Fixed TQ4 failed to spill to SSD");
            ++cases;
        }
    }
    auto pinned=blocks;for(auto& b:pinned)b.pinned=true;
    auto plan=tq::plan_ram_pressure(pinned,MiB,{2048,true});
    require(!plan.fits() && plan.needs_readers && plan.actions.empty(),"Active readers were discarded under pressure");++cases;
    auto tail=blocks;tail.back().tokens=3;
    plan=tq::plan_ram_pressure(tail,16*MiB,{2048,true});check_plan(tail,plan,{2048,true});
    for(const auto& action:plan.actions)
        require(std::find(action.blocks.begin(),action.blocks.end(),tail.size()-1)==action.blocks.end(),"An incomplete microblock was requantized");++cases;
    auto existing=blocks;for(std::size_t i=0;i<existing.size()/2;++i)existing[i].tier=tq::RamKvTier::Ssd;
    plan=tq::plan_ram_pressure(existing,8*MiB,{2048,true});check_plan(existing,plan,{2048,true});++cases;
    for(std::size_t count:{std::size_t(0),std::size_t(1),std::size_t(17),std::size_t(512)}) {
        auto small=fixture(count);plan=tq::plan_ram_pressure(small,0,{2048,true});check_plan(small,plan,{2048,true});++cases;
    }
    bool rejected=false;
    try{tq::ram_kv_budget(std::numeric_limits<std::size_t>::max(),1);}catch(const std::overflow_error&){rejected=true;}
    require(rejected,"Overflowed RAM availability was accepted");++cases;
    rejected=false;try{tq::plan_ram_pressure(blocks,MiB,{2047,true});}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"Partial quantization batches were accepted");++cases;
    require(tq::ram_precision_target(0,{0,0,0},tq::RamKvTier::Fp16)==0,"Empty KV retained a RAM reservation");
    require(tq::ram_precision_target(1,{1024,512,256},tq::RamKvTier::Fp16)==1024,"Short KV lost its bounded precision tail");
    require(tq::ram_precision_target(10000,{10240000,5120000,2560000},tq::RamKvTier::Tq4)==2560000,"TQ4 reservation grew beyond its payload");
    std::printf("QSA RAM pressure reserve, LRU, 2048-token batches, protected tiers and SSD policy PASS cases=%d\n",cases);
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
