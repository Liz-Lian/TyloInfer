#include "tyloinfer/qsa_precision_policy.h"
#include <cassert>
#include <iostream>

using namespace tyloinfer::qsa;

static std::vector<BlockPrecisionHistory> blocks(std::size_t count) {
    std::vector<BlockPrecisionHistory> result(count);
    for (std::size_t i=0;i<count;++i) {
        result[i].token_begin=i*4;
        result[i].bytes={100,120,140,160,180,340};
        result[i].distortion={16,12,8,6,4,0};
    }
    return result;
}

int main() {
    PrecisionPolicy policy({8,128,128});
    auto b=blocks(8);
    std::vector<double> hits(8),mass(8);
    hits[0]=100;mass[0]=80;hits[1]=20;mass[1]=2;
    policy.observe(b,hits,mass,128);
    // At 32 tokens, the two most recent blocks are protected irrespective of
    // observed frequency. A tight budget spends upgrades on the hot old block.
    auto p=policy.plan(b,32,6*100+2*340+80);
    assert(p.bytes<=1360 && p.minimum_bytes==1280);
    assert(p.tiers[6]==5 && p.tiers[7]==5 && p.tiers[0]>p.tiers[1]);
    assert(p.tiers[0]==4 && p.tiers[1]==0);
    bool rejected=false;
    try {policy.plan(b,32,1279);}catch(const std::runtime_error&){rejected=true;}
    assert(rejected);
    // A drifted query distribution makes a previously cold block valuable.
    for (std::size_t i=0;i<b.size();++i)b[i].current=p.tiers[i];
    hits.assign(8,0);mass.assign(8,0);hits[4]=100;mass[4]=90;
    policy.observe(b,hits,mass,384);
    auto drift=policy.plan(b,32,1360);
    assert(drift.tiers[4]>drift.tiers[0] && drift.promotions>0 && drift.demotions>0);
    // Enough budget restores every block to canonical TQ8.
    auto all=policy.plan(b,32,8*340);
    for (auto tier:all.tiers)assert(tier==5);
    assert(precisions[1].key_bits==2 && precisions[1].value_bits==3);
    assert(precisions[3].key_bits==3 && precisions[3].value_bits==4);
    assert(!policy.update_due(511) && policy.update_due(512));
    PrecisionPolicy every256({8192,256,4096});
    assert(!every256.update_due(255) && every256.update_due(256));
    // A block straddling the recent-window boundary must stay high precision.
    auto boundary=blocks(4);
    auto bp=policy.plan(boundary,17,4*340);
    assert(bp.tiers[2]==5 && bp.tiers[3]==5);
    std::cout<<"QSA precision budget, recent window, drift promotion and intervals PASS\n";
}
