#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

// Extracted from Strata generate.cpp's adaptive expert tier (MIT; see
// third_party/strata.LICENSE). The selection rule and defaults are unchanged.
namespace mfq::cuda::gguf {
struct AdaptiveSlotSwap {float gain;std::int32_t layer,in,out;};
template<class Resident>
std::vector<AdaptiveSlotSwap> adaptive_slot_swaps(
        const std::vector<std::vector<float>>& usage,Resident resident,std::size_t cap=96) {
    std::vector<AdaptiveSlotSwap> swaps;
    std::vector<std::pair<float,std::int32_t>> candidates,victims;
    for(std::size_t layer=0;layer<usage.size();++layer) {
        candidates.clear();victims.clear();
        const auto& counts=usage[layer];
        for(std::size_t expert=0;expert<counts.size();++expert) {
            if(!resident(layer,expert)) {if(counts[expert]>=2.f)candidates.emplace_back(counts[expert],std::int32_t(expert));}
            else victims.emplace_back(counts[expert],std::int32_t(expert));
        }
        if(candidates.empty() || victims.empty())continue;
        std::sort(candidates.begin(),candidates.end(),[](auto a,auto b){return a.first>b.first;});
        const auto count=std::min(candidates.size(),victims.size());
        std::partial_sort(victims.begin(),victims.begin()+std::ptrdiff_t(count),victims.end(),
            [](auto a,auto b){return a.first<b.first;});
        for(std::size_t i=0;i<count;++i) {
            if(candidates[i].first<victims[i].first+1.5f)break;
            swaps.push_back({candidates[i].first-victims[i].first,std::int32_t(layer),candidates[i].second,victims[i].second});
        }
    }
    std::sort(swaps.begin(),swaps.end(),[](auto a,auto b){return a.gain>b.gain;});
    if(swaps.size()>cap)swaps.resize(cap);
    return swaps;
}
}
