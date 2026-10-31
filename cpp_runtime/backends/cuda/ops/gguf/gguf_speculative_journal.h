#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

struct GgufSpeculativeTouch {
    std::int64_t expert=0;
    std::uint64_t uses=0;
    bool missed=false;
};

// Residency must stay fixed during collect(). Reverse traversal records each
// expert's last access first and evaluates its miss state only once. Reversing
// the result when committing preserves the original final LRU order.
class GgufSpeculativeJournal {
    struct Index {std::uint64_t epoch=0;std::size_t offset=0;};
    std::vector<Index> index_;
    std::uint64_t epoch_=0;
public:
    template<class Missed>
    void collect(std::span<const std::int64_t> ids,std::size_t experts,
            std::vector<GgufSpeculativeTouch>& touches,Missed&& missed) {
        if(index_.size()<experts)index_.resize(experts);
        if(!++epoch_) {
            std::fill(index_.begin(),index_.end(),Index{});
            epoch_=1;
        }
        touches.clear();touches.reserve(std::min(ids.size(),experts));
        for(auto it=ids.rbegin();it!=ids.rend();++it) {
            const auto expert=*it;
            auto offset=touches.size();
            if(expert>=0 && std::uint64_t(expert)<experts) {
                auto& entry=index_[std::size_t(expert)];
                if(entry.epoch==epoch_)offset=entry.offset;
                else {entry.epoch=epoch_;entry.offset=offset;}
            } else {
                // Preserve arbitrary-ID behavior without indexing scratch
                // outside the bank's actual expert geometry.
                const auto found=std::find_if(touches.begin(),touches.end(),
                    [&](const auto& item){return item.expert==expert;});
                if(found!=touches.end())offset=std::size_t(found-touches.begin());
            }
            if(offset==touches.size())touches.push_back({expert,1,missed(expert)});
            else ++touches[offset].uses;
        }
    }
};
