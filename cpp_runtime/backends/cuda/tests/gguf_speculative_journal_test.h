#pragma once
#include "../ops/gguf/gguf_speculative_journal.h"
#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <list>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace gguf_journal_test {
struct State {
    std::vector<float> heat;
    std::unordered_map<std::int64_t,std::uint64_t> scores;
    std::list<std::uint64_t> lru;
    std::unordered_map<std::uint64_t,std::list<std::uint64_t>::iterator> resident;
    explicit State(std::size_t experts):heat(experts) {
        for(std::size_t e=0;e<experts;++e) {
            heat[e]=e%7?float(e):16777216.f;
            scores[std::int64_t(e)]=std::numeric_limits<std::uint64_t>::max()-2;
            for(unsigned role=0;role<3;++role)if((e+role)%5) {
                const auto key=(std::uint64_t(role)<<32)|e;
                lru.push_front(key);resident[key]=lru.begin();
            }
        }
    }
    bool missed(std::int64_t expert)const {
        for(unsigned role=0;role<3;++role)
            if(!resident.count((std::uint64_t(role)<<32)|std::uint64_t(expert)))return true;
        return false;
    }
    void touch(std::int64_t expert) {
        for(unsigned role=0;role<3;++role) {
            const auto key=(std::uint64_t(role)<<32)|std::uint64_t(expert);
            if(auto p=resident.find(key);p!=resident.end())lru.splice(lru.begin(),lru,p->second);
        }
    }
    void heat_one(std::int64_t expert) {
        if(expert>=0 && std::uint64_t(expert)<heat.size())heat[std::size_t(expert)]+=1.f;
    }
    void remove(std::int64_t expert) {
        const auto key=std::uint64_t(expert);
        if(auto p=resident.find(key);p!=resident.end()){lru.erase(p->second);resident.erase(p);}
    }
};
using Rows=std::vector<std::vector<std::pair<std::int64_t,bool>>>;
inline Rows observe_reference(const State& state,const std::vector<std::int64_t>& ids,int rows,int routes) {
    Rows result(rows);
    for(int t=0;t<rows;++t)for(int r=0;r<routes;++r)
        result[t].emplace_back(ids[std::size_t(t)*routes+r],state.missed(ids[std::size_t(t)*routes+r]));
    return result;
}
inline void resolve_reference(State& state,const Rows& rows) {
    for(const auto& row:rows)for(const auto& [expert,miss]:row) {
        state.heat_one(expert);
        if(miss)++state.scores[expert];else state.scores.erase(expert);
        state.touch(expert);
    }
}
inline void resolve_compact(State& state,const std::vector<GgufSpeculativeTouch>& touches) {
    for(auto it=touches.rbegin();it!=touches.rend();++it) {
        for(std::uint64_t i=0;i<it->uses;++i)state.heat_one(it->expert);
        if(it->missed)state.scores[it->expert]+=it->uses;else state.scores.erase(it->expert);
        state.touch(it->expert);
    }
}
inline void check_ordinary() {
    GgufSpeculativeJournal journal;std::size_t cases=0;
    std::vector<GgufSpeculativeTouch> touches,other_window;
    for(int experts:{1,17,257,1024,7}) {
        State expected(experts),actual(experts);
        for(int rows:{0,1,2,3,4,8,32})for(int routes:{1,5,10})for(int diversity:{0,3,53}) {
            std::vector<std::int64_t> ids(std::size_t(rows)*routes);
            for(std::size_t i=0;i<ids.size();++i)ids[i]=std::int64_t(i*31+diversity)%experts;
            if(!ids.empty() && diversity==53)ids.front()=-1;
            std::unordered_set<std::int64_t> seen;
            for(auto expert:ids) {
                expected.heat_one(expert);
                if(seen.insert(expert).second) {
                    if(expected.missed(expert))++expected.scores[expert];else expected.scores.erase(expert);
                }
            }
            std::size_t lookups=0;
            journal.collect(ids,std::size_t(experts),touches,[&](auto expert){++lookups;return actual.missed(expert);});
            // Other captured/speculative windows share the scratch index.
            // Their collection must not mutate an already collected journal.
            const std::int64_t other[]{0,0,experts-1};
            journal.collect(other,std::size_t(experts),other_window,[](auto){return true;});
            for(const auto& touch:touches) {
                for(std::uint64_t i=0;i<touch.uses;++i)actual.heat_one(touch.expert);
                if(touch.missed)++actual.scores[touch.expert];else actual.scores.erase(touch.expert);
            }
            if(lookups!=seen.size() || actual.scores!=expected.scores || actual.lru!=expected.lru ||
                    std::memcmp(actual.heat.data(),expected.heat.data(),actual.heat.size()*sizeof(float)))
                throw std::runtime_error("Ordinary route journal changed unique demand, heat rounding or residency");
            ++cases;
        }
    }
    std::cout<<"GGUF compact ordinary journal CPU cases="<<cases
        <<" heat-per-route/demand-once-per-expert/shared-scratch/FP32 saturation/overflow exact PASS\n";
}
inline void check() {
    check_ordinary();
    GgufSpeculativeJournal journal;
    std::size_t cases=0;
    for(int experts:{1,17,257,1024,7}) {
        State expected(experts),actual(experts);
        std::vector<GgufSpeculativeTouch> touches;
        for(int rows:{0,1,2,3,4,8,32})for(int routes:{1,5,10})for(int diversity:{0,3,53})
        for(int kept:{0,1,rows})for(bool cancelled:{false,true}) {
            std::vector<std::int64_t> ids(std::size_t(rows)*routes);
            for(int t=0;t<rows;++t)for(int r=0;r<routes;++r)
                ids[std::size_t(t)*routes+r]=((t*diversity+r)*31+7)%experts;
            if(!ids.empty() && diversity==53)ids.front()=-1;
            const auto original=observe_reference(expected,ids,rows,routes);
            std::size_t lookups=0;
            journal.collect(ids,std::size_t(experts),touches,[&](auto e){++lookups;return actual.missed(e);});
            const std::unordered_set<std::int64_t> distinct(ids.begin(),ids.end());
            if(lookups!=distinct.size() || touches.size()!=distinct.size())
                throw std::runtime_error("Compact journal repeated a residency lookup");
            // A later admission can change residency after observation. Keep
            // the original snapshot's miss flags and touch only current keys.
            const auto removed=ids.empty()?std::int64_t(0):ids.back();
            expected.remove(removed);actual.remove(removed);
            if(!cancelled && kept>0) {
                resolve_reference(expected,original);resolve_compact(actual,touches);
            }
            if(expected.scores!=actual.scores || expected.lru!=actual.lru ||
                    std::memcmp(expected.heat.data(),actual.heat.data(),expected.heat.size()*sizeof(float)))
                throw std::runtime_error("Compact journal changed heat, demand or final LRU order");
            ++cases;
        }
    }
    std::cout<<"GGUF compact speculative journal CPU cases="<<cases
        <<" miss snapshot/one lookup per expert/partial-full-cancel/FP32 saturation/uint64 overflow/last-use LRU exact PASS\n";
}
} // namespace gguf_journal_test
