#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#include "models/common/causal_forward.h"

namespace {
void check(bool value,const char* message){if(!value)throw std::runtime_error(message);}
struct Model;
struct Block {
    Model& owner;int id;
    void commit_speculative();
    void rollback_speculative(int64_t keep);
};
struct Model {
    int64_t speculative_start=4,speculative_confirmed=1,cache_pos=9,rolled_to=-1;
    bool guarded=false,needs_flush=false,flushed=false,fail_block=false;
    std::vector<int> events;
    std::vector<std::unique_ptr<Block>> blocks;
    Model(){for(int i=0;i<2;++i)blocks.push_back(std::make_unique<Block>(Block{*this,i}));}
    void adapter_commit_speculative() {
        check(!guarded && (!needs_flush || flushed),"Adapter observed unpublished state");events.push_back(20);
    }
    void adapter_rollback_speculative(int64_t keep) {
        check(!guarded && (!needs_flush || flushed),"Rollback adapter observed unpublished state");
        rolled_to=keep;events.push_back(21);
    }
};
void Block::commit_speculative() {
    check(owner.guarded,"Commit lost its block guard");
    if(owner.fail_block && id==1)throw std::runtime_error("test block failure");
    owner.events.push_back(id);
}
void Block::rollback_speculative(int64_t keep) {
    check(owner.guarded && keep==7,"Rollback boundary or guard differs");owner.events.push_back(id+2);
}
struct Guard {
    bool& active;
    explicit Guard(bool& value):active(value){check(!active,"Overlapping block scopes");active=true;}
    ~Guard(){active=false;}
    Guard(const Guard&)=delete;
};
auto guard_for(Model& model){return [&model](const auto&){return Guard(model.guarded);};}
}
int main()try {
    {
        Model m;mfq::models::finish_speculative(m,true,0,guard_for(m));
        check(m.events==std::vector<int>({0,1,20}) && m.speculative_start==-1 && !m.speculative_confirmed,
              "Legacy commit changed");
    }
    {
        Model m;m.needs_flush=true;
        mfq::models::finish_speculative(m,true,0,guard_for(m),[&]{
            check(!m.guarded && m.events==std::vector<int>({0,1}),"Flush ran before all layers completed");
            m.flushed=true;m.events.push_back(10);
        });
        check(m.events==std::vector<int>({0,1,10,20}) && m.cache_pos==9 && m.speculative_start==-1,
              "Commit publication order changed");
    }
    {
        Model m;mfq::models::finish_speculative(m,false,2,guard_for(m));
        check(m.events==std::vector<int>({2,3,21}) && m.rolled_to==7 && m.cache_pos==7 && m.speculative_start==-1,
              "Legacy rollback changed");
    }
    {
        Model m;m.speculative_start=-1;bool rejected=false;
        try{mfq::models::finish_speculative(m,true,0,guard_for(m),[&]{m.flushed=true;});}
        catch(const std::runtime_error&){rejected=true;}
        check(rejected && !m.flushed && m.events.empty(),"Invalid transaction performed work");
    }
    {
        Model m;m.fail_block=true;bool rejected=false;
        try{mfq::models::finish_speculative(m,true,0,guard_for(m),[&]{m.flushed=true;});}
        catch(const std::runtime_error&){rejected=true;}
        check(rejected && !m.guarded && !m.flushed && m.events==std::vector<int>({0}),
              "Block failure published a partial transaction");
    }
    {
        Model m;bool rejected=false;
        try{mfq::models::finish_speculative(m,true,0,guard_for(m),[]{throw std::runtime_error("test flush failure");});}
        catch(const std::runtime_error&){rejected=true;}
        check(rejected && !m.guarded && m.events==std::vector<int>({0,1}) && m.speculative_start==4,
              "Failed publication reached the adapter");
    }
    std::cout<<"Speculative commit publication order and failure checks PASS cases=6\n";return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
