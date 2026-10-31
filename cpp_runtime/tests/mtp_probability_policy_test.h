#pragma once
#include "mtp_probability_policy.h"

void check_probability_depth_policy() {
    using namespace mfq::engine::mtp;
    const std::array<float,3> probabilities{.8f,.8f,.8f};
    DraftProbabilityOptions options;
    require(!options.enabled() && options.prefix(probabilities)==3);
    options.validate(false,false);
    options.minimum_probability=.8;
    require(options.prefix(probabilities)==3);
    options.minimum_probability=.81;
    require(options.prefix(probabilities)==0);
    options.chain_confidence=.5;
    require(options.prefix(probabilities)==3); // chain confidence overrides min-p
    options.chain_confidence=.52;
    require(options.prefix(probabilities)==2);
    rejects([&]{options.validate(false,true);});
    rejects([&]{options.validate(true,false);});
    options={};options.minimum_probability=std::numeric_limits<double>::quiet_NaN();
    rejects([&]{options.validate(false,true);});
    options={};options.cost_aware=true;
    rejects([&]{options.validate(true,true);});
    options.validate(false,true);
    ProbabilityDepthPolicy policy(3);
    const std::array<float,3> confident{.99f,.99f,.99f};
    policy.observe(0,0,{},16.,true);
    for(int depth=1;depth<=3;++depth)for(int i=0;i<80;++i)
        policy.observe(depth,depth,confident,16.+4.*depth,true);
    ProbabilityDepthPolicy::Chain chain;
    for(int position=0;position<3;++position)
        if(!policy.keep_draft(chain,position,.99f,3))break;
    require(chain.kept==3);
    const double before=policy.cost(3);
    policy.observe(3,0,confident,10000.,false);
    require(policy.cost(3)==before);
    for(int i=0;i<2000;++i)policy.observe(3,0,confident,28.,true);
    require(policy.rate(.99)<.02);
    for(int position=0;position<3;++position)
        if(!policy.keep_draft(chain,position,.99f,3))break;
    require(chain.kept==0);
    rejects([&]{policy.observe(4,4,confident,20.,true);});
    rejects([&]{policy.observe(3,4,confident,20.,true);});
    rejects([&]{policy.cost(-1);});
    require(policy.rate(std::numeric_limits<double>::quiet_NaN())==0.);
    ProbabilityDepthPolicy maximum(7);
    for(int position=0;position<7;++position)
        if(!maximum.keep_draft(chain,position,.99f,7))break;
    require(chain.kept==7);
    std::cout<<"MTP probability thresholds, chain confidence and native cost policy checks passed\n";
}
