#pragma once
#include "mtp_limits.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <string>

namespace mfq::engine::mtp {
struct DraftProbabilityOptions {
    double minimum_probability=0.;
    double chain_confidence=0.;
    bool cost_aware=false;
    bool enabled() const noexcept {return minimum_probability>0. || chain_confidence>0. || cost_aware;}
    static DraftProbabilityOptions from_environment() {
        const auto probability=[](const char* name) {
            const auto* text=std::getenv(name);
            if(!text || !*text)return 0.;
            char* end=nullptr;
            const double value=std::strtod(text,&end);
            if(end==text || *end || !std::isfinite(value) || value<0. || value>1.)
                throw std::invalid_argument(std::string(name)+" requires a probability in [0, 1]");
            return value;
        };
        DraftProbabilityOptions result;
        result.minimum_probability=probability("TYLOINFER_MTP_SPEC_MIN_P");
        result.chain_confidence=probability("TYLOINFER_DFLASH2_DRAFT_CONF");
        if(const auto* cost=std::getenv("TYLOINFER_MTP_SPEC_COST");cost && *cost) {
            if(std::string(cost)!="0" && std::string(cost)!="1")
                throw std::invalid_argument("TYLOINFER_MTP_SPEC_COST requires 0 or 1");
            result.cost_aware=cost[0]=='1';
        }
        return result;
    }
    void validate(bool blockwise,bool greedy) const {
        if(!std::isfinite(minimum_probability) || minimum_probability<0. || minimum_probability>1. ||
                !std::isfinite(chain_confidence) || chain_confidence<0. || chain_confidence>1.)
            throw std::invalid_argument("MTP probability thresholds must be in [0, 1]");
        if(enabled() && !greedy)
            throw std::invalid_argument("explicit probability-based MTP policies currently require greedy decoding");
        if(cost_aware && blockwise)
            throw std::invalid_argument("cost-aware chain drafting requires native MTP; DFlash2 uses --draft-conf");
        if(chain_confidence>0. && !blockwise)
            throw std::invalid_argument("--draft-conf requires a blockwise DFlash2 drafter");
    }
    int prefix(std::span<const float> probabilities) const {
        double chain=1.;
        int kept=0;
        for(float value:probabilities) {
            const double p=std::isfinite(value)?std::clamp(double(value),0.,1.):0.;
            chain*=p;
            const bool rejected=chain_confidence>0. ? chain<chain_confidence : p<minimum_probability;
            if(rejected)break;
            ++kept;
        }
        return kept;
    }
};

// Cost-aware native chain selection follows Strata DraftPolicy (MIT,
// third_party/strata.LICENSE): calibrated draft probabilities, expected
// committed tokens per measured round time, and two-step lookahead. Unknown
// row costs use a neutral linear prior rather than a particular GPU's curve.
class ProbabilityDepthPolicy {
public:
    struct Chain {double reach=1.,expected=1.,best_value=0.;int kept=0;};
    explicit ProbabilityDepthPolicy(int maximum):maximum_(std::clamp(maximum,1,kMaximumReportedDraftDepth)) {}
    double cost(int depth) const {
        if(depth<0 || depth>maximum_)throw std::out_of_range("MTP cost depth is outside the request cap");
        if(samples_[depth]>0.)return costs_[depth];
        double weighted=0.,weight=0.;
        for(int other=0;other<=maximum_;++other)if(samples_[other]>0.) {
            const double n=std::min(samples_[other],20.);
            weighted+=n*costs_[other]*(depth+1.)/(other+1.);weight+=n;
        }
        return weight>0.?weighted/weight:depth+1.;
    }
    double rate(double p) const {
        if(!std::isfinite(p))return 0.;
        p=std::clamp(p,0.,1.);
        const int bucket=std::min(7,int(p*8.));
        return (accepted_[bucket]+4.*p)/(reached_[bucket]+4.);
    }
    bool keep_draft(Chain& chain,int position,float probability,int cap) const {
        cap=std::clamp(cap,1,maximum_);
        if(position==0){chain={};chain.best_value=1./cost(0);}
        const double p=rate(probability);
        chain.reach*=p;chain.expected+=chain.reach;
        const int depth=position+1;
        const double value=chain.expected/cost(depth);
        if(value>chain.best_value*1.01){chain.best_value=value;chain.kept=depth;}
        if(samples_[depth]<3. && chain.reach>=.15)chain.kept=depth;
        if(depth>=cap)return false;
        const double next=(chain.expected+chain.reach*p)/cost(depth+1);
        const double after=depth+2<=cap?(chain.expected+chain.reach*p+chain.reach*p*p)/cost(depth+2):0.;
        return std::max(next,after)>chain.best_value*1.01 ||
            (samples_[depth+1]<3. && chain.reach*p>=.15);
    }
    void observe(int depth,int accepted,std::span<const float> probabilities,double milliseconds,bool time_sample) {
        if(depth<0 || depth>maximum_ || accepted<0 || accepted>depth || probabilities.size()<std::size_t(depth))
            throw std::invalid_argument("invalid cost-aware MTP observation");
        for(int i=0;i<depth && i<=accepted;++i) {
            const double p=std::isfinite(probabilities[i])?std::clamp(double(probabilities[i]),0.,1.):0.;
            const int bucket=std::min(7,int(p*8.));
            accepted_[bucket]=.995*accepted_[bucket]+double(i<accepted);
            reached_[bucket]=.995*reached_[bucket]+1.;
        }
        if(!time_sample || !std::isfinite(milliseconds) || milliseconds<=0.)return;
        const double known=cost(depth);
        if(samples_[depth]==0.) {
            const bool other=std::any_of(samples_.begin(),samples_.end(),[](double n){return n>0.;});
            if(other)milliseconds=known;
        } else milliseconds=std::min(milliseconds,2.*known);
        const double alpha=std::max(.1,1./(samples_[depth]+1.));
        costs_[depth]=samples_[depth]>0.?(1.-alpha)*costs_[depth]+alpha*milliseconds:milliseconds;
        samples_[depth]+=1.;
    }
private:
    int maximum_;
    std::array<double,kMaximumReportedDraftDepth+1> costs_{},samples_{};
    std::array<double,8> accepted_{},reached_{};
};
}
