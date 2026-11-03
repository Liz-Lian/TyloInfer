#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <vector>

namespace tyloinfer::qsa {

struct Precision {
    std::uint8_t key_bits, value_bits;
};
// Ordered by storage cost. Half-bit modes assign the extra bit to V.
inline constexpr std::array<Precision, 6> precisions{{{2,2},{2,3},{3,3},{3,4},{4,4},{8,8}}};

struct PrecisionPolicyConfig {
    std::uint64_t recent_tokens = 8192;
    std::uint32_t update_tokens = 128;
    double history_half_life_tokens = 4096;

    void validate() const {
        if (!recent_tokens || recent_tokens % 4 ||
            (update_tokens != 128 && update_tokens != 256) ||
            !std::isfinite(history_half_life_tokens) || history_half_life_tokens <= 0)
            throw std::invalid_argument("invalid adaptive QSA precision policy");
    }
};

struct BlockPrecisionHistory {
    std::uint64_t token_begin = 0;
    // Both sums are exponentially decayed. Attention mass is the sum of
    // token softmax probabilities, averaged over every Q head in the layer.
    double hits = 0, attention_mass = 0;
    // Distortion is measured against the canonical high-precision image.
    std::array<double, 6> distortion{};
    std::array<std::size_t, 6> bytes{};
    std::uint8_t current = 5;
};

struct PrecisionPlan {
    std::vector<std::uint8_t> tiers;
    std::size_t bytes = 0, minimum_bytes = 0;
    std::size_t promotions = 0, demotions = 0;
};

class PrecisionPolicy {
public:
    explicit PrecisionPolicy(PrecisionPolicyConfig config = {}) : config_(config) { config_.validate(); }

    bool update_due(std::uint64_t tokens) const {
        return tokens >= last_update_ && tokens - last_update_ >= config_.update_tokens;
    }

    void observe(std::vector<BlockPrecisionHistory>& blocks,
                 const std::vector<double>& interval_hits,
                 const std::vector<double>& interval_mass,
                 std::uint64_t tokens) {
        if (interval_hits.size() != blocks.size() || interval_mass.size() != blocks.size() ||
            tokens < last_update_)
            throw std::invalid_argument("invalid adaptive QSA observations");
        const auto decay = std::exp2(-double(tokens-last_update_) / config_.history_half_life_tokens);
        for (std::size_t i=0;i<blocks.size();++i) {
            if (!std::isfinite(interval_hits[i]) || !std::isfinite(interval_mass[i]) ||
                interval_hits[i]<0 || interval_mass[i]<0)
                throw std::invalid_argument("invalid QSA hit count or attention mass");
            blocks[i].hits = blocks[i].hits * decay + interval_hits[i];
            blocks[i].attention_mass = blocks[i].attention_mass * decay + interval_mass[i];
        }
        last_update_ = tokens;
    }

    PrecisionPlan plan(const std::vector<BlockPrecisionHistory>& blocks,
                       std::uint64_t tokens, std::size_t byte_budget,bool allow_promotions=true) const {
        PrecisionPlan result;
        result.tiers.resize(blocks.size());
        std::size_t full_bytes=0;
        for(const auto& block:blocks) {
            if(block.current>=precisions.size())throw std::invalid_argument("invalid adaptive QSA block");
            const auto bytes=block.bytes[allow_promotions?5:block.current];
            if(bytes>std::numeric_limits<std::size_t>::max()-full_bytes)
                throw std::overflow_error("adaptive QSA byte count overflow");
            full_bytes+=bytes;
        }
        if(full_bytes<=byte_budget) {
            result.bytes=full_bytes;
            for(std::size_t i=0;i<blocks.size();++i) {
                if(blocks[i].current>=precisions.size() || blocks[i].token_begin>=tokens || !blocks[i].bytes.back())
                    throw std::invalid_argument("invalid adaptive QSA block");
                result.tiers[i]=allow_promotions?5:blocks[i].current;
                result.promotions+=result.tiers[i]>blocks[i].current;
            }
            return result;
        }
        struct Upgrade { double benefit; std::size_t block; std::uint8_t to; };
        struct LowerPriority {
            bool operator()(const Upgrade& a,const Upgrade& b) const {
                if (a.benefit != b.benefit) return a.benefit < b.benefit;
                return a.block > b.block;
            }
        };
        std::priority_queue<Upgrade,std::vector<Upgrade>,LowerPriority> pending;
        double maximum_importance = 0;
        for (const auto& block:blocks)
            maximum_importance = std::max(maximum_importance,importance(block));
        // Unobserved blocks retain a small nonzero value without dominating
        // observed blocks. Otherwise a cold region could never be promoted.
        const auto prior = std::max(maximum_importance * 1e-6, 1e-12);
        const auto upgrade = [&](std::size_t index,std::uint8_t to) {
            const auto& block=blocks[index];
            if(!allow_promotions && to>block.current)return;
            const auto from=std::uint8_t(to-1);
            const auto saved=std::max(0.,block.distortion[from]-block.distortion[to]);
            const auto extra=block.bytes[to]-block.bytes[from];
            pending.push({(importance(block)+prior)*saved/double(extra),index,to});
        };
        const auto add_bytes = [&](std::size_t bytes) {
            if (bytes > std::numeric_limits<std::size_t>::max()-result.bytes)
                throw std::overflow_error("adaptive QSA byte count overflow");
            result.bytes+=bytes;
        };
        for (std::size_t i=0;i<blocks.size();++i) {
            const auto& block=blocks[i];
            if (block.current >= precisions.size() || block.token_begin >= tokens || !block.bytes[0])
                throw std::invalid_argument("invalid adaptive QSA block");
            for (std::size_t tier=0;tier<precisions.size();++tier) {
                if (!std::isfinite(block.distortion[tier]) || block.distortion[tier]<0 ||
                    (tier && block.bytes[tier]<=block.bytes[tier-1]))
                    throw std::invalid_argument("invalid adaptive QSA distortion or byte cost");
            }
            // A block partially overlapping the protected window stays 8-bit.
            const bool recent = tokens <= config_.recent_tokens ||
                block.token_begin + 4 > tokens-config_.recent_tokens;
            result.tiers[i]=recent?(allow_promotions?5:block.current):0;
            add_bytes(block.bytes[result.tiers[i]]);
            if (!recent) upgrade(i,1);
        }
        result.minimum_bytes=result.bytes;
        if (result.bytes>byte_budget)
            throw std::runtime_error("adaptive QSA VRAM budget cannot hold recent 8-bit and historical TQ2");
        while (!pending.empty()) {
            auto next=pending.top();pending.pop();
            const auto& block=blocks[next.block];
            const auto from=result.tiers[next.block];
            if (from+1 != next.to) continue;
            const auto extra=block.bytes[next.to]-block.bytes[from];
            if (extra>byte_budget-result.bytes) continue;
            result.tiers[next.block]=next.to;result.bytes+=extra;
            if (next.to+1<precisions.size()) upgrade(next.block,std::uint8_t(next.to+1));
        }
        for (std::size_t i=0;i<blocks.size();++i) {
            result.promotions+=result.tiers[i]>blocks[i].current;
            result.demotions+=result.tiers[i]<blocks[i].current;
        }
        return result;
    }

    const PrecisionPolicyConfig& config() const { return config_; }

private:
    // Hit rate times sqrt(mean conditional attention mass). The common query
    // denominator cancels when ranking blocks in one update interval.
    static double importance(const BlockPrecisionHistory& block) {
        return std::sqrt(std::max(0.,block.hits)*std::max(0.,block.attention_mass));
    }
    PrecisionPolicyConfig config_;
    std::uint64_t last_update_ = 0;
};
} // namespace tyloinfer::qsa
