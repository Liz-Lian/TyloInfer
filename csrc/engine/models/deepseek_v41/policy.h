#pragma once

#include "engine/generation/mtp/policy.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <optional>
#include <vector>

namespace mfq::models::deepseek_v41 {

class DepthController {
public:
    explicit DepthController(int maximum_depth = 3, int initial_depth = 2)
        : maximum_depth_(std::clamp(maximum_depth, 1, mfq::engine::mtp::kMaximumDraftDepth)),
          current_depth_(std::clamp(initial_depth, 1, maximum_depth_)),
          acceptance_(static_cast<std::size_t>(maximum_depth_), 0.6),
          warmup_accepts_(static_cast<std::size_t>(maximum_depth_), 0),
          warmup_trials_(static_cast<std::size_t>(maximum_depth_), 0),
          cycle_ms_(static_cast<std::size_t>(maximum_depth_ + 1)),
          cycle_age_ms_(static_cast<std::size_t>(maximum_depth_ + 1)) {
        // A wide cold window can load several GiB of unrelated SSD experts.
        // Match Metal's shallow start, measure plain decode, then refresh that
        // width after residency has warmed. The normal cost model and probes
        // still explore every allowed depth without a forced full-width sweep.
        warmup_.insert(warmup_.end(), {current_depth_, current_depth_});
        warmup_.insert(warmup_.end(), {0, 0, 0});
        warmup_.push_back(current_depth_);
        std::fprintf(stderr,"runtime_dspark_warmup initial_depth=%d maximum_depth=%d strategy=shallow_baseline_refresh\n",
            current_depth_,maximum_depth_);
    }

    int depth() const noexcept { return current_depth_; }
    int maximum_depth() const noexcept { return maximum_depth_; }

    void observe(
            int used_depth,
            int accepted_drafts,
            double cycle_ms,
            bool time_sample = true) {
        ++cycles_;
        used_depth = std::clamp(used_depth, 0, maximum_depth_);
        accepted_drafts = std::clamp(accepted_drafts, 0, used_depth);
        for (int position = 0; position < used_depth; ++position) {
            const double hit = position < accepted_drafts ? 1.0 : 0.0;
            const auto index = static_cast<std::size_t>(position);
            auto& estimate = acceptance_[index];
            if (!warmup_.empty()) {
                warmup_accepts_[index] += hit > 0.0 ? 1 : 0;
                ++warmup_trials_[index];
                estimate = static_cast<double>(warmup_accepts_[index]) /
                    static_cast<double>(warmup_trials_[index]);
            } else {
                estimate = (1.0 - kAcceptanceAlpha) * estimate +
                    kAcceptanceAlpha * hit;
            }
            if (position >= accepted_drafts) break;
        }

        cycle_ms = std::max(0.0, cycle_ms);
        if (time_sample) update_time(used_depth, cycle_ms);
        for (auto& age : cycle_age_ms_) {
            if (age) *age += cycle_ms;
        }
        if (time_sample) {
            cycle_age_ms_[static_cast<std::size_t>(used_depth)] = 0.0;
        }
        milliseconds_since_probe_ += cycle_ms;
        milliseconds_since_explore_ += cycle_ms;
        if (!warmup_.empty()) {
            // Maintenance and output/context-limited widths cannot satisfy the
            // requested calibration sample.
            if (!time_sample || used_depth != warmup_.front()) return;
            warmup_.erase(warmup_.begin());
            if (!warmup_.empty()) {
                current_depth_ = warmup_.front();
                return;
            }
            current_depth_ = best_depth();
            milliseconds_since_probe_ = 0.0;
            return;
        }
        if (probe_left_ > 0) {
            if (!time_sample || used_depth != current_depth_) return;
            --probe_left_;
            if (probe_left_ == 0) {
                current_depth_ = best_depth();
                milliseconds_since_probe_ = 0.0;
            }
            return;
        }

        current_depth_ = best_depth();
        if (maximum_depth_ <= 1) return;
        const double period = std::max(
            kProbePeriodMs,
            static_cast<double>(kProbeLength) * cycle_ms / kProbeDuty);
        if (milliseconds_since_probe_ < period) return;
        const bool explore_due = milliseconds_since_explore_ >=
            std::max(kExplorePeriodMs, 2.0 * period);
        const auto target = explore_due ? most_stale() : best_rival();
        if (!target) return;
        // A wide SSD verification may cost several ordinary cycles. Budget
        // the probe's own cost rather than assuming its width is equally cheap.
        const double target_period=std::max(period,
            static_cast<double>(kProbeLength)*time_estimate(*target)/kProbeDuty);
        if(milliseconds_since_probe_<target_period)return;
        if(!probe_budget_reported_) {
            probe_budget_reported_=true;
            std::fprintf(stderr,"runtime_dspark_probe_budget scope=first_probe enabled=1 strategy=candidate_cost depth=%d candidate_cycle_ms=%.3f minimum_gap_ms=%.3f\n",
                *target,time_estimate(*target),target_period);
        }
        current_depth_ = *target;
        probe_left_ = kProbeLength;
        milliseconds_since_probe_ = 0.0;
        if (explore_due) milliseconds_since_explore_ = 0.0;
    }

    double conditional_acceptance(int position) const {
        if (position < 0 || position >= maximum_depth_) {
            throw std::out_of_range("MTP acceptance position is out of range");
        }
        return acceptance_[static_cast<std::size_t>(position)];
    }

    std::optional<double> measured_cycle_ms(int depth) const {
        if (depth < 0 || depth > maximum_depth_) {
            throw std::out_of_range("MTP measured depth is out of range");
        }
        return cycle_ms_[static_cast<std::size_t>(depth)];
    }

private:
    static constexpr double kAcceptanceAlpha = 0.08;
    static constexpr double kTimeTauMs = 400.0;
    static constexpr double kProbePeriodMs = 1000.0;
    static constexpr double kExplorePeriodMs = 5000.0;
    static constexpr int kProbeLength = 4;
    static constexpr double kProbeDuty = 0.15;
    static constexpr double kProbeMargin = 1.15;
    static constexpr double kSpikeRatio = 2.0;
    static constexpr double kSpikeDamp = 0.25;
    static constexpr double kMarginalMs = 7.0;
    static constexpr double kHysteresis = 1.03;
    void update_time(int depth, double cycle_ms) {
        auto& estimate = cycle_ms_[static_cast<std::size_t>(depth)];
        if (!estimate) {
            estimate = cycle_ms;
            return;
        }
        if (!warmup_.empty()) {
            *estimate = std::min(*estimate, cycle_ms);
            return;
        }
        double alpha = 1.0 - std::exp(-cycle_ms / kTimeTauMs);
        if (cycle_ms > kSpikeRatio * *estimate) alpha *= kSpikeDamp;
        *estimate = (1.0 - alpha) * *estimate + alpha * cycle_ms;
    }

    double marginal_estimate() const {
        int low = -1;
        int high = -1;
        for (int depth = 0; depth <= maximum_depth_; ++depth) {
            if (!cycle_ms_[static_cast<std::size_t>(depth)]) continue;
            if (low < 0) low = depth;
            high = depth;
        }
        if (low >= 0 && high > low) {
            const double slope =
                (*cycle_ms_[static_cast<std::size_t>(high)] -
                 *cycle_ms_[static_cast<std::size_t>(low)]) /
                static_cast<double>(high - low);
            if (slope > 0.0) return slope;
        }
        return kMarginalMs;
    }

    double time_estimate(int depth) const {
        if (cycle_ms_[static_cast<std::size_t>(depth)]) {
            return *cycle_ms_[static_cast<std::size_t>(depth)];
        }
        int reference = -1;
        for (int candidate = 0; candidate <= maximum_depth_; ++candidate) {
            if (!cycle_ms_[static_cast<std::size_t>(candidate)]) continue;
            if (reference < 0 ||
                std::abs(candidate - depth) < std::abs(reference - depth)) {
                reference = candidate;
            }
        }
        if (reference < 0) return 30.0 + kMarginalMs * depth;
        if (depth == 0) {
            double minimum = std::numeric_limits<double>::infinity();
            for (const auto& measured : cycle_ms_) {
                if (measured) minimum = std::min(minimum, *measured);
            }
            return minimum;
        }
        return std::max(
            1.0e-3,
            *cycle_ms_[static_cast<std::size_t>(reference)] +
                marginal_estimate() * (depth - reference));
    }

    double score(int depth) const {
        double expected = 1.0;
        double run = 1.0;
        for (int position = 0; position < depth; ++position) {
            run *= acceptance_[static_cast<std::size_t>(position)];
            expected += run;
        }
        return expected / std::max(1.0e-6, time_estimate(depth));
    }

    int best_depth() const {
        int best = current_depth_;
        double best_score = -1.0;
        const int start = cycle_ms_.front() ? 0 : 1;
        for (int depth = start; depth <= maximum_depth_; ++depth) {
            const double candidate = score(depth);
            if (candidate > best_score) {
                best = depth;
                best_score = candidate;
            }
        }
        if (best != current_depth_ &&
            best_score < score(current_depth_) * kHysteresis) {
            return current_depth_;
        }
        return best;
    }

    std::optional<int> best_rival() const {
        const double current_score = score(current_depth_);
        std::optional<int> rival;
        double rival_score = 0.0;
        for (int depth = 1; depth <= maximum_depth_; ++depth) {
            if (depth == current_depth_) continue;
            const double candidate = score(depth);
            if (candidate > rival_score) {
                rival = depth;
                rival_score = candidate;
            }
        }
        if (current_depth_ != 0) {
            const double candidate = score(0);
            if (candidate > rival_score) {
                rival = 0;
                rival_score = candidate;
            }
        }
        if (current_score <= 0.0) return most_stale();
        return rival && rival_score >= current_score / kProbeMargin
            ? rival
            : std::nullopt;
    }

    std::optional<int> most_stale() const {
        std::optional<int> result;
        double oldest = -1.0;
        const auto consider = [&](int depth) {
            if (depth == current_depth_) return;
            const auto& age = cycle_age_ms_[static_cast<std::size_t>(depth)];
            const double value = age
                ? *age
                : std::numeric_limits<double>::infinity();
            if (value > oldest) {
                result = depth;
                oldest = value;
            }
        };
        for (int depth = 1; depth <= maximum_depth_; ++depth) consider(depth);
        consider(0);
        return result;
    }

    int maximum_depth_ = 1;
    int current_depth_ = 1;
    int cycles_ = 0;
    int probe_left_ = 0;
    bool probe_budget_reported_ = false;
    double milliseconds_since_probe_ = 0.0;
    double milliseconds_since_explore_ = 0.0;
    std::vector<double> acceptance_;
    std::vector<int> warmup_accepts_;
    std::vector<int> warmup_trials_;
    std::vector<std::optional<double>> cycle_ms_;
    std::vector<std::optional<double>> cycle_age_ms_;
    std::vector<int> warmup_;
};

} // namespace mfq::models::deepseek_v41
