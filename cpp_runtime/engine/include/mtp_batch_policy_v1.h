#pragma once
#include "mtp_limits.h"

// CUDA retains v1 exploration and parking. Each width learns its own actual
// committed-token yield and complete cycle cost. Ordinary timing is refreshed
// after cache warmup so drafts are compared with current ordinary execution.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <map>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace mfq::engine::mtp::v1 {

class BatchDepthPolicy {
public:
    using Clock = std::chrono::steady_clock;
    BatchDepthPolicy(std::vector<std::string> requests, int maximum, std::size_t ordinary_requests = 0,
            bool refresh_yields = true)
        : requests_(std::move(requests)), ordinary_requests_(ordinary_requests),
          maximum_(std::clamp(maximum, 1, kMaximumReportedDraftDepth)), current_(maximum_),
          refresh_yields_(refresh_yields) { committed_mean_.fill(1.); }
    const auto& requests() const noexcept { return requests_; }
    std::size_t ordinary_requests() const noexcept { return ordinary_requests_; }
    int maximum() const noexcept { return maximum_; }
    bool refresh_yields() const noexcept { return refresh_yields_; }
    void interrupt() noexcept { finished_.reset(); last_depth_ = -1; }
    int depth() const noexcept {
        return standard_warmup_ || standard_.size() < 3 || remaining_ ? 0 : current_;
    }
    std::optional<double> cost(int depth) const {
        const auto& values = depth ? costs_.at(depth) : standard_;
        if (values.empty()) return std::nullopt;
        // Cache exchange and CPU/DMA scheduling can make every few windows
        // expensive. Those windows still consume generation time; a median
        // would price a depth using only its inexpensive windows. Keep the
        // bounded recent history, including every eligible complete cycle.
        return std::accumulate(values.begin(), values.end(), 0.) / values.size();
    }
    double committed_tokens(int depth) const {
        if(depth<0 || depth>maximum_)throw std::out_of_range("MTP committed-token depth is out of range");
        return depth?committed_mean_[depth]:1.;
    }
    void observe(int depth, std::span<const int> accepted, Clock::time_point started,
        Clock::time_point finished, bool stable, bool interrupted) {
        if (interrupted) { interrupt(); return; }
        ++yield_epoch_;
        // A completed window still supplies valid accepted tokens when its
        // timing includes graph initialization or cache maintenance. Exclude
        // that cost without discarding the observed yield of this width.
        if (depth && !accepted.empty()) {
            const double committed = 1. +
                std::accumulate(accepted.begin(), accepted.end(), 0.) / accepted.size();
            auto& weight = committed_weight_[depth];
            // A dormant width must not retain the confidence of its early
            // samples while the response moves on. Consecutive observations
            // preserve the original EMA; missing cycles decay only its mass.
            const auto previous=last_yield_epoch_[depth];
            if(refresh_yields_ && previous && yield_epoch_>previous+1)
                weight*=std::pow(.92,double(yield_epoch_-previous-1));
            last_yield_epoch_[depth]=yield_epoch_;
            weight += .08 * (1. - weight);
            committed_mean_[depth] += .08 / weight * (committed - committed_mean_[depth]);
        }
        if (!stable) { interrupt(); return; }
        const auto begin = last_depth_ == depth && finished_ ? *finished_ : started;
        finished_ = finished;
        last_depth_ = depth;
        const auto ms = std::max(1e-6, std::chrono::duration<double, std::milli>(finished - begin).count());
        if (!depth) {
            if (standard_warmup_) --standard_warmup_;
            else append(standard_, ms);
            if (remaining_ && !--remaining_) {
                costs_ = {};
                last_seen_ = {};
                committed_mean_.fill(1.);
                committed_weight_.fill(0.);
                current_ = maximum_;
                losing_ = clear_losing_ = 0;
            }
            return;
        }
        elapsed_ms_ += ms;
        // started/finished delimit a complete stable cycle even immediately
        // after switching widths. No additional warmup cycle is needed after
        // the caller has already excluded initialization. Consecutive cycles
        // retain their existing inter-window host work in the measured cost.
        append(costs_[depth], ms);
        last_seen_[depth] = elapsed_ms_;
        for (int candidate = maximum_; candidate > 0; --candidate) if (costs_[candidate].size() < 3) {
            current_ = candidate;
            return;
        }
        // The initial ordinary samples precede speculative cache warming.
        // Refresh once after a second of useful measurements, then sparsely
        // (the same five-second age used for stale draft alternatives). Keep
        // the measured draft costs and acceptance through this short probe.
        if(elapsed_ms_ >= next_baseline_refresh_ms_) {
            refresh_standard();
            return;
        }
        ++decisions_;
        int best = 1;
        for (int candidate = 2; candidate <= maximum_; ++candidate)
            if (score(candidate) > score(best)) best = candidate;
        if (score(best) > score(current_) * 1.03) current_ = best;
        if (score(best) > score(0) * 1.03) confirm_loss_ = false;
        losing_ = score(best) <= score(0) * 1.03 ? losing_ + 1 : 0;
        clear_losing_ = score(best) < score(0) * .9 ? clear_losing_ + 1 : 0;
        if (elapsed_ms_ - last_probe_ms_ >= 1000) {
            // Low early acceptance can recover within a short response.
            // Delay only widths whose measured cost cannot compete even at
            // full acceptance, retaining the existing stale-cost refresh.
            int rival = 0;
            for (int candidate = 1; candidate <= maximum_; ++candidate) {
                if (candidate == current_ || (score(candidate,refresh_yields_) * 1.15 < score(best) &&
                    elapsed_ms_ - last_seen_[candidate] < 5000)) continue;
                if (!rival || last_seen_[candidate] < last_seen_[rival]) rival = candidate;
            }
            if (rival) {
                current_ = rival;
                last_probe_ms_ = elapsed_ms_;
            }
        }
        if (losing_ >= 16 || clear_losing_ >= 4) {
            // Cache and I/O pressure can change after the ordinary probe.
            // Recheck the current baseline before suspending speculation.
            if (!confirm_loss_) {
                refresh_standard();
                confirm_loss_ = true;
                current_ = best;
                return;
            }
            start_parked(cooldown_);
            standard_warmup_ = 2;
            losing_ = clear_losing_ = 0;
            ++parks_;
        }
    }
    void start_parked(std::uint64_t remaining, std::uint64_t next = 0) {
        remaining_ = remaining;
        confirm_loss_ = false;
        cooldown_ = next ? next : std::min<std::uint64_t>(4096, remaining * 2);
        decisions_ = 0;
    }
    std::uint64_t remaining() const noexcept { return remaining_; }
    std::uint64_t cooldown() const noexcept { return cooldown_; }
    std::uint64_t parks() const noexcept { return parks_; }
    bool held_up() const noexcept { return decisions_ >= 32; }

private:
    void refresh_standard() {
        next_baseline_refresh_ms_ = elapsed_ms_ + 5000.;
        standard_.clear();
        standard_warmup_ = 2;
        losing_ = clear_losing_ = 0;
    }
    static void append(std::deque<double>& values, double value) {
        if (values.size() == 8) values.pop_front();
        values.push_back(value);
    }
    double score(int depth,bool optimistic=false) const {
        const auto milliseconds = cost(depth);
        if (!milliseconds) return 0;
        const double expected = depth ? (optimistic?depth+1.:committed_mean_[depth]) : 1.;
        return (ordinary_requests_ + requests_.size() * expected) / *milliseconds;
    }
    std::vector<std::string> requests_;
    std::size_t ordinary_requests_;
    int maximum_, current_, standard_warmup_ = 2, losing_ = 0, clear_losing_ = 0, last_depth_ = -1;
    bool refresh_yields_, confirm_loss_ = false;
    std::array<double, kMaximumReportedDraftDepth + 1> committed_mean_{};
    std::array<double, kMaximumReportedDraftDepth + 1> committed_weight_{};
    std::array<std::deque<double>, kMaximumReportedDraftDepth + 1> costs_;
    std::deque<double> standard_;
    std::array<double, kMaximumReportedDraftDepth + 1> last_seen_{};
    std::array<std::uint64_t,kMaximumReportedDraftDepth+1> last_yield_epoch_{};
    std::uint64_t yield_epoch_=0;
    double elapsed_ms_ = 0, last_probe_ms_ = 0, next_baseline_refresh_ms_ = 1000;
    std::optional<Clock::time_point> finished_;
    std::uint64_t remaining_ = 0, cooldown_ = 128, decisions_ = 0, parks_ = 0;
};

class BatchParkMemory {
public:
    void tick() noexcept { ++clock_; }
    void seed(BatchDepthPolicy& policy) const {
        const auto found = verdicts_.find(key(policy));
        if (found != verdicts_.end() && found->second.until > clock_)
            policy.start_parked(found->second.until - clock_, found->second.cooldown);
    }
    void parked(const BatchDepthPolicy& policy) {
        verdicts_[key(policy)] = {clock_ + policy.remaining(), policy.cooldown()};
    }
    void retired(const BatchDepthPolicy& policy) {
        if (policy.held_up()) verdicts_.erase(key(policy));
    }
private:
    using Key = std::tuple<std::size_t, std::size_t, int>;
    static Key key(const BatchDepthPolicy& policy) {
        return {policy.requests().size(), policy.ordinary_requests(), policy.maximum()};
    }
    struct Verdict { std::uint64_t until, cooldown; };
    std::uint64_t clock_ = 0;
    std::map<Key, Verdict> verdicts_;
};

} // namespace mfq::engine::mtp::v1
