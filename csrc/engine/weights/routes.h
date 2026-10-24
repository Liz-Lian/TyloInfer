#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace mfq::engine {

class RouteHistory {
public:
    enum class HeatMode { count, unique, normalized };
    struct Bundle { int experts; bool verified; };
    struct Route { std::size_t bundle; int32_t expert; };
    struct Stats {
        std::uint64_t observed = 0, windows = 0, committed = 0, discarded = 0,
                      cancels = 0, heat_routes = 0;
    };
    struct Accepted { std::uint64_t routes = 0, heat_routes = 0; };

    RouteHistory(const std::vector<Bundle>& layout, HeatMode mode, float retention)
        : mode_(mode), retention_(retention) {
        const auto mode_value = static_cast<int>(mode);
        if (mode_value < static_cast<int>(HeatMode::count) ||
            mode_value > static_cast<int>(HeatMode::normalized) ||
            !std::isfinite(retention) || retention < 0 || retention > 1)
            throw std::invalid_argument("invalid expert heat policy");
        histories_.reserve(layout.size());
        for (const auto& bundle : layout) {
            if (bundle.experts <= 0)
                throw std::invalid_argument("invalid expert history layout");
            histories_.push_back({bundle.verified,
                std::vector<float>(std::size_t(bundle.experts)),
                std::vector<float>(std::size_t(bundle.experts)), {},
                std::vector<std::uint64_t>((std::size_t(bundle.experts) + 63) / 64)});
        }
    }

    void begin(int rows) {
        if (rows < 1 || rows > 32)
            throw std::invalid_argument("speculative route row count is out of range");
        if (pending_rows_)
            throw std::logic_error("speculative routes already pending");
        for (auto& history : histories_) {
            history.ids.clear();
            history.rows = 0;
        }
        pending_rows_ = rows;
    }
    bool pending() const noexcept { return pending_rows_ != 0; }

    void observe(std::size_t bundle, std::span<const int32_t> ids, int rows,
                 bool independent_sequences = false) {
        auto& history = histories_.at(bundle);
        const bool deferred = pending() && history.verified;
        if (deferred && (independent_sequences || rows < 1 || rows > pending_rows_ || ids.size() % rows))
            throw std::logic_error("speculative routes do not describe one aligned sequence");
        validate_experts(history, ids);
        if (deferred) {
            // Graph setup can publish a layer repeatedly; only the latest routes count.
            history.ids.assign(ids.begin(), ids.end());
            history.rows = rows;
        } else {
            for (const auto expert : ids)
                history.observations[expert] += 1.0f;
            observed_ += ids.size();
        }
    }

    // The returned view remains valid until the next prefix query.
    std::span<const Route> prefix(int rows) {
        validate_prefix(rows);
        accepted_.clear();
        if (!pending())
            return accepted_;
        for (int row = 0; row < rows; ++row)
            for (std::size_t bundle = 0; bundle < histories_.size(); ++bundle) {
                const auto& history = histories_[bundle];
                if (!history.rows)
                    continue;
                const auto routes = history.ids.size() / history.rows;
                for (std::size_t route = 0; route < routes; ++route)
                    accepted_.push_back({bundle, history.ids[std::size_t(row) * routes + route]});
            }
        return accepted_;
    }

    Accepted commit(int rows) {
        validate_prefix(rows);
        Accepted result;
        if (!pending())
            return result;
        for (auto& history : histories_) {
            if (!history.rows)
                continue;
            const auto count = std::size_t(rows) * (history.ids.size() / history.rows);
            if (mode_ != HeatMode::count)
                std::fill(history.seen.begin(), history.seen.end(), std::uint64_t(0));
            std::size_t distinct = 0;
            for (std::size_t i = 0; i < count; ++i) {
                const auto expert = unsigned(history.ids[i]);
                if (mode_ != HeatMode::count) {
                    auto& word = history.seen[expert / 64];
                    const auto bit = std::uint64_t(1) << (expert % 64);
                    if (word & bit)
                        continue;
                    word |= bit;
                }
                ++distinct;
            }
            // Preserve total token heat while valuing each saved transfer equally.
            const float weight = mode_ == HeatMode::normalized && distinct
                ? float(count) / float(distinct) : 1.0f;
            for (std::size_t i = 0; i < count; ++i) {
                const auto expert = unsigned(history.ids[i]);
                if (mode_ != HeatMode::count) {
                    auto& word = history.seen[expert / 64];
                    const auto bit = std::uint64_t(1) << (expert % 64);
                    if (!(word & bit))
                        continue;
                    word &= ~bit;
                }
                history.observations[expert] += weight;
            }
            result.routes += count;
            result.heat_routes += distinct;
            discarded_ += history.ids.size() - count;
        }
        observed_ += result.routes;
        committed_ += result.routes;
        heat_routes_ += result.heat_routes;
        ++windows_;
        pending_rows_ = 0;
        return result;
    }

    void discard() noexcept {
        if (!pending())
            return;
        for (auto& history : histories_) {
            discarded_ += history.ids.size();
            history.ids.clear();
            history.rows = 0;
        }
        ++cancels_;
        pending_rows_ = 0;
    }

    void prefill(std::size_t bundle, std::span<const int32_t> ids, int rows, int routes) {
        auto& history = histories_.at(bundle);
        if (rows < 1 || routes < 1 || ids.size() != std::size_t(rows) * routes)
            throw std::invalid_argument("invalid prefill route geometry");
        validate_experts(history, ids);
        std::fill(history.heat.begin(), history.heat.end(), 0.0f);
        std::fill(history.observations.begin(), history.observations.end(), 0.0f);
        // Seed about six recent token-equivalents so long prompts cannot dominate decode.
        for (int row = 0; row < rows; ++row) {
            const float weight = std::exp2(-float(rows - 1 - row) / 4.0f);
            for (int route = 0; route < routes; ++route)
                history.heat[ids[std::size_t(row) * routes + route]] += weight;
        }
    }

    void flush() {
        for (auto& history : histories_)
            for (std::size_t expert = 0; expert < history.heat.size(); ++expert) {
                history.heat[expert] += history.observations[expert];
                history.observations[expert] = 0;
            }
    }
    void decay() {
        for (auto& history : histories_)
            for (float& heat : history.heat)
                heat *= retention_;
    }
    float heat(std::size_t bundle, int expert) const {
        return histories_.at(bundle).heat.at(std::size_t(expert));
    }
    float total_heat(std::size_t bundle, int expert) const {
        const auto& history = histories_.at(bundle);
        return history.heat.at(std::size_t(expert)) + history.observations.at(std::size_t(expert));
    }
    Stats stats() const noexcept {
        return {observed_.load(), windows_.load(), committed_.load(), discarded_.load(),
                cancels_.load(), heat_routes_.load()};
    }

private:
    struct History {
        bool verified;
        std::vector<float> heat, observations;
        std::vector<int32_t> ids;
        std::vector<std::uint64_t> seen;
        int rows = 0;
    };

    static void validate_experts(const History& history, std::span<const int32_t> ids) {
        for (const auto expert : ids)
            if (unsigned(expert) >= history.heat.size())
                throw std::out_of_range("expert route is out of range");
    }

    void validate_prefix(int rows) const {
        if (!pending())
            return;
        if (rows < 0 || rows > pending_rows_)
            throw std::invalid_argument("committed route prefix is out of range");
        for (const auto& history : histories_)
            if (history.rows && history.rows < rows)
                throw std::logic_error("speculative route publication lost row alignment");
    }

    std::vector<History> histories_;
    std::vector<Route> accepted_;
    HeatMode mode_;
    float retention_;
    int pending_rows_ = 0;
    std::atomic<std::uint64_t> observed_{0}, windows_{0}, committed_{0}, discarded_{0},
        cancels_{0}, heat_routes_{0};
};

} // namespace mfq::engine
