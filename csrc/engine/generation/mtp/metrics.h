#pragma once

#include <array>
#include <cstdint>

#include <string>
#include <utility>
#include <vector>

namespace mfq::engine::mtp {

inline constexpr int kMaximumReportedDraftDepth = 7;

struct GenerationStats {
    bool available = false;
    bool used = false;
    uint64_t cycles = 0;
    uint64_t drafted_tokens = 0;
    uint64_t accepted_tokens = 0;
    std::array<uint64_t, kMaximumReportedDraftDepth + 1> depth_cycles{};
    std::array<uint64_t, kMaximumReportedDraftDepth> position_drafted{};
    std::array<uint64_t, kMaximumReportedDraftDepth> position_accepted{};
    std::array<double, kMaximumReportedDraftDepth + 1> measured_depth_ms{};
    int selected_depth = 0;
    uint64_t standard_tokens = 0;
    uint64_t park_count = 0;
    uint64_t reentry_probes = 0;
    // Host wall time, including waits observed in each phase; no extra GPU
    // synchronization is inserted for measurement.
    double target_ms = 0;
    double head_ms = 0;
    double rollback_ms = 0;
};

void append_generation_metrics(
    std::vector<std::pair<std::string, double>>& metrics,
    const GenerationStats& stats);

} // namespace mfq::engine::mtp
