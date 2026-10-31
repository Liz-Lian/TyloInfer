#pragma once

namespace mfq::engine::mtp {
inline constexpr int kMaximumDraftDepth = 5;
// A block drafter produces seven proposals plus the target anchor row.
inline constexpr int kMaximumBlockwiseDraftDepth = 7;
inline constexpr int kMaximumReportedDraftDepth = kMaximumBlockwiseDraftDepth;
}
