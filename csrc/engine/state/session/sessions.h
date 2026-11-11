#pragma once

#include "engine/requests/result.h"
#include "engine/tensor.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

struct TextSessionState;
namespace tyloi { class Executor; }
namespace tyloi::models { class State; struct Mtp; }
namespace tyloi::weights { class Cache; }

namespace tyloi::state {

class Sessions {
public:
    /** Restored prefix length and the predictor's last target hidden state, if any. */
    struct Restore {
        std::size_t tokens = 0;
        Tensor mtp_last_target_hidden;
    };
    virtual ~Sessions() = default;
    /** Report whether anonymous persistent prefix reuse is enabled. */
    virtual bool persistent_prefix_enabled() const noexcept = 0;
    /** Restore the best compatible prefix in the owner's execution scope. */
    virtual Restore restore(models::State& state, tyloi::Executor& executor, weights::Cache& weights,
        models::Mtp* mtp, const std::string& requested_session,
        const std::vector<std::int64_t>& prompt, std::size_t maximum_prefix_tokens,
        const std::string& input_key = {}) = 0;
    /** Store an owned snapshot under the requested session or prefix policy. */
    virtual void store(const std::string& session_id, TextSessionState snapshot) = 0;
    virtual engine::SessionResult control(const engine::SessionCommand& command) = 0;
    virtual std::vector<std::pair<std::string, double>> metrics() const = 0;
    virtual std::map<int, std::size_t> device_hot_bytes() const = 0;
    virtual std::uint64_t trim_hot(std::uint64_t bytes) = 0;
    virtual void reserve_ram(std::uint64_t bytes) = 0;
};

} // namespace tyloi::state
