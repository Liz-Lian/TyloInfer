#pragma once

#include "backend.h"
#include "backends/cuda/engine/executor/execution.h"
#include "backends/cuda/engine/state/session/state.h"
#include "engine/executor/mtp.h"
#include "engine/state/sessions.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace mfq {
class ModelSource;
namespace cache {
class PagedPrefixCache;
}
}

namespace mfq::engine {
struct PrefixCacheConfig;
struct SessionCacheConfig;
}

namespace mfq::cuda::internal {

bool cuda_has_separate_memory();

struct TextSessionRestore {
    std::size_t tokens = 0;
    mfq_tensor_backend::Tensor mtp_last_target_hidden;
};

struct CudaSessionOps {
    using Snapshot = TextSessionState;
    using Restore = TextSessionRestore;
    using Predictor = MtpModule;
    using StateError = CudaSessionStateError;

    static TextSessionState decode(
        const std::vector<CudaPagedPayload>& payloads,
        const std::vector<std::int64_t>& tokens,
        std::size_t block_size);
    static CudaPagedPayload encode(
        const Snapshot& state, std::size_t block_size, std::size_t index);
    static bool paged_mtp(const Snapshot& state);
    static bool partial_blocks(const Snapshot& state);
    static bool paged_media(const Snapshot& state);
    static bool checkpoint_needs_update(
        const CudaPagedPayload& payload, const Snapshot& state);
    static std::size_t decodable_blocks(
        const std::vector<CudaPagedPayload>& payloads);
    static std::size_t token_count(const CudaPagedPayload& payload);
    static void release_host_cache();

    template <class State>
    static void restore(
        State& state, const Snapshot& snapshot, CudaExecutionContext& execution) {
        if constexpr (requires(decltype(state.prepare(snapshot))& candidate) {
            state.stage(candidate, snapshot);
            { state.publish(std::move(candidate)) } noexcept;
        }) {
            auto candidate = state.prepare(snapshot);
            try {
                CudaScope scope(execution);
                state.stage(candidate, snapshot);
                execution.drain();
                state.publish(std::move(candidate));
            } catch (...) {
                const auto error = std::current_exception();
                try {
                    execution.drain();
                } catch (...) {
                }
                std::rethrow_exception(error);
            }
        } else {
            state.restore_text_session_state(snapshot);
        }
    }
};

std::shared_ptr<mfq::cache::PagedPrefixCache>
make_cuda_paged_prefix_cache(
    const mfq::ModelSource& source,
    std::int64_t max_position_embeddings,
    bool supports_paged_text_session_state,
    const mfq::engine::PrefixCacheConfig& config);

class TextSessionCache {
public:
    explicit TextSessionCache(
        const mfq::engine::SessionCacheConfig& session_config,
        const mfq::engine::PrefixCacheConfig& prefix_config,
        std::shared_ptr<mfq::cache::PagedPrefixCache> paged_cache = {},
        bool supported = true,
        int disabled_reason = 0,
        bool native_hot_snapshots = false);
    ~TextSessionCache();

    TextSessionCache(const TextSessionCache&) = delete;
    TextSessionCache& operator=(const TextSessionCache&) = delete;

    bool persistent_prefix_enabled() const noexcept;

    template <typename State>
    TextSessionRestore restore_best(
        State& state,
        CudaExecutionContext& execution,
        MtpModule* mtp,
        const std::string& requested_session,
        const std::vector<std::int64_t>& prompt,
        std::size_t maximum_prefix_tokens,
        const std::string& input_key = {});

    void store(const std::string& session_id, TextSessionState state);
    std::size_t fork_session(
        const std::string& source_session,
        const std::string& target_session);
    std::size_t close_session(const std::string& session_id);
    std::vector<std::pair<std::string, double>> metrics() const;
    std::map<int, std::size_t> device_hot_bytes() const;
    std::size_t clear_live_sessions() noexcept;
    std::size_t clear();
    std::uint64_t trim_hot(std::uint64_t target_bytes);
    std::uint64_t set_hot_limit(std::uint64_t max_bytes);
    std::uint64_t set_ram_limit(std::uint64_t max_bytes);
    void reserve_ram(std::uint64_t bytes);
    std::uint64_t set_disk_limit(std::uint64_t max_bytes);
    std::uint64_t refresh_disk_index();

private:
    struct Impl : mfq::engine::SessionCache<CudaSessionOps> {
        using SessionCache::SessionCache;
    };
    std::unique_ptr<Impl> impl_;
};

template <typename State>
TextSessionRestore TextSessionCache::restore_best(
    State& state,
    CudaExecutionContext& execution,
    MtpModule* mtp,
    const std::string& requested_session,
    const std::vector<std::int64_t>& prompt,
    std::size_t maximum_prefix_tokens,
    const std::string& input_key) {
    return impl_->restore_best(
        state, execution, mtp, requested_session, prompt, maximum_prefix_tokens, input_key);
}

} // namespace mfq::cuda::internal
