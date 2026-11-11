#pragma once

#include "backend.h"
#include "engine/models/session/snapshot.h"
#include "engine/models/state.h"
#include "engine/state/session/cache.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <map>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace tyloi {
class ModelSource;
namespace cache {
class PagedPrefixCache;
}
}

namespace tyloi::engine {
struct PrefixCacheConfig;
struct SessionCacheConfig;
}

namespace tyloi::cuda::internal {

bool cuda_has_separate_memory();

struct CudaSessionOps {
    using Snapshot = TextSessionState;
    using Restore = tyloi::state::Sessions::Restore;
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

    template <class State, class Predictor>
    static void reset_predictor(State& state, Predictor* mtp) {
        if (!mtp) return;
        if constexpr (requires { state.predictor; }) {
            if (!state.predictor)
                throw CudaSessionStateError("predictor state is unavailable");
            mtp->reset(*state.predictor, 1);
        } else {
            throw CudaSessionStateError("model state has no predictor state");
        }
    }

    template <class State, class Predictor>
    static void restore(
        State& state, const Snapshot& snapshot, const std::function<void()>& drain,
        Predictor* mtp) {
        // The caller owns the CUDA scope for the entire restore operation.
        if constexpr (std::is_same_v<std::remove_cvref_t<State>, tyloi::models::State>) {
            if (mtp)
                throw CudaSessionStateError("public model state has no joint predictor restore");
            drain();
            try {
                state.restore_text_session_state(snapshot);
                drain();
            } catch (...) {
                const auto error = std::current_exception();
                try { drain(); } catch (...) {}
                std::rethrow_exception(error);
            }
        } else if constexpr (requires { restore_session(state, mtp, snapshot, drain); }) {
            restore_session(state, mtp, snapshot, drain);
        } else {
            if constexpr (requires(decltype(state.prepare(snapshot))& candidate) {
                state.stage(candidate, snapshot);
                { state.publish(std::move(candidate)) } noexcept;
            }) {
                auto candidate = state.prepare(snapshot);
                try {
                    state.stage(candidate, snapshot);
                    drain();
                    state.publish(std::move(candidate));
                } catch (...) {
                    const auto error = std::current_exception();
                    try {
                        drain();
                    } catch (...) {
                    }
                    std::rethrow_exception(error);
                }
            } else if constexpr (requires { state.restore_text_session_state(snapshot); }) {
                state.restore_text_session_state(snapshot);
            } else {
                throw CudaSessionStateError("text session state is unsupported by this model");
            }
            if (mtp) {
                if constexpr (requires { state.predictor; })
                    mtp->restore_session_state(*state.predictor, *snapshot.mtp);
                else
                    throw CudaSessionStateError("model state has no predictor state");
            }
        }
    }
};

std::shared_ptr<tyloi::cache::PagedPrefixCache>
make_cuda_paged_prefix_cache(
    const tyloi::ModelSource& source,
    std::int64_t max_position_embeddings,
    bool supports_paged_text_session_state,
    const tyloi::engine::PrefixCacheConfig& config);

std::unique_ptr<tyloi::state::Sessions> make_sessions(
    const tyloi::ModelSource& source,
    const tyloi::engine::SessionCacheConfig& session_config,
    const tyloi::engine::PrefixCacheConfig& prefix_config,
    std::int64_t maximum_context, bool supported, bool paged);

} // namespace tyloi::cuda::internal
