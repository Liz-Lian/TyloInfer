#pragma once

#include "engine/models/common/generation_ops.h"
#include "engine/models/common/mtp.h"

#include <algorithm>
#include <chrono>
#include <optional>
#include <utility>

namespace mfq::cuda {

template<class Model> bool mtp_stream_prompt_history(Model&, int64_t, const char*) { return false; }
template<class Model> bool mtp_host_prompt_history(Model&, int64_t) { return false; }
template<class Model> bool mtp_bootstrap_before_publication(Model&, const char*) { return false; }
template<class Model, class Mtp> bool mtp_retains_partial_target_prefix(Model&, Mtp& mtp, int) { return mtp.retains_partial_target_prefix(); }
template<class Model> std::size_t mtp_history_merge_limit(Model&) { return 0; }
template<class Model> std::size_t mtp_history_flush_threshold(Model&, const char*) { return 512; }
template<class Model> std::size_t mtp_parked_history_flush_threshold(Model&, const char*) { return 512; }
template<class Model> int64_t mtp_predictor_prefill_chunk_size(Model&, int64_t requested) { return std::min<int64_t>(512,requested); }
template<class Model, class Ops> bool mtp_cache_only_prompt_history(Model&, Ops&) { return false; }
template<class Model, class Ops> void mtp_initialize(Model&, Ops&) {}

template<class Model, class Ops, class Tensor, class Finish>
std::optional<Tensor> mtp_target_forward(Model&, Ops&, Tensor&, Tensor*, int64_t, std::uint64_t, const Finish&) { return std::nullopt; }

template<class Model, class Ops, class Chunk, class Tensor, class Timer, class Finish>
std::optional<typename Ops::PrefillResult> mtp_prefill(Model&, Ops&, const Chunk&, Tensor&, Tensor&, Timer&, const Finish&) { return std::nullopt; }

template<class Model, class Ops, class Tensor>
std::optional<MtpStep> mtp_predictor_step(Model&, Ops&, Tensor&, Tensor&, Tensor&) { return std::nullopt; }
template<class Model, class Ops> void mtp_predictor_compact(Model&, Ops&, MtpStep&) {}
template<class Model, class Ops, class Tensor>
bool mtp_predictor_history_step(Model&, Ops&, Tensor&, Tensor&, Tensor&, bool) { return false; }
template<class Model, class Ops, class Started>
double mtp_begin_predictor_prefill(Model&, Ops&, int64_t, Started started) {
    return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
}
template<class Model, class Ops, class Tensor> void mtp_predictor_prefill_begin(Model&, Ops&, Tensor&) {}
template<class Model, class Ops, class Tensor, class Timer>
void mtp_predictor_prefill(Model&, Ops& ops, Tensor& hidden, Tensor& ids, Tensor& positions, Timer&, bool&) {
    (void)ops.predictor_step(std::move(hidden),std::move(ids),std::move(positions));
}

} // namespace mfq::cuda
