#include "attn.h"
#include "tensor.h"
#include "engine/backends/cuda/kernels/mfq_cuda_attention_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_cache_ops.h"
#include "engine/backends/cuda/ops/attention/kv.h"
#include "engine/state/kv_view.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tyloi::diagnostics::numerics {
namespace {
/** Append the saved chunks through the native KV view and return its visible prefix. */
std::pair<tyloi::DenseTensor, tyloi::DenseTensor> append(TensorWork& work, const Json& config) {
    auto key = work.input("initial_k");
    auto val = work.input("initial_v");
    tyloi::state::KvView cache;
    cache.physical_key = key;
    cache.physical_value = val;
    cache.dtype = key.scalar_type();
    const auto keys = config.at("append_k").get<std::vector<std::string>>();
    const auto values = config.at("append_v").get<std::vector<std::string>>();
    if (keys.empty() || keys.size() != values.size())
        throw std::invalid_argument("invalid KV append chunks");
    auto start = config.at("initial_valid_length").get<std::int64_t>();
    std::pair<tyloi::DenseTensor, tyloi::DenseTensor> visible;
    for (std::size_t i = 0; i < keys.size(); ++i) {
        auto chunk_k = work.input(keys[i]);
        auto chunk_v = work.input(values[i]);
        const auto count = chunk_k.size(2);
        if (start < 0 || start > key.size(2) || count > key.size(2) - start)
            throw std::invalid_argument("KV append exceeds fixed capacity");
        const auto end = start + count;
        auto positions = work.keep(tyloi::cuda::arange(
            start, end, tyloi::TensorOptions().dtype(tyloi::kInt64).device(work.device())));
        visible = tyloi::cuda::ops::append_kv(cache, chunk_k, chunk_v, positions, start, true);
        work.keep(visible.first);
        work.keep(visible.second);
        start = end;
    }
    return {work.keep(visible.first.contiguous()), work.keep(visible.second.contiguous())};
}
} // namespace

bool run_attn(Case& value) {
    const auto& data = value.data();
    const auto op = data.at("op").get<std::string>();
    if (op != "attention" && op != "kv_write" && op != "kv_grow" && op != "kv_rejection")
        return false;
    if (data.at("source").value("generator", std::string()) != "tests/numerics/attn_ref.py")
        return false;
    if (data.value("actual_role", std::string("native")) != "native")
        throw std::invalid_argument("Python KV owner cases require the candidate producer");
    if (op == "kv_grow" || op == "kv_rejection")
        throw std::invalid_argument("native KV growth and admission behavior is not implemented");

    TensorWork work(value);
    const auto& config = data.at("config");
    if (config.value("observe_default_f16", false))
        throw std::invalid_argument("default KV dtype requires the Python constructor candidate");
    if (op == "kv_write") {
        auto key = work.input("initial_k");
        auto val = work.input("initial_v");
        auto k = work.input("k");
        auto v = work.input("v");
        auto positions = work.input("positions");
        auto written = kv_cache_write_cuda(key, val, k, v, positions);
        for (auto& tensor : written)
            work.keep(tensor);
        work.output("k", key);
        work.output("v", val);
        work.finish("kv_cache_write_cuda");
        return true;
    }

    auto q = work.input("q");
    const double scale = 1.0 / std::sqrt(double(q.size(-1)));
    tyloi::DenseTensor result;
    std::string entry;
    Json details = Json::object();
    if (config.value("append", false)) {
        auto [k, v] = append(work, config);
        result = work.keep(attention_cuda(q, k, v, scale, config.at("native_causal").get<bool>()));
        entry = "append_kv+attention_cuda";
    } else if (config.at("mode") == "ring") {
        auto k = work.input("k");
        auto v = work.input("v");
        auto cache_k = work.input("initial_k");
        auto cache_v = work.input("initial_v");
        auto length = work.input("seq_len");
        for (const auto& range : config.at("ring_writes")) {
            const auto start = range.at(0).get<std::int64_t>();
            const auto end = range.at(1).get<std::int64_t>();
            if (start < 0 || end < start || end > k.size(2) || end > v.size(2))
                throw std::invalid_argument("invalid ring write range");
            auto slice_k = work.keep(k.narrow(2, start, end - start));
            auto slice_v = work.keep(v.narrow(2, start, end - start));
            auto chunk_k = work.keep(slice_k.contiguous());
            auto chunk_v = work.keep(slice_v.contiguous());
            auto written = kv_cache_write_ring_cuda(cache_k, cache_v, chunk_k, chunk_v, start);
            for (auto& tensor : written)
                work.keep(tensor);
        }
        result = work.keep(attention_cache_swa_cuda(q, cache_k, cache_v, length, scale,
                                                    config.at("window").get<std::int64_t>()));
        entry = "kv_cache_write_ring_cuda+attention_cache_swa_cuda";
    } else {
        auto k = work.input("k");
        auto v = work.input("v");
        if (config.at("mode") == "window") {
            result = work.keep(
                attention_swa_cuda(q, k, v, scale, config.at("window").get<std::int64_t>()));
            entry = "attention_swa_cuda";
        } else {
            result =
                work.keep(attention_cuda(q, k, v, scale, config.at("native_causal").get<bool>()));
            entry = "attention_cuda";
        }
    }
    work.output("output", result);
    work.finish(entry, std::move(details));
    return true;
}
} // namespace tyloi::diagnostics::numerics
