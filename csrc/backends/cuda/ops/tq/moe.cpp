#include "backends/cuda/ops/tq/include/moe.h"

#include "backends/cuda/ops/tq/include/detail.h"
#include "native/cuda/blas.h"
#include "mfq_cuda_activation_ops.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace tb = mfq_tensor_backend;

namespace tq_ops {

IndexedJobs make_indexed_jobs(
        std::span<const TqWeight> gate, std::span<const TqWeight> up,
        std::span<const TqWeight> down,
        const std::vector<std::vector<std::int64_t>>& slots,
        std::size_t begin, std::int64_t routes, std::int64_t input,
        std::int64_t hidden, std::int64_t output, bool wide) {
    IndexedTiles gate_up, down_tiles;
    WideTiles wide_tiles;
    std::vector<TrellisIndexedWidePairJob> wide_gate_up;
    for (std::size_t i = 0; i < gate.size(); ++i) {
        const auto& rows = slots[begin + i];
        const auto complete = wide ? rows.size() / 8 * 8 : 0;
        for (std::size_t start = 0; start < complete; start += 8) {
            TrellisIndexedWidePairJob job{};
            job.weight = gate[i].view;
            job.second = up[i].view;
            job.count = 8;
            for (int t = 0; t < 8; ++t) {
                const auto slot = rows[start + t];
                job.input_offsets[t] = (slot / routes) * input;
                job.output_offsets[t] = slot * hidden;
            }
            wide_gate_up.push_back(job);
        }
        // Keep incomplete MMA groups in the short SIMT kernel.
        auto chunks = (rows.size() - complete + 3) / 4;
        for (std::size_t start = complete; start < rows.size(); --chunks) {
            const auto tile = (rows.size() - start + chunks - 1) / chunks;
            TrellisIndexedProjectionJob gu{}, dn{};
            gu.weight = gate[i].view;
            gu.second = up[i].view;
            dn.weight = down[i].view;
            gu.count = dn.count = int(tile);
            for (int t = 0; t < gu.count; ++t) {
                const auto slot = rows[start + t];
                gu.input_offsets[t] = (slot / routes) * input;
                gu.output_offsets[t] = slot * hidden;
                dn.input_offsets[t] = slot * hidden;
                dn.output_offsets[t] = slot * output;
            }
            gate_up[tile - 1].push_back(gu);
            if (!wide) down_tiles[tile - 1].push_back(dn);
            start += tile;
        }
        if (wide) {
            auto down_chunks = (rows.size() + 7) / 8;
            for (std::size_t start = 0; start < rows.size(); --down_chunks) {
                const auto tile = (rows.size() - start + down_chunks - 1) / down_chunks;
                TrellisIndexedWideProjectionJob job{};
                job.weight = down[i].view;
                job.count = int(tile);
                for (int t = 0; t < job.count; ++t) {
                    const auto slot = rows[start + t];
                    job.input_offsets[t] = slot * hidden;
                    job.output_offsets[t] = slot * output;
                }
                if (tile > 4) wide_tiles[tile - 5].push_back(job);
                else {
                    TrellisIndexedProjectionJob short_job{};
                    short_job.weight = job.weight;
                    short_job.count = job.count;
                    std::copy_n(job.input_offsets, job.count, short_job.input_offsets);
                    std::copy_n(job.output_offsets, job.count, short_job.output_offsets);
                    down_tiles[tile - 1].push_back(short_job);
                }
                start += tile;
            }
        }
    }
    return {std::move(gate_up), std::move(down_tiles),
            std::move(wide_tiles), std::move(wide_gate_up)};
}

void routed_projection(const std::vector<TqWeight>& weights,
        tb::Tensor source, tb::Tensor output) {
    if (source.dim() != 2 && source.dim() != 3)
        throw std::invalid_argument("invalid prepared TQ routed activation geometry");
    if (output.dim() != 3)
        throw std::invalid_argument("invalid prepared TQ routed output geometry");
    const auto tokens = source.size(0), routes = output.size(1);
    if (tokens < 1 || routes < 1 || output.size(0) != tokens || weights.size() !=
            static_cast<std::size_t>(tokens * routes))
        throw std::invalid_argument("invalid prepared TQ routed projection");
    if (source.dim() == 3 && source.size(1) != routes)
        throw std::invalid_argument("TQ routed source and output route counts differ");
    source = source.contiguous();
    const auto out = weights.front().out;
    if (output.size(2) != out || !output.is_cuda() || output.device() != source.device())
        throw std::invalid_argument("invalid prepared TQ routed output tensor");
    std::vector<TrellisProjectionView> views;
    views.reserve(weights.size());
    for (const auto& weight : weights) views.push_back(weight.view);
    const auto bytes = static_cast<std::int64_t>(views.size() * sizeof(views[0]));
    const auto metadata = tb::from_blob(
        views.data(), {bytes}, tb::TensorOptions().dtype(tb::kUInt8))
        .clone().to(source.device(), false, false).contiguous();
    const auto* data = reinterpret_cast<const TrellisProjectionView*>(metadata.data_ptr());
    const auto element = source.element_size();
    for (std::int64_t token = 0; token < tokens; ++token) {
        const auto input_offset = token * (source.dim() == 3 ? routes : 1) *
            weights.front().neuron_len * element;
        const auto output_offset = token * routes * out * element;
        tq_detail::status(mfq_tq_gemv_batch(
            data + token * routes,
            static_cast<const std::uint8_t*>(source.data_ptr()) + input_offset,
            static_cast<std::uint8_t*>(output.data_ptr()) + output_offset,
            static_cast<int>(routes), static_cast<int>(out),
            source.dim() == 3 ? static_cast<int>(weights.front().neuron_len) : 0,
            static_cast<int>(out), tq_detail::dtype(source.scalar_type()),
            0, 0, mfq_current_cuda_stream()));
    }
}

void run_long_expert(const TqWeight& gate, const TqWeight& up,
        const TqWeight& down, const std::vector<std::int64_t>& slots,
        std::int64_t routes, const tb::Tensor& source, tb::Tensor output,
        mfq::cuda::BlasResources* blas, float limit) {
    std::vector<std::int64_t> token_ids;
    token_ids.reserve(slots.size());
    for (const auto slot : slots) token_ids.push_back(slot / routes);
    const auto token_index = tb::from_blob(
        token_ids.data(), {static_cast<std::int64_t>(token_ids.size())},
        tb::TensorOptions().dtype(tb::kInt64)).clone().to(source.device(), false, false);
    const auto slot_index = tb::from_blob(
        const_cast<std::int64_t*>(slots.data()), {static_cast<std::int64_t>(slots.size())},
        tb::TensorOptions().dtype(tb::kInt64)).clone().to(source.device(), false, false);
    const auto gathered = source.index_select(0, token_index);
    tb::Tensor activated;
    if (limit > 0) {
        const auto gv = gate.forward(blas, gathered);
        const auto uv = up.forward(blas, gathered);
        if (tq_detail::dtype(source.scalar_type()) == 0)
            activated = clipped_silu_mul_cuda(gv, uv, limit);
        else
            activated = (tb::silu(tb::clamp_max(gv.to(tb::kFloat32), limit)) *
                tb::clamp(uv.to(tb::kFloat32), -limit, limit))
                .to(source.scalar_type()).contiguous();
    } else {
        activated = TqProjectionGroup({gate, up}).swiglu(blas, gathered);
    }
    output.index_copy_(0, slot_index, down.forward(blas, std::move(activated)));
}

void run_indexed(std::span<const TqWeight> gate,
        std::span<const TqWeight> up, std::span<const TqWeight> down,
        const std::vector<std::vector<std::int64_t>>& slots, std::size_t begin,
        std::int64_t routes, const tb::Tensor& source, tb::Tensor activated,
        tb::Tensor pairs, float limit, bool wide) {
    const auto input = source.size(-1);
    const auto hidden = activated.size(-1);
    const auto output = pairs.size(-1);
    const auto type = tq_detail::dtype(source.scalar_type());
    const auto jobs = make_indexed_jobs(gate, up, down, slots, begin, routes,
        input, hidden, output, wide);
    std::vector<TrellisIndexedProjectionJob> gate_up_jobs, down_jobs;
    std::vector<TrellisIndexedWideProjectionJob> wide_jobs;
    for (int tile = 0; tile < 4; ++tile) {
        gate_up_jobs.insert(gate_up_jobs.end(), jobs.gate_up[tile].begin(), jobs.gate_up[tile].end());
        down_jobs.insert(down_jobs.end(), jobs.down[tile].begin(), jobs.down[tile].end());
        wide_jobs.insert(wide_jobs.end(), jobs.wide[tile].begin(), jobs.wide[tile].end());
    }
    const auto upload_plan = [&](const auto& values) {
        if (values.empty()) return tb::Tensor{};
        const auto bytes = static_cast<std::int64_t>(values.size() * sizeof(values[0]));
        return tb::from_blob(const_cast<void*>(static_cast<const void*>(values.data())),
            {bytes}, tb::TensorOptions().dtype(tb::kUInt8))
            .clone().to(source.device(), false, false).contiguous();
    };
    if (gate_up_jobs.size() > INT_MAX)
        throw std::overflow_error("too many TQ indexed jobs");
    const auto gate_plan = upload_plan(gate_up_jobs);
    const auto down_plan = upload_plan(down_jobs);
    const auto wide_plan = upload_plan(wide_jobs);
    const auto wide_gate_up_plan = upload_plan(jobs.wide_gate_up);
    const auto* gate_ptr = gate_plan.defined()
        ? reinterpret_cast<const TrellisIndexedProjectionJob*>(gate_plan.data_ptr()) : nullptr;
    const auto* down_ptr = down_plan.defined()
        ? reinterpret_cast<const TrellisIndexedProjectionJob*>(down_plan.data_ptr()) : nullptr;
    const auto* wide_ptr = wide_plan.defined()
        ? reinterpret_cast<const TrellisIndexedWideProjectionJob*>(wide_plan.data_ptr()) : nullptr;
    std::size_t first = 0;
    for (int tile = 0; tile < 4; ++tile) {
        const auto count = static_cast<int>(jobs.gate_up[tile].size());
        if (!count) continue;
        tq_detail::status(mfq_tq_indexed_swiglu(gate_ptr + first, source.data_ptr(),
            activated.data_ptr(), count, static_cast<int>(hidden), type, tile + 1,
            limit, mfq_current_cuda_stream()));
        first += jobs.gate_up[tile].size();
    }
    if (!jobs.wide_gate_up.empty())
        tq_detail::status(mfq_tq_indexed_wide_swiglu(
            reinterpret_cast<const TrellisIndexedWidePairJob*>(wide_gate_up_plan.data_ptr()),
            source.data_ptr(), activated.data_ptr(), static_cast<int>(jobs.wide_gate_up.size()),
            static_cast<int>(hidden), type, 8, limit, mfq_current_cuda_stream()));
    first = 0;
    for (int tile = 0; tile < 4; ++tile) {
        const auto count = static_cast<int>(jobs.down[tile].size());
        if (!count) continue;
        tq_detail::status(mfq_tq_indexed(down_ptr + first, activated.data_ptr(),
            pairs.data_ptr(), count, static_cast<int>(output), type, tile + 1,
            mfq_current_cuda_stream()));
        first += jobs.down[tile].size();
    }
    first = 0;
    for (int tile = 0; tile < 4; ++tile) {
        const auto count = static_cast<int>(jobs.wide[tile].size());
        if (!count) continue;
        tq_detail::status(mfq_tq_indexed_wide(wide_ptr + first, activated.data_ptr(),
            pairs.data_ptr(), count, static_cast<int>(output), type, tile + 5,
            mfq_current_cuda_stream()));
        first += jobs.wide[tile].size();
    }
}

} // namespace tq_ops
