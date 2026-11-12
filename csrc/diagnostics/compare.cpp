#include "diagnostics/compare.h"
#include "engine/program.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/executor/config.h"
#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/models/causal_model.h"
#include "engine/models/load.h"
#include "engine/models/source.h"
#include "engine/ops.h"
#include "engine/weights/loader.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tyloi::cuda::diagnostics {
namespace {
namespace ops = ::tyloi::ops;
using Clock = std::chrono::steady_clock;

/** Own all submitted inputs and observations until the execution owner drains. */
struct Work {
    Program& program;
    std::vector<Tensor> values;
    std::vector<models::Input::Trace> traces;
    std::vector<ops::Trace> stages;
    ~Work() noexcept {
        try { program.drain(); }
        catch (const std::exception& e) { std::cerr << "compare_cleanup error=" << e.what() << '\n'; }
        catch (...) { std::cerr << "compare_cleanup error=unknown\n"; }
    }
    Tensor keep(Tensor value) { values.push_back(value); return value; }
};

models::ForwardOptions options(Program& program) {
    const int primary = program.config().layer_placement.primary_device();
    const auto mode = program.executor().forward_mode(primary);
    return {mode.kv_capacity, mode.trace, mode.trace_layer, primary};
}

Tensor hidden(Work& work, Tensor ids, Tensor lengths = {}, std::int64_t planned = 0,
        models::Input::Trace* trace = nullptr, ops::Trace* stages = nullptr) {
    models::Input input;
    input.ids = work.keep(std::move(ids)); input.lengths = work.keep(std::move(lengths));
    input.advance = true; input.kv_length = planned; input.trace = trace;
    auto mode = options(work.program);
    if (stages) { mode.trace = stages; mode.trace_layer = 0; }
    return work.keep(work.program.model().forward(std::move(input), work.program.state(), mode));
}

Tensor last(Work& work, const Tensor& value, ScalarType dtype) {
    auto row = work.keep(value.dense().select(1, value.size(1)-1));
    return work.keep(row.dense().to(dtype).contiguous());
}

Tensor project(Work& work, const Tensor& value, ScalarType dtype, bool capped) {
    auto row = last(work, value, dtype);
    return work.keep(capped ? work.program.model().logits(row, true) : work.program.model().project(row));
}

models::Backbone family(Program& program) {
    auto loader = program.loader();
    const auto* source = dynamic_cast<const ModelSource*>(&loader.source);
    if (!source) throw std::runtime_error("diagnostic model source is unavailable");
    return models::model_plan(models::graph(*source)).backbone;
}

Tensor family_last_logits(Work& work, const Tensor& value, models::Backbone backbone) {
    if (backbone == models::Backbone::qwen4_exp || backbone == models::Backbone::glm5_next) {
        auto row = work.keep(value.dense().select(1, value.size(1)-1));
        return work.keep(work.program.model().logits(row, true));
    }
    return project(work, value, backbone == models::Backbone::minicpmo45 ? kBFloat16 : kFloat16, true);
}

Tensor next(Work& work, const Tensor& value, models::Backbone backbone) {
    const bool direct = backbone == models::Backbone::qwen4_exp || backbone == models::Backbone::glm5_next;
    const bool mini = backbone == models::Backbone::minicpmo45;
    auto row = work.keep(value.dense().select(1, value.size(1)-1));
    if (!direct) row = work.keep(row.dense().to(mini ? kBFloat16 : kFloat16).contiguous());
    auto logits = work.keep(work.program.model().logits(row, false));
    if (direct || mini) return work.keep(ops::argmax(logits, -1));
    return work.keep(ops::sample_greedy(work.keep(ops::reshape(work.keep(ops::contiguous(logits)), {logits.size(0), -1}))));
}

Tensor length(Work& work) {
    return work.keep(ops::full(std::array<std::int64_t, 1>{1}, work.program.state().position()+1, work.program.executor().device()));
}

void statistics(const Tensor& reference, const Tensor& candidate, const char* prefix, bool relative) {
    auto ref = reference.dense(), test = candidate.dense();
    auto ref_logp = tyloi::cuda::log_softmax(ref, -1);
    auto test_logp = tyloi::cuda::log_softmax(test, -1);
    auto delta = (ref-test).abs();
    std::cout << prefix << "_kl=" << (ref_logp.exp()*(ref_logp-test_logp)).sum(-1).item<float>() << '\n';
    if (relative) std::cout << prefix << "_rel=" << ((test-ref).norm()/ref.norm()).item<float>() << '\n';
    if (relative) {
        std::cout << prefix << "_mean_abs=" << delta.mean().item<float>() << '\n'
                  << prefix << "_max_abs=" << delta.max().item<float>() << '\n';
    } else {
        std::cout << prefix << "_max_logit_abs=" << delta.max().item<float>() << '\n'
                  << prefix << "_mean_logit_abs=" << delta.mean().item<float>() << '\n';
    }
    std::cout << prefix << "_same_top=" << (ref.argmax(-1).eq(test.argmax(-1)).item<bool>() ? 1 : 0) << '\n';
}
} // namespace

int compare_decode_splitk(Program& program, Tensor ids) {
    auto& executor = program.executor(); auto scope = executor.enter(); Work work{program}; ids = work.keep(std::move(ids));
    auto kernels = executor.attention_kernels(); kernels.split_k = false;
    Tensor ref, candidate;
    for (int pass = 0; pass < 2; ++pass) {
        kernels.split_k = pass != 0; AttentionScope policy(executor, kernels);
        program.drain(); program.state().reset(1);
        (void)hidden(work, ids);
        auto decode = work.keep(ids.dense().select(1, ids.size(1)-1).reshape({1,1}));
        auto output = work.keep(project(work, hidden(work, decode, length(work)), kFloat16, false).dense().to(kFloat32));
        program.drain();
        if (pass == 0) ref = output; else candidate = output;
    }
    statistics(ref, candidate, "decode_splitk_compare", true);
    return 0;
}

int compare_mma_decode(Program& program, Tensor ids, int steps, std::int64_t planned_length) {
    if (steps < 1) throw std::invalid_argument("--compare-mma-decode-steps must be positive");
    auto& executor = program.executor(); auto scope = executor.enter(); Work work{program}; ids = work.keep(std::move(ids));
    auto kernels = executor.attention_kernels(); auto backbone = family(program);
    std::vector<Tensor> reference; std::vector<std::int64_t> teachers; std::int64_t first = 0;
    kernels.mma_decode = false;
    {
        AttentionScope policy(executor, kernels); program.drain(); program.state().reset(1);
        auto input = work.keep(ops::reshape(next(work, hidden(work, ids), backbone), {1,1}));
        first = ops::item(input);
        for (int step = 0; step < steps; ++step) {
            auto logits = project(work, hidden(work, input, length(work)), kFloat16, false);
            reference.push_back(work.keep(ops::clone(work.keep(logits.dense().to(kFloat32)))));
            auto token = ops::item(work.keep(ops::argmax(reference.back(), -1))); teachers.push_back(token);
            input = work.keep(ops::reshape(work.keep(ops::full(std::array<std::int64_t, 1>{1}, token, executor.device())), {1,1}));
            work.values.clear();
        }
    }
    const auto planned = planned_length > 0 ? planned_length : ids.size(1) + steps;
    if (planned > program.model().info().max_context) throw std::runtime_error("decode comparison planned length exceeds context capacity");
    kernels.mma_decode = true;
    AttentionScope policy(executor, kernels); program.drain(); program.state().reset(1);
    (void)next(work, hidden(work, ids), backbone);
    auto input = work.keep(ops::reshape(work.keep(ops::full(std::array<std::int64_t, 1>{1}, first, executor.device())), {1,1}));
    double kl_sum=0., kl_max=0., abs_sum=0., delta_sq=0., reference_sq=0., max_abs=0.;
    int same=0, first_difference=-1; std::int64_t values=0;
    for (int step=0; step<steps; ++step) {
        auto test = work.keep(project(work, hidden(work, input, length(work), planned), kFloat16, false).dense().to(kFloat32));
        auto ref = reference[step].dense(); auto actual = test.dense();
        auto ref_logp = tyloi::cuda::log_softmax(ref,-1), test_logp = tyloi::cuda::log_softmax(actual,-1);
        const double kl = (ref_logp.exp()*(ref_logp-test_logp)).sum(-1).item<double>();
        auto delta = actual-ref; kl_sum += kl; kl_max=std::max(kl_max,kl);
        abs_sum+=delta.abs().sum().item<double>(); delta_sq+=delta.square().sum().item<double>();
        reference_sq+=ref.square().sum().item<double>(); max_abs=std::max(max_abs,delta.abs().max().item<double>());
        values+=delta.numel(); const bool equal=actual.argmax(-1).eq(ref.argmax(-1)).item<bool>();
        same += equal ? 1 : 0; if (!equal && first_difference<0) first_difference=step;
        input = work.keep(ops::reshape(work.keep(ops::full(std::array<std::int64_t, 1>{1}, teachers[step], executor.device())), {1,1}));
        work.values.clear();
    }
    program.drain();
    std::cout << std::setprecision(10) << "mma_decode_compare_steps=" << steps << '\n'
        << "mma_decode_compare_mean_kl=" << kl_sum/steps << '\n' << "mma_decode_compare_max_kl=" << kl_max << '\n'
        << "mma_decode_compare_rel=" << std::sqrt(delta_sq/reference_sq) << '\n'
        << "mma_decode_compare_mean_abs=" << abs_sum/values << '\n' << "mma_decode_compare_max_abs=" << max_abs << '\n'
        << "mma_decode_compare_same_top=" << same << '\n' << "mma_decode_compare_first_top_difference=" << first_difference << '\n';
    return 0;
}

int repeat_prefill(Program& program, Tensor ids, int count, bool trace) {
    if (count <= 0) return 0;
    auto& executor = program.executor(); auto scope = executor.enter(); Work work{program}; ids=work.keep(std::move(ids));
    const auto backbone = family(program); work.traces.resize(count); work.stages.resize(count);
    for (int i=0; i<count; ++i) {
        program.drain(); program.state().reset(1); const auto start=Clock::now();
        auto value = hidden(work, ids, {}, 0, trace ? &work.traces[i] : nullptr, trace ? &work.stages[i] : nullptr);
        auto logits = trace ? project(work, value, kFloat16, false) : family_last_logits(work, value, backbone);
        executor.synchronize(); const auto end=Clock::now();
        std::cout << "prefill_repeat=" << i+1 << " sec=" << std::chrono::duration<double>(end-start).count()
                  << " top=" << ops::item(work.keep(ops::argmax(logits,-1))) << '\n';
        if (trace && i>0) {
            if (work.stages[i].size()!=work.stages[0].size() || work.traces[i].size()!=work.traces[0].size())
                throw std::runtime_error("prefill repeat trace count changed");
            for (std::size_t s=0; s<work.stages[i].size(); ++s) {
                auto diff=(work.stages[i][s].second.dense()-work.stages[0][s].second.dense()).abs();
                const float maximum=diff.max().item<float>();
                if (maximum!=0.f) { std::cout << "prefill_repeat_first_gemma_stage=" << work.stages[i][s].first
                    << " max_abs=" << maximum << " mean_abs=" << diff.mean().item<float>() << '\n'; break; }
            }
            for (std::size_t s=0; s<work.traces[i].size(); ++s) {
                auto diff=(work.traces[i][s].dense()-work.traces[0][s].dense()).abs();
                const float maximum=diff.max().item<float>();
                if (maximum!=0.f) { std::cout << "prefill_repeat_first_difference=" << s << " layer=" << static_cast<std::int64_t>(s)-1
                    << " max_abs=" << maximum << " mean_abs=" << diff.mean().item<float>() << '\n'; break; }
            }
        }
        work.values.clear();
        if (i > 0) { work.traces[i].clear(); work.stages[i].clear(); }
    }
    return 0;
}

int compare_mma_attention(Program& program, Tensor ids) {
    auto& executor=program.executor(); auto scope=executor.enter(); Work work{program}; ids=work.keep(std::move(ids));
    auto kernels=executor.attention_kernels(); auto backbone=family(program); Tensor ref, candidate;
    for (int pass=0; pass<2; ++pass) {
        kernels.mma_prefill=pass!=0; kernels.bf16_flash128=pass!=0; AttentionScope policy(executor,kernels);
        if (pass) { program.drain(); program.state().reset(1); }
        auto output=work.keep(family_last_logits(work,hidden(work,ids),backbone).dense().to(kFloat32));
        program.drain(); if (pass) candidate=output; else ref=output;
    }
    statistics(ref,candidate,"attention_compare",false);
    return 0;
}
} // namespace tyloi::cuda::diagnostics
