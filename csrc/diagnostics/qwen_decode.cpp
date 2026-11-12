#include "diagnostics/qwen_decode.h"

#include "engine/program.h"
#include "engine/backends/cuda/executor/config.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/native/env.h"
#include "engine/backends/cuda/native/profile.h"
#include "engine/generation/decode.h"
#include "engine/models/causal_model.h"
#include "engine/models/source.h"
#include "engine/models/source/model_graph.h"
#include "engine/ops.h"
#include "engine/weights/cache.h"
#include "engine/weights/loader.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tyloi::cuda::diagnostics {
namespace {
namespace ops = ::tyloi::ops;

void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }

/** Keep asynchronous graph and eager inputs and observations alive until drain. */
struct Work {
    Program& program;
    std::vector<std::vector<std::int64_t>> values;
    std::vector<Tensor> tensors;
    ~Work() noexcept { try { program.drain(); } catch (...) {} }
    Tensor keep(Tensor value) { tensors.push_back(value); return value; }
    Tensor ids(std::vector<std::int64_t> value) {
        values.push_back(std::move(value));
        return keep(ops::reshape(keep(ops::tensor(values.back(), program.executor().device())), {1, -1}));
    }
    Tensor host(const Tensor& value) { return keep(ops::contiguous(keep(ops::copy(value, Device(kCPU))))); }
    void clear() { program.drain(); tensors.clear(); values.clear(); }
};

struct Context {
    Program& program;
    models::CausalModel& model;
    models::State& state;
    models::ForwardOptions options;
    explicit Context(Program& p) : program(p), model(p.model()), state(p.state()) {
        auto loader = p.loader();
        auto* source = dynamic_cast<const ModelSource*>(&loader.source);
        require(source && models::graph(*source).backbone == "qwen4_exp", "hidden graph diagnostic requires Qwen4");
        auto mode = p.executor().forward_mode(p.executor().device().index);
        options = {mode.kv_capacity, mode.trace, mode.trace_layer, mode.primary};
    }
    Tensor forward(const Tensor& ids, Tensor* raw = nullptr, std::int64_t confirmed = 0) {
        models::Input input; input.ids = ids; input.advance = true; input.raw_hidden = raw; input.confirmed = confirmed;
        return model.forward(std::move(input), state, options);
    }
    std::optional<models::Decoded> decode(const Tensor& ids, Tensor* raw = nullptr,
            std::int64_t confirmed = 0, const Tensor& host_ids = {}, bool project = false) {
        models::Input input; input.ids = ids; input.advance = true; input.raw_hidden = raw;
        input.confirmed = confirmed; input.host_ids = host_ids;
        models::DecodeOptions decode;
        decode.projection = project ? models::DecodeOptions::Projection::hidden : models::DecodeOptions::Projection::hidden_only;
        decode.routed = program.weights().decode() == weights::Cache::Decode::routed;
        return engine::decode(model, state, program.executor(), program.weights(), options, std::move(input), decode);
    }
};

/** Restore one environment setting after graph consumers have been drained. */
struct Environment {
    const char* name;
    std::optional<std::string> previous;
    explicit Environment(const char* key) : name(key) {
        if (const auto* value=env_lookup(key)) previous=std::string(value);
    }
    void set(const char* value) { if (value) env_set(name, value); else env_clear(name); }
    ~Environment() noexcept { try { if (previous) env_set(name, previous->c_str()); else env_clear(name); } catch (...) {} }
};

/** Restore profiler configuration after a diagnostic. */
struct ProfileScope {
    CudaProfiler& profiler;
    bool enabled, graph_events;
    explicit ProfileScope(CudaProfiler& p) : profiler(p), enabled(p.enabled), graph_events(p.graph_events) {}
    ~ProfileScope() { profiler.enabled = enabled; profiler.graph_events = graph_events; profiler.reset(); }
};

/** Retire variant-specific captures before restoring environment settings. */
struct GraphRetire {
    Program& program;
    ~GraphRetire() noexcept {
        try { program.drain(); program.state().release_graphs(); }
        catch (...) {}
    }
};

bool bits(const Tensor& a, const Tensor& b) {
    if (a.scalar_type() != b.scalar_type() || a.dim() != b.dim()) return false;
    for (std::int64_t i=0; i<a.dim(); ++i) if (a.size(i)!=b.size(i)) return false;
    return std::memcmp(a.dense().data_ptr(), b.dense().data_ptr(), std::size_t(a.numel())*a.element_size()) == 0;
}

void phases(const models::State& state) {
    for (const auto& p : state.decode_profile())
        std::cout << "mtp_graph_phase tokens=" << p.tokens << " confirmed=" << p.confirmed
            << " layer=" << p.layer << " from=" << p.from << " to=" << p.to
            << " samples=" << p.samples << " total_ns=" << p.total_ns << '\n';
}
} // namespace

int run_qwen4_decode_hidden_check(Program& program) {
    auto scope = program.executor().enter();
    Work w{program}; Context c{program};
    const auto info = c.model.info();
    require(info.max_context>=24 && info.vocab_size>=8, "Qwen4 hidden graph check requires context>=24 and vocab>=8");
    const auto prompt = w.ids({1,2});
    const auto ids_for = [&](int step) { return w.ids({3+step%5}); };
    std::vector<Tensor> reference_hidden, reference_raw, retained_raw;
    c.state.reset(1); (void)w.keep(c.forward(prompt));
    for (int step=0; step<20; ++step) {
        Tensor raw; auto hidden=w.keep(c.forward(ids_for(step), &raw)); w.keep(raw);
        reference_hidden.push_back(w.host(ops::cast(hidden, kFloat32)));
        reference_raw.push_back(w.host(ops::cast(raw, kFloat32)));
        w.clear();
    }
    double maximum_error=0;
    const auto compare = [&](const Tensor& value, const Tensor& reference) {
        require(value.dim()==reference.dim(), "hidden graph shape mismatch");
        for (std::int64_t i=0; i<value.dim(); ++i) require(value.size(i)==reference.size(i), "hidden graph shape mismatch");
        const auto a=ops::floats(value), b=ops::floats(reference);
        for (std::size_t i=0; i<a.size(); ++i) {
            const double error=std::abs(double(a[i])-b[i]); maximum_error=std::max(maximum_error,error);
            require(std::isfinite(a[i]) && error<=.015+.015*std::abs(b[i]), "hidden graph differs from eager execution");
        }
    };
    program.drain(); c.state.reset(1); (void)w.keep(c.forward(prompt));
    int graph_steps=0;
    for (int step=0; step<20; ++step) {
        Tensor raw, hidden;
        if (step==7) hidden=c.forward(ids_for(step), &raw);
        else {
            auto result=c.decode(ids_for(step), &raw);
            require(result.has_value(), "hidden graph was not exercised");
            hidden=result->hidden; ++graph_steps;
        }
        w.keep(hidden); w.keep(raw); compare(hidden,reference_hidden[step]); compare(raw,reference_raw[step]);
        retained_raw.push_back(raw);
        require(c.state.position()==step+3, "hidden graph advanced KV more than once");
        w.clear();
    }
    for (int step=0; step<20; ++step) compare(retained_raw[step],reference_raw[step]);
    const auto metrics=c.state.metrics(); bool registry=false;
    if (metrics) for (const auto& [key,value] : *metrics) if (key=="decode_graph_count") registry=value>0;
    require(registry, "hidden graph did not survive KV growth");
    std::cout << "qwen4_decode_hidden_check PASS graph_steps=" << graph_steps
        << " retained_steps=" << retained_raw.size() << " max_abs_error=" << maximum_error << '\n';
    retained_raw.clear(); reference_hidden.clear(); reference_raw.clear();
    int transactions=0;
    for (int tokens=2; tokens<=6; ++tokens) {
        std::vector<std::int64_t> values; for (int i=0;i<tokens;++i) values.push_back(i==1 && info.vocab_size>15 ? 15 : 3+i%5);
        const auto input=w.ids(values);
        std::vector<Tensor> expected, expected_raw, expected_next;
        const std::vector<int> actions{0,-1,tokens>2?1:0,tokens>3?2:-1};
        program.drain(); c.state.reset(1); (void)w.keep(c.forward(prompt));
        for (int accepted : actions) {
            Tensor raw; auto hidden=w.keep(c.forward(input,&raw,1)); w.keep(raw);
            expected.push_back(w.host(ops::cast(hidden,kFloat32))); expected_raw.push_back(w.host(ops::cast(raw,kFloat32)));
            require(!c.decode(ids_for(0)), "hidden graph accepted an active speculative transaction");
            if (accepted<0) c.state.commit_speculative();
            else if (c.state.retains_partial_prefix(accepted)) c.state.rollback_speculative(accepted);
            else {
                c.state.rollback_speculative(0);
                if (accepted) (void)w.keep(c.forward(w.keep(ops::slice(input,1,1,accepted+1))));
            }
            expected_next.push_back(w.host(ops::cast(w.keep(c.forward(ids_for(0))),kFloat32)));
            w.clear();
        }
        program.drain(); c.state.reset(1); (void)w.keep(c.forward(prompt));
        for (std::size_t i=0;i<actions.size();++i) {
            Tensor raw; const bool project=i%2==0;
            auto host_input=w.host(input);
            auto decoded=c.decode(input,&raw,1,host_input,project);
            require(decoded && c.state.speculating() && c.state.confirmed_prefix()==1,
                "MTP graph did not open its verification transaction");
            w.keep(raw); w.keep(decoded->hidden); if (decoded->logits) w.keep(decoded->logits);
            compare(decoded->hidden,expected[i]); compare(raw,expected_raw[i]);
            if (project) {
                auto logits=w.keep(c.model.logits(decoded->hidden)); compare(decoded->logits,logits);
                auto values_cpu=ops::floats(decoded->logits);
                const auto width=decoded->logits.size(decoded->logits.dim()-1);
                require(decoded->tokens.size()==static_cast<std::size_t>(tokens), "graph greedy row count changed");
                for (int row=0;row<tokens;++row) {
                    const auto begin=values_cpu.begin()+row*width;
                    const auto argmax=std::distance(begin,std::max_element(begin,begin+width));
                    require(decoded->tokens[row]==argmax, "graph returned stale greedy tokens");
                }
            }
            if (actions[i]<0) c.state.commit_speculative();
            else { require(c.state.retains_partial_prefix(actions[i]), "MTP graph omitted prefix checkpoints"); c.state.rollback_speculative(actions[i]); }
            auto next=c.decode(ids_for(0)); require(next.has_value(), "MTP graph did not resume after commit/rollback");
            w.keep(next->hidden); compare(next->hidden,expected_next[i]); ++transactions;
            w.clear();
        }
    }
    program.drain();
    std::cout << "qwen4_mtp_graph_check PASS transactions=" << transactions << " max_abs_error=" << maximum_error << '\n';
    return 0;
}

namespace {
void sync_stream() {
    const auto status=cudaStreamSynchronize(tyloi::cuda::current_stream().stream());
    if (status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

int variant_profile(Program& program, const std::vector<std::int64_t>& prompt_ids, bool short_path) {
    Environment compact("TYLOINFER_MFE_TOKEN_COMPACT_PLAN"), aligned("TYLOINFER_MFE_TOKEN_ALIGNED_GU"),
        gr("TYLOINFER_GR_BATCH_SHARED_DOWN"), capacity("TYLOINFER_MFE_TOKEN_GROUP_CAPACITY"),
        gdn("TYLOINFER_GDN_SHORT_FUSION"), inplace("TYLOINFER_GDN_SHORT_INPLACE");
    GraphRetire graph_retire{program};
    ProfileScope profile(program.executor().profiler());
    Work w{program}; Context c{program};
    struct Variant { const char* name; const char* compact; const char* aligned; const char* gr; const char* capacity; const char* gdn=nullptr; };
    const Variant variants[]{ {"baseline","0","0","0","2"}, {"compact","1","0","0","2"},
        {"aligned","1","1","0","2"}, {"shared_gr","1","1","1","2"}, {"capacity4","1","1","1","4"},
        {"compact_shared_gr","1","0","1","2"}, {"baseline_repeat","0","0","0","2"} };
    const Variant short_variants[]{ {"short_legacy","1",nullptr,nullptr,"2","0"},
        {"short_fused","1",nullptr,nullptr,"2","1"}, {"short_fused_repeat","1",nullptr,nullptr,"2","1"},
        {"short_legacy_repeat","1",nullptr,nullptr,"2","0"} };
    const auto* active=short_path?short_variants:variants;
    const auto count=short_path?std::size(short_variants):std::size(variants);
    std::array<Tensor,5> expected_raw, expected_logits;
    for (std::size_t arm=0;arm<count;++arm) for (int tokens=2;tokens<=6;++tokens) {
        w.clear(); c.state.release_graphs();
        const auto& variant=active[arm];
        compact.set(variant.compact); aligned.set(variant.aligned); gr.set(variant.gr); capacity.set(variant.capacity);
        gdn.set(variant.gdn); inplace.set(variant.gdn);
        c.state.reset(1); program.executor().profiler().enabled=false;
        auto prompt=w.ids(prompt_ids.empty()?std::vector<std::int64_t>{1,2,3,4,5,6,7}:prompt_ids);
        (void)w.keep(c.forward(prompt));
        std::vector<std::int64_t> values; for(int i=0;i<tokens;++i) values.push_back(3+i%5);
        const auto input=w.ids(values);
        double elapsed=0; Tensor raw, logits;
        for (int repeat=0;repeat<10;++repeat) {
            sync_stream(); const auto start=std::chrono::steady_clock::now();
            auto result=c.decode(input,&raw,1,{},true);
            require(result.has_value(), "MTP variant profile did not use the graph");
            logits=result->logits; sync_stream();
            const auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            if(repeat>=3) elapsed+=ms;
            c.state.rollback_speculative(0);
        }
        auto raw_host=w.host(raw), logits_host=w.host(logits);
        const auto index=tokens-2;
        if(arm==0) { expected_raw[index]=ops::clone(raw_host); expected_logits[index]=ops::clone(logits_host); }
        else require(bits(raw_host,expected_raw[index]) && bits(logits_host,expected_logits[index]), "MTP variant changed raw hidden or target logits");
        std::cout << "mtp_verify_variant name=" << variant.name << " tokens=" << tokens
            << " repeats=7 wall_avg_ms=" << elapsed/7 << " exact=1\n";
    }
    return 0;
}
} // namespace

int run_qwen4_mtp_profile(Program& program, const std::vector<std::int64_t>& prompt_ids) {
    auto scope=program.executor().enter();
    const auto* variants=env_lookup("TYLOINFER_MTP_PROFILE_VARIANTS");
    if (variants && (variants[0]=='1' || variants[0]=='2'))
        return variant_profile(program,prompt_ids,variants[0]=='2');
    ProfileScope profile(program.executor().profiler());
    Work w{program}; Context c{program};
    auto& profiler=program.executor().profiler();
    std::cout << std::unitbuf;
    for (bool graph : {false,true}) for (int tokens=2;tokens<=6;++tokens) {
        w.clear(); c.state.reset(1); profiler.enabled=false;
        const auto prompt=w.ids(prompt_ids.empty()?std::vector<std::int64_t>{1,2,3,4,5,6,7}:prompt_ids);
        (void)w.keep(c.forward(prompt));
        std::vector<std::int64_t> values; for(int i=0;i<tokens;++i) values.push_back(3+i%5);
        const auto input=w.ids(values); double elapsed=0;
        for (int repeat=0;repeat<8;++repeat) {
            profiler.enabled=!graph && repeat>=3; profiler.reset();
            sync_stream(); const auto start=std::chrono::steady_clock::now();
            Tensor raw;
            if (graph) {
                auto result=c.decode(input,&raw,1);
                require(result.has_value(), "MTP profile did not use the graph");
            } else (void)c.forward(input,&raw,1);
            sync_stream();
            const auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            if(repeat>=3) elapsed+=ms;
            if(profiler.enabled) profiler.report("mtp_verify_m"+std::to_string(tokens));
            c.state.rollback_speculative(0);
        }
        std::cout << "mtp_verify_profile tokens=" << tokens << " graph=" << graph
            << " repeats=5 wall_avg_ms=" << elapsed/5 << '\n';
        if(graph) phases(c.state);
    }
    return 0;
}
} // namespace tyloi::cuda::diagnostics
