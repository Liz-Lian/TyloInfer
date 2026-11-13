
#include "engine/models/qwen35/model.h"
#include "engine/models/qwen35/state.h"
#include "engine/models/qwen35/forward.h"
#include "engine/models/batch.h"
#include "engine/state/paged_kv.h"
#include "engine/weights/moe/experts.h"
#include "engine/backends/cuda/executor/decode_window.h"
#include "engine/backends/cuda/executor/decode_graph.h"
#include "engine/ops.h"

#include <algorithm>
#include <array>
#include <exception>
#include <limits>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace tyloi::cuda::qwen35 {
namespace {
using Growth = tyloi::models::BatchGrowth;
void add(Growth& out, Device device, std::size_t bytes) {
    if (!device.is_cuda() || !bytes) return;
    auto found=std::find_if(out.begin(),out.end(),[&](const auto& value){return value.first==device;});
    if (found==out.end()) out.emplace_back(device,bytes);
    else { if (found->second>std::numeric_limits<std::size_t>::max()-bytes) throw std::overflow_error("Qwen batch growth overflow"); found->second+=bytes; }
}
void merge(Growth& out,const Growth& more) { for (const auto& [device,bytes]:more) add(out,device,bytes); }
std::size_t mul(std::size_t a,std::size_t b) {
    if (b && a>std::numeric_limits<std::size_t>::max()/b) throw std::overflow_error("Qwen batch growth overflow");
    return a*b;
}
bool cached(const ::tyloi::ops::FfnWeights& f) {
    return f.is_moe && (f.moe_split_gate_up ?
        (tyloi::weights::cached(f.moe_gate) || tyloi::weights::cached(f.moe_up) || tyloi::weights::cached(f.moe_down)) :
        (tyloi::weights::cached(f.moe_gate_up) || tyloi::weights::cached(f.moe_down)));
}
class Batch final : public tyloi::models::Batch {
    struct Request final : tyloi::models::BatchState {
        Batch* owner;
        State state;
        PagedKv::Sequence pages;
        std::int64_t prompt;
        int slot=-1;
        std::int64_t active_position=0;
        bool initialized=false;
        Request(Batch& batch,std::int64_t tokens) : owner(&batch),state(batch.model_,batch.model_.transposed),prompt(tokens) {}
        ~Request() override {
            if (!owner) return;
            try { owner->release(*this); } catch (...) {}
            std::erase(owner->requests_,this);
        }
        std::int64_t position() const override { return slot>=0 ? active_position : state.position(); }
        std::size_t resident_bytes() const override {
            std::map<const void*,std::size_t> seen;
            for (const auto& t:state.storage()) if (t.defined() && t.device().is_cuda()) {
                const auto& d=t.dense(); const auto& storage=d.storage();
                if (storage) { const void* id=storage->owner ? storage->owner.get() : storage->base; if (id) seen[id]=std::max(seen[id],storage->bytes); }
            }
            std::size_t result=0; for (const auto& [_,bytes]:seen) { if (result>std::numeric_limits<std::size_t>::max()-bytes) throw std::overflow_error("Qwen batch resident overflow"); result+=bytes; } return result;
        }
        Growth initial_growth_bytes() const override {
            Growth result;
            if (owner->paged_) { const std::array requests{std::pair<const PagedKv::Sequence*,int64_t>{&pages,prompt}}; merge(result,owner->paged_->growth(requests)); }
            return result;
        }
        void initialize() override {
            if (initialized) throw std::logic_error("Qwen batch request already initialized");
            state.reset(1);
            if (owner->paged_) { owner->paged_->ensure_tokens(pages,prompt); owner->bind(state,{&pages}); }
            initialized=true;
        }
        Growth prefill_growth_bytes(std::int64_t tokens) const override {
            if (!initialized) throw std::logic_error("Qwen batch request is not initialized");
            if (slot>=0) return {};
            tyloi::models::ForwardOptions options; options.primary=owner->model_.primary_device;
            return state.growth(1,tokens,options,0);
        }
    };
    struct Graph {
        std::unique_ptr<DecodeWindow> window;
        std::int64_t length=0;
        std::vector<const void*> addresses;
        Tensor ids, positions, lengths, output;
        std::vector<Tensor> checkpoint;
        bool valid=false;
        explicit Graph(std::int64_t n,std::vector<const void*> a) : window(DecodeWindow::batch()),length(n),addresses(std::move(a)) {}
    };
    struct GraphPlan final : Plan {
        const Batch* owner;
        std::int64_t length;
        std::vector<const void*> addresses;
        std::shared_ptr<Graph> graph;
        GraphPlan(const Batch& value,std::int64_t n,std::vector<const void*> a,std::shared_ptr<Graph> g)
            : owner(&value),length(n),addresses(std::move(a)),graph(std::move(g)) { rebuild=!graph; }
    };
public:
    Batch(Model& model,const tyloi::models::BatchConfig& config) : model_(model),config_(config),state_(model,model.transposed) {
        if (!config.slots || config.slots>std::size_t(std::numeric_limits<int32_t>::max())) throw std::invalid_argument("continuous batching max sequences must be positive int32");
        if (model.layers.empty()) throw std::runtime_error("continuous batching requires at least one model block");
        if (config.cpu_offload) throw std::runtime_error("continuous batching requires GPU-resident model blocks");
        for (const auto& layer:model.layers) {
            if (layer.cpu) throw std::runtime_error("continuous batching cannot use CPU-offloaded blocks");
            std::visit([&](const auto& value) {
                using T=std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T,Full>) {
                    if (value.attention.sliding) throw std::runtime_error("continuous batching requires non-sliding Qwen attention");
                    layouts_.push_back({layer.device,value.attention.kv_heads>0?value.attention.kv_heads:model.metadata.num_key_value_heads,
                        value.attention.attention_head_dim>0?value.attention.attention_head_dim:model.metadata.head_dim,value.attention.kv_quantization});
                }
                moe_ = moe_ || value.ffn.is_moe;
                const bool hit=cached(value.ffn); cached_=cached_ || hit;
                if (hit && value.ffn.moe_top_k>config.cached_slots) throw std::runtime_error("continuous batching requires one cached slot per routed expert");
            },layer.weights);
        }
        slots_.resize(config.slots,nullptr);
        if (config.paged) pages_.resize(config.slots);
    }
    ~Batch() override {
        try { clear(); } catch (...) {}
        for (auto* request:requests_) { request->slot=-1; request->owner=nullptr; }
    }
    Growth initial_growth_bytes() const override {
        return config_.paged ? PagedKv::initial_growth(layouts_,model_.metadata.max_position_embeddings,int32_t(config_.slots),config_.preallocate) : Growth{};
    }
    void initialize() override {
        if (initialized_) throw std::logic_error("Qwen batch already initialized");
        state_.reset(std::int64_t(slots_.size()));
        if (config_.paged) {
            paged_=std::make_unique<PagedKv>(layouts_,model_.metadata.max_position_embeddings,int32_t(slots_.size()),config_.preallocate);
            paged_->initialize();
        }
        initialized_=true;
    }
    std::unique_ptr<tyloi::models::BatchState> make_state(std::int64_t prompt) override {
        if (!initialized_ || prompt<=0 || prompt>model_.metadata.max_position_embeddings) throw std::invalid_argument("invalid Qwen batch prompt");
        auto request=std::make_unique<Request>(*this,prompt);
        requests_.push_back(request.get());
        return request;
    }
    std::size_t prefill_row_bytes() const override { return mul(mul(std::size_t(model_.metadata.hidden_size),std::size_t(model_.metadata.hc_mult)),sizeof(float)*8); }
    Growth activate_growth(const tyloi::models::BatchState& value) const override {
        const auto& request=checked(value);
        if (!request.initialized || request.slot>=0) throw std::logic_error("invalid Qwen batch activation");
        Growth result;
        for (std::size_t i=0;i<state_.layers.size();++i) {
            if (const auto* linear=std::get_if<LinearState>(&request.state.layers[i])) {
                const auto& target=std::get<LinearState>(state_.layers[i]);
                if (state_.batch!=std::int64_t(slots_.size()) || !target.attention.defined()) {
                    const auto source=linear->attention.view();
                    if (source.convolution.defined()) add(result,source.convolution.device(),mul(source.convolution.nbytes(),slots_.size()));
                    if (source.recurrent.defined()) add(result,source.recurrent.device(),mul(source.recurrent.nbytes(),slots_.size()));
                }
            } else if (!paged_) {
                const auto& source=std::get<tyloi::state::KvView>(request.state.layers[i]);
                const auto& target=std::get<tyloi::state::KvView>(state_.layers[i]);
                if (state_.batch!=std::int64_t(slots_.size()) || !target.defined()) {
                    if (source.physical_key.defined()) add(result,source.physical_key.device(),mul(source.physical_key.nbytes(),slots_.size()));
                    if (source.physical_value.defined()) add(result,source.physical_value.device(),mul(source.physical_value.nbytes(),slots_.size()));
                }
            }
        }
        if (paged_) {
            std::vector<std::pair<const PagedKv::Sequence*,int64_t>> needs;
            for (const auto& page:pages_) needs.emplace_back(&page,1);
            merge(result,paged_->growth(needs));
        }
        return result;
    }
    void activate(tyloi::models::BatchState& value) override {
        auto& request=checked(value);
        if (!request.initialized || request.slot>=0) throw std::logic_error("invalid Qwen batch activation");
        auto free=std::find(slots_.begin(),slots_.end(),nullptr);
        if (free==slots_.end()) throw std::runtime_error("continuous batching has no free stable slot");
        const auto slot=std::size_t(free-slots_.begin());
        if (state_.batch!=std::int64_t(slots_.size())) {
            if (active()) throw std::logic_error("continuous batching aggregate state has active slots at the wrong batch size");
            state_.reset(std::int64_t(slots_.size()));
        }
        if (paged_) bind_slots();
        copy_slot(request,slot);
        bool swapped=false;
        try {
            if (paged_) { paged_->release(pages_[slot]); std::swap(pages_[slot],request.pages); swapped=true; }
            request.active_position=request.state.position();
            slots_[slot]=&request; request.slot=int(slot);
            if (paged_) bind_slots();
            request.state=State(model_,model_.transposed);
        } catch (...) {
            if (request.slot==int(slot)) { slots_[slot]=nullptr; request.slot=-1; }
            if (swapped) { detach(state_); detach(request.state); std::swap(pages_[slot],request.pages); }
            throw;
        }
    }
    void release(tyloi::models::BatchState& value) override {
        auto& request=checked(value);
        if (request.slot>=0) {
            const auto slot=std::size_t(request.slot);
            if (slot>=slots_.size() || slots_[slot]!=&request) throw std::logic_error("continuous batching stable slot ownership changed");
            if (paged_) paged_->release(pages_[slot]);
            slots_[slot]=nullptr; request.slot=-1; ++releases_;
        }
        if (paged_) paged_->release(request.pages);
    }
    void resume(std::int64_t position) override { state_.cache_pos=position; if (paged_ && active()) bind_slots(); }
    void retire(std::int64_t position) override {
        state_.cache_pos=position;
        if (!active()) { clear(); state_.reset(1); if (paged_) { detach(state_); for (auto& p:pages_) paged_->release(p); } }
        else if (paged_) bind_slots();
    }
    void recover() override { clear(); paused_.clear(); for (auto* r:slots_) if (r) { r->slot=-1; } std::fill(slots_.begin(),slots_.end(),nullptr); if (paged_) for (auto& p:pages_) paged_->release(p); state_.reset(1); }
    Growth decode_growth(std::span<const tyloi::models::BatchRow> rows) const override {
        Growth result;
        add(result,Device(kCUDA,model_.primary_device),mul(slots_.size(),sizeof(int64_t)*3));
        if (paged_) { std::vector<std::pair<const PagedKv::Sequence*,int64_t>> needs;
            for (std::size_t i=0;i<slots_.size();++i) {
                int64_t count=1;
                for (const auto& row:rows) { const auto& r=checked(*row.state); if (r.slot==int(i)) count=std::max<int64_t>(count,row.position+1); }
                needs.emplace_back(&pages_[i],count);
            }
            merge(result,paged_->growth(needs));
        }
        std::size_t paused=0; for (const auto& row:rows) if (!row.eligible) ++paused;
        if (paused) for (const auto& layer:state_.layers) if (const auto* linear=std::get_if<LinearState>(&layer)) {
            const auto v=linear->attention.view();
            if (v.convolution.defined()) add(result,v.convolution.device(),mul(v.convolution.nbytes()/slots_.size(),paused));
            if (v.recurrent.defined()) add(result,v.recurrent.device(),mul(v.recurrent.nbytes()/slots_.size(),paused));
        }
        return result;
    }
    void prepare_decode(std::span<const tyloi::models::BatchRow> rows) override {
        int64_t position=0;
        for (const auto& row:rows) { auto& r=checked(*row.state); validate_slot(r); position=std::max(position,row.position); if (paged_) paged_->ensure_tokens(pages_[r.slot],row.position+1); }
        if (paged_) bind_slots(); state_.cache_pos=position;
    }
    tyloi::StepSequence<std::vector<Tensor>> prefill(std::vector<tyloi::models::BatchState*> states,std::vector<Tensor> ids) override {
        if (states.size()!=1 || ids.size()!=1) throw std::invalid_argument("Qwen batch prefill requires one request");
        auto& request=checked(*states[0]);
        if (!request.initialized || request.slot>=0) throw std::logic_error("invalid Qwen batch prefill state");
        if (paged_) bind(request.state,{&request.pages});
        tyloi::models::Input input; input.ids=ids[0];
        if (request.state.position()>0 && ids[0].size(1)==1)
            input.lengths=tyloi::ops::full(std::array{int64_t(1)},request.state.position()+1,Device(kCUDA,model_.primary_device));
        tyloi::models::ForwardOptions options; options.primary=model_.primary_device;
        auto hidden=qwen35::forward(model_,request.state,std::move(input),options);
        co_yield std::vector<Tensor>{std::move(hidden)};
    }
    Tensor decode(std::span<const tyloi::models::BatchRow> rows,Tensor,const Tensor&) override {
        ::tyloi::ops::FfnScope policy(::tyloi::ops::FfnScope::Role::batch,cached_,int64_t(slots_.size()),false);
        auto metadata=inputs(rows); ++metadata_batches_; save_paused(rows);
        try { auto output=compute(metadata.ids,metadata.positions,metadata.lengths,0,0,false); restore_paused(paused_); paused_.clear(); advance(rows); return select(output,rows); }
        catch (...) { auto error=std::current_exception(); try { clear(); } catch (...) {} std::rethrow_exception(error); }
    }
    tyloi::StepSequence<std::unique_ptr<Plan>> plan(std::span<const tyloi::models::BatchRow> rows,bool greedy) override {
        if (!config_.graph || !config_.greedy || !greedy || cached_ || slots_.size()<2 || !mfq_cuda_graph_capture_supported() || rows.empty()) co_return;
        int64_t requested=0, remaining=std::numeric_limits<int64_t>::max();
        for (const auto& row:rows) { validate_slot(checked(*row.state)); if (!row.greedy) co_return; requested=std::max(requested,row.position+row.remaining); remaining=std::min(remaining,row.remaining); }
        const auto length=decode_graph_bucket(requested,model_.metadata.max_position_embeddings);
        auto addresses=state_addresses();
        std::shared_ptr<Graph> found;
        for (auto& graph:graphs_) if (graph->valid && graph->length==length && graph->addresses==addresses) { found=graph; break; }
        if (!found && remaining<config_.minimum) co_return;
        co_yield tyloi::GraphPhase::prepare;
        if (!found) co_yield tyloi::GraphPhase::ready;
        auto result=std::make_unique<GraphPlan>(*this,length,std::move(addresses),std::move(found));
        if (result->rebuild) {
            const Device device(kCUDA,model_.primary_device);
            add(result->growth,device,mul(slots_.size(),sizeof(int64_t)*3));
            for (auto* tensor:state_.graph_state()) add(result->growth,tensor->device(),tensor->nbytes());
            add(result->growth,device,mul(slots_.size(),sizeof(int64_t)));
        }
        std::unique_ptr<Plan> output=std::move(result); co_yield std::move(output);
    }
    std::optional<Tensor> capture(std::span<const tyloi::models::BatchRow> rows,const Tensor&,const Tensor&,bool greedy,const Plan& raw) override {
        const auto* plan=dynamic_cast<const GraphPlan*>(&raw);
        if (!plan || plan->owner!=this || !greedy || plan->addresses!=state_addresses()) throw std::invalid_argument("Qwen batch graph plan changed");
        auto graph=plan->graph;
        if (!graph) {
            if (graphs_.size()>=8) clear();
            graph=std::make_shared<Graph>(plan->length,plan->addresses);
            graphs_.push_back(graph);
        }
        auto scope=graph->window->enter();
        if (!plan->graph) {
            graph->window->prepare();
            const Device device(kCUDA,model_.primary_device);
            graph->ids=tyloi::ops::zeros({int64_t(slots_.size()),1},TensorOptions().device(device).dtype(kInt64));
            graph->positions=tyloi::ops::zeros({int64_t(slots_.size()),1},TensorOptions().device(device).dtype(kInt64));
            graph->lengths=tyloi::ops::zeros({int64_t(slots_.size())},TensorOptions().device(device).dtype(kInt64));
        }
        auto metadata=inputs(rows); ++metadata_batches_;
        tyloi::ops::copy_into(graph->ids,metadata.ids); tyloi::ops::copy_into(graph->positions,metadata.positions); tyloi::ops::copy_into(graph->lengths,metadata.lengths);
        save_paused(rows);
        ::tyloi::ops::FfnScope policy(::tyloi::ops::FfnScope::Role::batch,cached_,int64_t(slots_.size()),true);
        if (!plan->graph) {
            auto members=state_.graph_state(); graph->checkpoint.reserve(members.size());
            for (auto* member:members) {
                graph->checkpoint.push_back(empty_like(*member));
                tyloi::ops::copy_into(graph->checkpoint.back(),*member);
            }
            auto restore=[&] { for (std::size_t i=0;i<members.size();++i) tyloi::ops::copy_into(*members[i],graph->checkpoint[i]); };
            auto driver=graph->window->capture();
            try { while (driver->next()) { graph->output=compute(graph->ids,graph->positions,graph->lengths,plan->length,tyloi::ops::attention_parts(plan->length,Device(kCUDA,model_.primary_device)),true);
                    driver->submitted(); restore(); if (driver->warmup()) graph->output={}; driver->restored(); } }
            catch (...) { auto error=driver->fail(std::current_exception()); restore(); std::rethrow_exception(error); }
            graph->addresses=state_addresses(); graph->valid=true; graph->checkpoint.clear(); ++captures_;
        }
        try { graph->window->run(); restore_paused(paused_); paused_.clear(); advance(rows); ++replays_; return select(graph->output,rows); }
        catch (...) { auto error=std::current_exception(); try { clear(); } catch (...) {} std::rethrow_exception(error); }
    }
    void clear() override {
        for (auto& graph:graphs_) { auto scope=graph->window->enter(false); graph->window->clear(); }
        graphs_.clear();
    }
    tyloi::engine::Metrics metrics() const override {
        const auto p=paged_?paged_->stats():PagedKv::Stats{};
        return {{"continuous_batching_compactions",0.0},{"continuous_batching_packed_metadata_batches",double(metadata_batches_)},
            {"continuous_batching_stable_slot_releases",double(releases_)},{"continuous_batching_cuda_graph_captures",double(captures_)},
            {"continuous_batching_cuda_graph_replays",double(replays_)},{"continuous_batching_moe",double(moe_)},
            {"continuous_batching_cuda_graph_eligible",double(config_.graph && config_.greedy && !cached_ && slots_.size()>=2 && mfq_cuda_graph_capture_supported())},
            {"continuous_batching_moe_cached_row_serial",double(cached_)},{"continuous_batching_paged_kv",double(bool(paged_))},
            {"paged_kv_page_size",double(p.page_size)},{"paged_kv_live_pages",double(p.live_pages)},
            {"paged_kv_peak_live_pages",double(p.peak_live_pages)},{"paged_kv_capacity_pages",double(p.capacity_pages)},
            {"paged_kv_reserved_bytes",double(p.reserved_bytes)},{"paged_kv_page_allocations",double(p.allocations)},
            {"paged_kv_page_reuses",double(p.reuses)},{"paged_kv_page_releases",double(p.releases)},{"paged_kv_table_updates",double(p.table_updates)}};
    }
private:
    struct Metadata { Tensor ids,positions,lengths; };
    struct Paused { int slot; std::vector<std::pair<Tensor,Tensor>> values; };
    Request& checked(tyloi::models::BatchState& v) const { auto* p=dynamic_cast<Request*>(&v); if (!p || p->owner!=this) throw std::invalid_argument("Qwen batch request belongs to another batch"); return *p; }
    const Request& checked(const tyloi::models::BatchState& v) const { return checked(const_cast<tyloi::models::BatchState&>(v)); }
    void validate_slot(const Request& r) const { if (r.slot<0 || std::size_t(r.slot)>=slots_.size() || slots_[r.slot]!=&r) throw std::logic_error("continuous batching request lost its stable slot"); }
    bool active() const { return std::any_of(slots_.begin(),slots_.end(),[](auto* v){return v;}); }
    void detach(State& state) { for (auto& layer:state.layers) if (auto* kv=std::get_if<tyloi::state::KvView>(&layer)) *kv={}; }
    void bind(State& state,const std::vector<const PagedKv::Sequence*>& pages) {
        paged_->update(pages); std::size_t page=0;
        for (auto& layer:state.layers) if (auto* kv=std::get_if<tyloi::state::KvView>(&layer)) *kv=paged_->view(page++,pages.size());
    }
    void bind_slots() { std::vector<const PagedKv::Sequence*> rows; for (auto& page:pages_) { paged_->ensure_tokens(page,1); rows.push_back(&page); } bind(state_,rows); }
    void copy_slot(Request& request,std::size_t slot) {
        if (request.state.batch!=1 || request.state.speculative_start>=0) throw std::logic_error("invalid Qwen batch source");
        for (std::size_t i=0;i<state_.layers.size();++i) {
            if (auto* target=std::get_if<LinearState>(&state_.layers[i])) {
                const auto& source=std::get<LinearState>(request.state.layers[i]); auto v=source.attention.view();
                if (!source.attention.defined() || source.attention.pending()) throw std::logic_error("invalid Qwen recurrent source");
                if (!target->attention.defined()) target->attention.initialize(v,slots_.size()); auto t=target->attention.view();
                if (v.transposed!=t.transposed) throw std::logic_error("continuous batching recurrent layout changed");
                tyloi::ops::copy_into(tyloi::ops::slice(t.convolution,0,slot,slot+1),v.convolution);
                tyloi::ops::copy_into(tyloi::ops::slice(t.recurrent,0,slot,slot+1),v.recurrent);
            } else if (!paged_) {
                auto& target=std::get<tyloi::state::KvView>(state_.layers[i]); const auto& source=std::get<tyloi::state::KvView>(request.state.layers[i]);
                if (!source.defined()) throw std::logic_error("invalid Qwen KV source");
                if (!target.defined()) { target=source; std::vector<int64_t> k,v; for(int j=0;j<source.physical_key.dim();++j) k.push_back(source.physical_key.size(j)); for(int j=0;j<source.physical_value.dim();++j) v.push_back(source.physical_value.size(j)); k[0]=v[0]=slots_.size(); target.physical_key=tyloi::ops::zeros(k,source.physical_key.options()); target.physical_value=tyloi::ops::zeros(v,source.physical_value.options()); target.batch=slots_.size(); }
                tyloi::ops::copy_into(tyloi::ops::slice(target.physical_key,0,slot,slot+1),source.physical_key);
                tyloi::ops::copy_into(tyloi::ops::slice(target.physical_value,0,slot,slot+1),source.physical_value);
            }
        }
    }
    Metadata inputs(std::span<const tyloi::models::BatchRow> rows) const {
        std::vector<int64_t> ids(slots_.size()),pos(slots_.size()),lengths(slots_.size(),1);
        for (const auto& row:rows) { const auto& request=checked(*row.state); validate_slot(request); ids[request.slot]=row.token; pos[request.slot]=row.position; lengths[request.slot]=row.position+1; }
        const Device device(kCUDA,model_.primary_device);
        return {tyloi::ops::reshape(tyloi::ops::tensor(ids,device),{int64_t(slots_.size()),1}),
            tyloi::ops::reshape(tyloi::ops::tensor(pos,device),{int64_t(slots_.size()),1}),tyloi::ops::tensor(lengths,device)};
    }
    static Tensor empty_like(const Tensor& source) {
        std::vector<int64_t> shape;
        for (int64_t axis=0;axis<source.dim();++axis) shape.push_back(source.size(axis));
        return tyloi::ops::empty(shape,source.options());
    }
    void save_paused(std::span<const tyloi::models::BatchRow> rows) {
        paused_.clear();
        paused_.reserve(rows.size());
        for (const auto& row:rows) if (!row.eligible) {
            paused_.push_back(Paused{checked(*row.state).slot,{}});
            auto& p=paused_.back();
            p.values.reserve(state_.layers.size());
            for (const auto& layer:state_.layers) if (const auto* linear=std::get_if<LinearState>(&layer)) {
                const auto v=linear->attention.view();
                auto conv=tyloi::ops::slice(v.convolution,0,p.slot,p.slot+1);
                auto recurrent=tyloi::ops::slice(v.recurrent,0,p.slot,p.slot+1);
                p.values.emplace_back(Tensor{},Tensor{});
                auto& targets=p.values.back();
                targets.first=empty_like(conv);
                targets.second=empty_like(recurrent);
                tyloi::ops::copy_into(targets.first,conv);
                tyloi::ops::copy_into(targets.second,recurrent);
            }
        }
    }
    void restore_paused(const std::vector<Paused>& rows) {
        for (const auto& row:rows) { std::size_t i=0; for (auto& layer:state_.layers) if (auto* linear=std::get_if<LinearState>(&layer)) {
            const auto v=linear->attention.view(); tyloi::ops::copy_into(tyloi::ops::slice(v.convolution,0,row.slot,row.slot+1),row.values[i].first);
            tyloi::ops::copy_into(tyloi::ops::slice(v.recurrent,0,row.slot,row.slot+1),row.values[i++].second);
        }}
    }
    Tensor compute(const Tensor& ids,const Tensor& pos,const Tensor& lengths,int64_t kv,int64_t parts,bool greedy) {
        tyloi::models::Input input; input.ids=ids; input.positions=pos; input.lengths=lengths; input.cache_positions=pos; input.kv_length=kv; input.attention_parts=parts;
        tyloi::models::ForwardOptions options; options.primary=model_.primary_device;
        auto hidden=qwen35::forward(model_,state_,std::move(input),options);
        auto last=tyloi::ops::contiguous(tyloi::ops::cast(tyloi::ops::select(hidden,1,-1),kFloat16));
        auto output=qwen35::logits(model_,last,true);
        if (greedy) output=tyloi::ops::sample_greedy(tyloi::ops::reshape(output,{int64_t(slots_.size()),-1}));
        return output;
    }
    void advance(std::span<const tyloi::models::BatchRow> rows) {
        state_.cache_pos=0;
        for (const auto& row:rows) {
            state_.cache_pos=std::max(state_.cache_pos,row.position+1);
            if (row.eligible) checked(*row.state).active_position=row.position+1;
        }
    }
    Tensor select(const Tensor& output,std::span<const tyloi::models::BatchRow> rows) const {
        std::vector<int64_t> indexes; for (const auto& row:rows) if (row.eligible) indexes.push_back(checked(*row.state).slot);
        return tyloi::ops::index_select(output,0,tyloi::ops::tensor(indexes,output.device()));
    }
    std::vector<const void*> state_addresses() const { std::vector<const void*> values; for (const auto& t:state_.storage()) if (t.defined()) values.push_back(t.dense().data_ptr()); return values; }
    Model& model_; tyloi::models::BatchConfig config_; State state_;
    std::vector<PagedKv::Layout> layouts_; std::vector<Request*> requests_,slots_; std::vector<PagedKv::Sequence> pages_; std::unique_ptr<PagedKv> paged_;
    std::vector<std::shared_ptr<Graph>> graphs_;
    std::vector<Paused> paused_;
    bool initialized_=false,moe_=false,cached_=false;
    int64_t releases_=0,captures_=0,replays_=0,metadata_batches_=0;
};
}
std::unique_ptr<tyloi::models::Batch> Model::batch(const tyloi::models::BatchConfig& config,tyloi::models::State* serial) {
    if (serial) { auto* state=dynamic_cast<State*>(serial); if (!state || !state->belongs_to(*this)) throw std::invalid_argument("Qwen batch serial state belongs to another model");
        if (config.preallocate) for (auto& layer:state->layers) if (auto* kv=std::get_if<tyloi::state::KvView>(&layer)) *kv={}; }
    return std::make_unique<Batch>(*this,config);
}
}
