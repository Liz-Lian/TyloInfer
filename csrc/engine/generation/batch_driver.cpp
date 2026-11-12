#include "engine/generation/batch_driver.h"
#include "engine/budget.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/executor.h"
#include "engine/generation/decode.h"
#include "engine/generation/prefill_budget.h"
#include "engine/weights/cache.h"

#include <algorithm>
#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <utility>

namespace tyloi::engine {
void reserve_batch(Budget& budget, Executor& executor, const models::BatchGrowth& values,
        std::size_t margin, bool primary) {
    models::BatchGrowth growth;
    for(const auto& [device,bytes]:values) {
        auto found=std::find_if(growth.begin(),growth.end(),[&](const auto& value){return value.first==device;});
        if(found==growth.end())growth.emplace_back(device,bytes);
        else found->second=prefill_budget_add(found->second,bytes);
    }
    if(primary && std::none_of(growth.begin(),growth.end(),[&](const auto& value){return value.first==executor.device();}))
        growth.emplace_back(executor.device(),0);
    for(const auto& [device,bytes]:growth)
        if(bytes || (primary && device==executor.device()))budget.reserve_kv(device,prefill_budget_add(bytes,margin));
}

std::vector<Tensor> prefill_batch(models::Batch& batch, Executor& executor, weights::Cache& weights,
        std::vector<models::BatchState*> states, std::vector<Tensor> ids) {
    auto work=executor.work();
    StepSequence<std::vector<Tensor>> job;
    std::optional<std::vector<Tensor>> result;
    try {
        job=batch.prefill(std::move(states),std::move(ids));
        for(;;) {
            StepResult<std::vector<Tensor>> step;
            { auto scope=work->enter(); step=job.next(); }
            if(step.workspace || step.graph ||
                (step.state==StepState::complete && (step.value || step.layer)) || (step.layer && step.value))
                throw std::logic_error("invalid batch prefill event");
            if(!step)break;
            if(step.state!=StepState::advanced || result)throw std::logic_error("invalid batch prefill event");
            if(step.layer)work->step(*step.layer);
            else if(step.value)result=std::move(*step.value);
        }
        if(!result)throw std::runtime_error("batch prefill ended without a result");
        work->finish(); job={}; return std::move(*result);
    } catch(...) {
        auto failure=std::current_exception();
        try { weights.drain(); } catch(...) {}
        try { executor.drain(); } catch(...) {}
        job={}; std::rethrow_exception(failure);
    }
}

std::unique_ptr<models::Batch::Plan> plan_batch(models::Batch& batch, Executor& executor,
        weights::Cache& weights, const std::vector<models::BatchRow>& rows, bool greedy) {
    Capture capture(executor,weights);
    std::unique_ptr<models::Batch::Plan> plan;
    StepSequence<std::unique_ptr<models::Batch::Plan>> job;
    bool observed=false;
    try {
        job=batch.plan(rows,greedy);
        for(;;) {
            auto step=job.next();
            if(step.layer || step.workspace || (step.graph && step.value) ||
                (step.state==StepState::complete && (step.graph || step.value)))
                throw std::logic_error("invalid batch graph plan event");
            if(!step)break;
            if(step.state!=StepState::advanced)throw std::logic_error("invalid batch graph plan event");
            observed=true;
            if(step.graph) {
                if(plan)throw std::logic_error("batch graph plan continued after its result");
                capture.step(*step.graph);
            } else if(step.value) {
                if(plan || !*step.value)throw std::logic_error("invalid batch graph plan result");
                plan=std::move(*step.value);
            } else if(plan)throw std::logic_error("batch graph plan continued after its result");
        }
        if(observed && !plan)throw std::logic_error("batch graph plan ended without a result");
        capture.finish(); job={}; return plan;
    } catch(...) {
        auto failure=std::current_exception();
        try { capture.drain(); } catch(...) {}
        job={}; std::rethrow_exception(failure);
    }
}

BatchDriver::BatchDriver(std::unique_ptr<models::Batch> batch, Executor& executor,
        weights::Cache& weights, Budget& budget)
    : executor_(executor), weights_(weights), budget_(budget), batch_(std::move(batch)) {
    if(!batch_)throw std::invalid_argument("model does not provide numerical batching");
}

void BatchDriver::initialize() {
    auto scope=executor_.enter();
    reserve_batch(budget_,executor_,batch_->initial_growth_bytes(),std::size_t(64)<<20);
    batch_->initialize();
    weights_.finish_exchange();
    executor_.drain();
}

BatchDriver::~BatchDriver() noexcept {
    std::unique_ptr<Executor::Scope> scope;
    try { scope=executor_.enter(); } catch(const std::exception& e) { std::cerr<<"batch_driver_cleanup error="<<e.what()<<'\n'; } catch(...) {}
    try { weights_.drain(); } catch(const std::exception& e) { std::cerr<<"batch_driver_cleanup error="<<e.what()<<'\n'; } catch(...) {}
    try { executor_.drain(); } catch(const std::exception& e) { std::cerr<<"batch_driver_cleanup error="<<e.what()<<'\n'; } catch(...) {}
    try { close(); } catch(const std::exception& e) { std::cerr<<"batch_driver_cleanup error="<<e.what()<<'\n'; } catch(...) {}
}

models::BatchState& BatchDriver::add(std::int64_t prompt_tokens) {
    auto scope=executor_.enter();
    if(states_.empty())weights_.finish_exchange();
    auto state=batch_->make_state(prompt_tokens);
    if(!state)throw std::runtime_error("batch state is unavailable");
    reserve_batch(budget_,executor_,state->initial_growth_bytes(),std::size_t(64)<<20);
    states_.push_back(std::move(state));
    states_.back()->initialize();
    return *states_.back();
}

Tensor BatchDriver::prefill(models::BatchState& state, Tensor ids) {
    auto scope=executor_.enter();
    if(ids.dim()!=2 || ids.size(0)!=1 || ids.size(1)<=0)
        throw std::invalid_argument("batch prompt must be nonempty [1,tokens]");
    if(state.position()!=0)throw std::runtime_error("batch prefill position mismatch");
    const auto count=ids.size(1);
    const auto growth=state.prefill_growth_bytes(count);
    std::size_t primary=0;
    for(const auto& [device,bytes]:growth) {
        if(device==executor_.device())primary=prefill_budget_add(primary,bytes);
        else if(bytes)budget_.reserve_kv(device,prefill_budget_add(bytes,std::size_t(64)<<20));
    }
    const auto chunk=budget_.batch(executor_.device(),batch_->prefill_row_bytes(),{0,count,true},primary);
    if(chunk.count!=count)throw std::runtime_error("batch prefill admission cannot preserve the full prompt");
    if(count>1024)weights_.workspace(cuda::options::workspace_reserve_bytes());
    reserve_batch(budget_,executor_,growth,std::size_t(256)<<20,true);
    auto outputs=prefill_batch(*batch_,executor_,weights_,{&state},{std::move(ids)});
    if(outputs.size()!=1)throw std::runtime_error("invalid batch prefill output count");
    return std::move(outputs.front());
}

std::optional<Tensor> BatchDriver::capture(const std::vector<models::BatchRow>& rows,
        const Tensor& ids, const Tensor& host_ids, bool greedy) {
    auto scope=executor_.enter();
    std::unique_ptr<models::Batch::Plan> plan;
    try {
        reserve_batch(budget_,executor_,batch_->decode_growth(rows),std::size_t(64)<<20);
        batch_->prepare_decode(rows);
        plan=plan_batch(*batch_,executor_,weights_,rows,greedy);
        if(!plan)return {};
        if(plan->rebuild) {
            executor_.trim(executor_.device());
            reserve_batch(budget_,executor_,plan->growth,std::size_t(64)<<20);
        }
        return batch_->capture(rows,ids,host_ids,greedy,*plan);
    } catch(...) {
        auto failure=std::current_exception();
        try { weights_.drain(); } catch(...) {}
        try { executor_.drain(); } catch(...) {}
        plan.reset();
        std::rethrow_exception(failure);
    }
}

void BatchDriver::close() {
    if(!batch_)return;
    std::exception_ptr failure;
    try { batch_->clear(); } catch(...) { failure=std::current_exception(); }
    for(auto& state:states_)if(state)try { batch_->release(*state); }
        catch(...) { if(!failure)failure=std::current_exception(); }
    batch_.reset(); states_.clear();
    if(failure)std::rethrow_exception(failure);
}
}
