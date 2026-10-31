#include "gguf_cpu_executor.h"
#include "gguf_rotation.h"
#include "gguf_cpu_activation.h"
#include "tyloinfer/cpu_expert_pool.h"
#include "tyloinfer/cpu_wait.h"
#include "ggml.h"
#include "mfq_native_tensor.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <stdexcept>
#include <type_traits>

GgufCpuExecutor::GgufCpuExecutor(std::vector<int> cores)
    :cores_(std::move(cores)),pool_([this](std::size_t index) {
        if(index+1<cores_.size())(void)mfq::cpu::pin_current_thread(cores_[index]);
    },false,20000000) {
    worker_=std::thread([this]{run();});
}
GgufCpuExecutor::~GgufCpuExecutor() {
    {std::lock_guard lock(mutex_);stopping_=true;}
    keep_awake();worker_.join();
    mfq::cuda::tensor_host_bytes.fetch_sub(workspace_bytes_);
}
void GgufCpuExecutor::submit(std::shared_ptr<GgufCpuWork> job) {
    if(!job || !job->ready || job->submitted.exchange(true))throw std::invalid_argument("invalid or repeated CPU expert work");
    {std::lock_guard lock(mutex_);
        if(stopping_)throw std::logic_error("CPU expert executor is stopping");
        job->queued_at=std::chrono::steady_clock::now();
        queue_.push_back(std::move(job));}
    keep_awake();
}
void GgufCpuExecutor::keep_awake() noexcept {
    activity_.fetch_add(1,std::memory_order_release);activity_.notify_one();
}
void GgufCpuExecutor::run() {
    try{if(!cores_.empty())(void)mfq::cpu::pin_current_thread(cores_.back());}catch(const std::bad_alloc&){}
    while(true) {
        // Observe before checking the queue so a concurrent submission cannot
        // be missed between that check and the atomic wait.
        const auto observed=activity_.load(std::memory_order_acquire);
        std::shared_ptr<GgufCpuWork> work;
        {std::lock_guard lock(mutex_);
            if(!queue_.empty()){work=std::move(queue_.front());queue_.pop_front();}
            else if(stopping_)return;}
        if(!work) {
            const auto spin_begin=std::chrono::steady_clock::now();
            while(activity_.load(std::memory_order_acquire)==observed &&
                std::chrono::steady_clock::now()-spin_begin<std::chrono::milliseconds(20))
                mfq::cpu::spin_pause();
            activity_.wait(observed,std::memory_order_acquire);
            continue;
        }
        const auto started=std::chrono::steady_clock::now();
        work->queue_ms=std::chrono::duration<double,std::milli>(started-work->queued_at).count();
        std::exception_ptr failure;
        try {execute(*work);}catch(...){failure=std::current_exception();}
        work->elapsed_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
        // Captured work objects can outlive this execution. Weight owners are
        // needed only through the CPU dot products, not through the graph cache.
        for(auto& expert:work->experts)for(auto& owner:expert.owners)owner.reset();
        // Release CPU writes before the mapped GPU consumer can observe ready.
        // Even failures publish, so a captured GPU graph can exit and report it.
        work->completed_at=std::chrono::steady_clock::now();
        std::atomic_thread_fence(std::memory_order_seq_cst);
        std::atomic_ref<std::uint32_t>(*work->ready).store(1,std::memory_order_release);
        if(failure)work->completion.set_exception(failure);else work->completion.set_value();
    }
}
void GgufCpuExecutor::execute(GgufCpuWork& w) {
    namespace cpu=mfq::cpu::gguf;
    const auto dispatch=cpu::projection_dispatch();
    if(dispatch.trace) {
        static std::atomic<unsigned> seen{0};const unsigned bit=w.half?2:1;
        if(!(seen.load(std::memory_order_relaxed)&bit) && !(seen.fetch_or(bit,std::memory_order_relaxed)&bit))
            std::fprintf(stderr,"native_gguf_cpu_prepare actual_match=1 activation=%s hidden_quant=%s\n",
                w.half && cpu::available()?"f16c8":"scalar",cpu::available()?"avx2_exact":"reference");
    }
    if(w.tokens<1 || w.tokens>4 || w.routes<=0 || !w.input || !w.output || !w.ready)
        throw std::invalid_argument("CPU expert work geometry");
    const int down_type=w.experts.empty()?20:w.experts.front().projections[2].type;
    const auto input_bytes=cpu::input_bytes(w.input_width),hidden_bytes=cpu::hidden_bytes(w.hidden_width,down_type);
    // Charge capacity as each allocation succeeds, including partial growth
    // when a later allocation fails. Storage is reused across layer jobs.
    const auto resize=[&](auto& buffer,std::size_t count) {
        using Value=typename std::decay_t<decltype(buffer)>::value_type;
        const auto previous=buffer.capacity()*sizeof(Value);buffer.resize(count);
        const auto growth=buffer.capacity()*sizeof(Value)-previous;
        if(growth){mfq::cuda::charge_tensor_host_bytes(growth);workspace_bytes_+=growth;}
    };
    resize(prepared_,w.experts.size());std::fill(prepared_.begin(),prepared_.end(),PreparedExpert{});
    auto& prepared=prepared_;
    const auto values=std::size_t(w.tokens)*w.hidden_width;
    resize(qinput_,w.tokens*input_bytes);resize(qhidden_,w.experts.size()*w.tokens*hidden_bytes);
    resize(gate_,w.experts.size()*values);resize(up_,gate_.size());resize(hidden_,gate_.size());
    for(int t=0;t<w.tokens;++t)cpu::quantize_input(w.input+t*w.input_width,qinput_.data()+t*input_bytes,w.input_width);
    if(w.up_input && w.up_input!=w.input) {
        resize(qup_,w.tokens*input_bytes);
        for(int t=0;t<w.tokens;++t)cpu::quantize_input(w.up_input+t*w.input_width,qup_.data()+t*input_bytes,w.input_width);
    }
    const int threads=std::max(1,int(cores_.size()));
    for(std::size_t e=0;e<w.experts.size();++e) {
        const auto& expert=w.experts[e];auto& p=prepared[e];
        if(expert.projections[0].columns!=w.input_width || expert.projections[1].columns!=w.input_width ||
            expert.projections[0].rows!=w.hidden_width || expert.projections[1].rows!=w.hidden_width ||
            expert.projections[2].columns!=w.hidden_width || expert.projections[2].rows!=w.output_width ||
            expert.projections[2].type!=down_type)throw std::invalid_argument("CPU expert projection geometry differs");
        for(int slot:expert.slots) {
            if(slot<0 || slot>=w.tokens*w.routes)throw std::out_of_range("CPU expert route");
            const int t=slot/w.routes;
            if(std::find(p.token.begin(),p.token.begin()+p.count,t)==p.token.begin()+p.count) {
                p.token[p.count]=t;p.first_slot[p.count++]=slot;
            }
        }
        for(int t=0;t<p.count;++t) {
            p.x[t]=qinput_.data()+p.token[t]*input_bytes;
            p.u[t]=w.up_input && w.up_input!=w.input?qup_.data()+p.token[t]*input_bytes:p.x[t];
            p.h[t]=qhidden_.data()+(e*w.tokens+t)*hidden_bytes;
            p.gate[t]=gate_.data()+e*values+t*w.hidden_width;
            p.up[t]=up_.data()+e*values+t*w.hidden_width;
            p.out[t]=w.output+p.first_slot[t]*w.output_width;
        }
    }
    // Dispatch all selected experts together. Separate row ranges preserve
    // each dot product's arithmetic order while sharing one worker barrier
    // for Gate/Up and one for Down, rather than two barriers per expert.
    pool_.run(0,std::int64_t(w.experts.size())*w.hidden_width,32,threads,
        [&](std::int64_t begin,std::int64_t end) {
            while(begin<end) {
                const auto e=std::size_t(begin/w.hidden_width);const auto& p=prepared[e];
                const int first=int(begin%w.hidden_width);
                const int last=first+int(std::min(end-begin,std::int64_t(w.hidden_width-first)));
                if(p.count) {
                    cpu::project(w.experts[e].projections[0],p.x.data(),p.count,p.gate.data(),first,last,dispatch);
                    cpu::project(w.experts[e].projections[1],p.u.data(),p.count,p.up.data(),first,last,dispatch);
                    for(int t=0;t<p.count;++t)cpu::swiglu(p.gate[t]+first,p.up[t]+first,
                        hidden_.data()+e*values+t*w.hidden_width+first,last-first,w.half,w.limit);
                }
                begin+=last-first;
            }
        });
    for(std::size_t e=0;e<prepared.size();++e)for(int t=0;t<prepared[e].count;++t) {
        if(w.down_rotation)w.down_rotation->apply_cpu(hidden_.data()+e*values+t*w.hidden_width,1,w.hidden_width,w.half);
        cpu::quantize_hidden(hidden_.data()+e*values+t*w.hidden_width,
            qhidden_.data()+(e*w.tokens+t)*hidden_bytes,w.hidden_width,down_type);
    }
    pool_.run(0,std::int64_t(w.experts.size())*w.output_width,32,threads,
        [&](std::int64_t begin,std::int64_t end) {
            while(begin<end) {
                const auto e=std::size_t(begin/w.output_width);const auto& p=prepared[e];
                const int first=int(begin%w.output_width);
                const int last=first+int(std::min(end-begin,std::int64_t(w.output_width-first)));
                if(p.count)cpu::project(w.experts[e].projections[2],p.h.data(),p.count,p.out.data(),first,last,dispatch);
                begin+=last-first;
            }
        });
    for(std::size_t e=0;e<prepared.size();++e) {
        const auto& p=prepared[e];
        for(int slot:w.experts[e].slots) {
            const auto index=std::find(p.token.begin(),p.token.begin()+p.count,slot/w.routes)-p.token.begin();
            if(slot!=p.first_slot[index])std::copy_n(p.out[index],w.output_width,w.output+slot*w.output_width);
        }
    }
}
