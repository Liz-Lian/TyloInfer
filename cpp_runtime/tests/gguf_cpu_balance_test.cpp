#include "../backends/cuda/ops/gguf/gguf_cpu_balance.h"
#include "../backends/cuda/ops/gguf/gguf_admission_executor.h"
#include <atomic>
#include <iostream>
#include <stdexcept>
#include <vector>

void require(bool condition,const char* message) {if(!condition)throw std::runtime_error(message);}
void check_risk_only_reprobe() {
    GgufCpuBalanceModel model;
    model.on=true;model.ncpu=8;model.ngpu_p=4;model.max_observed_cpu=3;
    model.a=.02;model.c=.12;model.d=.01;model.g0=.03;model.g=.06;
    // A transient delay may reject CPU work even though its fitted mean is
    // faster. Preserve that protection, but collect fresh evidence on time.
    model.tail_variance=100.;
    for(int i=0;i<63;++i)require(model.cpu_count(3)==0,"CPU tail protection was bypassed between probes");
    require(model.cpu_count(3)==1,"old tail uncertainty postponed a cheaper CPU lane's recovery probe");
}
void check_relative_tail_risk() {
    const auto trained=[] {
        GgufCpuBalanceModel model;
        for(int i=0;i<600;++i) {
            const int cpu=i%5,pcie=(i/5)%6;
            model.observe(cpu,pcie,cpu?.02+.12*cpu+.01*pcie:0.,.03+.06*pcie);
        }
        model.on=true;return model;
    };
    auto gpu_delayed=trained();
    gpu_delayed.observe(1,2,.16,1.15);
    require(gpu_delayed.tail_variance<1e-8,"GPU-only delay was priced as unexpected CPU lateness");
    auto cpu_delayed=trained();
    cpu_delayed.observe(1,2,1.16,.15);
    require(cpu_delayed.tail_variance>.01,"unexpected CPU lateness lost its risk penalty");
    auto shared_delay=trained();
    shared_delay.observe(1,2,1.16,1.15);
    require(shared_delay.tail_variance<1e-6,"a delay shared by both lanes became CPU tail risk");
}
void check_admission_worker() {
    GgufAdmissionExecutor worker;
    std::thread::id identity;int completed=0;
    std::vector<std::future<void>> jobs;
    for(int i=0;i<32;++i)jobs.push_back(worker.submit([&,i] {
        require(completed++==i,"cache admission changed submission order");
        if(i==0)identity=std::this_thread::get_id();
        require(identity==std::this_thread::get_id(),"cache admission recreated its host thread");
    }));
    for(auto& job:jobs)job.get();
    auto failed=worker.submit([]{throw std::runtime_error("admission failure");});
    bool propagated=false;
    try{failed.get();}catch(const std::runtime_error&){propagated=true;}
    require(propagated,"cache admission swallowed an asynchronous failure");
    worker.submit([&]{++completed;}).get();
    require(completed==33,"cache admission worker did not recover after a job failed");
    std::atomic<int> drained=0;
    std::promise<void> gate;const auto ready=gate.get_future().share();
    jobs.clear();
    {
        GgufAdmissionExecutor retiring;
        jobs.push_back(retiring.submit([&]{ready.wait();++drained;}));
        for(int i=0;i<32;++i)jobs.push_back(retiring.submit([&]{++drained;}));
        gate.set_value();
    }
    require(drained==33,"cache unload did not drain accepted admission jobs");
    for(auto& job:jobs)job.get();
}
int main() {
    try {
        check_admission_worker();
        check_relative_tail_risk();
        check_risk_only_reprobe();
        GgufCpuBalanceModel model;
        require(model.cpu_count(0)==0,"empty route set scheduled CPU work");
        for(int i=0;i<4;++i)require(model.cpu_count(1)==(i%2==0?1:0),"sole-miss calibration did not measure both devices");
        model={};model.on=true;model.ncpu=8;model.ngpu_p=4;model.max_observed_cpu=3;
        model.a=.02;model.c=.12;model.d=.01;model.g0=.03;model.g=.06;
        require(model.cpu_count(1)==0,"CPU took a sole miss whose complete GPU cost is lower");
        require(model.cpu_count(3)==1,"cost balance did not hide CPU work behind the remaining uploads");
        model.tail_variance=.04;
        require(model.cpu_count(3)==0,"a slow CPU tail was hidden by its cheaper mean cost");
        model.tail_variance=0;
        model.explore=63;require(model.cpu_count(1)==1,"bounded exploration stopped measuring a rejected lane");
        model.a=8.;model.c=.01;model.d=0.;model.g0=.02;model.g=.01;model.explore=0;
        int expensive_probes=0;double extra_ms=0.;
        for(int i=0;i<20000;++i) {
            const int cpu=model.cpu_count(1);expensive_probes+=cpu>0;
            extra_ms+=cpu?model.price(1,0)-model.price(1,1):0.;
        }
        require(expensive_probes>0 && extra_ms<20000.*model.price(1,1)*.016,
            "rejected CPU wake probes consumed more than their measured exploration budget");
        model.a=.001;model.c=.001;model.explore=0;
        require(model.cpu_count(1)==1,"a previously slow CPU lane could not recover after its measured cost fell");
        model={};model.on=true;model.ncpu=8;model.ngpu_p=4;model.max_observed_cpu=1;
        model.c=.001;model.g=1.;
        require(model.cpu_count(20)<=2,"one-expert calibration extrapolated to a whole CPU layer");
        model={};
        for(int i=0;i<600;++i) {
            const int cpu=i%5,pcie=(i/5)%6;
            model.observe(cpu,pcie,cpu?.02+.12*cpu+.01*pcie:0.,.03+.06*pcie);
        }
        model.on=true;
        require(model.ready(),"both fits did not become usable");
        require(std::abs(model.a-.02)<.01 && std::abs(model.c-.12)<.01 && std::abs(model.d-.01)<.01 &&
            std::abs(model.g0-.03)<.001 && std::abs(model.g-.06)<.001,"measured-cost fit changed its CPU/PCIe slopes");
        require(model.cpu_count(1)==0 && model.cpu_count(3)==1,"learned costs selected a slower split");
        model={};
        for(int i=0;i<300;++i) {
            const int cpu=i%3,pcie=3-cpu;
            model.observe(cpu,pcie,cpu?.02+.12*cpu+.01*pcie:0.,.03+.06*pcie);
        }
        model.on=true;
        require(std::abs(model.a+model.c+2*model.d-.16)<.002 &&
            std::abs(model.a+2*model.c+model.d-.27)<.002,
            "constant miss count made constrained CPU predictions drift");
        require(model.cpu_count(3)==1,"collinear counts made the CPU/PCIe split slower");
        model={};
        for(int i=0;i<300;++i) {
            const int pcie=3+i%6;
            model.sample(0,pcie,0.,.02*pcie-.05);
        }
        model.solve();
        require(model.g0==0 && model.g>0 && model.g<.02,
            "negative GPU intercept was independently clipped without refitting its slope");
        model.on=true;model.ncpu=8;model.max_observed_cpu=3;
        model.a=.045;model.c=.02;model.d=0.;
        require(model.cpu_count(4)==0,"biased GPU cost assigned a slower CPU expert");
        std::cout<<"GGUF CPU/PCIe measured-cost balance, calibration and bounded exploration PASS\n";
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
