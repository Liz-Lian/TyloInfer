#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "kernels/qsa_tq.h"
#include "kernels/qsa_tq_fa.h"
#include "kernels/gguf.h"
#include <vector>
#include <algorithm>
#include <cstring>
#include <numeric>
#include <cmath>
#include <iostream>
#include <stdexcept>
namespace tq=tyloinfer::qsa;
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>struct Device {
    T* p=nullptr;explicit Device(size_t count){ck(cudaMalloc(&p,std::max(size_t(1),count)*sizeof(T)));}
    ~Device(){cudaFree(p);}void put(const std::vector<T>& x){ck(cudaMemcpy(p,x.data(),x.size()*sizeof(T),cudaMemcpyHostToDevice));}
};
void check_materialized_gate() {
    constexpr int n=32771;
    std::vector<float> gate(n),up(n),out(n);std::vector<half> gh(n),uh(n),oh(n);
    for(int i=0;i<n;++i){gate[i]=32.f*std::sin(float(i)*.021f);up[i]=8.f*std::cos(float(i)*.073f);gh[i]=__float2half(gate[i]);uh[i]=__float2half(up[i]);}
    Device<float> dg(n),du(n),dy(n);Device<half> dgh(n),duh(n),dyh(n);
    dg.put(gate);du.put(up);dgh.put(gh);duh.put(uh);
    for(bool fp16:{false,true})for(bool silu:{false,true}) {
        mfq::cuda::gguf::materialize_gated_input(fp16?static_cast<void*>(duh.p):du.p,
            fp16?static_cast<void*>(dgh.p):dg.p,fp16,n,silu,fp16?static_cast<void*>(dyh.p):dy.p,nullptr);
        if(fp16)ck(cudaMemcpy(oh.data(),dyh.p,n*2,cudaMemcpyDeviceToHost));
        else ck(cudaMemcpy(out.data(),dy.p,n*4,cudaMemcpyDeviceToHost));
        for(int i=0;i<n;++i) {
            const float g=fp16?__half2float(gh[i]):gate[i],u=fp16?__half2float(uh[i]):up[i];
            float a=silu?g/(1.f+std::exp(-g)):1.f/(1.f+std::exp(-g));
            if(fp16)a=__half2float(__float2half(a));
            float expected=a*u;if(fp16)expected=__half2float(__float2half(expected));
            const float actual=fp16?__half2float(oh[i]):out[i];
            if(std::abs(actual-expected)>(fp16?.002f:2.e-6f)*std::max(1.f,std::abs(expected)))throw std::runtime_error("Materialized gate changed dtype rounding");
        }
    }
    std::cout<<"FP16/FP32 sigmoid and SiLU fused materialization, odd tails PASS"<<std::endl;
}
struct GraphFixture {
 int queries,rows,kvheads,heads,width,count,blocks;size_t scratch;
 Device<float> query,gate,sign,vsign,rot,temporary,logmass,norm,hits,mass;
 Device<half> keys,values,output;
 Device<tq::SelectedBlock> selected;Device<int64_t> positions;
 GraphFixture(int q,int r,int kv,int ratio,int d,int k):queries(q),rows(r),kvheads(kv),heads(kv*ratio),width(d),count(k),blocks(r/4),scratch(tq::tq_prefill_attention_scratch(q,heads,r)),
 query(size_t(q)*heads*d),gate(size_t(q)*heads*d),sign(256),vsign(256),rot(size_t(q)*heads*256),temporary((scratch+3)/4),logmass(size_t(q)*heads*k),norm(q*heads),hits(blocks),mass(blocks),keys(r*kv*256),values(r*kv*256),output(size_t(q)*heads*d),selected(size_t(q)*k),positions(q) {
  std::vector<float> x(size_t(q)*heads*d),g(x.size()),s(256),v(256);
  for(size_t i=0;i<x.size();++i){x[i]=.3f*std::sin(i*.071f);g[i]=.31f*std::cos(i*.049f);}
  for(int i=0;i<256;++i){s[i]=(i%3)?1.f:-1.f;v[i]=(i%5)?-1.f:1.f;}
  query.put(x);gate.put(g);sign.put(s);vsign.put(v);
  std::vector<half> kh(size_t(r)*kv*256),vh(kh.size());
  for(size_t i=0;i<kh.size();++i){kh[i]=__float2half(.21f*std::cos(i*.043f));vh[i]=__float2half(.17f*std::sin(i*.063f));}
  keys.put(kh);values.put(vh);
  std::vector<tq::SelectedBlock> sel(size_t(q)*k);std::vector<int64_t> pos(q);
  for(int t=0;t<q;++t){pos[t]=r-q+t;const int available=int(pos[t]/4+1);int step=17;while(std::gcd(step,available)!=1)step+=2;
   for(int j=0;j<k;++j)sel[size_t(t)*k+j]=j<available?tq::SelectedBlock{(j*step+(t/8)*3)%available,uint32_t(q<128?(j+t)%16:15)}:tq::SelectedBlock{-1,0};}
  // A complete empty query group must stay zero even when a larger tile
  // also contains queries which attend to actual keys.
  if(q<64 && ratio==6 && d==17)for(int t=0;t<8;++t)for(int j=0;j<k;++j)sel[t*k+j]={-1,0};
  selected.put(sel);positions.put(pos);ck(cudaMemset(hits.p,0,blocks*4));ck(cudaMemset(mass.p,0,blocks*4));
 }
 tq::TqPrefillAttention args(){
  tq::TqPrefillAttention a;a.keys=reinterpret_cast<uint16_t*>(keys.p);a.values=reinterpret_cast<uint16_t*>(values.p);a.query=query.p;a.key_signs=sign.p;a.value_signs=vsign.p;a.selected=selected.p;a.positions=positions.p;a.output=output.p;a.rotated=rot.p;a.temporary=temporary.p;a.temporary_bytes=scratch;a.logmass=logmass.p;a.norm=norm.p;a.hits=hits.p;a.mass=mass.p;a.queries=queries;a.heads=heads;a.kv_heads=kvheads;a.width=width;a.kv_rows=rows;a.count=count;a.query_token_stride=heads*width;a.query_head_stride=width;a.gate=gate.p;a.gate_token_stride=heads*width;a.gate_head_stride=width;a.output_half=true;return a;
 }
};

std::vector<half> read_graph_output(GraphFixture& f) {std::vector<half> x(size_t(f.queries)*f.heads*f.width);ck(cudaMemcpy(x.data(),f.output.p,x.size()*2,cudaMemcpyDeviceToHost));return x;}
void assert_graph_equal(const std::vector<half>& a,const std::vector<half>& b) {
 if(a.size()!=b.size())throw std::runtime_error("graph shape mismatch");
 if(std::memcmp(a.data(),b.data(),a.size()*sizeof(half)))throw std::runtime_error("Graph changed attention output");
}
struct GraphHistory {std::vector<float> hits,mass;};
GraphHistory read_graph_history(GraphFixture& f) {
 GraphHistory result{std::vector<float>(f.blocks),std::vector<float>(f.blocks)};
 ck(cudaMemcpy(result.hits.data(),f.hits.p,f.blocks*4,cudaMemcpyDeviceToHost));
 ck(cudaMemcpy(result.mass.data(),f.mass.p,f.blocks*4,cudaMemcpyDeviceToHost));return result;
}
void clear_graph_history(GraphFixture& f,cudaStream_t stream) {
 ck(cudaMemsetAsync(f.hits.p,0,f.blocks*4,stream));ck(cudaMemsetAsync(f.mass.p,0,f.blocks*4,stream));
}
void assert_graph_history(const GraphHistory& expected,GraphFixture& f) {
 const auto actual=read_graph_history(f);
 for(int i=0;i<f.blocks;++i)if(expected.hits[i]!=actual.hits[i] || !std::isfinite(actual.mass[i]) ||
     std::abs(expected.mass[i]-actual.mass[i])>1.e-5f)throw std::runtime_error("Graph changed attention history");
}
void check_prefill_graph() {

 cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
 tq::TqPrefillGraph graph;int cases=0;
 for(int q:{33,67,33})for(int width:{17,256})for(int ratio:{4,6,9})for(bool observe:{true,false}) {
  GraphFixture f(q,q==67?260:128,2,ratio,width,16);ck(cudaDeviceSynchronize());auto a=f.args();a.stream=stream;
  if(!observe)a.hits=a.mass=nullptr;
  ck(cudaError_t(tq::tq_sparse_attention_prefill(a)));ck(cudaStreamSynchronize(stream));auto expected=read_graph_output(f);
  const auto history=read_graph_history(f);clear_graph_history(f,stream);
  ck(cudaMemsetAsync(f.output.p,0x7f,expected.size()*2,stream));
  ck(cudaError_t(graph.run(a)));ck(cudaStreamSynchronize(stream));assert_graph_equal(expected,read_graph_output(f));assert_graph_history(history,f);++cases;
 }
 // Exercise compact unions and KV partitions with empty tiles and tails.
 {
  GraphFixture sparse(35,65536,1,6,17,16);ck(cudaDeviceSynchronize());auto a=sparse.args();a.stream=stream;
  ck(cudaError_t(tq::tq_sparse_attention_prefill(a)));ck(cudaStreamSynchronize(stream));
  const auto expected=read_graph_output(sparse);const auto history=read_graph_history(sparse);
  clear_graph_history(sparse,stream);ck(cudaError_t(graph.run(a)));ck(cudaStreamSynchronize(stream));
  assert_graph_equal(expected,read_graph_output(sparse));assert_graph_history(history,sparse);
  sparse.positions.put(std::vector<int64_t>(sparse.queries,-1));ck(cudaDeviceSynchronize());
  clear_graph_history(sparse,stream);ck(cudaError_t(graph.run(a)));ck(cudaStreamSynchronize(stream));
  const auto empty=read_graph_output(sparse);const auto empty_history=read_graph_history(sparse);
  for(auto value:empty)if(__half2float(value)!=0.f)throw std::runtime_error("Empty partitioned attention changed zero output");
  for(int i=0;i<sparse.blocks;++i)if(empty_history.hits[i]!=0.f || empty_history.mass[i]!=0.f)
   throw std::runtime_error("Empty partitioned attention recorded history");
 }
 // Update one executor again while its preceding launch remains in flight.
 GraphFixture first(67,260,2,6,17,16),second(67,260,2,6,17,16);
 ck(cudaDeviceSynchronize());
 auto a=first.args(),b=second.args();a.stream=b.stream=stream;
 ck(cudaError_t(tq::tq_sparse_attention_prefill(a)));ck(cudaError_t(tq::tq_sparse_attention_prefill(b)));ck(cudaStreamSynchronize(stream));
 auto ea=read_graph_output(first),eb=read_graph_output(second);
 const auto ha=read_graph_history(first),hb=read_graph_history(second);
 clear_graph_history(first,stream);clear_graph_history(second,stream);
 ck(cudaMemsetAsync(first.output.p,0x7f,ea.size()*2,stream));ck(cudaMemsetAsync(second.output.p,0x7f,eb.size()*2,stream));
 ck(cudaError_t(graph.run(a)));ck(cudaError_t(graph.run(b)));ck(cudaStreamSynchronize(stream));assert_graph_equal(ea,read_graph_output(first));assert_graph_equal(eb,read_graph_output(second));
 assert_graph_history(ha,first);assert_graph_history(hb,second);
 cudaGraph_t outer;cudaGraphExec_t replay;
 ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));ck(cudaError_t(graph.run(a)));
 ck(cudaStreamEndCapture(stream,&outer));ck(cudaGraphInstantiate(&replay,outer,0));
 clear_graph_history(first,stream);
 ck(cudaGraphLaunch(replay,stream));ck(cudaStreamSynchronize(stream));assert_graph_equal(ea,read_graph_output(first));assert_graph_history(ha,first);
 ck(cudaGraphExecDestroy(replay));ck(cudaGraphDestroy(outer));ck(cudaStreamDestroy(stream));
 a.stream=nullptr;clear_graph_history(first,nullptr);ck(cudaError_t(graph.run(a)));ck(cudaDeviceSynchronize());
 assert_graph_equal(ea,read_graph_output(first));assert_graph_history(ha,first);
 std::cout<<"QSA graph exact parity PASS cases="<<cases<<"; shape/pointer updates, observer topology changes, in-flight updates and outer capture PASS"<<std::endl;
}
int main()try {
    check_materialized_gate();
    constexpr int queries=33,kvheads=2,count=16,padded=256;
    int cases=0;double maxrel=0,maxmass=0;
    for(int rows:{128,260,65536}) {
    const int blocks=rows/4;
    for(int ratio:{4,5,6,8,9,12})for(int width:{17,256})for(bool query_half:{false,true})for(bool output_half:{false,true}) {
        if(rows==65536 && (ratio!=6 || width!=17 || query_half))continue;
        const int heads=kvheads*ratio,parts=tq::tq_attention_partitions(queries,heads,kvheads,padded,count,query_half,true);
        std::vector<float> q(queries*heads*width),gate(q.size()),signs(padded),vsigns(padded);
        std::vector<half> qh(q.size()),gh(q.size()),k(rows*kvheads*padded),v(k.size());
        for(size_t i=0;i<q.size();++i){q[i]=.25f*std::sin(float(i)*.071f);qh[i]=__float2half(q[i]);gate[i]=.31f*std::cos(float(i)*.049f);gh[i]=__float2half(gate[i]);}
        for(int d=0;d<padded;++d){signs[d]=(d%3)?1.f:-1.f;vsigns[d]=(d%5)?-1.f:1.f;}
        for(size_t i=0;i<k.size();++i){k[i]=__float2half(.21f*std::cos(float(i)*.043f));v[i]=__float2half(.17f*std::sin(float(i)*.063f));}
        std::vector<tq::SelectedBlock> selected(queries*count);std::vector<int64_t> position(queries);
        for(int t=0;t<queries;++t){position[t]=rows-queries+t;for(int slot=0;slot<count;++slot)selected[t*count+slot]={(slot*17+t/8*3)%blocks,uint32_t((slot+t)%16)};}
        if(ratio==6 && width==17 && !query_half && !output_half)for(int t=0;t<8;++t)for(int slot=0;slot<count;++slot)selected[t*count+slot]={-1,0};
        Device<float> dq(q.size()),dg(q.size()),dsign(padded),dvsign(padded),old_float(q.size()),new_float(q.size()),rot(int64_t(queries)*heads*padded),partial(int64_t(queries)*heads*parts*padded),stats(int64_t(queries)*heads*parts*2),lm(int64_t(queries)*heads*count),nlm(int64_t(queries)*heads*count),ln(queries*heads),nln(queries*heads),oh(blocks),om(blocks),nh(blocks),nm(blocks);
        Device<half> dqh(q.size()),dgh(q.size()),dk(k.size()),dv(v.size()),old_half(q.size()),new_half(q.size());Device<tq::SelectedBlock> dsel(selected.size());Device<int64_t> dp(position.size());Device<tq::TqBlockView> views(blocks);Device<uint16_t> relative(1);
        dq.put(q);dg.put(gate);dqh.put(qh);dgh.put(gh);dk.put(k);dv.put(v);dsign.put(signs);dvsign.put(vsigns);dsel.put(selected);dp.put(position);
        ck(cudaMemset(oh.p,0,blocks*4));ck(cudaMemset(om.p,0,blocks*4));ck(cudaMemset(nh.p,0,blocks*4));ck(cudaMemset(nm.p,0,blocks*4));
        const void* query=query_half?static_cast<void*>(dqh.p):dq.p;
        const void* g=query_half?static_cast<void*>(dg.p):dgh.p;
        void* old_output=output_half?static_cast<void*>(old_half.p):old_float.p;
        void* new_output=output_half?static_cast<void*>(new_half.p):new_float.p;
        ck(cudaError_t(tq::tq_sparse_attention_staged(reinterpret_cast<uint16_t*>(dk.p),reinterpret_cast<uint16_t*>(dv.p),query,dsign.p,dvsign.p,relative.p,views.p,dsel.p,dp.p,old_output,rot.p,partial.p,stats.p,lm.p,ln.p,oh.p,om.p,queries,heads,kvheads,width,padded,count,parts,heads*width,width,nullptr,false,g,!query_half,output_half,heads*width,width,query_half,true,false)));
        const size_t temporary_bytes=size_t(queries)*heads*padded*4+size_t(queries)*rows*2+size_t(queries)*blocks*4+(size_t((queries+7)/8)*rows+(queries+7)/8)*4;
        Device<float> temporary((temporary_bytes+3)/4);
        tq::TqPrefillAttention a;a.keys=reinterpret_cast<uint16_t*>(dk.p);a.values=reinterpret_cast<uint16_t*>(dv.p);a.query=query;a.key_signs=dsign.p;a.value_signs=dvsign.p;
        a.selected=dsel.p;a.positions=dp.p;a.output=new_output;a.rotated=rot.p;a.temporary=temporary.p;a.temporary_bytes=temporary_bytes;a.logmass=nlm.p;a.norm=nln.p;a.hits=nh.p;a.mass=nm.p;
        a.queries=queries;a.heads=heads;a.kv_heads=kvheads;a.width=width;a.kv_rows=rows;a.count=count;a.query_token_stride=heads*width;a.query_head_stride=width;
        a.query_half=query_half;a.output_half=output_half;a.gate=g;a.gate_half=!query_half;a.gate_token_stride=heads*width;a.gate_head_stride=width;
        const auto saved_bytes=a.temporary_bytes;
        a.temporary_bytes=0;
        if(tq::tq_sparse_attention_prefill(a)!=int(cudaErrorNotSupported))throw std::runtime_error("Insufficient scratch accepted");
        a.temporary_bytes=saved_bytes;
        const auto status=cudaError_t(tq::tq_sparse_attention_prefill(a));
        if(status==cudaErrorNotSupported && cases==0){std::cout<<"SKIP: sparse prefill requires an SM80 kernel image and sufficient dynamic shared memory"<<std::endl;return 77;}
        ck(status);ck(cudaDeviceSynchronize());
        std::vector<float> actual(q.size()),expected(q.size());
        if(output_half){std::vector<half> x(q.size()),y(q.size());ck(cudaMemcpy(x.data(),new_half.p,x.size()*2,cudaMemcpyDeviceToHost));ck(cudaMemcpy(y.data(),old_half.p,y.size()*2,cudaMemcpyDeviceToHost));for(size_t i=0;i<x.size();++i){actual[i]=__half2float(x[i]);expected[i]=__half2float(y[i]);}}
        else{ck(cudaMemcpy(actual.data(),new_float.p,actual.size()*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(expected.data(),old_float.p,expected.size()*4,cudaMemcpyDeviceToHost));}
        double e=0,n=0;for(size_t i=0;i<actual.size();++i){if(!std::isfinite(actual[i]))throw std::runtime_error("Nonfinite full adapter output");const double d=actual[i]-expected[i];e+=d*d;n+=double(expected[i])*expected[i];}
        const double rel=std::sqrt(e/std::max(n,1.e-30));maxrel=std::max(maxrel,rel);
        std::vector<float> xh(blocks),yh(blocks),xm(blocks),ym(blocks);ck(cudaMemcpy(xh.data(),nh.p,blocks*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(yh.data(),oh.p,blocks*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(xm.data(),nm.p,blocks*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(ym.data(),om.p,blocks*4,cudaMemcpyDeviceToHost));
        double me=0,mn=0;for(int b=0;b<blocks;++b){if(xh[b]!=yh[b])throw std::runtime_error("GQA adapter changed block hits");me+=double(xm[b]-ym[b])*(xm[b]-ym[b]);mn+=double(ym[b])*ym[b];}
        const double massrel=std::sqrt(me/std::max(mn,1.e-30));maxmass=std::max(maxmass,massrel);
        if(rel>.02 || massrel>.01){std::cerr<<"ratio="<<ratio<<" width="<<width<<" qhalf="<<query_half<<" ohalf="<<output_half<<" rel="<<rel<<" mass="<<massrel<<'\n';throw std::runtime_error("Full adapter GQA/codec-space mismatch");}
        ++cases;
        if(std::getenv("TYLO_TEST_PROGRESS"))std::cerr<<"completed_case="<<cases<<'\n';
    }
    }
    check_prefill_graph();
    std::cout<<"Generic GQA tails, packed KV/masks, signs/RHT, FP16/FP32, sigmoid gate and empty query tile PASS cases="<<cases<<" max_rel="<<maxrel<<" history_rel="<<maxmass<<std::endl;return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
