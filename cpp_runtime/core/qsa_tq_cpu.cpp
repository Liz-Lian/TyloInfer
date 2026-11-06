#include "tyloinfer/qsa_tq_cpu.h"
#include "tyloinfer/qsa_energy_allocation.h"
#include "ggml.h"
#include "tyloinfer/host_parallel.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>

namespace tyloinfer::qsa {
namespace {
const std::array<float,65536>& book() {
    static const auto values=[] {
        std::array<float,65536> result{};
        const float gain=ggml_fp16_to_fp32(0x1eee),bias=ggml_fp16_to_fp32(0xc931);
        for(unsigned state=0;state<65536;++state) {
            const std::uint32_t p=state*0x83dcd12du;
            const auto sum=(p&255)+((p>>8)&255)+((p>>16)&255)+(p>>24);
            result[state]=ggml_fp16_to_fp32(ggml_fp32_to_fp16(float(sum+1024)*gain+bias));
        }
        return result;
    }();
    return values;
}
float relative(int selector) {
    return ggml_fp16_to_fp32(ggml_fp32_to_fp16(std::exp2(-2.f+4.f*selector/15.f)));
}
struct Path {float cost;std::uint16_t state,parent;};
struct Scratch {
    std::array<std::uint16_t,65536> tags{},indices{};
    std::uint16_t epoch=0;
    std::vector<Path> beam,next,candidates;
    std::vector<std::uint16_t> parent,state;
};
thread_local Scratch scratch;
}

CpuTqRow fit_cpu_tq_row(const float* input,int bits,int budget,int state_bits) {
    if(!input || bits<1 || bits>8 || budget<2 || state_bits<8 || state_bits>16)
        throw std::invalid_argument("invalid online CPU TQ row fitting parameters");
    const unsigned state_mask=(1u<<state_bits)-1;
    CpuTqRow result;result.bits=std::uint8_t(bits);result.state_bits=std::uint8_t(state_bits);result.payload.resize(32*bits);
    double energy=0;
    for(int i=0;i<256;++i) {
        if(!std::isfinite(input[i]))throw std::invalid_argument("non-finite online TQ input");
        energy+=double(input[i])*input[i];
    }
    if(!energy)return result;
    const int branches=1<<bits,beam_width=std::max(8,std::min(256,budget/branches));
    auto& s=scratch;
    s.beam.clear();s.beam.push_back({0,0,0});s.next.reserve(beam_width);
    s.parent.resize(256*beam_width);s.state.resize(256*beam_width);
    s.candidates.reserve(beam_width*branches);
    const auto& codebook=book();
    const float gain=relative(7);
    const float initial=std::sqrt(float(energy/256))*1.1f;
    const auto initial_anchor=ggml_fp32_to_fp16(std::max(initial/gain,0x1p-24f));
    if((initial_anchor&0x7fff)>0x7bff)throw std::invalid_argument("online TQ scale exceeds FP16 range");
    const float scale=ggml_fp16_to_fp32(initial_anchor)*gain;
    const auto less=[](const Path& a,const Path& b) {return a.cost!=b.cost?a.cost<b.cost:a.state<b.state;};
    for(int step=0;step<288;++step) {
        const int t=step<32?224+step:step-32;
        if(step==32)for(auto& path:s.beam)path.cost=0;
        if(++s.epoch==0) {s.tags.fill(0);s.epoch=1;}
        s.candidates.clear();
        const float target=input[t]/scale;
        for(std::size_t previous=0;previous<s.beam.size();++previous) {
            const auto path=s.beam[previous];
            for(int symbol=0;symbol<branches;++symbol) {
                const auto state=std::uint16_t(((unsigned(path.state)<<bits)|symbol)&state_mask);
                const float delta=codebook[state]-target,cost=path.cost+delta*delta;
                if(s.tags[state]!=s.epoch) {
                    s.tags[state]=s.epoch;s.indices[state]=std::uint16_t(s.candidates.size());
                    s.candidates.push_back({cost,state,std::uint16_t(previous)});
                } else {
                    auto& candidate=s.candidates[s.indices[state]];
                    if(cost<candidate.cost)candidate={cost,state,std::uint16_t(previous)};
                }
            }
        }
        const auto count=std::min<std::size_t>(beam_width,s.candidates.size());
        if(count<s.candidates.size())std::nth_element(s.candidates.begin(),s.candidates.begin()+count,s.candidates.end(),less);
        s.next.assign(s.candidates.begin(),s.candidates.begin()+count);
        if(step>=32)for(std::size_t i=0;i<count;++i) {
            s.parent[(step-32)*beam_width+i]=s.next[i].parent;
            s.state[(step-32)*beam_width+i]=s.next[i].state;
        }
        s.beam.swap(s.next);
    }
    double best=std::numeric_limits<double>::infinity();
    std::array<std::uint8_t,256> symbols{},best_symbols{};
    std::array<float,256> decoded{};
    for(std::size_t end=0;end<s.beam.size();++end) {
        auto path=end;
        for(int t=255;t>=0;--t) {
            symbols[t]=std::uint8_t(s.state[t*beam_width+path]&(branches-1));
            path=s.parent[t*beam_width+path];
        }
        std::uint16_t state=0;
        for(int t=256-(16+bits-1)/bits;t<256;++t)state=std::uint16_t((unsigned(state)<<bits)|symbols[t]);
        double numerator=0,denominator=0;
        for(int t=0;t<256;++t) {
            state=std::uint16_t(((unsigned(state)<<bits)|symbols[t])&state_mask);decoded[t]=codebook[state];
            numerator+=double(input[t])*decoded[t];denominator+=double(decoded[t])*decoded[t];
        }
        const auto anchor=ggml_fp32_to_fp16(std::max(0.f,float(numerator/(denominator*gain))));
        const float fitted_scale=ggml_fp16_to_fp32(anchor)*gain;
        double loss=0;
        for(int t=0;t<256;++t) {
            const auto y=ggml_fp16_to_fp32(ggml_fp32_to_fp16(decoded[t]*fitted_scale));
            const double error=double(input[t])-y;loss+=error*error;
        }
        if(loss<best){best=loss;result.anchor=anchor;best_symbols=symbols;}
    }
    for(int i=0;i<256;++i) {
        const unsigned symbol=best_symbols[255-i];
        const int at=i*bits,byte=at/8,shift=at&7;
        result.payload[byte]|=std::uint8_t(symbol<<shift);
        if(shift+bits>8)result.payload[byte+1]|=std::uint8_t(symbol>>(8-shift));
    }
    result.sse=best;return result;
}

std::vector<float> decode_cpu_tq_row(const CpuTqRow& row) {
    if(row.bits<1 || row.bits>8 || row.selector>15 || row.payload.size()!=32*row.bits || row.state_bits<8 || row.state_bits>16)
        throw std::invalid_argument("invalid CPU TQ row image");
    const auto& values=book();
    const float scale=ggml_fp16_to_fp32(row.anchor)*relative(row.selector);
    std::vector<float> output(256);
    for(int i=0;i<256;++i) {
        const int bit=(255-i)*row.bits,byte=bit/8,shift=bit&7;
        const unsigned word=row.payload[byte]|unsigned(row.payload[(byte+1)%row.payload.size()])<<8|
            unsigned(row.payload[(byte+2)%row.payload.size()])<<16;
        output[i]=ggml_fp16_to_fp32(ggml_fp32_to_fp16(values[(word>>shift)&((1u<<row.state_bits)-1)]*scale));
    }
    return output;
}

struct CpuTqBlockFitter::Impl {
    MicroblockImage canonical;
    std::size_t rows=0;
    int workers=1;
    int state_bits=16;
    std::array<std::vector<float>,2> decoded;
    std::array<std::vector<double>,2> energy;
    std::array<std::vector<CpuTqRow>,2> fitted;
    std::array<int,2> fitted_bits{};
    std::array<std::uint8_t,2> packed_tags{};
    RowPrecisionAllocation allocation;
    mutable std::mutex mutex;
    Impl(const MicroblockImage& source,std::size_t count,int worker_count,RowPrecisionAllocation policy,int profile)
        :canonical(source),rows(count),workers(worker_count),state_bits(profile),allocation(policy) {
        if(!rows || rows%4 || rows>std::numeric_limits<std::size_t>::max()/520 ||
           workers<1 || profile<8 || profile>16 || source.key_state_bits!=16 || source.value_state_bits!=16 || source.key_bits!=source.value_bits ||
           (source.key_bits!=0 && source.key_bits!=8 && source.key_bits!=affine8_tag))
            throw std::invalid_argument("CPU QSA re-quantization requires FP16, Affine8 or canonical TQ8 modes");
        const bool fp16=source.key_bits==0;
        const bool affine=source.key_bits==affine8_tag;
        const auto metadata=fp16?0:affine?((3*rows+3)&~std::size_t(3)):((2*rows+(rows+1)/2+3)&~std::size_t(3));
        for(int kind=0;kind<2;++kind) {
            const auto& input=kind?source.value:source.key;
            if(input.size()!=metadata+rows*(fp16?512:256))throw std::invalid_argument("invalid canonical microblock geometry");
            decoded[kind].reserve(rows*256);
            for(std::size_t row=0;row<rows;++row) {
                if(fp16) {
                    for(std::size_t d=0;d<256;++d) {
                        const auto at=2*(row*256+d);
                        decoded[kind].push_back(ggml_fp16_to_fp32(input[at]|std::uint16_t(input[at+1])<<8));
                    }
                    continue;
                }
                if(affine) {
                    const float scale=ggml_fp16_to_fp32(input[row*2]|std::uint16_t(input[row*2+1])<<8);
                    const auto zero=std::int8_t(input[rows*2+row]);
                    for(std::size_t d=0;d<256;++d)
                        decoded[kind].push_back(float(int(std::int8_t(input[metadata+row*256+d]))-int(zero))*scale);
                    continue;
                }
                CpuTqRow c;c.bits=8;c.anchor=input[row*2]|std::uint16_t(input[row*2+1])<<8;
                c.selector=(input[2*rows+row/2]>>((row&1)*4))&15;
                c.payload.assign(input.begin()+metadata+row*256,input.begin()+metadata+(row+1)*256);
                const auto x=decode_cpu_tq_row(c);decoded[kind].insert(decoded[kind].end(),x.begin(),x.end());
            }
            energy[kind].resize(rows);
            for(std::size_t row=0;row<rows;++row)for(std::size_t d=0;d<256;++d) {
                const auto x=double(decoded[kind][row*256+d]);
                if(!std::isfinite(x))throw std::invalid_argument("non-finite canonical QSA modes");
                energy[kind][row]+=x*x;
            }
        }
    }
    std::vector<std::uint8_t> pack(int kind,int budget_bits) {
        if(budget_bits==8 && canonical.key_bits==0) {
            // Affine8 uses the GPU encoder's upward-rounded FP16 step and
            // includes zero in each row's range. No trellis fit is needed.
            const auto header=(3*rows+3)&~std::size_t(3);
            std::vector<std::uint8_t> output(header+rows*256);
            for(std::size_t row=0;row<rows;++row) {
                const auto* x=decoded[kind].data()+row*256;
                float lo=0,hi=0;
                for(int d=0;d<256;++d){lo=std::min(lo,x[d]);hi=std::max(hi,x[d]);}
                const float step=std::max((hi-lo)/255.f,0x1p-24f);
                auto encoded=ggml_fp32_to_fp16(step);
                if(ggml_fp16_to_fp32(encoded)<step)++encoded;
                const float scale=ggml_fp16_to_fp32(encoded);
                const int zero=std::max(-128,std::min(127,-128-int(std::nearbyint(lo/scale))));
                output[2*row]=std::uint8_t(encoded);output[2*row+1]=std::uint8_t(encoded>>8);
                output[2*rows+row]=std::uint8_t(std::int8_t(zero));
                for(int d=0;d<256;++d)output[header+row*256+d]=std::uint8_t(std::int8_t(
                    std::max(-128,std::min(127,int(std::nearbyint(x[d]/scale))+zero))));
            }
            packed_tags[kind]=affine8_tag;return output;
        }
        if(budget_bits==8){packed_tags[kind]=kind?canonical.value_bits:canonical.key_bits;return kind?canonical.value:canonical.key;}
        if(budget_bits<1 || budget_bits>7)throw std::invalid_argument("invalid CPU QSA target precision");
        auto bits=allocation==RowPrecisionAllocation::Energy?allocate_energy_bits(energy[kind],budget_bits):
            std::vector<std::uint8_t>(rows,std::uint8_t(budget_bits));
        if(fitted_bits[kind]!=budget_bits) {
            // Fit exactly one energy-assigned width per row. Keep at most
            // one encoded row in the memo, never a bank of candidate widths.
            fitted_bits[kind]=0;fitted[kind].resize(rows);
            mfq::host_parallel_for(0,std::int64_t(rows),1,workers,[&](std::int64_t begin,std::int64_t end) {
                for(auto row=begin;row<end;++row) {
                    auto& target=fitted[kind][row];
                    if(target.bits!=bits[row] || target.payload.size()!=32*bits[row])
                        target=fit_cpu_tq_row(decoded[kind].data()+row*256,bits[row],1024,state_bits);
                }
            });
            fitted_bits[kind]=budget_bits;
        }
        const bool mixed=std::any_of(bits.begin(),bits.end(),[&](auto bit){return bit!=budget_bits;});
        packed_tags[kind]=mixed?255:std::uint8_t(budget_bits);
        const auto map=2*rows+(rows+1)/2;
        const auto header=(map+(mixed?(3*rows+7)/8:0)+3)&~std::size_t(3);
        std::vector<std::uint8_t> output(header);
        for(std::size_t row=0;row<rows;++row) {
            const auto& c=fitted[kind][row];
            output[2*row]=std::uint8_t(c.anchor);output[2*row+1]=std::uint8_t(c.anchor>>8);
            output[2*rows+row/2]|=c.selector<<((row&1)*4);
            if(mixed) {
                const auto at=row*3,shift=at&7;
                output[map+at/8]|=std::uint8_t((bits[row]-1)<<shift);
                if(shift>5)output[map+at/8+1]|=std::uint8_t((bits[row]-1)>>(8-shift));
            }
            output.insert(output.end(),c.payload.begin(),c.payload.end());
        }
        return output;
    }
};
CpuTqBlockFitter::CpuTqBlockFitter(const MicroblockImage& canonical,std::size_t rows,int workers,RowPrecisionAllocation policy,int state_bits)
    :impl_(std::make_unique<Impl>(canonical,rows,workers,policy,state_bits)) {}
CpuTqBlockFitter::~CpuTqBlockFitter()=default;
MicroblockImage CpuTqBlockFitter::image(Precision p) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    MicroblockImage result;result.key=impl_->pack(0,p.key_bits);result.value=impl_->pack(1,p.value_bits);
    result.key_bits=impl_->packed_tags[0];result.value_bits=impl_->packed_tags[1];
    result.key_state_bits=p.key_bits==8?16:std::uint8_t(impl_->state_bits);
    result.value_state_bits=p.value_bits==8?16:std::uint8_t(impl_->state_bits);
    return result;
}
std::size_t CpuTqBlockFitter::bytes() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    std::size_t bytes=impl_->canonical.key.size()+impl_->canonical.value.size();
    for(const auto& kind:impl_->decoded)bytes+=kind.size()*sizeof(float);
    for(const auto& kind:impl_->energy)bytes+=kind.size()*sizeof(double);
    for(const auto& kind:impl_->fitted)for(const auto& row:kind)bytes+=sizeof(row)+row.payload.size();
    return bytes;
}
} // namespace tyloinfer::qsa
