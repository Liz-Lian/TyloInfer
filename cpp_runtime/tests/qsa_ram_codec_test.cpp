#include "tyloinfer/qsa_tq_cpu.h"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>
using namespace tyloinfer::qsa;
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main()try {
    constexpr std::size_t rows=4,header=12;
    MicroblockImage source;source.key_bits=source.value_bits=0;
    std::vector<float> expected(rows*256);
    for(auto* data:{&source.key,&source.value}) {
        data->resize(rows*512);
        for(std::size_t r=0;r<rows;++r)for(std::size_t d=0;d<256;++d) {
            const float x=r==0?0:r==1?1.f+float(d)/1024:r==2?-2.f-float(d)/512:
                std::sin(float(d)*.073f)*.7f;
            const auto h=ggml_fp32_to_fp16(x);const auto at=2*(r*256+d);
            (*data)[at]=std::uint8_t(h);(*data)[at+1]=std::uint8_t(h>>8);
            expected[r*256+d]=ggml_fp16_to_fp32(h);
        }
    }
    CpuTqBlockFitter first(source,rows,2,RowPrecisionAllocation::Uniform);
    const auto before=first.bytes();const auto affine=first.image({8,8});
    require(first.bytes()==before,"Affine8 encoding retained a trellis fit");
    require(affine.key_bits==affine8_tag && affine.value_bits==affine8_tag,"FP16 was not demoted to Affine8");
    for(const auto* data:{&affine.key,&affine.value}) {
        require(data->size()==header+rows*256,"Affine8 row storage differs");
        double error=0,energy=0;
        for(std::size_t r=0;r<rows;++r) {
            const float scale=ggml_fp16_to_fp32((*data)[2*r]|std::uint16_t((*data)[2*r+1])<<8);
            const int zero=std::int8_t((*data)[2*rows+r]);
            for(std::size_t d=0;d<256;++d) {
                const float x=expected[r*256+d],y=(int(std::int8_t((*data)[header+r*256+d]))-zero)*scale;
                require(std::isfinite(y) && std::abs(y-x)<=scale*.5001f,"Affine8 exceeded its scalar rounding error bound");
                error+=double(y-x)*(y-x);energy+=double(x)*x;
                if(!r)require(y==0,"A zero H4 mode became nonzero");
            }
        }
        require(error/energy<1.e-4,"FP16 to Affine8 reconstruction loss is excessive");
    }
    CpuTqBlockFitter second(affine,rows,2,RowPrecisionAllocation::Uniform,14);
    const auto low=second.image({4,4});
    require(low.key_bits==4 && low.value_bits==4 && low.key_state_bits==14 && low.value_state_bits==14,
        "TQ4 demotion changed its packed precision/profile");
    for(const auto* data:{&low.key,&low.value}) {
        require(data->size()==header+rows*128 && data->size()<affine.key.size(),"TQ4 did not reduce storage");
        double error=0,energy=0;
        for(std::size_t r=0;r<rows;++r) {
            CpuTqRow row;row.bits=4;row.state_bits=14;
            row.anchor=(*data)[2*r]|std::uint16_t((*data)[2*r+1])<<8;
            row.selector=((*data)[2*rows+r/2]>>((r&1)*4))&15;
            row.payload.assign(data->begin()+header+r*128,data->begin()+header+(r+1)*128);
            const auto decoded=decode_cpu_tq_row(row);
            for(std::size_t d=0;d<256;++d){const auto x=expected[r*256+d],y=decoded[d];
                require(std::isfinite(y),"TQ4 reconstruction is non-finite");
                error+=double(y-x)*(y-x);energy+=double(x)*x;}
        }
        require(error/energy<.05,"Chained FP16/Affine8/TQ4 reconstruction loss is excessive");
    }
    auto invalid=source;invalid.key[0]=0;invalid.key[1]=0x7e;
    bool rejected=false;try{CpuTqBlockFitter bad(invalid,rows,1);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"A non-finite FP16 source was accepted");
    std::puts("QSA CPU FP16 -> Affine8 -> TQ4 storage, scalar error bound, zero rows and finite-input validation PASS");
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
