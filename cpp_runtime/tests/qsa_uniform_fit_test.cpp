#include "tyloinfer/qsa_tq_cpu.h"
#include "tyloinfer/qsa_energy_allocation.h"
#include "ggml.h"
#include <cassert>
#include <cmath>
#include <iostream>

using namespace tyloinfer::qsa;
int main() {
    constexpr std::size_t rows=4,affine_header=12;
    MicroblockImage source;
    for(auto* bytes:{&source.key,&source.value}) {
        bytes->resize(affine_header+rows*256);
        for(std::size_t row=0;row<rows;++row) {
            const auto scale=ggml_fp32_to_fp16(.01f);
            (*bytes)[2*row]=std::uint8_t(scale);(*bytes)[2*row+1]=std::uint8_t(scale>>8);
            for(std::size_t d=0;d<256;++d)
                (*bytes)[affine_header+row*256+d]=row?std::uint8_t(std::int8_t(int((d*17+row*13)%201)-100)):0;
        }
    }
    CpuTqBlockFitter fitter(source,rows,2,RowPrecisionAllocation::Uniform);
    const auto before=fitter.bytes();
    const auto canonical=fitter.image({8,8});
    assert(canonical.key==source.key && canonical.value==source.value && fitter.bytes()==before);
    const auto first=fitter.image({3,4});
    assert(first.key_bits==3 && first.value_bits==4);
    // Only the requested widths have encoded rows; no retained width sweep.
    assert(fitter.bytes()-before==rows*(2*sizeof(CpuTqRow)+32*(3+4)));
    for(auto pair:{std::make_pair(&first.key,3),std::make_pair(&first.value,4)}) {
        const auto& image=*pair.first;
        assert(image.size()==12+rows*32*pair.second);
        for(std::size_t row=0;row<rows;++row) {
            CpuTqRow encoded;encoded.bits=std::uint8_t(pair.second);
            encoded.anchor=image[2*row]|std::uint16_t(image[2*row+1])<<8;
            encoded.selector=(image[8+row/2]>>((row&1)*4))&15;
            const auto begin=image.begin()+12+row*32*pair.second;
            encoded.payload.assign(begin,begin+32*pair.second);
            const auto decoded=decode_cpu_tq_row(encoded);
            double error=0,power=0;
            for(std::size_t d=0;d<256;++d) {
                assert(std::isfinite(decoded[d]));
                const auto x=row?(int((d*17+row*13)%201)-100)*ggml_fp16_to_fp32(ggml_fp32_to_fp16(.01f)):0;
                error+=(decoded[d]-x)*(decoded[d]-x);power+=x*x;
            }
            if(!row)assert(error==0);else assert(error<power);
        }
    }
    const auto lower=fitter.image({2,3});
    assert(lower.key_bits==2 && lower.value_bits==3);
    assert(fitter.bytes()-before==rows*(2*sizeof(CpuTqRow)+32*(2+3)));
    const auto repeated=fitter.image({3,4});
    assert(repeated.key==first.key && repeated.value==first.value);
    assert(allocate_energy_bits({1,1,1,1},3)==std::vector<std::uint8_t>({3,3,3,3}));
    auto allocation=allocate_energy_bits({0,1,16,256},3);
    assert(allocation[0]<=allocation[1] && allocation[1]<=allocation[2] && allocation[2]<=allocation[3]);
    assert(allocation[0]+allocation[1]+allocation[2]+allocation[3]==12);
    assert(allocate_energy_bits({0,0,0,0},1)==std::vector<std::uint8_t>({1,1,1,1}));
    assert(allocate_energy_bits({0,0,0,0},8)==std::vector<std::uint8_t>({8,8,8,8}));
    const std::array<double,4> power{{0,1,16,256}};
    double predicted=0;
    for(int i=0;i<4;++i)predicted+=power[i]*std::exp2(-2*allocation[i]);
    // Exhaustive independent oracle for the cheap allocator's proxy objective.
    for(int a=1;a<=8;++a)for(int b=1;b<=8;++b)for(int c=1;c<=8;++c)for(int d=1;d<=8;++d)
        if(a+b+c+d<=12)assert(predicted<=power[0]*std::exp2(-2*a)+power[1]*std::exp2(-2*b)+
            power[2]*std::exp2(-2*c)+power[3]*std::exp2(-2*d));
    CpuTqBlockFitter energy_fitter(source,rows,2);
    const auto energy_before=energy_fitter.bytes();
    const auto dynamic=energy_fitter.image({3,4});
    assert(dynamic.key_bits==255 && dynamic.value_bits==255);
    // Only one fitted precision per row is retained under energy allocation.
    assert(energy_fitter.bytes()-energy_before==rows*(2*sizeof(CpuTqRow)+32*(3+4)));
    for(const auto* data:{&dynamic.key,&dynamic.value}) {
        unsigned sum=0;
        for(std::size_t row=0;row<rows;++row) {
            const auto at=row*3,shift=at&7;
            unsigned code=(*data)[10+at/8]>>shift;
            if(shift>5)code|=unsigned((*data)[10+at/8+1])<<(8-shift);
            const auto bit=1+(code&7);sum+=bit;
            if(row==0)assert(bit==1);
        }
        assert(sum==rows*(data==&dynamic.key?3:4));
    }
    std::array<float,256> row{};
    for(int i=0;i<256;++i)row[i]=std::sin(i*.37f)+.2f*std::cos(i*.11f);
    const auto fast_row=fit_cpu_tq_row(row.data(),3,1024,14);
    assert(fast_row.state_bits==14 && fast_row.payload.size()==96);
    const auto decoded_fast=decode_cpu_tq_row(fast_row);
    double actual=0;
    for(int i=0;i<256;++i)actual+=double(decoded_fast[i]-row[i])*(decoded_fast[i]-row[i]);
    assert(std::abs(actual-fast_row.sse)<1e-5);
    auto wrong_profile=fast_row;wrong_profile.state_bits=16;
    assert(decode_cpu_tq_row(wrong_profile)!=decoded_fast);
    CpuTqBlockFitter fast_fitter(source,rows,2,RowPrecisionAllocation::Energy,14);
    const auto fast_image=fast_fitter.image({3,4});
    assert(fast_image.key_state_bits==14 && fast_image.value_state_bits==14);
    const auto restored=fast_fitter.image({8,8});
    assert(restored.key_state_bits==16 && restored.value_state_bits==16);
    assert(restored.key==source.key && restored.value==source.value);
    bool invalid=false;
    try{fit_cpu_tq_row(row.data(),3,1024,17);}catch(const std::invalid_argument&){invalid=true;}
    assert(invalid);
    std::cout<<"QSA energy allocation, TQ14/TQ16 profiles, compact layout and canonical restoration PASS\n";
}
