#pragma once
#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint>

// Valid compressed rows independent of either fitter. No model or GPU shape
// restrictions are inferred from these small correctness fixtures.
class RowResidualFixture {
    std::filesystem::path path_;
public:
    RowResidualFixture(std::uint64_t rows,std::uint64_t columns) {
        const auto stamp=std::chrono::steady_clock::now().time_since_epoch().count();
        path_=std::filesystem::current_path()/ ("row-residual-test-"+std::to_string(stamp)+".tqres");
        if(std::filesystem::exists(path_))throw std::runtime_error("residual fixture already exists");
        std::vector<std::uint8_t> data,descriptors;
        const auto append=[](std::vector<std::uint8_t>& out,std::uint64_t value,int count) {
            for(int i=0;i<count;++i)out.push_back(std::uint8_t(value>>(8*i)));
        };
        constexpr std::uint32_t formats[]={144,151,150,145,146,147,148,149,0,0};
        for(std::uint64_t row=0;row<rows;++row) {
            const auto type=formats[row%10];
            const auto bits=type==144?8:type==151?4:(type==145 || type==148)?6:(type==146 || type==149)?7:5;
            const auto steps=type>=147 && type<=149?64:32;
            const auto block=2+bits*steps/8;
            const auto bytes=type?(columns+127)/128*block:0;
            append(descriptors,data.size(),8);append(descriptors,type,4);append(descriptors,bytes,4);
            for(std::uint64_t at=0;at<bytes;++at) {
                const auto offset=at%block;
                // Small finite FP16 scale 0x2000; remaining bytes exercise the ring wrap.
                data.push_back(offset==0?0:offset==1?0x20:std::uint8_t(at*29+row*17+3));
            }
        }
        std::vector<std::uint8_t> header{'T','Q','R','E','S','0','0','1'};
        append(header,1,4);append(header,0,4);append(header,rows,8);append(header,columns,8);
        append(header,(columns+127)/128*128,8);append(header,data.size(),8);
        std::ofstream output(path_,std::ios::binary);
        output.write(reinterpret_cast<const char*>(header.data()),header.size());
        output.write(reinterpret_cast<const char*>(descriptors.data()),descriptors.size());
        output.write(reinterpret_cast<const char*>(data.data()),data.size());
        if(!output)throw std::runtime_error("write row residual fixture failed");
    }
    ~RowResidualFixture() { std::error_code error;std::filesystem::remove(path_,error); }
    RowResidualFixture(const RowResidualFixture&)=delete;
    RowResidualFixture& operator=(const RowResidualFixture&)=delete;
    const std::filesystem::path& path() const noexcept { return path_; }
};
