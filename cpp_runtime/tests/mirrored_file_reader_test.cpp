#include "tyloinfer/mirrored_file_reader.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void require(bool good,const char* message){if(!good)throw std::runtime_error(message);}
struct Setting {
    std::string old;bool present=false;
    explicit Setting(const std::string& value) {
        if(const auto* p=std::getenv("TYLOINFER_MODEL_MIRROR_DIRS")){present=true;old=p;}set(value.c_str());
    }
    static void set(const char* value) {
#if defined(_WIN32)
        _putenv_s("TYLOINFER_MODEL_MIRROR_DIRS",value?value:"");
#else
        if(value)setenv("TYLOINFER_MODEL_MIRROR_DIRS",value,1);else unsetenv("TYLOINFER_MODEL_MIRROR_DIRS");
#endif
    }
    ~Setting(){set(present?old.c_str():nullptr);}
};
void write(const std::filesystem::path& path,const std::vector<std::byte>& data) {
    std::ofstream file(path,std::ios::binary);file.write(reinterpret_cast<const char*>(data.data()),data.size());
    require(bool(file),"cannot write mirror fixture");
}
void exercise(const std::filesystem::path& primary,const std::filesystem::path& mirror,
    const std::vector<std::byte>& expected,mfq::FileReadMode mode) {
    mfq::MirroredFileReader reader({primary,mirror},mode,512);
    const auto before=reader.replica_stats();
    std::vector<std::vector<std::byte>> outputs;outputs.reserve(128);
    std::vector<mfq::ReadSpan> spans;
    for(int i=0;i<128;++i) {
        const auto offset=std::size_t(13+i*7919)%(expected.size()-2048);
        const auto bytes=31+(i*7)%513;
        outputs.emplace_back(bytes+2,std::byte{0x5a});spans.push_back({offset,outputs.back().data()+1,std::size_t(bytes)});
    }
    reader.read_coalesced(spans);
    for(std::size_t i=0;i<spans.size();++i) {
        require(outputs[i].front()==std::byte{0x5a} && outputs[i].back()==std::byte{0x5a},"mirror read changed guards");
        require(std::equal(outputs[i].begin()+1,outputs[i].end()-1,expected.begin()+spans[i].offset),"mirror batch changed bytes");
    }
    const auto after=reader.replica_stats();
    require(after.size()==2,"mirror count differs");
    for(int i=0;i<2;++i)require(after[i].logical_bytes>before[i].logical_bytes,"batch did not read both replicas");
    std::array<std::byte,17> tail{};reader.read(expected.size()-tail.size(),tail.data(),tail.size());
    require(std::equal(tail.begin(),tail.end(),expected.end()-tail.size()),"mirror EOF bytes changed");
    std::byte guard{0x5a};bool rejected=false;
    try{reader.read_coalesced({{0,&guard,1},{expected.size(),&guard,1}});}catch(const std::out_of_range&){rejected=true;}
    require(rejected && guard==std::byte{0x5a},"mirror batch validated bounds after writing earlier output");
    reader.read_coalesced({{expected.size(),nullptr,0}});
    reader.read(expected.size(),nullptr,0);
}
}
int main() {
    const auto parent=std::filesystem::current_path();
    const auto directory=parent/("mfq-mirror-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    if(!std::filesystem::create_directory(directory))return 1;
    struct Cleanup {
        std::filesystem::path path,parent;
        ~Cleanup(){if(path.parent_path()==parent && path.filename().string().rfind("mfq-mirror-test-",0)==0){std::error_code ignored;std::filesystem::remove_all(path,ignored);}}
    } cleanup{directory,parent};
    try {
        const auto original=directory/"primary",copy=directory/"copy",partial=directory/"partial";
        std::filesystem::create_directory(original);std::filesystem::create_directory(copy);std::filesystem::create_directory(partial);
        const auto primary=original/"weights.gguf",mirror=copy/primary.filename();
        std::vector<std::byte> expected((1u<<20)+333);
        for(std::size_t i=0;i<expected.size();++i)expected[i]=std::byte((i*37+i/257)&255);
        write(primary,expected);write(mirror,expected);
        for(auto mode:{mfq::FileReadMode::Buffered,mfq::FileReadMode::Direct})exercise(primary,mirror,expected,mode);
        {
            Setting setting(copy.string()+";"+partial.string()+";"+original.string());
            require(mfq::model_read_replica_paths(primary).size()==2,"mirror directories did not skip missing/duplicate shards");
        }
        bool rejected=false;
        try{mfq::MirroredFileReader invalid({primary,primary},mfq::FileReadMode::Buffered,512);}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"duplicate mirror accepted");
        expected[77]^=std::byte{1};write(mirror,expected);expected[77]^=std::byte{1};rejected=false;
        try{mfq::MirroredFileReader invalid({primary,mirror},mfq::FileReadMode::Buffered,512);}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"different mirror header accepted");
        expected.resize(129);write(mirror,expected);rejected=false;
        try{mfq::MirroredFileReader invalid({primary,mirror},mfq::FileReadMode::Buffered,512);}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"truncated mirror accepted");
        std::cout<<"Mirror buffered/direct bytes, independent batches, bounds, headers, duplicates and partial directories PASS\n";return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
