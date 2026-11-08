#include "../backends/cuda/apps/cuda_startup.h"
#include <iostream>
#include <optional>
#include <string>

namespace {
void set_mode(const char* value) {
#ifdef _WIN32
    if(_putenv_s("CUDA_MODULE_LOADING",value?value:""))throw std::runtime_error("Cannot change test environment");
#else
    if(value?setenv("CUDA_MODULE_LOADING",value,1):unsetenv("CUDA_MODULE_LOADING"))
        throw std::runtime_error("Cannot change test environment");
#endif
}
void expect(const char* mode) {
    mfq::cuda::configure_cuda_startup();
    const auto* value=std::getenv("CUDA_MODULE_LOADING");
    if(!value || std::string(value)!=mode)throw std::runtime_error("CUDA startup changed the module loading policy unexpectedly");
}
}
int main()try {
    const auto* current=std::getenv("CUDA_MODULE_LOADING");
    const auto saved=current?std::optional<std::string>(current):std::nullopt;
    struct Restore {const std::optional<std::string>& value;~Restore(){try{set_mode(value?value->c_str():nullptr);}catch(...){}}} restore{saved};
    set_mode(nullptr);expect("LAZY");expect("LAZY");
    set_mode("EAGER");expect("EAGER");
    set_mode("LAZY");expect("LAZY");
    std::cout<<"CUDA startup default and explicit module-loading overrides PASS without a GPU\n";
    return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
