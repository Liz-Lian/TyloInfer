#pragma once
#include "core/mtp.h"
#include <filesystem>
#include <memory>
#include <vector>

namespace mfq::cuda::dflash2 {
struct Config {
    int hidden, vocabulary, layers, intermediate, heads, kv_heads, dimension, block, window, group, rank, candidates, mask;
    float theta, epsilon;
    std::vector<int64_t> target_layers;
    static Config load(const std::filesystem::path&);
};
std::unique_ptr<MtpModule> load(CudaExecutionContext&, Qwen4CausalLm&, const std::filesystem::path&);
}
