#include "mlx_builtin_kernels.h"
#include "gguf_sources.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("usage: mfq-metal-source-export OUTPUT_DIRECTORY");
        const std::filesystem::path root(argv[1]);
        std::filesystem::create_directories(root);
        const std::pair<const char*, std::string (*)()> sources[]{
            {"mfq_qsa_prefill_indexer_v1", mfq::metal::qsa_prefill_indexer_source},
            {"mfq_sparse_attention_v1", mfq::metal::sparse_attention_source},
            {"tinfer_gguf_linear", [] { return std::string(mfq::metal::native_gguf::linear_source); }},
            {"tinfer_gguf_register", [] { return std::string(mfq::metal::native_gguf::register_source); }},
            {"tinfer_gguf_float", [] { return std::string(mfq::metal::native_gguf::float_source); }},
            {"tinfer_gguf_repack", [] { return std::string(mfq::metal::native_gguf::repack_source); }},
            {"tinfer_gguf_moe", [] { return std::string(mfq::metal::native_gguf::moe_source); }},
            {"tinfer_gguf_moe_projection", [] { return std::string(mfq::metal::native_gguf::moe_projection_source); }},
            {"tinfer_gguf_hyper", [] { return std::string(mfq::metal::native_gguf::hyper_source); }},
#if TYLOINFER_ENABLE_MFQ_COMPAT
            {"mfq_dsv4_mxfp4_blocks_v5", mfq::metal::dsv4_mxfp4_blocks_source},
#endif
        };
        for (const auto& [name, source] : sources) {
            std::ofstream output(root / (std::string(name) + ".metal"), std::ios::binary);
            output.exceptions(std::ios::badbit | std::ios::failbit);
            output << source();
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
