#include "diagnostics/source.h"
#include "engine/program.h"
#include "engine/models/source.h"
#include "engine/models/source/model_config.h"
#include "engine/requests/tokenizer.h"
#include "engine/weights/loader.h"
#include "engine/weights/source/read.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {
const tyloi::ModelSource& model_source(const tyloi::weights::Loader& loader) {
    const auto* source = dynamic_cast<const tyloi::ModelSource*>(&loader.source);
    if (!source) throw std::runtime_error("configured source is not a model source");
    return *source;
}
} // namespace

int run_container_check(tyloi::cuda::Program& program) {
    const auto loader = program.loader();
    const auto& source = model_source(loader);
    std::cout << "mfq_container_check=ok"
              << " shards=" << source.source_paths().size()
              << " tensors=" << source.tensors().size()
              << " assets=" << source.assets().size()
              << "\n";
    return 0;
}

int run_assets_check(tyloi::cuda::Program& program, const std::string& config_path,
    const std::string& tokenizer_model, const std::string& text) {
    const auto loader = program.loader();
    const auto& source = model_source(loader);
    const auto config = tyloi::models::ModelConfig::from_json(
        tyloi::models::config_json(source, config_path));
    if (!source.has_asset(tyloi::kTokenizerGgufAsset)) {
        throw std::runtime_error("model source has no tokenizer GGUF");
    }
    const auto tokenizer_blob = tyloi::read_asset(source, tyloi::kTokenizerGgufAsset);
    const auto embedded = probe_mfq_tokenizer(tokenizer_blob, text);
    if (embedded.vocab_size != config.vocab_size) {
        throw std::runtime_error(
            "embedded tokenizer/model vocabulary mismatch: tokenizer=" +
            std::to_string(embedded.vocab_size) + " model=" + std::to_string(config.vocab_size));
    }
    std::cout << "runtime_assets_check=ok"
              << " config=embedded"
              << " tokenizer=embedded"
              << " vocab_size=" << embedded.vocab_size
              << " chat_template=" << (!embedded.chat_template.empty() ? 1 : 0)
              << " bos=" << embedded.bos_token
              << " eos=" << embedded.eos_token
              << " eot=" << embedded.eot_token
              << " pad=" << embedded.pad_token
              << " token_count=" << embedded.tokens.size()
              << "\n";
    if (!tokenizer_model.empty()) {
        const auto external = probe_mfq_tokenizer(tokenizer_model, text);
        if (external.vocab_size != embedded.vocab_size ||
            external.bos_token != embedded.bos_token ||
            external.eos_token != embedded.eos_token ||
            external.eot_token != embedded.eot_token ||
            external.pad_token != embedded.pad_token ||
            external.chat_template != embedded.chat_template ||
            external.tokens != embedded.tokens) {
            throw std::runtime_error("embedded tokenizer differs from external GGUF");
        }
        std::cout << "runtime_assets_external_match=ok\n";
    }
    return 0;
}
