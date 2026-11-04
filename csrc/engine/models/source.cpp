#include "source.h"

#include "engine/weights/source/gguf.h"
#include "common/hf_source.h"
#include "common/gguf_source.h"
#include "common/mfq_source.h"
#include "common/tq_fixture_source.h"

#include <cstdlib>

namespace tyloi {

std::shared_ptr<const ModelSource> open_model_source(
        const std::filesystem::path& path) {
    for (const char* environment :
         {"TYLOINFER_TENSOR_OVERLAY", "TYLOINFER_EXPERT_OVERLAY"}) {
        const char* overlay = std::getenv(environment);
        if (overlay != nullptr && overlay[0] != '\0') {
            throw std::runtime_error(
                std::string(environment) +
                " is unsupported by the canonical ModelSource path");
        }
    }

    const auto filename=path.filename().u8string();
    const std::u8string fixture_suffix=u8".tq-fixture.json";
    if(filename.size()>=fixture_suffix.size() &&
       filename.compare(filename.size()-fixture_suffix.size(),fixture_suffix.size(),fixture_suffix)==0)
        return open_tq_fixture_source(path);
    std::error_code error;
    if (std::filesystem::is_directory(path, error) && !error) {
        return std::make_shared<models::HfSource>(path);
    }
    if (is_gguf_file(path)) return std::make_shared<models::GgufSource>(path);
    return std::make_shared<models::MfqSource>(path);
}

} // namespace tyloi
