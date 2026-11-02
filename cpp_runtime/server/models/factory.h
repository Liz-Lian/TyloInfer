#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <cstdint>

namespace mfq { class ModelSource; namespace engine { class Engine; } }

namespace mfq::server {
struct LoadOptions {
    // Zero leaves the context default to the native factory.
    std::int64_t context_size = 0;
};
struct EngineFactory {
    std::string backend;
    std::function<std::string(const ModelSource&)> unavailable_reason;
    // Loads own independent engines; another instance's failure or close must not invalidate them.
    std::function<std::unique_ptr<engine::Engine>(const std::filesystem::path&, const LoadOptions&)> load;
};
}
