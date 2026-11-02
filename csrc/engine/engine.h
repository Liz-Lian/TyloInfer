#pragma once

#include "requests/request.h"
#include "requests/result.h"

#include <chrono>
#include <optional>
#include <variant>

struct MfqModelCapabilities {
    std::string family = "unknown";
    bool text = true;
    bool image_input = false;
    bool video_input = false;
    bool audio_input = false;
    bool audio_output = false;
    bool full_duplex = false;
    bool mtp = false;
    std::string source;
};

namespace mfq::engine {

struct EngineInfo {
    std::size_t max_requests = 1;
    std::int64_t max_context = 0;
    std::int32_t vocab_size = 0;
    bool multimodal = false, duplex = false, reload = false;
    ChatTemplateCapabilities chat;
    std::string model_type;
    MfqModelCapabilities capabilities;
    bool probability = false;
    std::unordered_set<std::int64_t> stop_token_ids;
};
struct EngineStatus {
    std::size_t available = 0;
    bool healthy = true;
};
struct EngineStepResult {
    std::vector<RequestId> advanced;
    std::vector<EngineEvent> events;
    EngineStatus status;
    bool has_work = false;
    std::optional<Clock::time_point> wake_at;
};
enum class Admission { accepted, deferred };
// Only the scheduler loop calls these methods. Execution returns owned values;
// no caller code or response writer can run on the device execution stack.
class Engine {
public:
    virtual ~Engine() = default;
    virtual EngineInfo info() const = 0;
    virtual Admission admit(EngineRequest request) = 0;
    virtual void cancel(const RequestId& id) = 0;
    virtual EngineStepResult step(const std::vector<RequestId>& eligible) = 0;
    virtual EngineStatus status() const = 0;
    virtual SessionResult session(const SessionCommand& command) = 0;
    virtual std::int64_t reload(std::int64_t context) = 0;
    virtual void shutdown() = 0;
    virtual ControlResult control(ControlRequest request) = 0;
};

} // namespace mfq::engine
