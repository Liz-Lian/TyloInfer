#pragma once

#include "runtime_profile.h"
#include "scheduler.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct MfqHttpRuntimeTransportConfig : MfqRuntimeTransportConfig {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string api_key;
};

class MfqTransport {
public:
    virtual ~MfqTransport() = default;
    virtual int run(const MfqScheduler & scheduler) = 0;
};

class MfqRuntime {
public:
    MfqRuntime(
            std::unique_ptr<mfq::engine::Engine> engine,
            std::unique_ptr<MfqTransport> transport)
        : engine_(require_engine(std::move(engine))),
          transport_(require_transport(std::move(transport))),
          scheduler_(*engine_) {}

    int run() {
        return transport_->run(scheduler_);
    }

private:
    static std::unique_ptr<mfq::engine::Engine> require_engine(
            std::unique_ptr<mfq::engine::Engine> engine) {
        if (!engine) {
            throw std::invalid_argument("MFQ runtime requires an engine");
        }
        return engine;
    }
    static std::unique_ptr<MfqTransport> require_transport(
            std::unique_ptr<MfqTransport> transport) {
        if (!transport) {
            throw std::invalid_argument("MFQ runtime requires a transport");
        }
        return transport;
    }
    std::unique_ptr<mfq::engine::Engine> engine_;
    std::unique_ptr<MfqTransport> transport_;
    MfqScheduler scheduler_;
};

std::unique_ptr<MfqTransport> make_mfq_http_transport(
    MfqHttpRuntimeTransportConfig config);

// Reserve the original stdout pipe for NDJSON and redirect ordinary process
// output to stderr. Call before loading a model so startup logs cannot corrupt
// the protocol stream.
void prepare_mfq_stdio_transport();
std::unique_ptr<MfqTransport> make_mfq_stdio_transport(
    MfqRuntimeTransportConfig config);
