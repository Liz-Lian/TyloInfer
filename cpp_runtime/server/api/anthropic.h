#pragma once
#include <atomic>
#include <memory>
namespace httplib { class Server; }
namespace mfq::server {
class ModelInstances;
void register_anthropic(httplib::Server& http, ModelInstances& models, std::shared_ptr<std::atomic<std::size_t>> streams);
}
