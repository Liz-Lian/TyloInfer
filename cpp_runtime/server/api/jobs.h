#pragma once
#include <atomic>
#include <memory>
namespace httplib { class Server; }
namespace mfq::server {
class Jobs;
void register_jobs(httplib::Server& http, Jobs& jobs, std::shared_ptr<std::atomic<std::size_t>> streams);
}
