#pragma once

#include "httplib.h"
#include "sessions/sessions.h"

namespace mfq::server {
void register_sessions(httplib::Server& http, Sessions& sessions, std::shared_ptr<std::atomic<std::size_t>> active);
}
