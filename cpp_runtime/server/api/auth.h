#pragma once

#include "auth/auth.h"
#include "httplib.h"

namespace mfq::server {
Scope required_scope(const std::string& method, const std::string& path);
void register_auth(httplib::Server& http, Authentication& auth);
}
