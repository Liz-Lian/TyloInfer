#include "config.h"

namespace tyloi::models::glm5_next {

Config Config::from_json(std::string_view payload) {
    return parse(nlohmann::json::parse(payload));
}

} // namespace tyloi::models::glm5_next
