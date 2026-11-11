// Interface test for the single TYLOINFER_* knob seam: presence, strict parsing, the
// retired-name alias, frozen memoization and provenance recording.
#include "native/env.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        ++failures;
        std::cerr << "failed: " << what << '\n';
    }
}

}  // namespace

int main() {
    tyloi::cuda::env_clear("TYLOINFER_TEST_KNOB");
    check(tyloi::cuda::env_lookup("TYLOINFER_TEST_KNOB") == nullptr, "an unset knob reads as unset");
    check(tyloi::cuda::env_flag("TYLOINFER_TEST_KNOB", true), "an unset flag falls back");
    check(tyloi::cuda::env_number("TYLOINFER_TEST_KNOB", 7) == 7, "an unset number falls back");
    check(tyloi::cuda::env_text("TYLOINFER_TEST_KNOB", "d") == "d", "an unset text falls back");

    tyloi::cuda::env_set("TYLOINFER_TEST_KNOB", "0");
    check(!tyloi::cuda::env_flag("TYLOINFER_TEST_KNOB", true), "a set flag overrides the fallback");
    check(tyloi::cuda::env_number("TYLOINFER_TEST_KNOB", 7) == 0, "a set number is parsed");

    bool rejected = false;
    tyloi::cuda::env_set("TYLOINFER_TEST_KNOB", "on");
    try {
        tyloi::cuda::env_flag("TYLOINFER_TEST_KNOB", false);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    check(rejected, "a non-numeric value is rejected instead of guessed");

    tyloi::cuda::env_clear("TYLOINFER_DISABLE_NIQ_FUSION");
    check(tyloi::cuda::env_lookup("TYLOINFER_DISABLE_NVQ_FUSION") == nullptr,
          "the NVQ knob is unset when neither name is set");
    tyloi::cuda::env_set("TYLOINFER_DISABLE_NIQ_FUSION", "1");
    const auto* aliased = tyloi::cuda::env_lookup("TYLOINFER_DISABLE_NVQ_FUSION");
    check(aliased && std::string(aliased) == "1", "the retired NIQ name still steers NVQ");
    tyloi::cuda::env_clear("TYLOINFER_DISABLE_NIQ_FUSION");

    tyloi::cuda::env_freeze();
    check(tyloi::cuda::env_frozen(), "freeze is observable");
    tyloi::cuda::env_set("TYLOINFER_TEST_KNOB", "5");
    check(tyloi::cuda::env_number("TYLOINFER_TEST_KNOB", 0) == 5, "env_set overrides a frozen memo");
    const auto* retained = tyloi::cuda::env_lookup("TYLOINFER_TEST_KNOB");
#ifdef _WIN32
    _putenv_s("TYLOINFER_TEST_KNOB", "6");
#else
    ::setenv("TYLOINFER_TEST_KNOB", "6", 1);
#endif
    check(std::string(retained) == "5" && tyloi::cuda::env_number("TYLOINFER_TEST_KNOB", 0) == 5,
          "a frozen lookup retains its own value across outside environment changes");
    tyloi::cuda::env_clear("TYLOINFER_TEST_KNOB");
    check(tyloi::cuda::env_lookup("TYLOINFER_TEST_KNOB") == nullptr, "env_clear drops the memo entry");

    const auto snapshot = tyloi::cuda::env_snapshot();
    check(!snapshot.empty(), "every looked-up knob is recorded");
    tyloi::cuda::env_set("TYLOINFER_TEST_KNOB", "9");
    check(tyloi::cuda::env_dump().find("TYLOINFER_TEST_KNOB=9") != std::string::npos,
          "the dump lists the knobs that are set");
    tyloi::cuda::env_clear("TYLOINFER_TEST_KNOB");
    check(tyloi::cuda::env_dump().find("TYLOINFER_TEST_KNOB") == std::string::npos,
          "the dump omits knobs that are not set");

    if (failures) return 1;
    std::cout << "cuda env knobs passed\n";
    return 0;
}
