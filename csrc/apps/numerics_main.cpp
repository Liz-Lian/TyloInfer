#include "diagnostics/numerics/case.h"
#include "diagnostics/numerics/basic.h"
#include "diagnostics/numerics/tq.h"
#include <exception>
#include <iostream>

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "usage: mfq-numerics CASE_JSON RAW_DIR\n";
        return 2;
    }
    try {
        tyloi::diagnostics::numerics::Case value(argv[1], argv[2]);
        if (!tyloi::diagnostics::numerics::run_basic(value))
            tyloi::diagnostics::numerics::run_tq(value);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "numerics: " << error.what() << '\n';
        return 1;
    }
}
