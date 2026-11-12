#include "diagnostics/numerics/case.h"
#include "diagnostics/numerics/basic.h"
#include "diagnostics/numerics/attn.h"
#include "diagnostics/numerics/sample.h"
#include "diagnostics/numerics/gdn.h"
#include "diagnostics/numerics/flash.h"
#include "diagnostics/numerics/flash_graph.h"
#include "diagnostics/numerics/flash_reject.h"
#include "diagnostics/numerics/mfq.h"
#include "diagnostics/numerics/sq4.h"
#include "diagnostics/numerics/gemma.h"
#include "diagnostics/numerics/tq_load.h"
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
        if (!tyloi::diagnostics::numerics::run_basic(value) &&
            !tyloi::diagnostics::numerics::run_attn(value) &&
            !tyloi::diagnostics::numerics::run_sample(value) &&
            !tyloi::diagnostics::numerics::run_gdn(value) &&
            !tyloi::diagnostics::numerics::run_flash(value) &&
            !tyloi::diagnostics::numerics::run_flash_graph(value) &&
            !tyloi::diagnostics::numerics::run_flash_reject(value) &&
            !tyloi::diagnostics::numerics::run_mfq(value) &&
            !tyloi::diagnostics::numerics::run_sq4(value) &&
            !tyloi::diagnostics::numerics::run_gemma(value) &&
            !tyloi::diagnostics::numerics::run_tq_load(value))
            tyloi::diagnostics::numerics::run_tq(value);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "numerics: " << error.what() << '\n';
        return 1;
    }
}
