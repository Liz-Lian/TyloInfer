#include "cuda_startup.h"

namespace mfq::cuda::commands {
int run_eval(int argc, char** argv);
}

int main(int argc, char** argv) {
    mfq::cuda::configure_cuda_startup();
    return mfq::cuda::commands::run_eval(argc, argv);
}
