#include "commands/run.h"

int main(int argc, char** argv) {
    return tyloi::cuda::commands::run_runtime(argc, argv);
}
