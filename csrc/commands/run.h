#pragma once

namespace tyloi::cuda::commands {

int run_runtime(int argc, char** argv);
int run_reference(int argc, char** argv);
int run_diagnostics(int argc, char** argv);
int run_eval(int argc, char** argv);

} // namespace tyloi::cuda::commands
