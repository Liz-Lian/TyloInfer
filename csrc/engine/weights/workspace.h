#pragma once

#include "engine/tensor.h"

#include <cstddef>
#include <memory>

namespace tyloi::weights {
class Cache;

class Loan {
public:
    explicit Loan(Cache& cache);
    Loan(Cache& cache, std::size_t bytes);
    ~Loan();
    Loan(const Loan&) = delete;
    Loan& operator=(const Loan&) = delete;
    Loan(Loan&&) = delete;
    Loan& operator=(Loan&&) = delete;
    std::size_t begin(std::size_t bytes);
    std::size_t predictor(std::size_t bytes);
    void suspend();
    void finish();
    // Move the output out of borrowed storage before returning it to the cache.
    std::shared_ptr<Tensor> finish(std::shared_ptr<Tensor> output);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    bool active_=false;
};

} // namespace tyloi::weights
