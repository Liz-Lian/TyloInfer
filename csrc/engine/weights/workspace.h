#pragma once

#include "backend.h"

#include <cstddef>
#include <memory>
#include <utility>

class MoeExpertCache;

namespace tyloi::cuda {

class WeightLoan {
public:
    explicit WeightLoan(std::shared_ptr<MoeExpertCache> cache):cache_(std::move(cache)){}
    WeightLoan(std::shared_ptr<MoeExpertCache> cache, std::size_t bytes)
        : WeightLoan(std::move(cache)) { begin(bytes); }
    ~WeightLoan();
    WeightLoan(const WeightLoan&) = delete;
    WeightLoan& operator=(const WeightLoan&) = delete;
    WeightLoan(WeightLoan&&) = delete;
    WeightLoan& operator=(WeightLoan&&) = delete;
    std::size_t begin(std::size_t bytes);
    void suspend();
    void finish();
    // Move the output out of borrowed storage before returning it to the cache.
    ::tyloi::DenseTensor finish(::tyloi::DenseTensor output);
private:
    std::shared_ptr<MoeExpertCache> cache_;
    bool active_=false;
};

} // namespace tyloi::cuda
