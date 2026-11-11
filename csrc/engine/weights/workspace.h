#pragma once

#include "engine/tensor.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace tyloi::weights {
class Cache;

class KvRegion {
public:
    KvRegion() noexcept;
    KvRegion(KvRegion&&) noexcept;
    KvRegion& operator=(KvRegion&&);
    ~KvRegion() noexcept;
    KvRegion(const KvRegion&) = delete;
    KvRegion& operator=(const KvRegion&) = delete;
    void reset();
    void advance(std::int64_t end);
    void protect(std::int64_t end);
    std::size_t bytes() const;
private:
    friend class Cache;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

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
    std::size_t transfer_bytes() const;
    void suspend();
    void finish();
    /** Replace the output in place before returning borrowed storage; preserve it if cloning fails. */
    void finish(Tensor& output);
private:
    friend class Cache;
    struct Impl;
    std::unique_ptr<Impl> impl_;
    bool active_=false;
};

} // namespace tyloi::weights
