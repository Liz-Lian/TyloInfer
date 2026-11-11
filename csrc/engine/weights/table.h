#pragma once

#include "engine/tensor.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace tyloi::source { class Source; }

namespace tyloi::weights {

/** Own a streamed, row-cached embedding table shared by request states. */
class Table {
    struct Shared;
    struct Impl;
public:
    /** Own an independent row cache, worker, and device staging for one request. */
    class State {
    public:
        ~State();
        State(State&&) noexcept;
        State& operator=(State&&) noexcept;
        State(const State&) = delete;
        State& operator=(const State&) = delete;
        /** Return the currently staged device tensor, if one exists. */
        Tensor storage() const;
    private:
        friend class Table;
        explicit State(std::unique_ptr<Impl>);
        std::unique_ptr<Impl> impl_;
    };

    /** Open a table with the expected geometry; source must outlive this table and all its states. */
    static std::shared_ptr<Table> load(const source::Source& source, const std::string& name,
        std::int64_t expected_rows, int expected_width, std::size_t cache_capacity_rows);
    ~Table();
    /** Create independently owned mutable request resources. */
    std::unique_ptr<State> make_state() const;
    /** Start or reuse a pending lookup for the given host row IDs. */
    void prefetch(State&, const std::int64_t* rows, std::size_t count) const;
    /** Gather selected rows into a device tensor with shape [batch, tokens, columns * width]. */
    Tensor gather(State&, const std::int64_t* rows, int batch, int tokens, int hash_columns,
                  const Device& device) const;
private:
    explicit Table(std::shared_ptr<const Shared>);
    std::shared_ptr<const Shared> shared_;
};

} // namespace tyloi::weights
