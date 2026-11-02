#include "jobs.h"
#include "storage/sqlite.h"
#include "instance.h"
#include <limits>
#include <regex>
#include <cmath>
#include <iostream>

namespace mfq::server {
namespace {
using Json = nlohmann::json;
using sqlite::Statement;
constexpr std::size_t queue_limit = 16;
Json read_job(sqlite3* database, const std::string& id) {
    Statement query(database, "SELECT id,kind,status,payload_json,progress,progress_data_json,cancel_requested,"
        "result_json,error_json,created_at,updated_at,started_at,completed_at,"
        "COALESCE((SELECT MAX(sequence) FROM job_events WHERE job_id=jobs.id),0) FROM jobs WHERE id=? AND archived_at IS NULL");
    query.bind(1, id);
    if (!query.step()) throw Jobs::Error(Jobs::Error::not_found, "operation not found");
    auto parse = [&](int i) { const auto value = query.column(i); return value.is_null() ? value : Json::parse(value.get<std::string>()); };
    return {{"id", query.column(0)}, {"kind", query.column(1)}, {"status", query.column(2)},
        {"payload", parse(3)}, {"progress", std::stod(query.column(4).get<std::string>())},
        {"progress_data", parse(5)}, {"cancel_requested", query.column(6) == "1"},
        {"result", parse(7)}, {"error", parse(8)}, {"created_at", query.column(9)},
        {"updated_at", query.column(10)}, {"started_at", query.column(11)}, {"completed_at", query.column(12)},
        {"last_sequence", std::stoll(query.column(13).get<std::string>())}};
}
void event(sqlite3* database, const std::string& id, const char* type, const Json& data) {
    Statement insert(database, "INSERT INTO job_events(job_id,sequence,type,level,progress,data_json,created_at) "
        "SELECT ?,COALESCE(MAX(sequence),0)+1,?,'info',?,?,strftime('%Y-%m-%dT%H:%M:%f+00:00','now') "
        "FROM job_events WHERE job_id=?");
    insert.bind(1, id); insert.bind(2, type);
    insert.bind(3, data.contains("progress") ? Json(data.at("progress").dump()) : data.value("status", "") == "succeeded" ? Json("1") : Json(nullptr));
    insert.bind(4, data.dump()); insert.bind(5, id); insert.step();
    Statement trim(database, "DELETE FROM job_events WHERE job_id=? AND sequence <= "
        "(SELECT MAX(sequence)-256 FROM job_events WHERE job_id=?)");
    trim.bind(1, id); trim.bind(2, id); trim.step();
}
}

bool Jobs::terminal(const Json& job) {
    const auto& state = job.at("status");
    return state == "succeeded" || state == "failed" || state == "cancelled" || state == "interrupted";
}
Jobs::Jobs(Storage& storage, Failure failure) : storage_(storage), failure_(std::move(failure)) {
    auto access = storage_.access();
    sqlite::Transaction transaction(access.database);
    Statement pending(access.database, "SELECT id FROM jobs WHERE status IN ('queued','running','cancelling')");
    std::vector<std::string> interrupted;
    while (pending.step()) interrupted.push_back(pending.column(0).get<std::string>());
    for (const auto& id : interrupted) {
        Statement update(access.database, "UPDATE jobs SET status='interrupted',error_json=?,"
            "updated_at=strftime('%Y-%m-%dT%H:%M:%f+00:00','now'),completed_at=strftime('%Y-%m-%dT%H:%M:%f+00:00','now') WHERE id=?");
        update.bind(1, Json{{"code", "operation_interrupted"}, {"message", "server restarted before operation completion"},
            {"retryable", false}, {"details", Json::object()}}.dump());
        update.bind(2, id); update.step();
        event(access.database, id, "state", {{"status", "interrupted"}});
    }
    transaction.commit();
    worker_ = std::thread([this] { loop(); });
}
Jobs::~Jobs() { try { shutdown(); } catch (...) {} }

std::string Jobs::submit(std::string kind, Json payload, Work work) {
    auto task = std::make_shared<Task>(); task->id = normalize_uuid({}); task->work = std::move(work);
    std::lock_guard guard(mutex_);
    if (stopping_) throw Error(Error::stopping, "server is stopping");
    if (queued_.size() >= queue_limit) throw Error(Error::full, "background operation queue is full");
    queued_.push_back(task);
    try {
        auto access = storage_.access();
        sqlite::Transaction transaction(access.database);
        Statement insert(access.database, "INSERT INTO jobs(id,kind,status,payload_json,progress,cancel_requested,created_at,updated_at) "
            "VALUES(?,?,'queued',?,0,0,strftime('%Y-%m-%dT%H:%M:%f+00:00','now'),strftime('%Y-%m-%dT%H:%M:%f+00:00','now'))");
        insert.bind(1, task->id); insert.bind(2, kind); insert.bind(3, payload.dump()); insert.step();
        event(access.database, task->id, "state", {{"status", "queued"}});
        transaction.commit();
    } catch (...) { queued_.pop_back(); throw; }
    ready_.notify_one();
    return task->id;
}
Jobs::Json Jobs::get(const std::string& id) const { auto access = storage_.access(); return read_job(access.database, id); }
Jobs::Json Jobs::list(std::size_t limit, std::size_t offset, const std::string& status, const std::string& kind) const {
    static const std::regex states("queued|running|cancelling|succeeded|failed|cancelled|interrupted"), kinds("[a-z][a-z0-9_.-]{0,63}");
    if (!limit || limit > 1000 || offset > std::size_t(std::numeric_limits<std::int64_t>::max()) ||
        (!status.empty() && !std::regex_match(status, states)) || (!kind.empty() && !std::regex_match(kind, kinds)))
        throw Error(Error::invalid, "invalid job filter or pagination");
    auto access = storage_.access();
    std::string sql = "SELECT id FROM jobs WHERE archived_at IS NULL";
    if (!status.empty()) sql += " AND status=?";
    if (!kind.empty()) sql += " AND kind=?";
    sql += " ORDER BY created_at DESC,id DESC LIMIT ? OFFSET ?";
    Statement query(access.database, sql.c_str());
    int field = 1;
    if (!status.empty()) query.bind(field++, status);
    if (!kind.empty()) query.bind(field++, kind);
    query.bind(field++, std::to_string(limit)); query.bind(field, std::to_string(offset));
    auto data = Json::array();
    while (query.step()) data.push_back(read_job(access.database, query.column(0).get<std::string>()));
    return {{"data", std::move(data)}};
}
void Jobs::erase(const std::string& id) {
    auto access = storage_.access(); sqlite::Transaction transaction(access.database);
    if (!terminal(read_job(access.database, id))) throw Error(Error::conflict, "active jobs cannot be archived");
    Statement update(access.database, "UPDATE jobs SET archived_at=strftime('%Y-%m-%dT%H:%M:%f+00:00','now') WHERE id=?");
    update.bind(1, id); update.step(); transaction.commit();
}
std::size_t Jobs::clear() {
    auto access = storage_.access(); sqlite::Transaction transaction(access.database);
    sqlite::execute(access.database, "UPDATE jobs SET archived_at=strftime('%Y-%m-%dT%H:%M:%f+00:00','now') "
        "WHERE archived_at IS NULL AND status IN ('succeeded','failed','cancelled','interrupted')");
    const auto count = sqlite3_changes(access.database);
    transaction.commit(); return static_cast<std::size_t>(count);
}
void Jobs::update(const std::string& id, const char* state, Json result, Json error) {
    auto access = storage_.access();
    sqlite::Transaction transaction(access.database);
    Statement update(access.database, "UPDATE jobs SET status=?,result_json=?,error_json=?,"
        "progress=CASE WHEN ?='succeeded' THEN 1 ELSE progress END,"
        "started_at=CASE WHEN ?='running' THEN COALESCE(started_at,strftime('%Y-%m-%dT%H:%M:%f+00:00','now')) ELSE started_at END,"
        "completed_at=CASE WHEN ? IN ('succeeded','failed','cancelled') THEN strftime('%Y-%m-%dT%H:%M:%f+00:00','now') ELSE completed_at END,"
        "updated_at=strftime('%Y-%m-%dT%H:%M:%f+00:00','now') WHERE id=?");
    update.bind(1, state); update.bind(2, result.is_null() ? result : Json(result.dump()));
    update.bind(3, error.is_null() ? error : Json(error.dump()));
    for (int i = 4; i <= 6; ++i) update.bind(i, state);
    update.bind(7, id); update.step();
    event(access.database, id, "state", {{"status", state}, {"result", result}, {"error", error}});
    transaction.commit();
}
void Jobs::progress(const std::string& id, const std::string& phase, double value, Json data) {
    if (!std::isfinite(value) || value < 0 || value > 1 || !data.is_object()) throw Error(Error::invalid, "invalid job progress");
    auto access = storage_.access();
    sqlite::Transaction transaction(access.database);
    Statement update(access.database, "UPDATE jobs SET progress_data_json=?,progress=?,updated_at=strftime('%Y-%m-%dT%H:%M:%f+00:00','now') WHERE id=?");
    data["phase"] = phase; data["progress"] = value;
    update.bind(1, data.dump()); update.bind(2, Json(value).dump()); update.bind(3, id); update.step();
    event(access.database, id, "progress", data);
    transaction.commit();
}
Jobs::Json Jobs::cancel(const std::string& id) {
    std::lock_guard guard(mutex_);
    auto current = get(id);
    if (terminal(current)) return current;
    auto access = storage_.access();
    sqlite::Transaction transaction(access.database);
    Statement cancel(access.database, "UPDATE jobs SET cancel_requested=1,status='cancelling',"
        "updated_at=strftime('%Y-%m-%dT%H:%M:%f+00:00','now') WHERE id=?");
    cancel.bind(1, id); cancel.step();
    event(access.database, id, "state", {{"status", "cancelling"}, {"cancel_requested", true}});
    transaction.commit();
    access.lock.unlock();
    if (running_ && running_->id == id) *running_->cancelled = true;
    for (auto it = queued_.begin(); it != queued_.end(); ++it) if ((*it)->id == id) {
        *(*it)->cancelled = true; queued_.erase(it); update(id, "cancelled"); break;
    }
    return get(id);
}
Jobs::Json Jobs::events(const std::string& id, std::int64_t after, std::size_t limit) const {
    auto access = storage_.access();
    const auto job = read_job(access.database, id);
    Statement first(access.database, "SELECT MIN(sequence) FROM job_events WHERE job_id=?");
    first.bind(1, id); first.step();
    if (!first.column(0).is_null() && after < std::stoll(first.column(0).get<std::string>()) - 1)
        throw Error(Error::expired, "operation event cursor expired; fetch the operation snapshot");
    if (after > job.at("last_sequence").get<std::int64_t>()) throw Error(Error::invalid, "event cursor is ahead of the operation");
    Statement query(access.database, "SELECT sequence,type,level,message,progress,data_json,created_at FROM job_events "
        "WHERE job_id=? AND sequence>? ORDER BY sequence LIMIT ?");
    query.bind(1, id); query.bind(2, std::to_string(after)); query.bind(3, std::to_string(limit));
    auto data = Json::array();
    while (query.step()) data.push_back({{"job_id", id}, {"sequence", std::stoll(query.column(0).get<std::string>())},
        {"type", query.column(1)}, {"level", query.column(2)}, {"message", query.column(3)},
        {"progress", query.column(4).is_null() ? Json(nullptr) : Json(std::stod(query.column(4).get<std::string>()))},
        {"data", Json::parse(query.column(5).get<std::string>())}, {"created_at", query.column(6)}});
    return {{"data", std::move(data)}};
}
void Jobs::loop() noexcept {
    try {
        for (;;) {
            std::shared_ptr<Task> task;
            {
                std::unique_lock lock(mutex_);
                ready_.wait(lock, [&] { return stopping_ || !queued_.empty(); });
                if (stopping_) return;
                task = running_ = queued_.front(); queued_.pop_front();
                update(task->id, "running");
            }
            Json result, failure;
            bool cancelled = false;
            try {
                Control control{*task->cancelled, [this, id = task->id](const std::string& phase, double value, Json data) { progress(id, phase, value, std::move(data)); }};
                control.checkpoint(); result = task->work(control);
            } catch (const Cancelled&) { cancelled = true; }
            catch (const std::exception& error) {
                failure = failure_(std::current_exception());
                if (failure.value("code", "") == "internal_error")
                    std::cerr << Json{{"event", "operation_failed"}, {"operation_id", task->id}, {"detail", error.what()}}.dump() << '\n';
            }
            catch (...) {
                failure = failure_(std::current_exception());
                std::cerr << Json{{"event", "operation_failed"}, {"operation_id", task->id}, {"detail", "unknown exception"}}.dump() << '\n';
            }
            {
                std::lock_guard lock(mutex_);
                update(task->id, cancelled ? "cancelled" : failure.is_null() ? "succeeded" : "failed", result, failure);
                running_.reset();
            }
        }
    } catch (...) {
        std::lock_guard guard(mutex_); worker_error_ = std::current_exception(); stopping_ = true;
    }
}
void Jobs::request_stop() {
    {
        std::lock_guard guard(mutex_);
        stopping_ = true;
        if (running_) *running_->cancelled = true;
        for (const auto& task : queued_) *task->cancelled = true;
    }
    ready_.notify_all();
}
void Jobs::shutdown() {
    request_stop();
    std::exception_ptr failure;
    {
        std::lock_guard guard(mutex_);
        for (const auto& task : queued_) {
            try { update(task->id, "cancelled"); } catch (...) { if (!failure) failure = std::current_exception(); }
        }
        queued_.clear();
    }
    ready_.notify_all();
    if (worker_.joinable()) worker_.join();
    if (worker_error_) std::rethrow_exception(worker_error_);
    if (failure) std::rethrow_exception(failure);
}
}
