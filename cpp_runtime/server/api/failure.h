#pragma once
#include "errors.h"
#include "auth/auth.h"
#include "models/errors.h"
#include "jobs/jobs.h"
#include "hub/hub.h"
#include "sessions/sessions.h"
#include "protocol.h"
#include "httplib.h"

namespace mfq::server {
struct HttpFailure {
    int status;
    nlohmann::json detail;
    std::string param;
};
inline HttpFailure describe_failure(std::exception_ptr exception) {
    int status = 500;
    std::string code = "internal_error", message = "internal server error", param;
    try { std::rethrow_exception(exception); }
    catch (const ApiError& error) { status = error.status; code = error.code; message = error.what(); }
    catch (const AuthError& error) {
        code = error.code; message = error.what();
        switch (error.kind) {
            case AuthError::invalid: status = 400; break;
            case AuthError::not_found: status = 404; break;
            case AuthError::conflict: status = 409; break;
            case AuthError::unauthorized: status = 401; break;
            case AuthError::forbidden: status = 403; break;
            case AuthError::save_failed: break;
        }
    }
    catch (const ModelError& error) {
        code = error.code; message = error.what();
        switch (error.kind) {
            case ModelError::Kind::invalid: status = 400; break;
            case ModelError::Kind::not_found: status = 404; break;
            case ModelError::Kind::conflict:
            case ModelError::Kind::cancelled: status = 409; break;
            case ModelError::Kind::too_large: status = 413; break;
            case ModelError::Kind::unavailable: status = 501; break;
            case ModelError::Kind::stopping: status = 503; break;
            case ModelError::Kind::failure: break;
        }
    } catch (const Jobs::Error& error) {
        message = error.what();
        switch (error.kind) {
            case Jobs::Error::not_found: status = 404; code = "operation_not_found"; break;
            case Jobs::Error::full: status = 429; code = "operation_queue_full"; break;
            case Jobs::Error::expired: status = 410; code = "event_cursor_expired"; break;
            case Jobs::Error::invalid: status = 400; code = "invalid_request"; break;
            case Jobs::Error::conflict: status = 409; code = "job_state_conflict"; break;
            case Jobs::Error::stopping: status = 503; code = "server_stopping"; break;
        }
    } catch (const Hub::Error& error) {
        code = error.code; message = error.what();
        switch (error.kind) {
            case Hub::Error::invalid: status = 400; break;
            case Hub::Error::missing: status = 404; break;
            case Hub::Error::conflict:
            case Hub::Error::cancelled: status = 409; break;
            case Hub::Error::network:
            case Hub::Error::integrity: status = 502; break;
            case Hub::Error::unavailable: status = 501; break;
            case Hub::Error::io: break;
        }
    } catch (const Sessions::Error& error) {
        switch (error.kind) {
            case Sessions::Error::invalid: status = 400; break;
            case Sessions::Error::not_found: status = 404; break;
            case Sessions::Error::conflict: status = 409; break;
            case Sessions::Error::unavailable: status = 501; break;
            case Sessions::Error::too_large: status = 413; break;
            case Sessions::Error::stopping: status = 503; break;
            case Sessions::Error::save_failed: status = 500; break;
        }
        return {status, {{"code", error.code}, {"message", error.what()},
            {"retryable", error.kind == Sessions::Error::stopping}, {"details", error.details}}};
    } catch (const std::invalid_argument& error) { status = 400; code = "invalid_request"; message = error.what(); }
    catch (const mfq::protocol::ApiError& error) { status = error.status; code = error.type; message = error.what(); param = error.param; }
    catch (const MfqResourceExhausted& error) { status = 503; code = "resource_exhausted"; message = error.what(); }
    catch (...) {}
    return {status, {{"code", code}, {"message", message}, {"retryable", status == 429 || status == 503},
        {"details", nlohmann::json::object()}}, std::move(param)};
}
inline bool anthropic_request(const httplib::Request& request) {
    if (request.path == "/v1/messages" || request.path.rfind("/v1/messages/", 0) == 0 ||
        request.path.rfind("/v1/models/", 0) == 0) return true;
    return (request.path == "/v1" || request.path == "/v1/models") &&
        (request.has_header("Anthropic-Version") || request.has_header("Anthropic-Beta") || request.has_header("X-Api-Key"));
}
inline nlohmann::json compat_error(const HttpFailure& failure, bool anthropic) {
    if (!anthropic) {
        auto error = failure.detail;
        error["type"] = error["code"];
        error["param"] = failure.param.empty() ? nlohmann::json(nullptr) : nlohmann::json(failure.param);
        return {{"error", std::move(error)}};
    }
    const auto status = failure.status;
    const char* type = status == 401 ? "authentication_error" : status == 403 ? "permission_error" :
        status == 404 ? "not_found_error" : status == 413 ? "request_too_large" : status == 429 ? "rate_limit_error" :
        status == 503 || status == 529 ? "overloaded_error" : status >= 400 && status < 500 ? "invalid_request_error" : "api_error";
    return {{"type", "error"}, {"error", {{"type", type}, {"message", failure.detail.at("message")}}}};
}
}
