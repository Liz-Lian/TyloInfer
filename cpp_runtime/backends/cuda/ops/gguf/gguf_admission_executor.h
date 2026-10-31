#pragma once
#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

// Cache admission is serialized outside decode windows. Reuse its host thread
// and CUDA thread state across windows; destruction drains accepted work.
class GgufAdmissionExecutor {
    std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<std::packaged_task<void()>> queue_;
    bool stopping_=false;
    std::thread worker_;
    void run() {
        for(;;) {
            std::packaged_task<void()> task;
            {
                std::unique_lock lock(mutex_);
                changed_.wait(lock,[&]{return stopping_ || !queue_.empty();});
                if(queue_.empty())return;
                task=std::move(queue_.front());queue_.pop_front();
            }
            task(); // packaged_task forwards failures to the caller's future.
        }
    }
public:
    GgufAdmissionExecutor():worker_([this]{run();}){}
    GgufAdmissionExecutor(const GgufAdmissionExecutor&)=delete;
    GgufAdmissionExecutor& operator=(const GgufAdmissionExecutor&)=delete;
    ~GgufAdmissionExecutor() {
        {std::lock_guard lock(mutex_);stopping_=true;}
        changed_.notify_one();worker_.join();
    }
    template<class Function> std::future<void> submit(Function&& function) {
        std::packaged_task<void()> task(std::forward<Function>(function));
        auto result=task.get_future();
        {
            std::lock_guard lock(mutex_);
            if(stopping_)throw std::logic_error("cache admission worker has stopped");
            queue_.push_back(std::move(task));
        }
        changed_.notify_one();return result;
    }
};
