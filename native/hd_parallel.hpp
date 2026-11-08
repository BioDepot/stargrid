#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <thread>
#include <vector>

namespace visium_hd {

// Calls body(i) for every i in [0, count) on up to `threads` threads, handing out indices one at a
// time. The first exception thrown by any call is rethrown after all threads have stopped.
template <class Body>
void parallel_for(std::size_t count, int threads, Body&& body) {
    const std::size_t workers = std::min<std::size_t>(static_cast<std::size_t>(std::max(threads, 1)), count);
    if (workers <= 1) {
        for (std::size_t i = 0; i < count; ++i) body(i);
        return;
    }
    std::atomic<std::size_t> next{0};
    std::vector<std::exception_ptr> failures(workers);
    std::vector<std::thread> pool;
    pool.reserve(workers);
    for (std::size_t t = 0; t < workers; ++t) {
        pool.emplace_back([&, t] {
            try {
                for (std::size_t i; (i = next.fetch_add(1, std::memory_order_relaxed)) < count;) body(i);
            } catch (...) {
                failures[t] = std::current_exception();
                next.store(count, std::memory_order_relaxed);
            }
        });
    }
    for (std::thread& worker : pool) worker.join();
    for (const std::exception_ptr& failure : failures) {
        if (failure) std::rethrow_exception(failure);
    }
}

}  // namespace visium_hd
