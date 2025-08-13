#pragma once
#include <thread>
#include <atomic>
#include <memory>
#include <utility>

#if __has_include(<stop_token>) && (__cplusplus >= 202002L)
  #include <stop_token>
  namespace jthread_compat {
    using stop_token = std::stop_token;
    using jthread    = std::jthread;
  }
#else
  // Minimal C++17 stand-ins for std::stop_token and std::jthread.
  // Good enough for "stop_requested()" polling and auto-join on destruction.
  namespace jthread_compat {
    struct stop_state {
      std::atomic_bool stop{false};
    };
    struct stop_token {
      std::shared_ptr<stop_state> s;
      bool stop_requested() const noexcept { return s && s->stop.load(std::memory_order_relaxed); }
    };
    struct stop_source {
      std::shared_ptr<stop_state> s = std::make_shared<stop_state>();
      stop_token get_token() const noexcept { return stop_token{s}; }
      void request_stop() noexcept { if (s) s->stop.store(true, std::memory_order_relaxed); }
      bool stop_requested() const noexcept { return s && s->stop.load(std::memory_order_relaxed); }
    };
    struct jthread {
      std::thread t;
      stop_source src;
      jthread() = default;
      template <class F, class... Args>
      explicit jthread(F&& f, Args&&... args) {
        auto tok = src.get_token();
        t = std::thread(std::forward<F>(f), std::move(tok), std::forward<Args>(args)...);
      }
      jthread(jthread&&) noexcept = default;
      jthread& operator=(jthread&& other) noexcept {
        if (this != &other) { request_stop(); join(); t = std::move(other.t); src = std::move(other.src); }
        return *this;
      }
      ~jthread() { request_stop(); join(); }
      bool joinable() const noexcept { return t.joinable(); }
      void join() { if (t.joinable()) t.join(); }
      void request_stop() noexcept { src.request_stop(); }
      // expose a token if you need it
      stop_token get_token() const noexcept { return src.get_token(); }
    };
  }
#endif
