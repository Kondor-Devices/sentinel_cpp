#pragma once
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <unistd.h>

#include <opencv2/opencv.hpp>

#include "visualizer/ipc_common.hpp"


// ---------- ViewerProcess: forks a child that runs the OpenCV viewer ----------
class ViewerProcess {
public:
    struct Options {
        std::string shm_name = "/kdi_frames_batch";
        bool per_camera_windows = true;   // one window per camera
        int waitkey_ms = 10;              // UI poll interval
        int cascade_offset = 40;          // per-window offset when cascading
    };

    ViewerProcess() = default;
    explicit ViewerProcess(Options opt) : opt_(std::move(opt)) {}

    // Forks and starts the viewer loop in the child. Parent returns immediately.
    // Returns true in the parent on success.
    bool start() {
        if (running_) return true;
        pid_ = ::fork();
        if (pid_ < 0) {
            std::perror("[viewer] fork");
            return false;
        }
        if (pid_ == 0) {
            // --- Child: run the viewer loop and exit the process when done ---
            run_child();
            _exit(0);
        }
        // --- Parent ---
        running_ = true;
        std::cout << "[viewer] started child pid=" << pid_ << "\n";
        return true;
    }

    // Sends SIGTERM and waits briefly; SIGKILL if needed.
    void stop(int timeout_ms = 1500) {
        if (!running_) return;
        if (pid_ > 0) {
            ::kill(pid_, SIGTERM);
            int waited = 0, status = 0;
            while (waited < timeout_ms) {
                pid_t r = ::waitpid(pid_, &status, WNOHANG);
                if (r == pid_) { pid_ = -1; running_ = false; return; }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                waited += 50;
            }
            ::kill(pid_, SIGKILL);
            ::waitpid(pid_, &status, 0);
            pid_ = -1;
        }
        running_ = false;
    }

    bool is_running() const noexcept { return running_; }
    pid_t pid() const noexcept { return pid_; }

private:
    Options opt_;
    pid_t pid_{-1};
    bool running_{false};

    // -------- child-side helpers --------
    static bool map_frames_shm(const std::string& name, const void*& base, size_t& bytes) {
        int fd = ::shm_open(name.c_str(), O_RDONLY, 0660);
        if (fd < 0) { std::perror("[viewer] shm_open"); return false; }
        struct stat st{};
        if (::fstat(fd, &st) != 0) { std::perror("[viewer] fstat"); ::close(fd); return false; }
        bytes = static_cast<size_t>(st.st_size);
        void* addr = ::mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0);
        ::close(fd);
        if (addr == MAP_FAILED) { std::perror("[viewer] mmap"); return false; }
        base = addr;
        return true;
    }

    static cv::Mat nchw_fp16_rgb_to_bgr8(const __fp16* src, int /*C*/, int H, int W) {
        const int64_t strideC = static_cast<int64_t>(H) * W;
        const __fp16* r = src + 0*strideC;
        const __fp16* g = src + 1*strideC;
        const __fp16* b = src + 2*strideC;

        cv::Mat out(H, W, CV_8UC3);
        uint8_t* dst = out.data;

        auto sat = [](float v)->uint8_t {
            if (v < 0.f) { v = 0.f; }
            if (v > 1.f) { v = 1.f; }
            return static_cast<uint8_t>(v * 255.f + 0.5f);
        };

        for (int y=0; y<H; ++y) {
            for (int x=0; x<W; ++x) {
                const int64_t idx = static_cast<int64_t>(y)*W + x;
                uint8_t B = sat(static_cast<float>(b[idx]));
                uint8_t G = sat(static_cast<float>(g[idx]));
                uint8_t R = sat(static_cast<float>(r[idx]));
                uint8_t* px = dst + 3*idx;
                px[0]=B; px[1]=G; px[2]=R;
            }
        }
        return out;
    }

    [[noreturn]] void run_child() {
        // Map frames.
        const void* base = nullptr; size_t bytes = 0;
        if (!map_frames_shm(opt_.shm_name, base, bytes)) {
            std::cerr << "[viewer] cannot map frames shm: " << opt_.shm_name << "\n";
            _exit(2);
        }
        auto* hdr  = reinterpret_cast<const ipc::BatchHeader*>(base);
        if (bytes < sizeof(ipc::BatchHeader) || hdr->magic != ipc::kMagic || hdr->version != ipc::kVersion) {
            std::cerr << "[viewer] bad header/magic/version\n";
            _exit(3);
        }
        const int N = hdr->N, C = hdr->C, H = hdr->H, W = hdr->W;
        const size_t per_cam_bytes = static_cast<size_t>(C) * H * W * sizeof(__fp16);
        auto* data = reinterpret_cast<const uint8_t*>(base) + sizeof(ipc::BatchHeader);

        uint64_t mask = hdr->valid_mask;

        // Windows
        std::vector<std::string> names(N);
        for (int i=0; i<N; ++i) {
            if (((mask >> i) & 1ull) == 0ull) {
                // No fresh frame for this cam in this batch — show a placeholder
                cv::Mat blank(H, W, CV_8UC3, cv::Scalar(40,40,40));
                cv::putText(blank, "NO DATA", {20, 50}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {220,220,220}, 2);
                cv::imshow(names[i], blank);
                continue;
            }

            const auto* src = reinterpret_cast<const __fp16*>(
                data + static_cast<size_t>(i)*per_cam_bytes
            );
            cv::Mat img = nchw_fp16_rgb_to_bgr8(src, C, H, W);
            // (detections overlay goes here later)
            cv::imshow(names[i], img);
        }

        uint64_t last_seq = ~0ull;
        while (true) {
            uint64_t s1 = hdr->seq.load(std::memory_order_acquire);
            if ((s1 % 2ull) == 0ull && s1 != last_seq) {
                std::atomic_thread_fence(std::memory_order_acquire);
                uint64_t s2 = hdr->seq.load(std::memory_order_acquire);
                if (s1 == s2) {
                    for (int i=0; i<N; ++i) {
                        auto* src = reinterpret_cast<const __fp16*>(data + static_cast<size_t>(i)*per_cam_bytes);
                        cv::Mat img = nchw_fp16_rgb_to_bgr8(src, C, H, W);

                        // (detections overlay goes here later)

                        cv::imshow(names[i], img);
                    }
                    last_seq = s1;
                }
            }
            int key = cv::waitKey(opt_.waitkey_ms);
            if (key == 27 || key == 'q') break;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        // Cleanup and exit child.
        ::munmap(const_cast<void*>(base), bytes);
        _exit(0);
    }
};
