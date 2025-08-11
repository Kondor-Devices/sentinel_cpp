// yolo_manager.hpp
#pragma once
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <chrono>
#include <cuda_runtime.h>

#include "stereo_camera/stereo_camera.hpp"   // CameraModule, DetsGpuDesc, GpuFrameDesc

class YoloManager {
public:
    YoloManager() = default;

    // Non-copyable (owns CUDA events and a thread)
    YoloManager(const YoloManager&) = delete;
    YoloManager& operator=(const YoloManager&) = delete;
    YoloManager(YoloManager&&) = delete;
    YoloManager& operator=(YoloManager&&) = delete;

    ~YoloManager() { stop(); }

    // Register a camera to receive YOLO outputs (stub for now).
    // Safe to call before or after start(); not thread-safe with stop().
    void add_camera(CameraModule* cam) {
        if (!cam) return;
        slots_.push_back(CamSlot{cam});
    }

    void start() {
        if (running_.exchange(true)) return;

        // Create per-camera reusable events (one per camera)
        for (auto& s : slots_) {
            if (!s.ready_event) {
                // Disable timing for cheaper events
                cudaEventCreateWithFlags(&s.ready_event, cudaEventDisableTiming);
            }
        }

        worker_ = std::jthread([this](std::stop_token st){
            // Main loop: poll cameras and publish "empty" detections with a recorded event
            while (!st.stop_requested() && running_.load(std::memory_order_relaxed)) {
                bool did_something = false;

                for (auto& s : slots_) {
                    if (!s.cam) continue;

                    // Try snapshot a frame; if none yet, skip this cam
                    GpuFrameDesc frame{};
                    if (!s.cam->snapshotFrame(frame)) continue;

                    // Record an event on the **camera's stream** to mark "YOLO done" (stub)
                    // In the real path, you'd launch inference on a stream, then record event after it.
                    cudaEventRecord(s.ready_event, s.cam->stream());

                    // Publish an empty det descriptor with the event (wires the pipeline)
                    DetsGpuDesc dets{};
                    dets.dev_ptr     = nullptr;       // no output yet
                    dets.count       = 0;             // <- important: camera will skip ingest
                    dets.seq         = frame.seq;     // keep sequence aligned to frame
                    dets.ts_ns       = frame.ts_ns;
                    dets.ready_event = s.ready_event; // consumers can wait on this if they need

                    s.cam->publishDetectionsGpu(dets);

                    did_something = true;
                }

                // Backoff a touch if no frames were available this tick
                if (!did_something) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                }
            }
        });
    }

    void stop() {
        if (!running_.exchange(false)) return;
        if (worker_.joinable()) {
            worker_.request_stop();
            worker_.join();
        }
        // Destroy events
        for (auto& s : slots_) {
            if (s.ready_event) {
                cudaEventDestroy(s.ready_event);
                s.ready_event = nullptr;
            }
        }
    }

private:
    struct CamSlot {
        CameraModule* cam = nullptr;
        cudaEvent_t   ready_event = nullptr; // recorded each frame
    };

    std::vector<CamSlot> slots_;
    std::jthread         worker_;
    std::atomic<bool>    running_{false};
};
