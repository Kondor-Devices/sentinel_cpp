// stereo_camera.hpp
#pragma once


#include "compile_utils/zed_safe_include.hpp"
#include "stereo_camera_kernels.cuh"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <thread>
#include <stop_token>
#include <memory>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda.h>  // CUcontext, CUstream, cuCtxSetCurrent

// -----------------------------------------------------------------------------
// Small lock-free "latest only" mailbox
// -----------------------------------------------------------------------------
template <typename T>
class LatestOnlyT {
public:
    void publish(const T& v) {
        auto s = seq_.load(std::memory_order_relaxed);
        seq_.store(s + 1, std::memory_order_release); // odd = writing
        val_ = v;
        seq_.store(s + 2, std::memory_order_release); // even = stable
    }
    bool snapshot(T& out) const {
        for (int i = 0; i < 3; ++i) {
            auto s1 = seq_.load(std::memory_order_acquire);
            if (s1 & 1ULL) continue;      // writer in progress
            T tmp = val_;
            auto s2 = seq_.load(std::memory_order_acquire);
            if (s1 == s2 && !(s2 & 1ULL)) { out = std::move(tmp); return true; }
        }
        return false;
    }
private:
    std::atomic<uint64_t> seq_{0};
    T                     val_{};
};

// -----------------------------------------------------------------------------
// Public data descriptions
// -----------------------------------------------------------------------------
struct Trk2D {
    uint64_t id;
    int      label;
    float    conf;
    float    x0, y0, x1, y1;  // pixels
};

struct GpuFrameDesc {
    void*    dev_ptr{nullptr}; // __half* RGB NCHW (1,3,H,W)
    int64_t  shape[4]{1,3,0,0};
    int64_t  stride_bytes[4]{0,0,0,0};
    uint64_t seq{0};
    uint64_t ts_ns{0};
};

struct DetsGpuDesc {
    const void* dev_ptr{nullptr};  // float[N,6]: [x0,y0,x1,y1,conf,cls]
    int         count{0};          // N
    uint64_t    seq{0};
    uint64_t    ts_ns{0};
    cudaEvent_t ready_event{nullptr};   // recorded on the producing (inference) stream
};

// Convenience aliases for tracked detections
using TrkVec  = std::vector<Trk2D>;
using TrkSnap = std::shared_ptr<const TrkVec>;

// -----------------------------------------------------------------------------
// Camera module
// -----------------------------------------------------------------------------
class CameraModule {
public:
    struct Params {
        int width  = 960;
        int height = 600;
        int fps    = 15;
        sl::RESOLUTION open_resolution = sl::RESOLUTION::SVGA;
    };

    CameraModule() : p_{} {}
    explicit CameraModule(const Params& p) : p_(p) {}

    CameraModule(const CameraModule&) = delete;
    CameraModule& operator=(const CameraModule&) = delete;
    CameraModule(CameraModule&&) = delete;
    CameraModule& operator=(CameraModule&&) = delete;

    // --- external inputs ---
    void publishDetectionsGpu(const DetsGpuDesc& d) {
        // reuse GpuFrameDesc shape/seq/ts to publish metadata; event is stored separately
        GpuFrameDesc desc{};
        desc.dev_ptr = const_cast<void*>(d.dev_ptr);
        desc.shape[0] = d.count; desc.shape[1] = 6; desc.shape[2] = 1; desc.shape[3] = 1;
        desc.seq = d.seq; desc.ts_ns = d.ts_ns;
        latest_dets_gpu_.publish(desc);
        std::atomic_store_explicit(&ready_event_, d.ready_event, std::memory_order_release);
    }

    // --- startup / shutdown ---
    void start_from_svo(const std::string& path, bool recording, const std::string& out) {
        if (running_.exchange(true)) return;
        init_from_svo(path, recording, out);
        worker_ = std::jthread([this](std::stop_token st){ run_loop(st); });
    }

    void start_from_serial(uint32_t serial, bool recording, const std::string& out) {
        if (running_.exchange(true)) return;
        init_from_serial(serial, recording, out);
        worker_ = std::jthread([this](std::stop_token st){ run_loop(st); });
    }

    void stop() {
        if (!running_.exchange(false)) return;
        if (worker_.joinable()) {
            worker_.request_stop();
            worker_.join();
        }
        cleanup();
    }

    ~CameraModule() { stop(); }

    // --- queries ---
    bool is_running() const noexcept { return running_.load(); }
    cudaStream_t stream() const { return stream_; }
    bool snapshotFrame(GpuFrameDesc& out) const { return latest_.snapshot(out); }
    bool snapshotTracked(std::vector<Trk2D>& out) const {
        TrkSnap snap;
        if (!latest_trk_.snapshot(snap) || !snap) return false;
        out = *snap; // copy into caller's buffer
        return true;
    }

private:
    // --- helpers ---
    static uint64_t now_ns() {
        using namespace std::chrono;
        return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
    }

    void open_from_svo(sl::InitParameters& ip, const std::string& path) {
        ip.input.setFromSVOFile(sl::String(path.c_str()));
    }
    void open_from_serial(sl::InitParameters& ip, uint32_t serial) {
        ip.input.setFromSerialNumber(static_cast<unsigned int>(serial));
    }

    void init_from_svo(const std::string& path, bool recording, const std::string& out) {
        sl::InitParameters ip;
        open_from_svo(ip, path);
        ip.svo_real_time_mode = true;
        common_open(ip, recording, /*svo=*/true, out);
    }
    void init_from_serial(uint32_t serial, bool recording, const std::string& out) {
        sl::InitParameters ip;
        open_from_serial(ip, serial);
        common_open(ip, recording, /*svo=*/false, out);
    }

    void common_open(sl::InitParameters& ip, bool recording, bool svo, const std::string& out) {
        ip.camera_resolution = p_.open_resolution;
        ip.camera_fps        = p_.fps;

        if (zed_.open(ip) != sl::ERROR_CODE::SUCCESS) std::abort();
        runtime_ = sl::RuntimeParameters{};

        W_ = p_.width; H_ = p_.height;
        frame_gpu_ = sl::Mat(W_, H_, sl::MAT_TYPE::U8_C4, sl::MEM::GPU);

        init_object_detection();

        if (recording && !svo) {
            sl::RecordingParameters rp;
            rp.compression_mode = sl::SVO_COMPRESSION_MODE::H264;
            rp.video_filename   = sl::String(out.c_str());
            zed_.enableRecording(rp);
        }

        // adopt ZED stream (no context switch here)
        stream_ = reinterpret_cast<cudaStream_t>(zed_.getCUDAStream());

        // allocate outputs and prep descriptor template
        alloc_outputs();
        prep_desc_template();
    }

    void run_loop(std::stop_token st){
        // Make ZED CUDA context current on this thread
        CUcontext zed_ctx = zed_.getCUDAContext();
        cuCtxSetCurrent(zed_ctx);

        while (!st.stop_requested() && running_.load(std::memory_order_relaxed)) {
            if (!grab()) std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }

    void init_object_detection() {
        sl::ObjectDetectionParameters detection_parameters;
        detection_parameters.detection_model = sl::OBJECT_DETECTION_MODEL::CUSTOM_BOX_OBJECTS;
        detection_parameters.enable_tracking = true;
        if (detection_parameters.enable_tracking) {
            sl::PositionalTrackingParameters positional_tracking_parameters;
            zed_.enablePositionalTracking(positional_tracking_parameters);
        }
        sl::ERROR_CODE ec = zed_.enableObjectDetection(detection_parameters);
        if (ec != sl::ERROR_CODE::SUCCESS) {
            std::cout << "enableObjectDetection failed: " << ec << "\n";
            zed_.close();
            std::exit(-1);
        }
    }

    // Grab LEFT → convert → publish (async on stream_). Returns false if grab failed.
    bool grab() {
        // 1) Grab
        if (zed_.grab(runtime_) != sl::ERROR_CODE::SUCCESS) return false;

        // 2) Retrieve directly to GPU at W_×H_ on our stream
        if (zed_.retrieveImage(frame_gpu_, sl::VIEW::LEFT, sl::MEM::GPU,
                               sl::Resolution(W_, H_), stream_) != sl::ERROR_CODE::SUCCESS) {
            return false;
        }

        // 3) Source BGRA8 (pitched) on device
        const sl::uchar4* p4 = frame_gpu_.getPtr<sl::uchar4>(sl::MEM::GPU);
        const unsigned char* src = reinterpret_cast<const unsigned char*>(p4);
        const size_t pitch = frame_gpu_.getStepBytes(sl::MEM::GPU);

        // 4) Ping-Pong output buffer
        cur_ ^= 1;
        __half* dst = reinterpret_cast<__half*>(d_out_[cur_]);

        // 5) BGRA → RGB FP16 NCHW on the same stream (via launcher from .cu)
        launch_bgra_to_rgb_fp16_nchw(src, pitch, W_, H_, dst, stream_);

        // publish descriptor (consumers must use same stream or wait on an event)
        GpuFrameDesc d = desc_template_;
        d.dev_ptr = dst;
        d.seq     = ++seq_;
        d.ts_ns   = now_ns();
        latest_.publish(d);

        ingestAndRetrieve();
        return true;
    }

    void ingestAndRetrieve() {
        // 1) snapshot latest GPU detections
        GpuFrameDesc dets{};
        if (latest_dets_gpu_.snapshot(dets) && dets.dev_ptr && dets.shape[0] > 0) {
            const int n = static_cast<int>(dets.shape[0]);
            const size_t bytes = static_cast<size_t>(n) * 6 * sizeof(float);

            // wait for YOLO’s event before reading its output
            if (auto ev = std::atomic_load_explicit(&ready_event_, std::memory_order_acquire)) {
                cudaStreamWaitEvent(stream_, ev, 0);
            }

            // ensure pinned host buffer
            if (bytes > h_dets_bytes_) {
                if (h_dets_) cudaFreeHost(h_dets_);
                cudaHostAlloc(&h_dets_, bytes, cudaHostAllocPortable);
                h_dets_bytes_ = bytes;
            }

            // copy [N,6] from device to pinned host on camera stream
            cudaMemcpyAsync(h_dets_, dets.dev_ptr, bytes, cudaMemcpyDeviceToHost, stream_);
            cudaStreamSynchronize(stream_); // need CPU data below

            // pack into ZED custom boxes
            const float* f = static_cast<const float*>(h_dets_);
            std::vector<sl::CustomBoxObjectData> objs;
            objs.reserve(n);
            for (int i = 0; i < n; ++i) {
                float x0=f[0], y0=f[1], x1=f[2], y1=f[3], conf=f[4]; int cls=(int)f[5]; f+=6;
                sl::CustomBoxObjectData o;
                o.unique_object_id = sl::generate_unique_id();
                o.probability = conf; o.label = cls; o.is_grounded = true;
                o.bounding_box_2d = {
                    {static_cast<uint32_t>(x0), static_cast<uint32_t>(y0)},
                    {static_cast<uint32_t>(x1), static_cast<uint32_t>(y0)},
                    {static_cast<uint32_t>(x1), static_cast<uint32_t>(y1)},
                    {static_cast<uint32_t>(x0), static_cast<uint32_t>(y1)}
                };
                objs.push_back(o);
            }
            if (!objs.empty()) zed_.ingestCustomBoxObjects(objs);
        }

        // 2) retrieve tracked objects and publish a tiny CPU list
        sl::Objects objs_out;
        if (zed_.retrieveObjects(objs_out, sl::ObjectDetectionRuntimeParameters{}) ==
            sl::ERROR_CODE::SUCCESS) {
            TrkVec v; v.reserve(objs_out.object_list.size());
            for (const auto& o : objs_out.object_list) {
                const auto& bb = o.bounding_box_2d;
                float x0 = std::min({bb[0].x,bb[1].x,bb[2].x,bb[3].x});
                float y0 = std::min({bb[0].y,bb[1].y,bb[2].y,bb[3].y});
                float x1 = std::max({bb[0].x,bb[1].x,bb[2].x,bb[3].x});
                float y1 = std::max({bb[0].y,bb[1].y,bb[2].y,bb[3].y});
                v.push_back(Trk2D{
                    static_cast<uint64_t>(o.id), (int)o.label, o.confidence, x0,y0,x1,y1
                });
            }
            auto owned = std::make_shared<const TrkVec>(std::move(v));
            latest_trk_.publish(owned);
        }
    }

    void alloc_outputs() {
        const size_t bytes = static_cast<size_t>(3) * W_ * H_ * sizeof(__half);
        if (cudaMalloc(&d_out_[0], bytes) != cudaSuccess) std::abort();
        if (cudaMalloc(&d_out_[1], bytes) != cudaSuccess) std::abort();
        cur_ = 0;
    }

    void prep_desc_template() {
        desc_template_.dev_ptr = nullptr;
        desc_template_.shape[0] = 1;        // N
        desc_template_.shape[1] = 3;        // C
        desc_template_.shape[2] = H_;       // H
        desc_template_.shape[3] = W_;       // W
        const int64_t e = sizeof(__half);
        desc_template_.stride_bytes[3] = e;                 // W step
        desc_template_.stride_bytes[2] = W_ * e;            // H step
        desc_template_.stride_bytes[1] = H_ * W_ * e;       // C step
        desc_template_.stride_bytes[0] = 3  * H_ * W_ * e;  // N step
        desc_template_.seq   = 0;
        desc_template_.ts_ns = 0;
    }

    void cleanup() {
        if (d_out_[0]) { cudaFree(d_out_[0]); d_out_[0] = nullptr; }
        if (d_out_[1]) { cudaFree(d_out_[1]); d_out_[1] = nullptr; }
        if (h_dets_)   { cudaFreeHost(h_dets_); h_dets_ = nullptr; }
        if (frame_gpu_.isInit()) frame_gpu_.free();
        if (zed_.isOpened()) zed_.close();
        h_dets_bytes_ = 0;
    }

    // ---- state ----
    Params                p_;
    sl::Camera            zed_;
    sl::RuntimeParameters runtime_;
    sl::Mat               frame_gpu_;

    int                   W_{0}, H_{0};    // set in common_open()
    cudaStream_t          stream_{nullptr};
    void*                 d_out_[2]{nullptr, nullptr};
    int                   cur_{0};
    uint64_t              seq_{0};

    GpuFrameDesc                desc_template_{};
    LatestOnlyT<GpuFrameDesc>   latest_;          // frames
    LatestOnlyT<GpuFrameDesc>   latest_dets_gpu_; // YOLO dets descriptor (GPU)
    std::atomic<cudaEvent_t>    ready_event_{nullptr};

    LatestOnlyT<TrkSnap>        latest_trk_{};   // tracked det snapshots (const)

    void*   h_dets_ = nullptr;         // pinned host buffer for dets
    size_t  h_dets_bytes_ = 0;         // size of pinned host buffer

    std::jthread           worker_;
    std::atomic<bool>      running_{false};
};
