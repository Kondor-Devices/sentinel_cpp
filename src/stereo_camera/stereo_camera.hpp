// stereo_camera.hpp
#pragma once
#include <sl/Camera.hpp>
#include "stereo_camera_kernels.cuh"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <cstdlib>
#include <string>
#include <mutex>
#include <vector>
#include <algorithm>
#include <iostream>
#include <thread>
#include <mutex>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda.h>  // for CUcontext, CUstream, cuCtxSetCurrent


template <typename T>
class LatestOnlyT {
public:
    void publish(const T& v) {
        auto s = seq_.load(std::memory_order_relaxed);
        seq_.store(s + 1, std::memory_order_release);
        val_ = v;
        seq_.store(s + 2, std::memory_order_release);
    }
    bool snapshot(T& out) const {
        for (int i = 0; i < 3; ++i) {
            auto s1 = seq_.load(std::memory_order_acquire);
            if (s1 & 1ULL) continue;
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

// tracked-object snapshot we publish after retrieveObjects()
struct Trk2D {
    uint64_t id;
    int      label;
    float    conf;
    float    x0, y0, x1, y1;  // pixels
};

struct GpuFrameDesc {
    void*    dev_ptr;
    int64_t  shape[4];
    int64_t  stride_bytes[4];
    uint64_t seq;
    uint64_t ts_ns;
};

struct DetsGpuDesc {
    const void* dev_ptr;   // e.g., float[N,6] = [x0,y0,x1,y1,conf,class_id] in pixels
    int         count;     // N
    uint64_t    seq;
    uint64_t    ts_ns;
    cudaEvent_t ready_event;   // producing stream
};

class CameraModule {
public:
    struct Params {
        int width  = 960;
        int height = 600;
        int fps    = 15;
        sl::RESOLUTION open_resolution = sl::RESOLUTION::SVGA;
    };

    explicit CameraModule(const Params& p = Params()) : p_(p) {}

    // Forbid copying and moving, forces new instance creation
    CameraModule(const CameraModule&) = delete;
    CameraModule& operator=(const CameraModule&) = delete;
    CameraModule(CameraModule&&) = delete;
    CameraModule& operator=(CameraModule&&) = delete;


    void publishDetectionsGpu(const DetsGpuDesc& d) {
        // reuse GpuFrameDesc to store ptr/shape/seq/ts; store event separately
        GpuFrameDesc desc{};
        desc.dev_ptr = const_cast<void*>(d.dev_ptr);
        desc.shape[0] = d.count; desc.shape[1] = 6; desc.shape[2] = 1; desc.shape[3] = 1;
        desc.seq = d.seq; desc.ts_ns = d.ts_ns;
        latest_dets_gpu_.publish(desc);
        // stash the event atomically (simple 1-slot)
        std::atomic_store_explicit(&ready_event_, d.ready_event, std::memory_order_release);
    }
        
    void start(bool from_svo, auto input, bool recording, std::string outputPath) {
        if (running_.exchange(true)) return; // already running

        init(from_svo, input, recording, outputPath);

        worker_ = std::jthread([this](std::stop_token st){
            // Make ZED's CUDA context current on THIS thread once
            CUcontext zed_ctx = zed_.getCUDAContext();
            cuCtxSetCurrent(zed_ctx);

            // Main capture loop
            while (!st.stop_requested() && running_.load(std::memory_order_relaxed)) {
                if (!this->grab()) {
                    // brief backoff on grab failure to avoid hot spin
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                }
                // No sleep on success; ZED FPS naturally paces the loop
            }
        });
    }

    void stop() {
        if (!running_.exchange(false)) return;
        if (worker_.joinable()) {
            worker_.request_stop(); // cooperative stop
            worker_.join();         // wait for the thread to finish
        }
        cleanup();
    }

    ~CameraModule() {
        stop();
    }

    bool is_running() const noexcept { return running_.load(); }
    

    bool snapshotTracked(std::vector<Trk2D>& out) const {
        TrkSnapshot snap;
        if (!latest_trk_.snapshot(snap) || !snap) return false;
        out = *snap;  // copy to caller's vector
        return true;
    }
    cudaStream_t stream() const { return stream_; }

    bool snapshotFrame(GpuFrameDesc& out) const { return latest_.snapshot(out); }

private:
    void*      h_dets_{nullptr};
    size_t     h_dets_bytes_{0};

    std::atomic<cudaEvent_t> ready_event_{nullptr};

    std::jthread           worker_;
    std::atomic<bool>      running_{false};


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
                o.bounding_box_2d = { {x0,y0},{x1,y0},{x1,y1},{x0,y1} };
                objs.push_back(o);
            }
            if (!objs.empty()) zed_.ingestCustomBoxObjects(objs);
        }

        // 2) retrieve tracked objects and publish a tiny CPU list
        sl::Objects objs_out;
        if (zed_.retrieveObjects(objs_out, sl::ObjectDetectionRuntimeParameters{}) ==
            sl::ERROR_CODE::SUCCESS) {
            std::vector<Trk2D> v; v.reserve(objs_out.object_list.size());
            for (const auto& o : objs_out.object_list) {
                const auto& bb = o.bounding_box_2d;
                float x0 = std::min({bb[0].x,bb[1].x,bb[2].x,bb[3].x});
                float y0 = std::min({bb[0].y,bb[1].y,bb[2].y,bb[3].y});
                float x1 = std::max({bb[0].x,bb[1].x,bb[2].x,bb[3].x});
                float y1 = std::max({bb[0].y,bb[1].y,bb[2].y,bb[3].y});
                v.push_back(Trk2D{ o.id, (int)o.label, o.confidence, x0,y0,x1,y1 });
            }
            latest_trk_.publish(std::make_shared<std::vector<Trk2D>>(std::move(v)));
        }
    }

    void init_object_detection() {
        sl::ObjectDetectionParameters detection_parameters;
        detection_parameters.detection_model = sl::OBJECT_DETECTION_MODEL::CUSTOM_BOX_OBJECTS; // Mandatory for this mode
        detection_parameters.enable_tracking = true; // Objects will keep the same ID between frames
        if (detection_parameters.enable_tracking) {
            // Set positional tracking parameters
            sl::PositionalTrackingParameters positional_tracking_parameters;
            // Enable positional tracking
            zed_.enablePositionalTracking(positional_tracking_parameters);
        }
        sl::ERROR_CODE zed_error = zed_.enableObjectDetection(detection_parameters);
        if (zed_error != sl::ERROR_CODE::SUCCESS) {
            std::cout << "enableObjectDetection: " << zed_error << "\nExit program.";
            zed_.close();
            std::exit(-1);
        }
    }

    void open_camera(bool svo, auto input, bool recording, std::string outputPath) {
        sl::InitParameters ip;
        ip.camera_resolution = p_.open_resolution; // sensor mode (enum)
        ip.camera_fps        = p_.fps;
        if (svo) {
            ip.input.setFromSVOFile(input);
            ip.svo_real_time_mode = true;
        } else {
            ip.input.setFromSerialNumber(input);
        }


        if (zed_.open(ip) != sl::ERROR_CODE::SUCCESS) std::abort();
        runtime_ = sl::RuntimeParameters{};
        W_ = p_.width; H_ = p_.height;
        frame_gpu_ = sl::Mat(W_, H_, sl::MAT_TYPE::U8_C4, sl::MEM::GPU );

        init_object_detection();


        if (recording && !svo) {
            sl::RecordingParameters recordingParams;
            recordingParams.compression_mode = sl::SVO_COMPRESSION_MODE::H264;
            recordingParams.video_filename = outputPath;
            zed_.enableRecording(recordingParams);

        }

        // adopt ZED stream (no context switch here)
        CUstream  zed_cus = zed_.getCUDAStream();
        stream_ = reinterpret_cast<cudaStream_t>(zed_cus);

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
    static uint64_t now_ns() {
        using namespace std::chrono;
        return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
    }
    void cleanup() {
        if (d_out_[0]) { cudaFree(d_out_[0]); d_out_[0] = nullptr; }
        if (d_out_[1]) { cudaFree(d_out_[1]); d_out_[1] = nullptr; }
        if (h_dets_) { cudaFreeHost(h_dets_); h_dets_ = nullptr; }
        if (frame_gpu_.isInit()) frame_gpu_.free();
        if (zed_.isOpened()) zed_.close();
    }

     // Grab LEFT → convert → publish (async on stream_). Returns false if grab failed.
    bool grab() {
        // 1) Grab
        if (zed_.grab(runtime_) != sl::ERROR_CODE::SUCCESS) return false;

        // 2) Retrieve directly to GPU at 960x600 on our stream
        if (zed_.retrieveImage(frame_gpu_, sl::VIEW::LEFT, sl::MEM::GPU,
                               sl::Resolution(W_, H_), stream_) != sl::ERROR_CODE::SUCCESS) {
            return false;
        }
        // 3) Source BGRA8 (pitched) on device
        const unsigned char* src = reinterpret_cast<const unsigned char*>(
            frame_gpu_.getPtr<sl::uchar1>(sl::MEM::GPU)
        );
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

    void init(bool from_svo, auto input, bool recording, std::string outputPath) {
        open_camera(from_svo, input, recording, outputPath);
        CUcontext zed_ctx = zed_.getCUDAContext();
        cuCtxSetCurrent(zed_ctx);
        alloc_outputs();
        prep_desc_template();
    }

    // ---- state ----
    Params                p_;
    sl::Camera            zed_;
    sl::RuntimeParameters runtime_;
    sl::Mat               frame_gpu_;

    int                   W_{0}, H_{0};   // set in open_camera()
    cudaStream_t          stream_{nullptr};
    void*                 d_out_[2]{nullptr, nullptr};
    int                   cur_{0};
    uint64_t              seq_{0};

    GpuFrameDesc          desc_template_{};
    LatestOnlyT<GpuFrameDesc> latest_;          // frames (unchanged behavior)
    LatestOnlyT<GpuFrameDesc> latest_dets_gpu_; // YOLO dets descriptor (GPU)
    using TrkSnapshot = std::shared_ptr<const std::vector<Trk2D>>;
    LatestOnlyT<TrkSnapshot> latest_trk_;
};

