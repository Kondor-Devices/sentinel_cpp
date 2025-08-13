//batch_publisher.hpp
#pragma once
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "stereo_camera/stereo_camera.hpp" // for CameraModule, GpuFrameDesc
#include "visualizer/ipc_common.hpp"

namespace ipc {

inline std::string frames_shm_name() { return "/kdi_frames_batch"; }

static inline uint64_t now_ns() {
    using namespace std::chrono;
    return duration_cast<std::chrono::nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

struct HostPinned {
    void* ptr{nullptr};
    size_t bytes{0};
    ~HostPinned(){ if(ptr) cudaFreeHost(ptr); }
    bool ensure(size_t want){
        if (want <= bytes) return true;
        if (ptr) cudaFreeHost(ptr);
        if (cudaHostAlloc(&ptr, want, cudaHostAllocPortable) != cudaSuccess) return false;
        bytes = want; return true;
    }
};

inline bool create_or_resize_shm(const std::string& name, size_t bytes, int& fd, void*& addr) {
    ::shm_unlink(name.c_str());
    fd = ::shm_open(name.c_str(), O_CREAT|O_RDWR, 0660);
    if (fd < 0) { std::perror("shm_open"); return false; }
    if (::ftruncate(fd, bytes) != 0) { std::perror("ftruncate"); ::close(fd); return false; }
    addr = ::mmap(nullptr, bytes, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) { std::perror("mmap"); ::close(fd); return false; }

    return true;
}

inline std::thread start_frame_publisher(std::vector<std::unique_ptr<CameraModule>>& cams) {
    const int N = static_cast<int>(cams.size());
    const int C = 3;
    const int W = 960;
    const int H = 600;
    const size_t per_cam_bytes = static_cast<size_t>(C)*H*W*sizeof(__half);
    const size_t total_bytes   = sizeof(BatchHeader) + static_cast<size_t>(N)*per_cam_bytes;

    int shm_fd; void* shm_addr;
    if (!create_or_resize_shm(frames_shm_name(), total_bytes, shm_fd, shm_addr)) {
        throw std::runtime_error("Failed to create shm");
    }
    ::close(shm_fd);  // <-- add this after mmap succeeds
    auto* hdr  = reinterpret_cast<BatchHeader*>(shm_addr);
    auto* data = reinterpret_cast<uint8_t*>(shm_addr) + sizeof(BatchHeader);

    hdr->magic = kMagic; hdr->version = kVersion;
    hdr->N = N; hdr->C = C; hdr->H = H; hdr->W = W;
    hdr->ts_ns = 0;
    hdr->seq.store(0, std::memory_order_release);
    hdr->valid_mask = 0;

    std::vector<HostPinned> stages(N);
    for (auto& s : stages) s.ensure(per_cam_bytes);

    return std::thread([hdr, data, N, per_cam_bytes, &cams, stages = std::move(stages)]() mutable {
        while (true) { // you'll gate this with your g_running
            uint64_t s = hdr->seq.load(std::memory_order_acquire);
            if ((s & 1ull) == 0ull) hdr->seq.store(s+1, std::memory_order_release);

            uint64_t mask = 0;

            for (int i=0; i<N; ++i) {
                GpuFrameDesc desc{};
                if (!cams[i]->snapshotFrame(desc) || !desc.dev_ptr) continue;

                cudaStream_t stream = cams[i]->stream();
                cudaMemcpyAsync(stages[i].ptr, desc.dev_ptr, per_cam_bytes, cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                std::memcpy(data + static_cast<size_t>(i)*per_cam_bytes, stages[i].ptr, per_cam_bytes);

                mask |= (1ull << i);
            }

            hdr->ts_ns = now_ns();
            hdr->valid_mask = mask;
            
            s = hdr->seq.load(std::memory_order_acquire);
            if (s & 1ull) hdr->seq.store(s+1, std::memory_order_release);

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });
}


} // namespace ipc
