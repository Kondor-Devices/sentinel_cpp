#include "compile_utils/zed_safe_include.hpp"
#include "stereo_camera/stereo_camera.hpp"
#include "utils/input_sources.hpp"
#include "yolo_manager/yolo_manager.hpp"
#include "visualizer/batch_publisher.hpp"
#include "visualizer/batch_viewer.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

extern char **environ;

static void print_usage(const char* prog) {
    std::cout <<
        "Usage: " << prog << " [--svo] [--svo-dir=<path>] [--record]\n"
        "  --svo                 Use SVO files instead of live feed (default: false)\n"
        "  --svo-dir=<path>      Directory to scan (default: ./svos)\n"
        "  --record              Save recording to SVO file (default: false)\n"
        "  -h, --help            Show this help\n";
}

std::tuple<bool, std::string, bool> parse_args(int argc, char**argv) {
    bool use_svo = false;
    bool record = false;
    std::string svo_dir = "svos";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--svo") {
            use_svo = true;
        } else if (arg.rfind("--svo-dir=",0) == 0) {
            svo_dir = arg.substr(std::string("--svo-dir=").size());
        } else if  (arg == "--record") {
            record = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        }
    };
    return {use_svo, svo_dir, record};
}

std::vector<std::unique_ptr<CameraModule>> start_cameras(const InputSelection& inputs) {
    std::vector<std::unique_ptr<CameraModule>> cams;
    cams.reserve(inputs.from_svo ? inputs.svos.size() : inputs.serials.size());
    if (inputs.from_svo) {
        for (const std::string& svo : inputs.svos) {
            auto cam = std::make_unique<CameraModule>();
            cam->start_from_svo(svo, inputs.record, inputs.svo_dir);
            cams.emplace_back(std::move(cam));
        };
    } else {
        for (const uint64_t& serial : inputs.serials) {
            auto cam =  std::make_unique<CameraModule>();
            cam->start_from_serial(serial, inputs.record, inputs.svo_dir);
            cams.emplace_back(std::move(cam));
        };
    }
    return cams;
}

void yolo_manager_startup(YoloManager& yolo, const std::vector<std::unique_ptr<CameraModule>>& cams) {
    for (auto& cam : cams) {
        yolo.add_camera(cam.get());
    };
    yolo.start();

}

static std::atomic<bool> g_running{true};
static void on_sigint(int){ g_running=false; };


int main(int argc, char** argv) {
    std::signal(SIGINT, on_sigint);

    auto [from_svo, svo_dir, record] = parse_args(argc, argv);
    InputSelection inputs;
    inputs = resolve_inputs(from_svo, svo_dir, record);

    auto cams = start_cameras(inputs);
    if (cams.empty()) {
        std::cerr << "No cameras started.\n";
        return 1;
    }
    
    YoloManager yolo;
    yolo_manager_startup(yolo, cams);

    auto pub_thread = ipc::start_frame_publisher(cams);

    ViewerProcess viewer;
    viewer.start(); 

    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    for (auto& cam : cams) cam->stop();
    yolo.stop();
    return 0;

}
