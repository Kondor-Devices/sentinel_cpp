#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

static cv::Mat slMat2cvMat(sl::Mat &im) {
    // ZED returns BGRA by default (CV_8UC4)
    return cv::Mat((int)im.getHeight(), (int)im.getWidth(), CV_8UC4,
                   im.getPtr<sl::uchar1>(sl::MEM::CPU));
}

int main() {
    sl::Camera zed;

    sl::InitParameters init_params;
    init_params.camera_resolution = sl::RESOLUTION::SVGA; // fits Jetson nicely
    init_params.camera_fps = 30;
    init_params.depth_mode = sl::DEPTH_MODE::NONE;         // we only display RGB
    init_params.coordinate_units = sl::UNIT::METER;

    sl::ERROR_CODE err = zed.open(init_params);
    if (err != sl::ERROR_CODE::SUCCESS) {
        std::cerr << "ZED open error: " << sl::toString(err) << std::endl;
        return 1;
    }

    sl::RuntimeParameters runtime_params;
    sl::Mat zed_frame;

    std::cout << "Press 'q' to quit." << std::endl;
    while (true) {
        if (zed.grab(runtime_params) == sl::ERROR_CODE::SUCCESS) {
            // Get left image in CPU mem for OpenCV
            zed.retrieveImage(zed_frame, sl::VIEW::LEFT, sl::MEM::CPU);
            cv::Mat cv_frame = slMat2cvMat(zed_frame); // no-copy view

            // Optional: convert BGRA->BGR for OpenCV convenience
            cv::Mat bgr;
            cv::cvtColor(cv_frame, bgr, cv::COLOR_BGRA2BGR);

            cv::imshow("ZED Left", bgr);
            int key = cv::waitKey(1);
            if (key == 'q' || key == 27) break;
        } else {
            // small sleep to avoid busy loop on grab failure
            sl::sleep_ms(1);
        }
    }

    zed.close();
    return 0;
}
