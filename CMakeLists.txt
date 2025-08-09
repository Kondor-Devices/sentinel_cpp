cmake_minimum_required(VERSION 3.18)
project(zed_opencv_demo CXX)
set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_PREFIX_PATH "/usr/local/zed")

find_package(ZED 5 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)   # ← add this

add_executable(zed_opencv_app main.cpp)

target_include_directories(zed_opencv_app PRIVATE
    ${ZED_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}      # ← add this (/usr/local/cuda/include)
)

target_link_libraries(zed_opencv_app PRIVATE
    ${ZED_LIBRARIES}
    ${OpenCV_LIBS}
    CUDA::cudart                     # ← safe to link cudart
)

set(CMAKE_BUILD_RPATH "/usr/local/zed/lib;/usr/local/cuda/lib64")
set(CMAKE_INSTALL_RPATH "/usr/local/zed/lib;/usr/local/cuda/lib64")
