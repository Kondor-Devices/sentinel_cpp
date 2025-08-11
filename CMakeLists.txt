cmake_minimum_required(VERSION 3.24)
project(zed_opencv_demo LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Jetson arch: Orin (Ampere)=87, Xavier (Volta)=72, Nano (Maxwell)=53
# Set the one you have:
set(CMAKE_CUDA_ARCHITECTURES 87)

list(APPEND CMAKE_PREFIX_PATH "/usr/local/zed")

find_package(ZED 5 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(stereo_kernels STATIC stereo_camera_kernels.cu)
target_include_directories(stereo_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(stereo_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

add_executable(zed_opencv_app main.cpp)
target_include_directories(zed_opencv_app PRIVATE
    ${ZED_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(zed_opencv_app PRIVATE
    stereo_kernels
    ${ZED_LIBRARIES}
    ${OpenCV_LIBS}
    CUDA::cudart
)

# Runtime search paths (Jetson defaults)
set(CMAKE_BUILD_RPATH "/usr/local/zed/lib;/usr/local/cuda/lib64")
set(CMAKE_INSTALL_RPATH "/usr/local/zed/lib;/usr/local/cuda/lib64")
