cmake_minimum_required(VERSION 3.22)
project(SentinelCpp LANGUAGES CXX CUDA)

# ---- Build options ----
set(CMAKE_CXX_STANDARD 17)                # drop to 17 if ZED complains
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

# Warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wno-unknown-pragmas)
endif()

# ---- Dependencies ----
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui imgcodecs)
find_package(Threads REQUIRED)

# ZED SDK
find_path(ZED_INCLUDE_DIR
  NAMES sl/Camera.hpp
  HINTS /usr/local/zed/include /usr/local/include /opt/zed/include
)
find_library(ZED_LIBRARY
  NAMES sl_zed
  HINTS /usr/local/zed/lib /usr/local/lib /opt/zed/lib
)
if(NOT ZED_INCLUDE_DIR OR NOT ZED_LIBRARY)
  message(FATAL_ERROR "ZED SDK not found. Need headers in /usr/local/zed/include and libsl_zed.so in /usr/local/zed/lib")
endif()

# ---- Sources ----
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.cpp")
if(SRC_FILES STREQUAL "")
  message(FATAL_ERROR "No .cpp files found under src/.")
endif()

# ---- Executable ----
add_executable(sentinel_app ${SRC_FILES})

# make sure CUDA is enabled for the target
enable_language(CUDA)

# add your .cu files to the target
file(GLOB_RECURSE CUDA_SRC CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.cu")
target_sources(sentinel_app PRIVATE ${CUDA_SRC})

# link cudart (you already do, keep it)
# target_link_libraries(sentinel_app PRIVATE CUDA::cudart)

# optional but fine:
set_target_properties(sentinel_app PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

# Include paths (AFTER add_executable)
target_include_directories(sentinel_app
  PRIVATE
    ${CMAKE_SOURCE_DIR}/src         # so "compile_utils/..." resolves
    ${OpenCV_INCLUDE_DIRS}
)
# Treat ZED headers as system to hush their warnings
target_include_directories(sentinel_app SYSTEM PRIVATE ${ZED_INCLUDE_DIR})


# Link libraries
target_link_libraries(sentinel_app
  PRIVATE
    Threads::Threads
    CUDA::cudart
    CUDA::cuda_driver
    ${ZED_LIBRARY}
    ${OpenCV_LIBS}
)

# Position independent code
set_target_properties(sentinel_app PROPERTIES POSITION_INDEPENDENT_CODE ON)

# RPATH for ZED
set_target_properties(sentinel_app PROPERTIES
  BUILD_RPATH "/usr/local/zed/lib"
  INSTALL_RPATH "/usr/local/zed/lib"
)
