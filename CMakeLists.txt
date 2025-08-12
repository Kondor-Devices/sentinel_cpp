cmake_minimum_required(VERSION 3.22)

# Project
project(SentinelCpp LANGUAGES CXX CUDA)

# ---- Build options ----
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default to a sensible build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

# Warnings (tweak to taste)
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
  message(FATAL_ERROR "ZED SDK not found. Ensure headers in /usr/local/zed/include and libsl_zed.so in /usr/local/zed/lib")
endif()

# ---- Sources ----
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.cpp")
if(SRC_FILES STREQUAL "")
  message(FATAL_ERROR "No .cpp files found under src/.")
endif()

# ---- Executable ----
add_executable(sentinel_app ${SRC_FILES})

# Include paths
target_include_directories(sentinel_app
  PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
# Treat ZED headers as system to silence their warnings
target_include_directories(sentinel_app SYSTEM PRIVATE ${ZED_INCLUDE_DIR})

# Undef any stray Vector* macros for C++ compilations (fixes ZED constructor parse errors)
target_compile_options(sentinel_app PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-UVector2>
  $<$<COMPILE_LANGUAGE:CXX>:-UVector3>
  $<$<COMPILE_LANGUAGE:CXX>:-UVector4>
)

# Libraries
target_link_libraries(sentinel_app
  PRIVATE
    Threads::Threads
    CUDA::cudart
    CUDA::cuda_driver      # needed for cuCtxSetCurrent (CUDA Driver API)
    ${ZED_LIBRARY}
    ${OpenCV_LIBS}
)

# Position independent code
set_target_properties(sentinel_app PROPERTIES POSITION_INDEPENDENT_CODE ON)

# RPATH for ZED
set_target_properties(sentinel_app PROPERTIES
  BUILD_RPATH "${CMAKE_INSTALL_RPATH};/usr/local/zed/lib"
  INSTALL_RPATH "/usr/local/zed/lib"
)

# Faster NVCC compilation (optional)
if(CMAKE_CUDA_COMPILER)
  target_compile_options(sentinel_app PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
endif()

# ---- Install (optional) ----
install(TARGETS sentinel_app RUNTIME DESTINATION bin)
