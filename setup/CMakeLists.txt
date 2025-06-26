cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(ORB_SLAM2)

set(CMAKE_BUILD_TYPE Release)
message("Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native")
set(CMAKE_CXX_STANDARD 14)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++11" COMPILER_SUPPORTS_CXX11)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    add_definitions(-DCOMPILEDWITHC11)
    message(STATUS "Using flag -std=c++11.")
else()
    message(FATAL_ERROR "Compiler does not support C++11.")
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# OpenCV
find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
    find_package(OpenCV 2.4.3 QUIET)
    if(NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
    endif()
endif()

# Eigen & Pangolin
find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)

# Torch
set(Torch_DIR "/home/lab605/lab605/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

if(Torch_VERSION GREATER 1.0.1)
    add_definitions(-DTORCH_NEW_API)
    message(STATUS "Torch version is newer than v1.0.1, will use new API.")
endif()

# Include dirs
include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

# Output dirs
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/GCN2)

# ORB_SLAM2 Core Library
add_library(${PROJECT_NAME} SHARED
    src/System.cc
    src/Tracking.cc
    src/LocalMapping.cc
    src/LoopClosing.cc
    src/ORBextractor.cc
    src/GCNextractor.cc
    src/ORBmatcher.cc
    src/FrameDrawer.cc
    src/Converter.cc
    src/MapPoint.cc
    src/KeyFrame.cc
    src/Map.cc
    src/MapDrawer.cc
    src/Optimizer.cc
    src/PnPsolver.cc
    src/Frame.cc
    src/KeyFrameDatabase.cc
    src/Sim3Solver.cc
    src/Initializer.cc
    src/Viewer.cc
)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${Pangolin_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
    ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
)

# Executable
add_executable(rgbd_gcn GCN2/rgbd_gcn.cc)
target_link_libraries(rgbd_gcn ${PROJECT_NAME} ${TORCH_LIBRARIES})
set_property(TARGET rgbd_gcn PROPERTY CXX_STANDARD 14)


