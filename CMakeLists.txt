cmake_minimum_required(VERSION 3.8.3)
project(viocali_estimator)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(OpenCV REQUIRED)

message("OpenCV_VERSION: ${OpenCV_VERSION}")

find_package(Ceres REQUIRED)

find_package(Sophus REQUIRED)

find_package(FMT REQUIRED)

find_package(Eigen3)
include_directories(
  include
  ${CERES_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

add_executable(viocali_estimator
        viocali_imu_camera_node.cpp
        src/viocali_calibrate.cpp
        src/parameters.cpp
        src/calibrate_local_parameterization.cpp
        src/viocali_camera_factor.cpp
        src/viocali_imu_factor.cpp
        src/viocali_lidar_factor.cpp
        src/utility.cpp
        )

add_executable(liocali_estimator
        viocali_lidar_node.cpp
        src/viocali_calibrate.cpp
        src/parameters.cpp
        src/calibrate_local_parameterization.cpp
        src/viocali_camera_factor.cpp
        src/viocali_imu_factor.cpp
        src/viocali_lidar_factor.cpp
        src/utility.cpp
        )

target_link_libraries(viocali_estimator ${OpenCV_LIBS} ${CERES_LIBRARIES} fmt::fmt)

target_link_libraries(liocali_estimator ${OpenCV_LIBS} ${CERES_LIBRARIES} fmt::fmt)




