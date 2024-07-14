cmake_minimum_required(VERSION 2.8.3)
project(3d-mot-cpp)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
# find_package(Eigen3 REQUIRED)
# find_package(yaml-cpp REQUIRED)

include_directories(
        ${PROJECT_SOURCE_DIR}/include
        # ${EIGEN3_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIR}
        # /usr/local/include/yaml-cpp
)

add_executable(3d-mot-cpp
        main.cpp
        ${PROJECT_SOURCE_DIR}/src/kitti_dataset.cpp
        ${PROJECT_SOURCE_DIR}/src/tracker.cpp
        ${PROJECT_SOURCE_DIR}/src/trajectory.cpp
        ${PROJECT_SOURCE_DIR}/src/utils.cpp)


target_link_libraries(3d-mot-cpp ${OpenCV_LIBS}
        # /usr/local/lib/libyaml-cpp.so
        #stdc++fs
	)