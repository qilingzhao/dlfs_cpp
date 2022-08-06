cmake_minimum_required(VERSION 3.22)
project(dlfs_cpp)

set(CMAKE_CXX_STANDARD 14)

include_directories(
        "/usr/local/include/eigen3"
        "${PROJECT_SOURCE_DIR}/include"
)

SET(SOURCES main.cpp src/ch01/eigen_demo.cpp)

add_executable(dlfs_cpp ${SOURCES})