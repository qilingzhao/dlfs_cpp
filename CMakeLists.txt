cmake_minimum_required(VERSION 3.22)
project(dlfs_cpp)

set(CMAKE_CXX_STANDARD 14)

include_directories(
        "/usr/local/include/eigen3"
        "${PROJECT_SOURCE_DIR}/include"
)

set(MNIST_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/include)
set(MNIST_DATA_DIR ${CMAKE_CURRENT_LIST_DIR}/data/mnist)
set(MNIST_FOUND TRUE)

SET(SOURCES main.cpp
        src/ch01/eigen_demo.cpp
        src/ch02/perceptron.cpp
        src/ch03/inference.cpp)

add_executable(dlfs_cpp ${SOURCES})
target_compile_definitions(dlfs_cpp PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")
