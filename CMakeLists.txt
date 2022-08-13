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

set(INFER_PARAM_DIR ${CMAKE_CURRENT_LIST_DIR}/data/inference_param)


SET(SOURCES main.cpp
        src/ch01/eigen_demo.cpp
        src/ch02/perceptron.cpp
        src/ch03/inference.cpp
        src/ch04/nn_study.cpp)

add_executable(dlfs_cpp ${SOURCES})
target_compile_definitions(dlfs_cpp PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")
target_compile_definitions(dlfs_cpp PRIVATE INFER_PARAM_LOCATION="${INFER_PARAM_DIR}/ch03_mnist_weights.json")
