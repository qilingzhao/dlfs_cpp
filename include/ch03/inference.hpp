//
// Created by 赵其岭 on 2022/8/9.
//

#ifndef DLFS_CPP_INFERENCE_HPP
#define DLFS_CPP_INFERENCE_HPP

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <map>
#include "configor/json.hpp"

void inference();

// sigmoid: 1 / (1 + exp(-x))
Eigen::MatrixXf sigmoid(Eigen::MatrixXf input);

// softmax
Eigen::MatrixXf softmax(Eigen::MatrixXf input);


#endif //DLFS_CPP_INFERENCE_HPP
