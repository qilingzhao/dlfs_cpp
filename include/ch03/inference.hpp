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
Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& input);

// softmax
Eigen::MatrixXd softmax(const Eigen::MatrixXd& input);

// 加载mnist数据集
std::pair<Eigen::MatrixXd, Eigen::MatrixXi> load_mnist(std::string part= "test");

#endif //DLFS_CPP_INFERENCE_HPP
