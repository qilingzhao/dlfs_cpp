//
// Created by 赵其岭 on 2022/8/13.
//

#include "ch04/nn_study.hpp"

const float delta = 1e-5;

float cross_entropy_error(Eigen::MatrixXf y, Eigen::MatrixXi labels) {
    float sum = 0;
    for (int i = 0; i < y.rows(); i++) {
        float value = y(i, labels(0, i));
        sum += std::log(value + delta); // +delta 防止log(0)这种inf的出现
    }
    return -1.0f * sum / y.rows();
}

void test_cross_entropy_error() {
    Eigen::Matrix<float, 2, 10> y;
    y << 0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0,
            0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0;
    Eigen::Matrix<int, 1, 2> labels;
    labels << 2, 2;
    std::cout << cross_entropy_error(y, labels) << std::endl;

    Eigen::Matrix<float, 1, 10> y1;
    y1<< 0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0;
    Eigen::Matrix<int, 1, 1> labels1;
    labels1 << 2;
    std::cout << cross_entropy_error(y1, labels1) << std::endl;
}


void nn_study() {
    test_cross_entropy_error();
}