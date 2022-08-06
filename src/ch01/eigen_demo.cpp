//
// Created by 赵其岭 on 2022/8/6.
//
#include "ch01/eigen_demo.hpp"

void test_eigen() {
    Eigen::MatrixXd mat(3, 3);
    mat(0, 1) = 10.1;
    std::cout << mat << std::endl;
}