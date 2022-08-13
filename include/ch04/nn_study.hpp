//
// Created by 赵其岭 on 2022/8/13.
//

#ifndef DLFS_CPP_NN_STUDY_HPP
#define DLFS_CPP_NN_STUDY_HPP

#include <Eigen/Dense>
#include <iostream>
#include <cmath>

float cross_entropy_error(Eigen::MatrixXf y, Eigen::MatrixXi labels);

void nn_study();
#endif //DLFS_CPP_NN_STUDY_HPP
