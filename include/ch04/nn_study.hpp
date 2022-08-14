//
// Created by 赵其岭 on 2022/8/13.
//

#ifndef DLFS_CPP_NN_STUDY_HPP
#define DLFS_CPP_NN_STUDY_HPP

#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <map>
#include <random>

double cross_entropy_error(Eigen::MatrixXd y, Eigen::MatrixXi labels);

void nn_study();

class TwoLayerModel {
public:
    const int batch_size = 10;
    const int iter_times = 10000;
    const double learn_rate = 0.1f;
    std::map<std::string, Eigen::MatrixXd> network_params;
    Eigen::MatrixXd batch_images;
    Eigen::MatrixXi batch_labels;

    Eigen::MatrixXd w1;
    Eigen::MatrixXd w2;

    Eigen::MatrixXd b1;
    Eigen::MatrixXd b2;


public:
    // 随机选择batch_size个数据 作为一批训练数据
    void random_batch_dataset();
    // 随机选择batch_size个数据, 进行前向推理，返回损失函数
    Eigen::MatrixXd forward();
    double loss(Eigen::MatrixXd y, Eigen::MatrixXi labels);
    TwoLayerModel();
    Eigen::MatrixXd numerical_gradient(Eigen::MatrixXd&);
    double acc();
};
#endif //DLFS_CPP_NN_STUDY_HPP
