//
// Created by 赵其岭 on 2022/8/19.
//

#ifndef DLFS_CPP_COMPUTATIONAL_GRAPH_HPP
#define DLFS_CPP_COMPUTATIONAL_GRAPH_HPP

#include <Eigen/Dense>
#include <iostream>
#include <utility>
#include <map>
#include <random>
#include "ch03/inference.hpp"

class Op {

};

class AddOp {
public:
    Eigen::MatrixXf forward(const Eigen::MatrixXf& a, const Eigen::MatrixXf& b) const;
    Eigen::MatrixXf backward(const Eigen::MatrixXf& dOut) const;
};

class MulOp {
private:
    Eigen::MatrixXf a;
    Eigen::MatrixXf b;
public:
    Eigen::MatrixXf forward(const Eigen::MatrixXf& a, const Eigen::MatrixXf& b);
    std::pair<Eigen::MatrixXf, Eigen::MatrixXf> backward(const Eigen::MatrixXf& dOut) const;
};

class ReLUOp {
private:
    Eigen::MatrixXf x;
public:
    Eigen::MatrixXf forward(const Eigen::MatrixXf& x);
    Eigen::MatrixXf backward(const Eigen::MatrixXf& dOut) const;
};

class SigmoidOp {
private:
    Eigen::MatrixXf x;
    Eigen::MatrixXf y;
public:
    Eigen::MatrixXf forward(const Eigen::MatrixXf& x);
    Eigen::MatrixXf backward(const Eigen::MatrixXf& dOut) const;
};

class ThreeLayerNet;

class AffineOp {
private:
    Eigen::MatrixXf x;
    Eigen::MatrixXf w;
    Eigen::MatrixXf b;
    Eigen::MatrixXf dW;
    Eigen::MatrixXf dB;
public:
    AffineOp() {};
    // 之所以Affine需要构造函数，而relu, sigmoid这种不需要
    // 因为在图中流动的是x(图像矩阵), 经过一次正向和反向传播求出dW和dB, 然后对w和b进行微调.
    // 对于relu, 如果有个参数控制x>0时的斜率q, y = x > 0 ? qx : 0; 那么这个q也是需要在初始化时传入的。
    // 其实, AddOp, MulOp 是 AffineOp 的一种特化.
    AffineOp(Eigen::MatrixXf w, Eigen::MatrixXf b) : w(std::move(w)), b(std::move(b)) {};
    Eigen::MatrixXf forward(const Eigen::MatrixXf& x);
    Eigen::MatrixXf backward(const Eigen::MatrixXf& dOut);

    friend class ThreeLayerNet;
};


Eigen::MatrixXf softmax(const Eigen::MatrixXf& input);
float cross_entropy_error(Eigen::MatrixXf y, Eigen::MatrixXi labels);

class SoftmaxWithLossOp {
private:
    Eigen::MatrixXf y;
    Eigen::MatrixXi t;
    float loss;
public:
    float forward(const Eigen::MatrixXf& x, const Eigen::MatrixXi& labels);
    Eigen::MatrixXf backward();
};

class ThreeLayerNet {
private:
    Eigen::MatrixXf batch_images;
    Eigen::MatrixXi batch_labels;
    AffineOp affine1;
    ReLUOp relu1;
    AffineOp affine2;
    ReLUOp relu2;
    AffineOp affine3;
    SoftmaxWithLossOp lastLayer;
    std::pair<Eigen::MatrixXd, Eigen::MatrixXi> mnist_dataset;
public:
    static const int batch_size = 100;
    static const float learn_rate;
    static const float init_weight_std;
    ThreeLayerNet();
    void load_batch_data();
    Eigen::MatrixXf predict();
    float loss(Eigen::MatrixXf);
    void gradient();
    void learn();
    float acc(Eigen::MatrixXf predictOut);
};

void nn_graph_study();

#endif //DLFS_CPP_COMPUTATIONAL_GRAPH_HPP
