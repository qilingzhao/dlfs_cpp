//
// Created by 赵其岭 on 2022/8/19.
//


#include "ch05/computational_graph.hpp"

void nn_graph_study() {

}

// AddOp

Eigen::MatrixXf AddOp::forward(const Eigen::MatrixXf& a, const Eigen::MatrixXf& b) const {
    return a + b;
}

Eigen::MatrixXf AddOp::backward(const Eigen::MatrixXf &dOut) const {
    return dOut;
}

// MulOp

Eigen::MatrixXf MulOp::forward(const Eigen::MatrixXf &a, const Eigen::MatrixXf &b) {
    this->a = a;
    this->b = b;
    return a * b;
}


std::pair<Eigen::MatrixXf, Eigen::MatrixXf> MulOp::backward(const Eigen::MatrixXf &dOut) const {
    return std::make_pair(dOut * this->b, dOut * this->a);
}

// ReLUOp

Eigen::MatrixXf ReLUOp::forward(const Eigen::MatrixXf &xx) {
    this->x = xx;
    Eigen::MatrixXf res = (this->x.array() < 0.f).select(0.f, xx);
    return res;
}

Eigen::MatrixXf ReLUOp::backward(const Eigen::MatrixXf &dOut) const {
    Eigen::MatrixXf res = (this->x.array() < 0.f).select(0.f, dOut);
    return res;
}

// SigmoidOp
Eigen::MatrixXf SigmoidOp::forward(const Eigen::MatrixXf &x) {
    this->x = x;
    Eigen::MatrixXf y = 1.0f / (1.0f + (x.array() * (-1.0)).exp());
    this->y = y;
    return y;
}

Eigen::MatrixXf SigmoidOp::backward(const Eigen::MatrixXf &dOut) const {
    return dOut.array() * y.array() * (-1.f * y.array() + 1.f);
}

// AffineOp

Eigen::MatrixXf AffineOp::forward(const Eigen::MatrixXf& w, const Eigen::MatrixXf& b) {
    this->w = w;
    this->b = b;
    Eigen::MatrixXf temp = x * w;
    for (int i = 0; i < temp.rows(); i++) {
        for (int j = 0; j < temp.cols(); j++) {
            temp(i, j) += b(0, j);
        }
    }
    return temp;
}

Eigen::MatrixXf AffineOp::backward(const Eigen::MatrixXf &dOut) {
    this->dW = this->x.transpose() * dOut;
    this->dB = dOut.colwise().sum();
    return dOut * this->w.transpose();
}



Eigen::MatrixXf softmax(const Eigen::MatrixXf& input) {
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> result;
    result.resize(input.rows(), input.cols());

    for (int i = 0; i < input.rows(); i++) {
        float max = input.row(i).maxCoeff();
        float sum = (input.row(i).array() - max).exp().sum();
        for (int j = 0; j < input.cols(); j++) {
            result(i, j) = exp(input(i, j) - max) / sum;
        }
    }
    return result;
}

float cross_entropy_error(Eigen::MatrixXf y, Eigen::MatrixXi labels) {
    float sum = 0;
    for (int i = 0; i < y.rows(); i++) {
        float value = y(i, labels(0, i));
        sum += std::log(value + 0.001f); // +0.001f 防止log(0)这种inf的出现
    }
    return -1.f * sum / float (y.rows());
}

float SoftmaxWithLossOp::forward(const Eigen::MatrixXf& x, const Eigen::MatrixXi& labels) {
    this->t = labels;
    this->y = softmax(x);
    this->loss = cross_entropy_error(this->y, this->t);
    return this->loss;
}

Eigen::MatrixXf SoftmaxWithLossOp::backward() {
    int batch_size = this->t.size();
    Eigen::MatrixXf dX(this->y);
    for (int i = 0; i < dX.rows(); i++) {
        dX(i, this->t(0, i)) = (dX(i, this->t(0, i)) - 1) / batch_size;
    }
    return dX;
}

