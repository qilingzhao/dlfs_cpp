//
// Created by 赵其岭 on 2022/8/19.
//


#include "ch05/computational_graph.hpp"

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
    Eigen::MatrixXf ones = Eigen::MatrixXf::Ones(y.rows(), y.cols());
    Eigen::MatrixXf yy = 1.0f / (1.0f + (x.array() * (-1.0)).exp());
    this->y = yy;
    return yy;
}

Eigen::MatrixXf SigmoidOp::backward(const Eigen::MatrixXf &dOut) const {
    Eigen::MatrixXf ones = Eigen::MatrixXf::Ones(y.rows(), y.cols());
//    std::cout << "dOut size " << dOut.rows() << " " << dOut.cols() << std::endl;
//    std::cout << "y size " << y.rows() << " " << y.cols() << std::endl;
    return dOut *  (-1.f * y + ones) * y; // TODO
}

// AffineOp

Eigen::MatrixXf AffineOp::forward(const Eigen::MatrixXf& x) {
    this->x = x;
    Eigen::MatrixXf temp = x * this->w;
    for (int i = 0; i < temp.rows(); i++) {
        for (int j = 0; j < temp.cols(); j++) {
            temp(i, j) += this->b(0, j);
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
        dX(i, this->t(0, i)) = dX(i, this->t(0, i)) - 1;
        dX.row(i) /= float (batch_size);
    }

    return dX;
}

TwoLayerNet::TwoLayerNet() {
    static std::default_random_engine e(time(0));
    static std::normal_distribution<float> n(0,1);
    const float iws = TwoLayerNet::init_weight_std;
    auto func = [iws](float dummy){return float(n(e)) * iws;};

    Eigen::MatrixXf w1 = Eigen::MatrixXf::Zero(28*28, 50).unaryExpr(func);
    Eigen::MatrixXf b1 = Eigen::MatrixXf::Zero(1, 50);
    this->affine1 = AffineOp(w1, b1);

    this->relu1 = ReLUOp();

    Eigen::MatrixXf w2 = Eigen::MatrixXf::Zero(50, 10).unaryExpr(func);
    Eigen::MatrixXf b2 = Eigen::MatrixXf::Zero(1, 10);
    this->affine2 = AffineOp(w2, b2);

    this->lastLayer = SoftmaxWithLossOp();

    this->mnist_dataset = load_mnist("train");
}

void TwoLayerNet::load_batch_data() {
    Eigen::MatrixXd train_images = this->mnist_dataset.first;
    Eigen::MatrixXi train_labels = this->mnist_dataset.second;

    batch_images.resize(batch_size, train_images.cols());
    batch_labels.resize(1, batch_size);

    std::random_device rd;
    std::default_random_engine e(rd());
    std::uniform_int_distribution<> u(0,batch_size);
    for (int i = 0; i < batch_size; i++) {
        int idx = u(e);
//        std::cout << "chioce idx: " << idx << std::endl;
        batch_images.row(i) = train_images.row(idx).cast<float>();
        batch_labels(0, i) = train_labels(0, idx);
    }
}

Eigen::MatrixXf TwoLayerNet::predict() {
    Eigen::MatrixXf infer_data = this->batch_images;
    Eigen::MatrixXf af_1 = this->affine1.forward(infer_data);
    Eigen::MatrixXf re_1 = this->relu1.forward(af_1);

    Eigen::MatrixXf af_2 = this->affine2.forward(re_1);

    return af_2;
}

float TwoLayerNet::loss(Eigen::MatrixXf res) {
    float lo = this->lastLayer.forward(res, this->batch_labels);
    return lo;
}

void TwoLayerNet::gradient() {
    Eigen::MatrixXf dOut1 = this->lastLayer.backward();
    Eigen::MatrixXf dOut2 = this->affine2.backward(dOut1);
    Eigen::MatrixXf dOut3 = this->relu1.backward(dOut2);
    Eigen::MatrixXf dOut4 = this->affine1.backward(dOut3);
}

void TwoLayerNet::learn() {
    this->affine1.w -= this->affine1.dW * TwoLayerNet::learn_rate;
    this->affine1.b -= this->affine1.dB * TwoLayerNet::learn_rate;

    this->affine2.w -= this->affine2.dW * TwoLayerNet::learn_rate;
    this->affine2.b -= this->affine2.dB * TwoLayerNet::learn_rate;
}

float TwoLayerNet::acc(Eigen::MatrixXf predictOut) {
    int succ = 0;
    for (int row = 0; row < predictOut.rows(); row++) {
        Eigen::Index col_num;
        predictOut.row(row).maxCoeff(&col_num);
        if (col_num == this->batch_labels(0, row)) {
            succ++;
        }
    }
    return float(succ) / float(predictOut.rows());
}


const float TwoLayerNet::learn_rate = 0.01f;
const float TwoLayerNet::init_weight_std = 0.1f;

void nn_graph_study() {
    TwoLayerNet net;
//    net.load_batch_data();

    std::vector<float> accs;
    std::vector<float> losses;
    for (int i = 0; i < TwoLayerNet::iter_times; i++) {
        std::cout << "iter cnt: " << i << std::endl;
        net.load_batch_data();
        Eigen::MatrixXf predictOut = net.predict();
        float acc = net.acc(predictOut);
        std::cout << "acc: " << acc << std::endl;
        float loss = net.loss(predictOut);
        std::cout << "loss: " << loss << std::endl;
        net.gradient();
//        std::cout << "finish grad()" << std::endl;
        net.learn();
        accs.push_back(acc);
        losses.push_back(loss);
    }
    std::ofstream ofs1("./accs.json");
    configor::json accs_json = accs;
    ofs1 << accs_json << std::endl;

    std::ofstream ofs2("./losses.json");
    configor::json losses_json = losses;
    ofs2 << losses_json << std::endl;
}