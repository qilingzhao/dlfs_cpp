//
// Created by 赵其岭 on 2022/8/13.
//

#include "ch04/nn_study.hpp"
#include "ch03/inference.hpp"

const double delta = 1e-4;

double cross_entropy_error(Eigen::MatrixXd y, Eigen::MatrixXi labels) {
    double sum = 0;
    for (int i = 0; i < y.rows(); i++) {
        double value = y(i, labels(0, i));
        sum += std::log(value + 0.001f); // +0.001f 防止log(0)这种inf的出现
    }
    return -1.0f * sum / double(y.rows());
}

void test_cross_entropy_error() {
    Eigen::Matrix<double, 2, 10> y;
    y << 0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0,
            0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0;
    Eigen::Matrix<int, 1, 2> labels;
    labels << 2, 2;
    std::cout << cross_entropy_error(y, labels) << std::endl;

    Eigen::Matrix<double, 1, 10> y1;
    y1<< 0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0;
    Eigen::Matrix<int, 1, 1> labels1;
    labels1 << 2;
    std::cout << cross_entropy_error(y1, labels1) << std::endl;
}

double TwoLayerModel::loss(Eigen::MatrixXd y, Eigen::MatrixXi labels) {
    return cross_entropy_error(y, labels);
}

TwoLayerModel::TwoLayerModel() {
    static std::default_random_engine e(time(0));
    static std::normal_distribution<double> n(0,2);
    auto func = [](double dummy){return n(e);};
    w1 = Eigen::MatrixXd::Zero(28*28, 50).unaryExpr(func);
    w2 = Eigen::MatrixXd::Zero(50, 10).unaryExpr(func);
    b1 = Eigen::MatrixXd::Zero(1, 50);
    b2 = Eigen::MatrixXd::Zero(1, 10);
    network_params["w1"] = w1;
    network_params["w2"] = w2;
    network_params["b1"] = b1;
    network_params["b2"] = b2;
}

void TwoLayerModel::random_batch_dataset() {
    auto train_pair = load_mnist("train");
    Eigen::MatrixXd train_images = train_pair.first;
    Eigen::MatrixXi train_labels = train_pair.second;

    batch_images.resize(batch_size, train_images.cols());
    batch_labels.resize(1, batch_size);

    std::random_device rd;
    std::default_random_engine e(rd());
    std::uniform_int_distribution<> u(0,batch_size);
    for (int i = 0; i < batch_size; i++) {
        int idx = u(e);
//        std::cout << "chioce idx: " << idx << std::endl;
        batch_images.row(i) = train_images.row(i);
        batch_labels(0, i) = train_labels(0, idx);
    }
}
Eigen::MatrixXd TwoLayerModel::forward() {

    Eigen::MatrixXd  infer_data = batch_images;
    for (int layer_num = 1; layer_num <= 2; layer_num++) {
        std::string weight_name("w" + std::to_string(layer_num));
        std::string bias_name("b" + std::to_string(layer_num));
//        std::cout << "infer_data before is " << infer_data << std::endl;
        infer_data = infer_data * network_params[weight_name];
//        std::cout << "infer_data after is " << infer_data << std::endl;
        for (int row = 0; row < infer_data.rows(); row++) {
            infer_data.row(row) += network_params[bias_name];
        }
        if (layer_num == 2) {
            infer_data = softmax(infer_data);
        } else {
            infer_data = sigmoid(infer_data);
        }
    }
    return infer_data;
}

void test_forward() {
    TwoLayerModel model;
    model.random_batch_dataset();
    model.batch_images(0, 0) = 0.9;
    Eigen::MatrixXd& mat = model.network_params["w1"];
    int i = 0; int j = 0;
    double origin_value = mat(i, j);
    std::cout << "origin value: " << origin_value << std::endl;
    mat(i, j) = origin_value - delta;
//    std::cout << "w1(0,0) minus value: " << model.network_params["w1"] << std::endl;

    Eigen::MatrixXd  last_layer = model.forward();
//    std::cout << "last_layer: " << last_layer << std::endl;
    double h1 = model.loss(last_layer, model.batch_labels);

    mat(i, j) = origin_value + delta;
//    std::cout << "w1(0,0) add value: " << model.network_params["w1"] << std::endl;
    last_layer = model.forward();
//    std::cout << "last_layer2: " << last_layer << std::endl;
    double h2 = model.loss(last_layer, model.batch_labels);

    double diff = (h2 - h1) / (2 * delta);
    std::cout << "mat_grad(" << i << ", " << j << "): " << diff << " h1: " << h1 << " h2: " << h2 << std::endl;
    mat(i, j) = origin_value;
}

Eigen::MatrixXd TwoLayerModel::numerical_gradient(Eigen::MatrixXd& mat) {
    Eigen::MatrixXd mat_grad = Eigen::MatrixXd::Zero(mat.rows(), mat.cols());
    for (int i = 0; i < mat.rows(); i++) {
        for (int j = 0; j < mat.cols(); j++) {
            double origin_value = mat(i, j);
            mat(i, j) = origin_value - delta;
            double h1 = loss(forward(), batch_labels);

            mat(i, j) = origin_value + delta;
            double h2 = loss(forward(), batch_labels);

            double diff = (h2 - h1) / (2 * delta);
            mat_grad(i, j) = diff;
//            std::cout << "mat_grad(" << i << ", " << j << "): " << diff << " h1: " << h1 << " h2: " << h2 << std::endl;
            mat(i, j) = origin_value;
        }
    }
    return mat_grad;
}

double TwoLayerModel::acc() {
    random_batch_dataset();
    Eigen::MatrixXd res_mat = forward();
    int succ = 0;
    for (int row = 0; row < res_mat.rows(); row++) {
        Eigen::Index col_num;
        res_mat.row(row).maxCoeff(&col_num);
        if (col_num == batch_labels(0, row)) {
            succ++;
        }
    }
    std::cout << "succ num: " << succ << std::endl;
    return double(succ) /  double(batch_size);
}

void nn_study() {
//    test_forward();
//    return;

    TwoLayerModel model;
    for (int iter_cnt = 0; iter_cnt < model.iter_times; ++iter_cnt) {
        std::cout << "iter_cnt: " << iter_cnt << std::endl;
        model.random_batch_dataset();
        std::cout << "finish rand" << std::endl;
        for (int layer_num = 1; layer_num <= 2; layer_num++) {
            std::string weight_name("w" + std::to_string(layer_num));
            std::string bias_name("b" + std::to_string(layer_num));

            Eigen::MatrixXd weight_mat_grad = model.numerical_gradient(model.network_params[weight_name]);
            model.network_params[weight_name + "_grad"] = weight_mat_grad;

            Eigen::MatrixXd bias_mat_grad = model.numerical_gradient(model.network_params[bias_name]);
            model.network_params[bias_name + "_grad"] = bias_mat_grad;
        }

        std::cout << "finish numerical_gradient" << std::endl;

        for (int layer_num = 1; layer_num <= 2; layer_num++) {
            std::string weight_name("w" + std::to_string(layer_num));
            std::string bias_name("b" + std::to_string(layer_num));

            std::string w_grad_name = weight_name + "_grad";
            std::string b_grad_name = bias_name + "_grad";

            model.network_params[weight_name].cwiseProduct(model.network_params[w_grad_name] * model.learn_rate * (-1.0));
            model.network_params[bias_name].cwiseProduct(model.network_params[b_grad_name] * model.learn_rate * (-1.0));
        }

        std::cout << "acc: " << model.acc() << std::endl;
    }
}