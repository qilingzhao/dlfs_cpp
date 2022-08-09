//
// Created by 赵其岭 on 2022/8/9.
//
#include "ch03/inference.hpp"

std::pair<Eigen::MatrixXf, Eigen::MatrixXi> load_test_mnist() {
    // MNIST_DATA_LOCATION set by MNIST cmake config
    std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;

    // Load MNIST data
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
            mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

//    std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
//    std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
    std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;

    int cnt = 0;
    for (auto it = dataset.test_images[0].begin(); it != dataset.test_images[0].end(); ++it) {
        std::cout << float(*it) / 256 << " ";
        cnt++;
        if (cnt % 28 == 0) {
            std::cout << std::endl;
        }
    }

    Eigen::MatrixXf images;
    images.resize(dataset.test_images.size(), 28 * 28);
    for (int i = 0; i < dataset.test_images.size(); i++) {
        std::vector<uint8_t> image = dataset.test_images[i];
        for (int j = 0; j < 28 * 28; j++) {
            images(i, j) = image[j];
        }
    }
    images /= images.maxCoeff();

    Eigen::MatrixXi labels;
    labels.resize(1, dataset.test_labels.size());
    for (int i = 0; i < dataset.test_labels.size(); i++) {
        labels(0, i) = int(dataset.test_labels[i]);
    }

    std::pair<Eigen::MatrixXf, Eigen::MatrixXi> res(images, labels);
    return res;
}

Eigen::MatrixXf sigmoid(Eigen::MatrixXf input) {
    return 1.0f / (1.0f + (input.array() * (-1.0)).exp());
}


void test_sigmoid() {
    Eigen::Matrix3f mat_3f;
    mat_3f << 0.458, 2, 3, 40, 50, 60, -100, -200, 300;
    Eigen::MatrixXf sm_result = sigmoid(mat_3f);
    std::cout << mat_3f << std::endl;
    std::cout << "---------" << std::endl;
    std::cout << sm_result << std::endl;
}

Eigen::MatrixXf softmax(Eigen::MatrixXf input) {
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

void test_softmax() {
    Eigen::MatrixXf input;
    input.resize(1, 3);
    input << 0.3, 2.9, 4.0;
    std::cout << softmax(input);
}

struct NetworkParam {
    std::vector<std::vector<float> > W1;
    std::vector<std::vector<float> > W2;
    std::vector<std::vector<float> > W3;

    std::vector<float> b1;
    std::vector<float> b2;
    std::vector<float> b3;

//    CONFIGOR_BIND_ALL_REQUIRED(configor::json, NetworkParam, W1, W2, W3, b1, b2, b3);
    CONFIGOR_BIND(configor::json, NetworkParam, REQUIRED(W1, "W1"), REQUIRED(W2, "W2"),REQUIRED(W3, "W3"),
                            REQUIRED(b1, "b1"),REQUIRED(b2, "b2"),REQUIRED(b3, "b3"));
};

std::map<std::string, Eigen::MatrixXf> load_network_param() {
    std::ifstream ifs(INFER_PARAM_LOCATION);
    std::cout << INFER_PARAM_LOCATION << std::endl;
    NetworkParam np;

//    std::cout << "hello world" << std::endl;
    ifs >> configor::json::wrap(np);
//    std::cout << "hello world" << std::endl;

//    std::cout << np.W1.size() << np.b1.size() << std::endl;
//    for (int i = 0; i < np.W1.size(); i++) {
//        for (int j = 0; j < np.W1[i].size(); j++) {
//            std::cout << np.W1[i][j] << " ";
//        }
//        std::cout << std::endl;
//    }

    Eigen::MatrixXf w1 = Eigen::Map<Eigen::MatrixXf>(np.W1.data()->data(), int(np.W1.size()), np.W1.front().size());
    Eigen::MatrixXf w2 = Eigen::Map<Eigen::MatrixXf>(np.W2.data()->data(), int(np.W2.size()), np.W2.front().size());
    Eigen::MatrixXf w3 = Eigen::Map<Eigen::MatrixXf>(np.W3.data()->data(), int(np.W3.size()), np.W3.front().size());
    Eigen::MatrixXf b1 = Eigen::Map<Eigen::MatrixXf>(np.b1.data(), 1, np.b1.size());
    Eigen::MatrixXf b2 = Eigen::Map<Eigen::MatrixXf>(np.b2.data(), 1, np.b2.size());

    std::map<std::string, Eigen::MatrixXf> resultMap;
    resultMap["w1"] = w1;
    resultMap["w2"] = w2;
    resultMap["w3"] = w3;
    resultMap["b1"] = b1;
    resultMap["b2"] = b2;
    return resultMap;
}

void inference() {
    std::map<std::string, Eigen::MatrixXf> network_params = load_network_param();
    for (auto it = network_params.begin(); it != network_params.end(); ++it) {
        std::cout << it->first << " rows: " << it->second.rows() << "  cols: " << it->second.cols() << std::endl;
    }

}